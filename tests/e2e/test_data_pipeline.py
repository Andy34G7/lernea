"""
Data pipeline: disjoint de-duplicated splits, cache keying, atomic writes, provenance,
streaming failures and SIGMORPHON parsing. No network: real sources are faked by
monkeypatching ``pipeline._iter_hf_rows``; see test_data_network.py for the live run.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import subprocess
import sys

import pytest

from tokenisers.data import (
    DataPipeline,
    DeterministicSplitter,
    IncompleteDatasetError,
    SigmorphonLoader,
)
from tokenisers.data import pipeline as pl

SPLITS = pl.SPLITS


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _assert_disjoint_and_deduplicated(ds) -> None:
    texts = {s: ds.texts(s) for s in SPLITS}
    for s in SPLITS:
        assert texts[s], f"{s} split is empty"
        assert len(texts[s]) == len(set(texts[s])), f"duplicate documents inside {s}"
        assert [d["sha256"] for d in ds.docs(s)] == [_sha(t) for t in texts[s]]
    for a, b in itertools.combinations(SPLITS, 2):
        assert not set(texts[a]) & set(texts[b]), f"{a} and {b} share documents"


def test_offline_dataset_is_disjoint_deduplicated_and_labelled_synthetic(tmp_path):
    ds = DataPipeline(cache_dir=tmp_path, force_offline=True).prepare(train_bytes=50_000)
    _assert_disjoint_and_deduplicated(ds)

    m = ds.manifest
    assert ds.synthetic and m["synthetic"] is True and ds.root.name.startswith("synthetic-")
    assert {d["source"] for s in SPLITS for d in ds.docs(s)} == {"synthetic"}
    assert m["splits"]["train"]["text_bytes"] >= 50_000
    assert {d["domain"] for d in ds.docs("train")} == {"english", "multilingual", "code"}
    for lang in pl.FINEWEB2_LANGUAGES:  # every language has held-out text
        assert ds.texts("test", lang=lang), lang
    assert m["complete"] and m["errors"] == [] and m["fallbacks"] == []


def test_split_depends_only_on_seed_and_text_across_processes(tmp_path):
    a = DataPipeline(cache_dir=tmp_path / "a", force_offline=True).prepare(train_bytes=30_000)
    script = (
        "import json,sys; from tokenisers.data import DataPipeline; "
        "ds = DataPipeline(cache_dir=sys.argv[1], force_offline=True).prepare(train_bytes=30_000); "
        "print(json.dumps({s: v['sha256'] for s, v in ds.manifest['splits'].items()}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "b")],
        capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": "12345"},
    ).stdout
    assert json.loads(out) == {s: v["sha256"] for s, v in a.manifest["splits"].items()}

    other_seed = DataPipeline(cache_dir=tmp_path / "a", force_offline=True).prepare(train_bytes=30_000, seed=1)
    assert other_seed.root != a.root
    assert set(other_seed.texts("val")) != set(a.texts("val"))

    splitter = DeterministicSplitter(seed=7)
    counts = {s: 0 for s in SPLITS}
    for i in range(4000):
        counts[splitter.assign_split(f"doc {i}")] += 1
    assert 120 < counts["val"] < 280 and 120 < counts["test"] < 280  # ~5% each


def test_cache_key_covers_size_and_seed(tmp_path):
    """D1: a cached 20KB corpus must not be returned for a 200KB request."""
    pipeline = DataPipeline(cache_dir=tmp_path, force_offline=True)
    small = pipeline.prepare(train_bytes=20_000)
    big = pipeline.prepare(train_bytes=200_000)
    assert big.root != small.root
    assert big.manifest["splits"]["train"]["text_bytes"] >= 200_000
    assert pipeline.sample_preset("smoke", max_bytes=20_000) == small.texts("train")
    assert pipeline.sample_preset("smoke", max_bytes=20_000, split="val") == small.texts("val")
    with pytest.raises(ValueError):
        pipeline.sample_preset("no-such-preset")


def test_corrupted_cache_file_is_detected_and_rebuilt(tmp_path):
    pipeline = DataPipeline(cache_dir=tmp_path, force_offline=True)
    ds = pipeline.prepare(train_bytes=20_000)
    good = ds.path("val").read_bytes()
    ds.path("val").write_text("CORRUPTED", encoding="utf-8")
    again = pipeline.prepare(train_bytes=20_000)
    assert again.root == ds.root
    assert again.path("val").read_bytes() == good


def test_failed_write_leaves_no_half_written_dataset(tmp_path, monkeypatch):
    pipeline = DataPipeline(cache_dir=tmp_path, force_offline=True)
    real_sha = pl._sha256_file
    calls = []

    def disk_full_on_second_file(path):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("No space left on device")
        return real_sha(path)

    monkeypatch.setattr(pl, "_sha256_file", disk_full_on_second_file)
    with pytest.raises(OSError):
        pipeline.prepare(train_bytes=20_000)
    assert list(tmp_path.iterdir()) == []

    monkeypatch.undo()
    assert pipeline.prepare(train_bytes=20_000).manifest["complete"]


def _gated_error():
    """The error huggingface_hub raises for a gated repo without access (built with its own HTTP lib)."""
    from huggingface_hub import errors

    http = sys.modules[errors.Response.__module__.split(".")[0]]
    request = http.Request("GET", "https://huggingface.co/datasets/bigcode/starcoderdata")
    return errors.GatedRepoError("Cannot access gated repo", response=http.Response(401, request=request))


SHARED_DOC = "This exact document is published by several sources and must be kept once."

# Row labels of the mixed code fallback in github-code-clean's proportions (train-00001-of-00880,
# per 1000 rows of these six languages: java 172, javascript 97, cpp 67, python 64, go 19, rust 3).
GITHUB_CODE_MIX = [
    lang for i in range(1000) for lang, n in
    (("java", 172), ("javascript", 97), ("cpp", 67), ("python", 64), ("go", 19), ("rust", 3))
    if i < n
]


def _fake_hub(src):
    """Stand-in for the HF reader: endless unique docs, plus duplicates; starcoder is gated."""
    if src.repo == "bigcode/starcoderdata":
        raise _gated_error()
    labels = GITHUB_CODE_MIX if src.lang == "mixed" else [src.lang]

    def rows():
        yield SHARED_DOC, labels[0]
        yield SHARED_DOC, labels[0]
        for i in itertools.count():
            lang = labels[i % len(labels)]
            yield f"{src.repo} {lang} document {i}: " + "lorem ipsum " * 20, lang

    return rows()


def test_real_path_dedups_records_provenance_and_falls_back_from_gated_code(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "_iter_hf_rows", _fake_hub)
    ds = DataPipeline(cache_dir=tmp_path).prepare(train_bytes=40_000, languages=("cmn_Hani", "deu_Latn"))
    m = ds.manifest
    _assert_disjoint_and_deduplicated(ds)

    assert not ds.synthetic and ds.root.name.startswith("hf-") and m["complete"]
    every_doc = [d for s in SPLITS for d in ds.docs(s)]
    assert [d["text"] for d in every_doc].count(SHARED_DOC) == 1
    assert sum(r["duplicates_skipped"] for r in m["sources"]) >= 4
    assert "synthetic" not in {d["source"] for d in every_doc}

    assert len(m["fallbacks"]) == 1
    fb = m["fallbacks"][0]
    assert (fb["domain"], fb["wanted"], fb["used"]) == ("code", "bigcode/starcoderdata", "codeparrot/github-code-clean")
    assert "gated" in fb["reason"]
    assert {d["source"] for d in every_doc if d["domain"] == "code"} == {"codeparrot/github-code-clean"}
    assert all(r["revision"] == pl.REVISIONS[r["repo"]] for r in m["sources"])

    # The mixed fallback keeps starcoder's per-language budgets: rare languages (rust is 0.7% of
    # the rows) still fill their share of train and get held-out val/test documents.
    fallback = {r["lang"]: r for r in m["sources"] if r["repo"] == "codeparrot/github-code-clean"}
    assert sorted(fallback) == sorted(pl.CODE_LANGUAGES)
    starcoder = next(r for r in m["sources"] if r["repo"] == "bigcode/starcoderdata")
    for lang, rec in fallback.items():
        assert rec["budget_bytes"] == starcoder["budget_bytes"]
        assert all(rec["bytes"][s] >= rec["budget_bytes"][s] for s in SPLITS), (lang, rec["bytes"])
        for s in SPLITS:
            assert ds.texts(s, domain="code", lang=lang), f"no {lang} code in {s}"
    assert (ds.root / "manifest.json").is_file()
    assert json.loads((ds.root / "manifest.json").read_text(encoding="utf-8")) == m


def test_stream_failure_is_logged_raised_and_partial_data_kept(tmp_path, monkeypatch, caplog):
    """D3: a dropped connection must not be silently replaced by synthetic text."""

    def flaky_hub(src):
        def rows():
            n = 50 if src.lang == "cmn_Hani" else None
            for i in itertools.count():
                if n is not None and i == n:
                    raise ConnectionError("connection reset by peer")
                lang = pl.CODE_LANGUAGES[i % len(pl.CODE_LANGUAGES)] if src.lang == "mixed" else src.lang
                yield f"{src.lang} document {i} " * 10, lang

        return rows()

    monkeypatch.setattr(pl, "_iter_hf_rows", flaky_hub)
    pipeline = DataPipeline(cache_dir=tmp_path)
    with caplog.at_level(logging.WARNING, logger="tokenisers.data.pipeline"):
        with pytest.raises(IncompleteDatasetError) as info:
            pipeline.prepare(train_bytes=200_000, languages=("cmn_Hani",))
    assert "connection reset by peer" in caplog.text

    partial = info.value.dataset
    m = partial.manifest
    assert m["complete"] is False
    assert [(e["lang"], e["repo"]) for e in m["errors"]] == [("cmn_Hani", "HuggingFaceFW/fineweb-2")]
    assert "connection reset by peer" in m["errors"][0]["error"]
    assert partial.texts("train", lang="cmn_Hani"), "rows read before the failure must be kept"
    assert "synthetic" not in {d["source"] for s in SPLITS for d in partial.docs(s)}

    kept = pipeline.prepare(train_bytes=200_000, languages=("cmn_Hani",), allow_partial=True)
    assert kept.manifest["complete"] is False and kept.texts("train", lang="cmn_Hani")


def _assert_pool(pipeline, base, total_bytes):
    """The LM text pool leaves base's splits as they were and shares no document with them."""
    splits = {s: v["sha256"] for s, v in base.manifest["splits"].items()}
    pool, prov = pipeline.lm_pool(base, total_bytes)
    cfg = base.manifest["config"]
    again = pipeline.prepare(train_bytes=cfg["train_bytes"], languages=cfg["languages"])
    assert again.root == base.root and {s: v["sha256"] for s, v in again.manifest["splits"].items()} == splits
    held = {d["sha256"] for s in SPLITS for d in base.docs(s)}
    assert pool and not {_sha(t) for t in pool} & held and len(set(pool)) == len(pool)
    lm_bytes = sum(len(t.encode()) for t in base.texts("train") + pool)
    assert lm_bytes >= total_bytes and prov["text_bytes"] == lm_bytes - base.manifest["splits"]["train"]["text_bytes"]
    assert prov["base"] == base.root.name and prov["dataset"]["config"]["train_bytes"] == total_bytes
    assert prov["dataset"]["root"] != base.root.name and prov["docs"] == len(pool)
    return pool, prov


def test_lm_pool_is_disjoint_and_leaves_the_default_splits_unchanged(tmp_path):
    pipeline = DataPipeline(cache_dir=tmp_path, force_offline=True)
    base = pipeline.prepare(train_bytes=20_000)
    pool, prov = _assert_pool(pipeline, base, 60_000)
    assert {d["domain"] for d in prov["dataset"]["sources"]} == {"english", "multilingual", "code"}
    assert prov["excluded_base_docs"] > 0  # base train documents are in the larger train split too
    with pytest.raises(ValueError, match="must exceed"):
        pipeline.lm_pool(base, 20_000)
    with pytest.raises(ValueError, match="same sources"):  # a pool must not come from other sources
        DataPipeline(cache_dir=tmp_path, force_offline=False).lm_pool(base, 60_000)


def test_lm_pool_on_the_real_path_dedups_against_every_split(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "_iter_hf_rows", _fake_hub)
    pipeline = DataPipeline(cache_dir=tmp_path)
    base = pipeline.prepare(train_bytes=40_000, languages=("cmn_Hani", "deu_Latn"))
    pool, prov = _assert_pool(pipeline, base, 120_000)
    assert SHARED_DOC not in pool and prov["dataset"]["fallbacks"] == base.manifest["fallbacks"]


def test_sigmorphon_tsv_marks_canonical_segmentations(tmp_path):
    tsv = tmp_path / "x.tsv"
    tsv.write_text(
        "undiscounted\tun @@discount @@ed\t110\n"
        "subsidised\tsubside @@y @@ise @@ed\t110\n"
        "szépség\tszép @@ség\t100\n"
        "malformed_row_single_col\n\n",
        encoding="utf-8",
    )
    entries = SigmorphonLoader.load_tsv(tsv)
    assert [e["word"] for e in entries] == ["undiscounted", "subsidised", "szépség"]
    assert entries[0]["surface"] and entries[0]["gold_cuts"] == {2, 10}
    assert entries[0]["segments"] == ["un", "discount", "ed"] and entries[0]["category"] == "110"
    assert not entries[1]["surface"] and entries[1]["gold_cuts"] is None
    assert entries[2]["gold_cuts"] == {len("szép".encode("utf-8"))}


def test_sigmorphon_loader_reads_cache_offline_and_never_invents_gold(tmp_path):
    loader = SigmorphonLoader(cache_dir=tmp_path, offline=True)
    with pytest.raises(FileNotFoundError):
        loader.load("eng", "test")
    assert loader.url("eng", "test").endswith(f"{pl.SIGMORPHON_COMMIT}/data/eng.word.test.gold.tsv")
    assert loader.url("mon", "train").endswith("/data/surprise/mon.word.train.tsv")
    with pytest.raises(ValueError):
        loader.url("xx", "test")

    loader.cache_dir.mkdir(parents=True)
    (loader.cache_dir / "eng.word.test.gold.tsv").write_text(
        "undiscounted\tun @@discount @@ed\t110\nsubsidised\tsubside @@y @@ise @@ed\t110\n", encoding="utf-8"
    )
    gold = loader.load("eng", "test")
    assert [(e["word"], e["lang"], e["split"]) for e in gold] == [("undiscounted", "eng", "test")]
    assert len(loader.load("eng", "test", surface_only=False)) == 2


def test_offline_morpheme_fixture_segments_join_to_their_words():
    fixture = SigmorphonLoader.load_offline_fixture()
    assert len(fixture) == 12
    for e in fixture:
        assert "".join(e["segments"]) == e["word"], e["word"]
