"""
Live-data checks (download from Hugging Face and GitHub). Skipped unless
TOKENISERS_NETWORK_TESTS=1, so the default suite never touches the network.
"""

from __future__ import annotations

import os

import pytest

from tokenisers.data import DataPipeline, SigmorphonLoader
from tokenisers.data import pipeline as pl

pytestmark = pytest.mark.skipif(
    os.environ.get("TOKENISERS_NETWORK_TESTS") != "1",
    reason="set TOKENISERS_NETWORK_TESTS=1 to run tests that download real data",
)


def test_real_dataset_sources_split_and_provenance(tmp_path):
    ds = DataPipeline(cache_dir=tmp_path).prepare(train_bytes=1_000_000)
    m = ds.manifest
    assert m["complete"] and not m["synthetic"], m["errors"]

    docs = {s: ds.docs(s) for s in pl.SPLITS}
    texts = {s: {d["text"] for d in docs[s]} for s in pl.SPLITS}
    assert not texts["train"] & texts["val"] and not texts["train"] & texts["test"] and not texts["val"] & texts["test"]

    sources = {d["source"] for s in pl.SPLITS for d in docs[s]}
    assert {"HuggingFaceFW/fineweb-edu", "HuggingFaceFW/fineweb-2"} <= sources
    if pl._hf_token_present() and not m["fallbacks"]:
        assert "bigcode/starcoderdata" in sources
    else:
        assert [fb["used"] for fb in m["fallbacks"]] == ["codeparrot/github-code-clean"]
        assert "codeparrot/github-code-clean" in sources
    for lang in pl.FINEWEB2_LANGUAGES:
        assert ds.texts("test", lang=lang), f"no held-out {lang} text"
    # Every code language, rare ones included (rust is ~0.3% of github-code-clean), gets its own
    # budget and held-out documents, whether starcoder or the mixed fallback supplied the code.
    for lang in pl.CODE_LANGUAGES:
        for split in pl.SPLITS:
            assert ds.texts(split, domain="code", lang=lang), f"no {lang} code in {split}"


def test_real_lm_pool_is_disjoint_and_from_the_same_sources(tmp_path):
    pipeline = DataPipeline(cache_dir=tmp_path)
    base = pipeline.prepare(train_bytes=1_000_000)
    pool, prov = pipeline.lm_pool(base, 2_000_000)
    held = {d["sha256"] for s in pl.SPLITS for d in base.docs(s)}
    assert not {pl.DeterministicSplitter().hash_document(t) for t in pool} & held
    assert prov["text_bytes"] > 900_000 and prov["dataset"]["complete"]
    used = lambda m: {(r["repo"], r["lang"]) for r in m["sources"] if r["docs"]["train"]}  # noqa: E731
    assert used(prov["dataset"]) == used(base.manifest)


@pytest.mark.parametrize("lang", ["eng", "hun", "mon"])
def test_sigmorphon_2022_download(tmp_path, lang):
    loader = SigmorphonLoader(cache_dir=tmp_path)
    test = loader.load(lang, "test", surface_only=False)
    train = loader.load(lang, "train", surface_only=False)
    dev = loader.load(lang, "dev", surface_only=False)
    assert len(test) > 1000 and len(train) > len(test) and dev
    surface = [e for e in test if e["surface"]]
    assert surface and all(0 < c < len(e["word"].encode("utf-8")) for e in surface for c in e["gold_cuts"])
    assert not {e["word"] for e in train} & {e["word"] for e in test}
    prov = loader.provenance(lang, "test")
    assert prov["commit"] == pl.SIGMORPHON_COMMIT and len(prov["sha256"]) == 64
