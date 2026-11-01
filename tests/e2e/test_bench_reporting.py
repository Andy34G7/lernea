"""Shared benchmark plumbing: duplicate tokenizers, README tables from JSON, memory preflight (B10)."""

from __future__ import annotations

import importlib.util
import inspect
import json
import warnings

import pytest

from tokenisers.benchmarks import common
from tokenisers.benchmarks.report import render_colab, render_colab_compare, render_intrinsic, render_tokka


def _gen_module(repo_root):
    spec = importlib.util.spec_from_file_location("gen_readme_tables", repo_root / "scripts" / "gen_readme_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readme_tables_match_committed_results(repo_root):
    """B10: README leaderboards are exactly what scripts/gen_readme_tables.py renders from the result JSON."""
    gen = _gen_module(repo_root)
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert gen.render_readme(readme, repo_root) == readme, "run: uv run python scripts/gen_readme_tables.py"
    assert "_Not yet run." not in readme  # every section, the GPT-2 tokenizer swap included, renders committed JSON


def _meta(**kw):
    meta = {"requested_vocab_size": 300, "synthetic_data": False, "train_bytes": 4_000_000, "eval_split": "test",
            "eval_bytes": 300_000, "git_commit": "0123456789abcdef", "git_dirty": False,
            "created_utc": "2026-10-30T00:00:00Z", "duplicates": {"wp_bpe_ft": "wordpiece_pmi"}, "host": {"cpu": "x"}}
    meta.update(kw)
    return meta


def test_generator_renders_json_and_detects_stale_readme(repo_root, tmp_path):
    """B10: a section is replaced by the rendered JSON; --check fails on a README that differs."""
    gen = _gen_module(repo_root)
    row = {"vocab_size": 300, "en": {"bytes_per_token": 4.0}, "macro_avg": {
        "bytes_per_token": 3.25, "subword_fertility": 1.5, "continued_word_rate": 20.0},
        "micro_avg": {"bytes_per_token": 3.5}, "unique_tokens": 250}
    (tmp_path / "benchmarks/results/tokka").mkdir(parents=True)
    (tmp_path / "benchmarks/results/tokka/tokka_matrix.json").write_text(json.dumps(
        {"meta": _meta(), "models": {"wordpiece_pmi": row, "wp_bpe_ft": row}}))
    readme = "x\n" + "".join(f"<!-- BEGIN GENERATED: {s} -->\nold\n<!-- END GENERATED: {s} -->\n"
                             for s in gen.SECTIONS)
    out = gen.render_readme(readme, tmp_path)
    assert "| `wordpiece_pmi` = `wp_bpe_ft` | 300 | 3.250 | 3.500 | 1.500 | 20.00 | 250 |" in out
    assert out.count("`wp_bpe_ft`") == 1  # one row for the duplicate pair
    assert "No results committed" in out.split("BEGIN GENERATED: intrinsic")[1]
    path = tmp_path / "README.md"
    path.write_text(readme)
    assert gen.main(["--check", "--readme", str(path)]) == 1


def test_find_duplicates_detects_identical_tokenizers(require_tokenisers):
    """Identical trainings are one result; different ones are kept apart."""
    tk = require_tokenisers
    corpus = ["the cat sat on the mat with the hat " * 20]
    models = {
        "a": tk.BpeTokenizer.train(corpus, vocab_size=266),
        "b": tk.BpeTokenizer.train(corpus, vocab_size=266),
        "c": tk.BpeTokenizer.train(corpus, vocab_size=262),
        "u": tk.UnigramTokenizer.train(corpus, vocab_size=266),
    }
    assert common.find_duplicates(models) == {"b": "a"}


def test_result_files_record_every_training_parameter(require_tokenisers):
    """meta.variants (variant_config) holds every parameter training takes: none is left to a
    library default, except scorer parameters the variant's scorer rejects. The recorded BPE values
    are the scorer defaults, so making them explicit changed no tokenizer."""
    from tokenisers import models

    corpus = ["the cat sat on the mat with the hat, a bat ate a gnat " * 5, "zyx qwv jk"]
    not_params = {"corpus", "vocab_size", "v_final", "morpheme_lexicon", "special_tokens"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # training may stop short of vocab_size on this corpus
        for name in common.VARIANTS:
            cfg = common.variant_config(name, 290)
            cls = getattr(models, cfg.pop("class"))
            extra = {"morpheme_lexicon": {}} if name == "morpheme_bpe" else {}
            left = set(inspect.signature(cls.train).parameters) - set(cfg) - not_params
            assert left <= {"alpha", "beta", "penalty"}, (name, left)
            for p in left:
                with pytest.raises(ValueError, match="does not (take|apply)"):
                    cls.train(corpus, vocab_size=290, **cfg, **extra, **{p: 0.5})
            if name in common._BPE_SCORERS:
                default = models.BpeTokenizer.train(corpus, vocab_size=290, scorer=cfg["scorer"])
                trained = common.train_variant(name, corpus, 290, {})
                assert common.tokenizer_fingerprint(trained) == common.tokenizer_fingerprint(default), name


def test_leaderboards_state_data_and_backend():
    """B11: the rendered header says whether data were synthetic and which backend ran the LM."""
    intr = {"meta": _meta(synthetic_data=True, duplicates={}), "models": {"bpe": {
        "vocab_size": 300, "bytes_per_token": 1.0, "fertility": 2.0, "renyi_e_1_0": 0.5, "renyi_e_2_5": 0.4,
        "morph_f1": 0.3, "identifier_f1": 0.2, "encode_mb_per_sec": 1.0, "decode_mb_per_sec": 2.0}}}
    assert "SYNTHETIC" in render_intrinsic(intr)
    col = {"meta": _meta(duplicates={}, eval_split="val", val_domains=["english"], seeds=[0, 1, 2], byte_budget=10,
                         backend={"kind": "colab", "device": "cuda", "gpu": "Tesla T4"}, lm={"d_model": 8}),
           "models": {"bpe": {"vocab_size": 300, "train_bytes_per_token": 3.0, "tokens_consumed": 10,
                              "val_domains": {"english": {"bpb": 1.25, "bpb_sd": 0.01}},
                              "macro_val_bpb": 1.25, "macro_val_bpb_sd": 0.01}}}
    md = render_colab(col)
    assert "Colab GPU `Tesla T4`" in md and "1.250 ± 0.010" in md
    col["meta"]["backend"] = {"kind": "local", "device": "cpu", "cpu": "Some CPU"}
    assert "local CPU `Some CPU`" in render_colab(col)
    assert render_tokka({"meta": _meta(duplicates={}), "models": {}}).count("|") > 0
    # The seeds vary only the LM's initialization (one LM text, one order): the sd says so.
    assert "the seed sets only the LM's initialization: every seed trains on the same LM text in the same order" in md

    # A --resume of a job prepared (tokenizers, LM text, ids) at another, dirty commit says so.
    assert "resuming a job" not in md
    col["meta"]["job_prepared"] = {"git_commit": "3186c1ba815c3136", "git_dirty": True}
    assert ("commit `0123456789`, resuming a job (tokenizers, LM text and its ids) prepared at commit `3186c1ba81` "
            "from a dirty tree (uncommitted changes: not reproducible from any commit)") in render_colab(col)
    col["meta"]["job_prepared"] = {"git_commit": "0123456789abcdef", "git_dirty": False}  # the same commit
    assert "resuming a job" not in render_colab(col)


def test_colab_compare_ranks_both_tiers_against_bpe_frequency():
    """Small vs big proxy LM: rank per tier and delta to bpe_frequency ± its standard error over the seeds."""
    def doc(bpb, gpu):  # name -> (mean, sd) over 4 seeds
        return {"meta": {"lm": {"d_model": 8, "n_layers": 1}, "byte_budget": 10, "seeds": [0, 1, 2, 3],
                         "backend": {"gpu": gpu}},
                "models": {n: {"macro_val_bpb": m, "macro_val_bpb_sd": sd,
                               "macro_val_bpb_by_seed": dict.fromkeys("0123")} for n, (m, sd) in bpb.items()}}
    small = doc({"bpe_frequency": (2.0, 0.04), "a": (1.9, 0.03), "only_small": (5.0, 0.1)}, "T4")
    big = doc({"bpe_frequency": (1.5, 0.0), "a": (1.6, 0.02)}, "L4")
    md = render_colab_compare(small, big)
    assert ("| `bpe_frequency` | 2 | reference | 1 | reference |\n"
            "| `a` | 1 | -0.100 ± 0.025 | 2 | +0.100 ± 0.010 |") in md
    assert "- small LM: 1 tokenizers beat `bpe_frequency`, by 0.100–0.100 BPB, i.e. 4.0–4.0 standard errors" in md
    assert "- big LM:" not in md and "only_small" not in md
    assert "seeds vary only the LM's initialization (same LM text, same order)" in md


def test_memory_preflight_refuses_runs_that_would_not_fit(monkeypatch):
    """The CLIs stop before training when the estimated peak exceeds available memory."""
    mb = 2**20
    monkeypatch.setattr(common, "available_memory", lambda: 3 * 1024 * mb)
    common.check_memory(["bpe_frequency", "super_bpe"], 4 * mb)  # ~0.9 GiB: fine
    with pytest.raises(SystemExit, match="super_bpe"):
        common.check_memory(["bpe_frequency", "super_bpe"], 50 * mb)  # ~4.3 GiB
    common.check_memory(["bpe_frequency"], 35 * mb)  # BPE alone: ~2.6 GiB
    monkeypatch.setattr(common, "available_memory", lambda: None)  # unknown platform: no check
    common.check_memory(["unigram_lm"], 10**12)


# Whole-process peak RSS (VmHWM) of training one variant at V=65536 on the real corpus (train split
# of `python -m tokenisers.data --train-bytes N`, commit 1e3b4da): {variant: [(text bytes, peak GiB)]}.
# Re-measure when training code changes: the earlier constants (unigram_lm 175 B/B) predated the
# unigram memory fix and refused the default 50 MB run on a 16 GB machine.
MEASURED_TRAIN_PEAKS = {
    "super_bpe": [(4_256_447, 0.42), (10_105_464, 0.79), (25_101_597, 1.53), (52_512_567, 2.92)],
    "wp_bpe_pig": [(4_256_447, 0.20), (25_101_597, 0.67), (52_512_567, 1.25)],
    "unigram_lm": [(4_256_447, 0.29), (10_105_464, 0.32), (25_101_597, 0.40), (52_512_567, 0.52)],
}


def test_memory_estimate_covers_the_largest_vocabulary():
    """The estimate covers every measured peak at V=65536 (a supported size) with a 20% margin."""
    for variant, points in MEASURED_TRAIN_PEAKS.items():
        for text_bytes, peak_gib in points:
            assert common.estimate_train_memory([variant], text_bytes) >= 1.2 * peak_gib * 2**30, (variant, text_bytes)


def test_memory_estimate_does_not_refuse_the_default_run_for_no_reason():
    """The full suite on the default 50 MB split is estimated at no more than 1.6x its measured peak
    (super_bpe at V=65536), so the default run is not refused on a machine that can hold it."""
    worst = max(peak for points in MEASURED_TRAIN_PEAKS.values() for n, peak in points if n >= common.DEFAULT_TRAIN_BYTES)
    assert common.estimate_train_memory(common.VARIANTS, common.DEFAULT_TRAIN_BYTES) <= 1.6 * worst * 2**30


def test_system_memory_keeps_a_reserve_for_the_desktop():
    """Outside a cgroup limit the benchmark may not plan on all of MemAvailable."""
    gib = 2**30
    meminfo = f"MemTotal: {15 * gib // 1024} kB\nMemFree: 1 kB\nMemAvailable: {int(9.5 * gib) // 1024} kB\n"
    usable = common.usable_system_memory(meminfo)
    assert usable == int(9.5 * gib) - int(0.15 * 15 * gib)  # 15% of 15 GiB > the 2 GiB floor
    assert common.usable_system_memory(f"MemTotal: {4 * gib // 1024} kB\nMemAvailable: {3 * gib // 1024} kB\n") == gib
    assert common.usable_system_memory("MemTotal: 1 kB\n") is None
    # a ~7.9 GiB estimate (super_bpe on 100 MiB) fits the 9.5 GiB available but not what is usable
    assert int(9.5 * gib) > common.estimate_train_memory(["super_bpe"], 100 * 2**20) > usable


def test_memory_preflight_runs_before_the_dataset_is_prepared(monkeypatch, tmp_path):
    """A run that cannot fit stops before downloading/preparing its dataset."""
    from tokenisers.benchmarks.colab import __main__ as colab_cli
    from tokenisers.benchmarks.intrinsic import __main__ as intrinsic_cli
    from tokenisers.benchmarks.tokka import __main__ as tokka_cli

    monkeypatch.setattr(common, "available_memory", lambda: 2**30)
    for cli in (tokka_cli, intrinsic_cli, colab_cli):
        monkeypatch.setattr(cli, "load_dataset", lambda args: pytest.fail("dataset prepared before the check"))
        with pytest.raises(SystemExit, match="needs ~"):
            cli.main(["--offline", "--cache-dir", str(tmp_path), "--train-bytes", str(50 * 2**20)])


def test_memory_preflight_counts_what_the_process_already_holds(monkeypatch):
    """The estimate is the run's total peak; memory the process already uses (interpreter, torch) is
    both part of it and missing from what is available, so only the difference must fit."""
    gib = 2**30
    need = common.estimate_train_memory(["unigram_lm"], 10**7)  # ~1.1 GiB
    monkeypatch.setattr(common, "available_memory", lambda: need - gib // 2)
    monkeypatch.setattr(common, "own_rss", lambda: gib // 2)
    common.check_memory(["unigram_lm"], 10**7)
    monkeypatch.setattr(common, "own_rss", lambda: gib // 4)
    with pytest.raises(SystemExit, match="unigram_lm"):
        common.check_memory(["unigram_lm"], 10**7)
    assert 0 < common.own_rss() < 10 * gib  # this process, from /proc/self/statm


def test_provenance_is_taken_when_the_run_starts(require_tokenisers, tmp_path, monkeypatch):
    """A commit made while a long run is going must not be recorded as the code that produced it."""
    from tokenisers.benchmarks.tokka import __main__ as cli

    monkeypatch.setattr(common, "_git", lambda *a: "a" * 40 if a[0] == "rev-parse" else "")

    def train_then_commit(*a, **kw):
        monkeypatch.setattr(common, "_git", lambda *a: "b" * 40 if a[0] == "rev-parse" else " M x.py")
        return common.train_suite(*a, **kw)

    monkeypatch.setattr(cli, "train_suite", train_then_commit)
    argv = ["--offline", "--cache-dir", str(tmp_path / "cache"), "--train-bytes", "30000", "--vocab-size", "270",
            "--variants", "bpe_frequency", "--output-dir", str(tmp_path / "out"),
            "--min-docs-per-lang", "1", "--min-bytes-per-lang", "0"]
    cli.main(argv)
    meta = json.loads((tmp_path / "out" / "tokka_matrix.json").read_text())["meta"]
    assert meta["git_commit"] == "a" * 40 and meta["git_dirty"] is False
    assert meta["command"] == ["python", "-m", "tokenisers.benchmarks.tokka", *argv]
    assert "- arguments: `--offline --cache-dir" in (tmp_path / "out" / "tokka_leaderboard.md").read_text()


def test_assert_disjoint():
    common.assert_disjoint(["a", "b"], ["c"])
    with pytest.raises(RuntimeError, match="1 test documents"):
        common.assert_disjoint(["a", "b"], ["b", "c"], "test")


def test_readme_setup_paragraph_accounts_for_every_results_section(repo_root):
    """'Scale and setup' says where each numbered results section's data and runs come from, so it
    cannot claim one suite run for all of them while sections are added; the limitations pointer
    names the last section."""
    import re

    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    results = readme[readme.index("## Benchmark Results"):readme.index("### Known limitations")]
    sections = {int(n) for n in re.findall(r"^### (\d+)\. ", results, re.M)}
    setup = re.search(r"\*\*Scale and setup\.\*\*(.+?)\n\n", results, re.S)[1]
    named = set()
    for m in re.finditer(r"[Ss]ections? (\d+(?:(?:-| and |, )\d+)*)", setup.replace("\n", " ")):
        for part in re.split(r" and |, ", m[1]):
            lo, _, hi = part.partition("-")
            named |= set(range(int(lo), int(hi or lo) + 1))
    assert sections - named == set(), f"Scale and setup says nothing about sections {sorted(sections - named)}"
    assert f"listed after section {max(sections)}." in " ".join(setup.split())
