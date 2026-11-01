"""The tokenizer-swap ablation plumbing: the ``_g2`` variants, tokenizers trained on more text on a
Colab VM (``train_tokenizers``, fake colab CLI), jobs assembled from prepared pools
(``--from-pool``) that run like any other, and the data-driven README section."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tarfile

import pytest

torch = pytest.importorskip("torch")

from tokenisers.benchmarks import colab  # noqa: E402
from tokenisers.benchmarks import common  # noqa: E402
from tokenisers.benchmarks.nanogpt_swap import __main__ as swap  # noqa: E402
from tokenisers.benchmarks.nanogpt_swap import swap_job as job_mod  # noqa: E402
from tokenisers.benchmarks.nanogpt_swap import train_tokenizers as tt  # noqa: E402
from tokenisers.benchmarks.report import render_nanogpt_swap, render_nanogpt_swap_ablation  # noqa: E402
from test_bench_nanogpt_swap import TINY_ARGS, _cpu_trainer, _source, shards, toks  # noqa: E402,F401 (fixtures)
from test_bench_proxy_lm import _fake_colab  # noqa: E402


def test_g2_variants_and_names():
    assert swap.G2_VARIANTS == ("bpe_frequency_g2", "wp_bpe_pig_g2", "wp_bpe_ppig_g2", "wp_bpe_npmi_g2",
                                "unigram_lm_g2", "length_penalized_g2", "morpheme_bpe_g2", "mdl_bits_g2")
    for ok in ("gpt2", "mdl_bits", "mdl_bits_sp", "mdl_bits_g2", "wp_bpe_pig_sp300", "unigram_lm_g2300"):
        assert swap.tokenizer_name(ok) == ok
    for bad in ("super_bpe_g2", "wordpiece_pmi_g2", "mdl_bits_x", "gpt3"):  # wordpiece_pmi: not a swap variant
        with pytest.raises(Exception, match="unknown tokenizer"):
            swap.tokenizer_name(bad)
    assert tt.names(["bpe_frequency", "mdl_bits"], "space_prefix", 300_000_000) == ["bpe_frequency_sp300",
                                                                                     "mdl_bits_sp300"]
    assert tt.names(["unigram_lm"], "gpt2", 50_000_000) == ["unigram_lm_g250"]


def test_prepare_refuses_tokenizers_trained_on_more_text(tmp_path):
    with pytest.raises(SystemExit, match="train_tokenizers"):
        swap.main(["--job-dir", str(tmp_path / "j"), "--prepare-only", "--tokenizers", "gpt2", "mdl_bits_sp300"])


def test_train_tokenizers_cli_checks_its_arguments(tmp_path, capsys):
    for bad in (["--train-bytes", "2500000"], ["--variants", "super_bpe"], ["--pretokenizer", "whitespace"]):
        with pytest.raises(SystemExit):
            tt.main(["--job-dir", str(tmp_path / "p"), "--variants", "mdl_bits", *bad])
    (tmp_path / "p" / "remote" / "runs").mkdir(parents=True)
    (tmp_path / "p" / "remote" / "runs" / "mdl_bits_sp300.json").write_text("{}")
    with pytest.raises(SystemExit, match="pass --resume"):
        tt.main(["--job-dir", str(tmp_path / "p"), "--variants", "mdl_bits"])


# ----------------------------------------------------------------------------
# Colab training (fake CLI) -> pool -> assembled job -> swap run
# ----------------------------------------------------------------------------

NAMES = ["bpe_frequency_sp1", "unigram_lm_sp1"]


def _remote(tmp_path: Path, toks, shards) -> tuple:
    """A pool dir whose remote/ job trains NAMES on 30 kB of the stand-in source at V=300."""
    root, specs = toks
    d, window, *_ = shards
    s = _source(toks, shards)
    s.extend()
    pool = tmp_path / "pool"
    remote = pool / "remote"
    remote.mkdir(parents=True)
    with tarfile.open(remote / "tokenizers.tar.gz", "w:gz") as tar:  # the stand-in source tokenizer
        tar.add(root / "tokenizers" / "gpt2", arcname="tokenizers/gpt2")
    ref = {"val_docs_sha256": job_mod.docs_sha256(s.val),
           "tokenizer_text": {"docs": 20, "docs_sha256": job_mod.docs_sha256(s.train[:20])}}
    cfg = tt.vm_config(NAMES, 30_000, ref, source_dir=d, vocab_size=300)
    cfg["source"].update(val_shard="val.bin", val_tokens=window, train_shards=["t1.bin", "t2.bin"],
                         tokenizer=specs["gpt2"])
    cfg["pip"] = []
    (remote / "config.json").write_text(json.dumps(cfg))
    files = [remote / "config.json", swap.JOB_SCRIPT, remote / "tokenizers.tar.gz"]
    return pool, remote, cfg, files, s


def _train(tmp_path, remote, files):
    return colab.ColabOrchestrator(colab_bin=str(_fake_colab(tmp_path))).run_job(
        remote, gpu="L4", poll_interval=0.2, script=tt.VM_SCRIPT, files=files)


START = {"command": ["x"], "git_commit": "abc", "git_dirty": False, "native_extension": None}


def test_colab_training_downloads_each_tokenizer_resumes_and_makes_a_pool(tmp_path, monkeypatch, toks, shards):
    pool, remote, cfg, files, s = _remote(tmp_path, toks, shards)
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm1"))
    result = _train(tmp_path, remote, files)
    assert result["runs"] == sorted(NAMES) and result["backend"]["kind"] == "colab"
    text = result["data"]["tokenizer_text"]
    docs, nbytes = s.first_docs(30_000)
    assert text["docs"] == len(docs) > 20 and text["bytes"] == nbytes <= 30_000
    assert text["docs_sha256"] == job_mod.docs_sha256(docs) and text["starts_with"]["docs"] == 20
    calls = [json.loads(line) for line in (tmp_path / "vm1" / "calls.log").read_text().splitlines()]
    assert calls[-1][0] == "stop" and sum(c[0] == "download" and "/runs/" in c[3] for c in calls) == 2

    # Dropped after the first tokenizer: a new session gets it uploaded and trains only the second
    (remote / "runs" / "unigram_lm_sp1.json").unlink()
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm2"))
    result = _train(tmp_path, remote, files)
    log = (tmp_path / "vm2" / "content" / "job.log").read_text()
    assert "bpe_frequency_sp1: finished in an earlier session, skipped" in log and "unigram_lm_sp1: training" in log

    cfg_pool = tt.finish(pool, cfg, result, START)
    assert [t["name"] for t in cfg_pool["tokenizers"]] == NAMES
    from tokenisers import models

    for spec in cfg_pool["tokenizers"]:
        tok = getattr(models, spec["class"]).from_pretrained(str(pool / spec["path"]))
        assert tok.pretokenizer == "space_prefix" and spec["tokenizer_text"] == text
        assert job_mod.ids_sha256(tok.encode(d) for d in s.train[:spec["ids_docs"]]) == spec["ids_sha256"]
        rec = cfg_pool["prepared"]["variants"][spec["name"]]
        assert rec["train_bytes"] == nbytes and rec["train_docs"] == len(docs) and rec["pretokenizer"] == "space_prefix"
    assert cfg_pool["source"] == swap.SOURCE and (pool / "tokenizers.tar.gz").is_file()


def test_failed_training_keeps_finished_tokenizers_and_stops(tmp_path, monkeypatch, toks, shards):
    pool, remote, cfg, files, _ = _remote(tmp_path, toks, shards)
    cfg["reference_text"]["docs_sha256"] = "0" * 64  # another stream than the swap's
    (remote / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm"))
    with pytest.raises(colab.ColabError, match="differ from the swap's tokenizer text"):
        _train(tmp_path, remote, files)
    calls = [json.loads(line)[0] for line in (tmp_path / "vm" / "calls.log").read_text().splitlines()]
    assert calls[-1] == "stop"


def _local_pool(tmp_path: Path, toks, shards, names: dict) -> Path:
    """A pool as ``--prepare-only`` writes it: names -> stand-in tokenizer, the job's 20-doc text."""
    root, specs = toks
    s = _source(toks, shards)
    s.extend()
    pool = tmp_path / "local_pool"
    specs_out = []
    for name, src in names.items():
        shutil.copytree(root / "tokenizers" / src, pool / "tokenizers" / name)
        gpt2 = job_mod.load_tokenizer({**specs[src], "path": f"tokenizers/{name}"}, pool)
        specs_out.append({**specs[src], "name": name, "path": f"tokenizers/{name}", "pretokenizer": "gpt2",
                          "ids_docs": 5, "ids_sha256": job_mod.ids_sha256(gpt2.encode_batch(s.train[:5]))})
    tt20 = s.train[:20]
    cfg = {"source": swap.SOURCE, "val_docs_sha256": job_mod.docs_sha256(s.val),
           "tokenizer_text": {"docs": 20, "bytes": sum(len(x.encode()) for x in tt20),
                              "docs_sha256": job_mod.docs_sha256(tt20)},
           "pip": swap.PIP, "headroom": 0.01, "tokenizers": specs_out,
           "prepared": {"command": ["prepare"], "tokenmonster_why": "n/a", "val_docs": len(s.val),
                        "variants": {n: {"pretokenizer": "gpt2"} for n in names}, "morphology": None}}
    (pool / "config.json").write_text(json.dumps(cfg))
    return pool


def test_assembled_job_runs_like_a_prepared_one(tmp_path, monkeypatch, toks, shards):
    """Two pools -> one job (--from-pool): specs, files and each tokenizer's training text; the
    swap VM checks every training text and trains both."""
    pool, remote, cfg, files, _ = _remote(tmp_path, toks, shards)
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm"))
    tt.finish(pool, cfg, _train(tmp_path, remote, files), START)
    local = _local_pool(tmp_path, toks, shards, {"bpe_frequency_g2": "lp", "length_penalized_g2": "lp"})
    job = tmp_path / "job"
    job.mkdir()
    swap.assemble([local, pool], ["bpe_frequency_g2", "unigram_lm_sp1", "length_penalized_g2"], job, START)
    a = json.loads((job / "config.json").read_text())
    assert [t["name"] for t in a["tokenizers"]] == ["bpe_frequency_g2", "unigram_lm_sp1", "length_penalized_g2"]
    assert a["tokenizer_text"] == json.loads((local / "config.json").read_text())["tokenizer_text"]
    by = {t["name"]: t for t in a["tokenizers"]}
    assert by["bpe_frequency_g2"]["tokenizer_text"]["docs"] == 20 and by["unigram_lm_sp1"]["tokenizer_text"]["docs"] > 20
    assert a["prepared"]["duplicates"] == {"length_penalized_g2": "bpe_frequency_g2"}
    assert set(a["prepared"]["variants"]) == set(by)
    with tarfile.open(job / "tokenizers.tar.gz") as tar:
        assert {m.name for m in tar.getmembers() if m.name.count("/") == 1} == {f"tokenizers/{n}" for n in by}
    other = tmp_path / "other"
    shutil.copytree(local, other)
    c = json.loads((other / "config.json").read_text())
    c["val_docs_sha256"] = "x"
    (other / "config.json").write_text(json.dumps(c))
    with pytest.raises(SystemExit, match="other val documents"):
        swap.assemble([local, other], ["bpe_frequency_g2"], tmp_path / "job2", START)
    with pytest.raises(SystemExit, match="in none of"):
        swap.assemble([local], ["mdl_bits_g2"], tmp_path / "job3", START)

    # The swap VM on the assembled job (the stand-in source, the tiny CPU trainer).
    root, specs = toks
    d, window, *_ = shards
    _cpu_trainer(job / "train_gpt2_swap.py")
    shutil.copytree(root / "tokenizers" / "gpt2", job / "tokenizers" / "gpt2")  # the stand-in source tokenizer
    (job / "tokenizers.tar.gz").unlink()
    with tarfile.open(job / "tokenizers.tar.gz", "w:gz") as tar:
        tar.add(job / "tokenizers", arcname="tokenizers")
    a["tokenizers"] = [by["bpe_frequency_g2"], by["unigram_lm_sp1"]]
    a.update(source={"local_dir": str(d), "val_shard": "val.bin", "val_tokens": window,
                     "train_shards": ["t1.bin", "t2.bin"], "tokenizer": specs["gpt2"]},
             train=["bpe_frequency_g2", "unigram_lm_sp1"], trainer_args={**swap.recipe_args(2, 2), **TINY_ARGS},
             trainer_env={"CUDA_VISIBLE_DEVICES": ""}, mbs=2, mode="full", gpu="L4", eval_docs=None, pip=[])
    (job / "config.json").write_text(json.dumps(a))
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm_swap"))
    result = colab.ColabOrchestrator(colab_bin=str(_fake_colab(tmp_path))).run_job(
        job, gpu="L4", poll_interval=0.2, script=swap.JOB_SCRIPT,
        files=[job / "config.json", job / "train_gpt2_swap.py", job / "tokenizers.tar.gz"])
    assert result["runs"] == ["bpe_frequency_g2", "unigram_lm_sp1"]
    doc = swap.collect(job, a, result, "L4", START)
    assert doc["models"]["unigram_lm_sp1"]["spec"]["tokenizer_text"]["docs"] > 20
    md = render_nanogpt_swap(doc)
    assert "trained on the first 20 train documents" in md and " / " in md.split("our tokenizers:")[1].split("\n")[0]

    # A training text the VM decodes differently is refused before any training.
    a["tokenizers"][1]["tokenizer_text"] = {**by["unigram_lm_sp1"]["tokenizer_text"], "docs_sha256": "0" * 64}
    (job / "config.json").write_text(json.dumps(a))
    shutil.rmtree(job / "runs")
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm_swap2"))
    with pytest.raises(colab.ColabError, match="differ from a tokenizer training text"):
        colab.ColabOrchestrator(colab_bin=str(_fake_colab(tmp_path))).run_job(
            job, gpu="L4", poll_interval=0.2, script=swap.JOB_SCRIPT,
            files=[job / "config.json", job / "train_gpt2_swap.py", job / "tokenizers.tar.gz"])


def test_swap_cli_assembles_with_from_pool(tmp_path, toks, shards):
    local = _local_pool(tmp_path, toks, shards, {"bpe_frequency_g2": "lp", "unigram_lm_g2": "uni"})
    job = tmp_path / "j"
    with pytest.raises(SystemExit):  # which ones?
        swap.main(["--job-dir", str(job), "--prepare-only", "--from-pool", str(local)])
    swap.main(["--job-dir", str(job), "--prepare-only", "--from-pool", str(local), "--tokenizers", "unigram_lm_g2"])
    cfg = json.loads((job / "config.json").read_text())
    assert [t["name"] for t in cfg["tokenizers"]] == ["unigram_lm_g2"]
    assert cfg["prepared"]["command"][-4:] == ["--from-pool", str(local), "--tokenizers", "unigram_lm_g2"]


# ----------------------------------------------------------------------------
# README section
# ----------------------------------------------------------------------------

def test_readme_ablation_section_is_data_driven(repo_root, tmp_path):
    import importlib.util

    rel_base = "benchmarks/results/nanogpt_swap/nanogpt_swap.json"
    base = json.loads((repo_root / rel_base).read_text(encoding="utf-8"))
    pending = render_nanogpt_swap_ablation(base, None, None)
    assert "Not yet run" in pending and not any(c.isdigit() for c in pending)

    abl = {"meta": copy.deepcopy(base["meta"]), "models": {}}
    big = {"docs": 97_000, "bytes": 299_999_000, "docs_sha256": "big",
           "starts_with": {"docs": 16157, "docs_sha256": base["meta"]["tokenizer_text"]["docs_sha256"]}}
    for v, suf in (("bpe_frequency", "_g2"), ("mdl_bits", "_g2"), ("bpe_frequency", "_sp300"), ("unigram_lm", "_sp300")):
        m = copy.deepcopy(base["models"][v])
        m.update(val_bytes_per_token=m["val_bytes_per_token"] + 2, final_bpb=m["final_bpb"] - 0.3)
        if suf == "_sp300":
            m["spec"] = {**m["spec"], "tokenizer_text": big}
        abl["models"][v + suf] = m
    md = render_nanogpt_swap_ablation(base, None, abl)
    old = base["models"]["bpe_frequency"]
    assert ("| Variant | whitespace, 50 MB | space_prefix, 50 MB | gpt2, 50 MB | space_prefix, 300 MB |") in md
    assert (f"| `bpe_frequency` | **{old['final_bpb']:.4f}** ({old['val_bytes_per_token']:.2f}) | - | "
            f"**{old['final_bpb'] - 0.3:.4f}** ({old['val_bytes_per_token'] + 2:.2f}) | "
            f"**{old['final_bpb'] - 0.3:.4f}** ({old['val_bytes_per_token'] + 2:.2f}) |") in md
    assert "| `mdl_bits` |" in md and md.split("| `mdl_bits` |")[1].split("\n")[0].endswith("| - |")
    assert "`bpe_frequency_sp300` 97,000 docs" in md and "NOT starting" not in md
    assert "same recipe and val documents" in md and "not committed yet" in md
    assert md.count("\n| `") == 4  # 3 variants + GPT-2
    other = copy.deepcopy(abl)
    other["meta"]["recipe"]["args"]["learning_rate"] = "0.0018"
    assert "DIFFERENT recipe" in render_nanogpt_swap_ablation(base, None, other)

    spec = importlib.util.spec_from_file_location("gen_readme_tables", repo_root / "scripts" / "gen_readme_tables.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    rels = gen.SECTIONS["nanogpt_swap_ablation"][0]
    assert rels[0] == rel_base and rels[2].endswith("nanogpt_swap_ablation/nanogpt_swap.json")
    (tmp_path / rels[0]).parent.mkdir(parents=True)
    (tmp_path / rels[0]).write_text(json.dumps(base))
    (tmp_path / rels[2]).parent.mkdir(parents=True)
    (tmp_path / rels[2]).write_text(json.dumps(abl))
    readme = "".join(f"<!-- BEGIN GENERATED: {s} -->\nold\n<!-- END GENERATED: {s} -->\n" for s in gen.SECTIONS)
    section = gen.render_readme(readme, tmp_path).split("BEGIN GENERATED: nanogpt_swap_ablation -->")[1]
    assert section.startswith("\n" + md + "\n<!-- END")
