"""Tokenizer swap on llm.c's GPT-2 recipe: recipe invariance, micro-batch accumulation, BPB byte
accounting, identical val bytes, separators, .bin files, and the Colab job (fake CLI, tiny CPU model)."""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import tarfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokenisers.benchmarks import colab  # noqa: E402
from tokenisers.benchmarks.nanogpt_swap import __main__ as swap  # noqa: E402
from tokenisers.benchmarks.nanogpt_swap import swap_job as job_mod  # noqa: E402
from tokenisers.benchmarks.report import render_nanogpt_swap  # noqa: E402
from test_bench_proxy_lm import _fake_colab  # noqa: E402  (tests/e2e is on sys.path)

# The CPU stand-in for the recipe: the same trainer file with a 1-layer, 32-wide model, 32-token
# windows and 128 tokens per step (micro-batch and accumulation still apply), uncompiled.
D12 = '"d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),'
TINY_ARGS = {"sequence_length": "32", "total_batch_size": "128", "compile": "0"}


def _cpu_trainer(path: Path) -> Path:
    src = swap.TRAINER.read_text()
    assert src.count(D12) == 1
    path.write_text(src.replace(D12, D12.replace("n_layer=12, n_head=12, n_embd=768", "n_layer=1, n_head=2, n_embd=32")))
    return path


def _trainer_module():
    spec = importlib.util.spec_from_file_location("train_gpt2_swap", swap.TRAINER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------
# Recipe invariance
# ----------------------------------------------------------------------------

def test_trainer_differs_from_llmc_only_inside_swap_blocks():
    """Every change to llm.c's train_gpt2.py is an inserted, delimited SWAP block, except the one
    replaced line that loads the GPT-2 tokenizer (kept for runs without SWAP_CONFIG)."""
    up = swap.UPSTREAM_TRAINER.read_text().splitlines()
    ours = swap.TRAINER.read_text().splitlines()
    replaced = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, up, ours, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        assert tag in ("insert", "replace"), (tag, up[i1:i2])
        block = "\n".join(ours[j1:j2]).strip().splitlines()  # blank lines around a block are not code
        assert block[0].strip().startswith("# >>> SWAP") and block[-1].strip() == "# <<< SWAP", block
        replaced += up[i1:i2]
    assert replaced == ['    enc = tiktoken.get_encoding("gpt2")']
    assert swap.LLMC_COMMIT in (swap.VENDOR / "README.md").read_text()
    assert "MIT License" in (swap.VENDOR / "LICENSE").read_text()


def test_run_arguments_differ_from_llmc_only_in_allowed_fields():
    """llm.c's pyrun_gpt2_124M.sh with only the allowed changes: 600 steps, the warmup scaled by the
    same fraction (700 / 18,865 of the run), val every 50 steps; micro-batch and paths per run."""
    up = swap.upstream_args()
    assert up["learning_rate"] == "0.0006" and up["warmup_iters"] == "700" and up["num_iterations"] == "18865"
    assert up["learning_rate_decay_frac"] == "0.0" and up["total_batch_size"] == "524288"
    ours = swap.recipe_args(600, 50)
    changed = {k for k in set(up) | set(ours) if up.get(k) != ours.get(k)}
    assert changed <= swap.ALLOWED_OVERRIDES, changed - swap.ALLOWED_OVERRIDES
    assert ours["num_iterations"] == "600" and ours["warmup_iters"] == "22" and ours["val_loss_every"] == "50"
    argv = job_mod.trainer_argv({"trainer_args": ours}, 16, Path("/r"))
    flags = dict(zip(argv[::2], argv[1::2]))
    assert set(flags) == {f"--{k}" for k in set(ours) | {"batch_size", "input_bin", "input_val_bin", "output_dir"}}
    assert flags["--batch_size"] == "16" and flags["--input_bin"] == "/r/train.bin"


def test_lr_override_changes_only_the_learning_rate():
    """--lr is the one opt-in deviation: same recipe, peak lr replaced, and it reaches the trainer."""
    base, ours = swap.recipe_args(600, 50), swap.recipe_args(600, 50, lr=1.8e-3)
    assert {k for k in set(base) | set(ours) if base.get(k) != ours.get(k)} == {swap.LR_OVERRIDE}
    assert ours["learning_rate"] == "0.0018" and base["learning_rate"] == "0.0006"
    argv = job_mod.trainer_argv({"trainer_args": ours}, 16, Path("/r"))
    assert dict(zip(argv[::2], argv[1::2]))["--learning_rate"] == "0.0018"


def test_padded_vocabulary_takes_no_part_in_the_softmax():
    """The vocabulary is padded to a multiple of 128 for speed only: the padded logits are -inf, so
    the loss equals the unpadded model's."""
    mod = _trainer_module()
    torch.manual_seed(0)
    cfg = mod.GPTConfig(block_size=16, vocab_size=128, n_layer=1, n_head=2, n_embd=32)
    model = mod.GPT(cfg)
    x = torch.randint(0, 101, (2, 16))
    y = torch.randint(0, 101, (2, 16))
    mod.SWAP = {"vocab_size": 101}
    logits, loss = model(x, y)
    assert torch.isinf(logits[..., 101:]).all() and not torch.isinf(logits[..., :101]).any()
    ref = torch.nn.functional.cross_entropy(logits[..., :101].reshape(-1, 101), y.reshape(-1))
    assert loss.item() == pytest.approx(ref.item(), rel=1e-6)
    loss.backward()
    assert model.lm_head.weight.grad[101:].abs().sum() == 0  # padded rows never learn


# ----------------------------------------------------------------------------
# Data: documents, separators, .bin files
# ----------------------------------------------------------------------------

WORDS = ("the of and to in data token model byte text naïve café déjà Ünïcödé 東京 🙂 tab\there "
         "line\nbreak <|endoftext|> sep").split(" ")


def _docs(n: int, seed: int) -> list:
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 60))) for _ in range(n)]


@pytest.fixture(scope="module")
def toks(require_tokenisers, tmp_path_factory):
    """Three tiny tokenizers saved like the benchmark's: 'gpt2' (the source shards' stand-in), a
    length-penalized BPE and a unigram model."""
    from tokenisers.models import BpeTokenizer, UnigramTokenizer

    root = tmp_path_factory.mktemp("toks")
    corpus = _docs(300, 1)
    trained = {"gpt2": BpeTokenizer.train(corpus, vocab_size=300),
               "lp": BpeTokenizer.train(corpus, vocab_size=320, scorer="length_penalized"),
               "uni": UnigramTokenizer.train(corpus, vocab_size=290)}
    specs = {}
    for name, tok in trained.items():
        tok.save(str(root / "tokenizers" / name))
        specs[name] = {"name": name, "kind": "tokenisers", "class": type(tok).__name__, "path": f"tokenizers/{name}"}
    return root, specs


@pytest.fixture(scope="module")
def shards(toks, tmp_path_factory):
    """Source shards like the speedrun's, made with the 'gpt2' stand-in: [sep d sep d ...], 100
    val documents (the window cuts the 101st), train documents split across two shards (one
    document spans the boundary, one is a copy of a val document)."""
    root, specs = toks
    src = job_mod.load_tokenizer(specs["gpt2"], root)
    val_docs, train_docs = _docs(110, 2), _docs(400, 3)
    train_docs[5] = val_docs[7]
    val_stream = job_mod.token_stream([src.encode_batch([d])[0] for d in val_docs], src.sep)
    cuts = np.flatnonzero(val_stream == src.sep)
    window = int(cuts[100] + 3)  # mid-document 101
    train_stream = job_mod.token_stream([src.encode_batch([d])[0] for d in train_docs], src.sep)
    half = int(np.flatnonzero(train_stream == src.sep)[200] + 2)  # inside train document 200
    d = tmp_path_factory.mktemp("shards")
    rest = val_stream[cuts[106] + 4:cuts[107]]  # the val shard's last document continues in train shard 1
    assert len(rest)
    job_mod.write_bin(d / "val.bin", val_stream[:cuts[106] + 4])
    job_mod.write_bin(d / "t1.bin", np.concatenate([rest, train_stream[:half]]))
    job_mod.write_bin(d / "t2.bin", train_stream[half:])
    return d, window, val_docs, train_docs


def _source(toks, shards) -> job_mod.SourceDocs:
    root, specs = toks
    d, window, *_ = shards
    src = job_mod.load_tokenizer(specs["gpt2"], root)
    return job_mod.SourceDocs(lambda n: d / n, src.decode, src.sep, "val.bin", window, ["t1.bin", "t2.bin"])


def test_bin_files_round_trip_through_llmc_reader(tmp_path):
    tokens = np.array([50256, 0, 1, 65535, 50256], dtype=np.uint16)
    job_mod.write_bin(tmp_path / "x.bin", tokens)
    header = np.fromfile(tmp_path / "x.bin", dtype=np.int32, count=256)
    assert list(header[:3]) == [20240520, 1, 5] and not header[3:].any()
    assert (job_mod.read_bin(tmp_path / "x.bin") == tokens).all()
    assert (_trainer_module()._load_data_shard(str(tmp_path / "x.bin")) == tokens).all()  # llm.c's own reader
    assert (job_mod.read_bin(tmp_path / "x.bin", max_tokens=2) == tokens[:2]).all()
    with pytest.raises(ValueError):
        job_mod.write_bin(tmp_path / "y.bin", np.array([1 << 16]))


def test_source_documents_decode_exactly_and_train_excludes_val(toks, shards):
    _, _, val_docs, train_docs = shards
    s = _source(toks, shards)
    assert s.val == val_docs[:100]  # the document cut by the val window is not a val document
    while s.extend():
        pass
    # shard 1 starts inside the val shard's last document (dropped), document 200 spans t1/t2, and
    # the copy of a val document is dropped
    assert s.train == [d for i, d in enumerate(train_docs) if i != 5] and s.dropped_as_val == 1
    s.assert_disjoint()
    assert s.train_bytes == [len(d.encode()) for d in s.train]


def test_every_tokenizer_scores_the_same_val_bytes_with_one_separator_per_document(toks, shards):
    root, specs = toks
    s = _source(toks, shards)
    records = {}
    for name in ("gpt2", "lp", "uni"):
        tok = job_mod.load_tokenizer(specs[name], root)
        stream, rec = job_mod.encode_val(tok, s.val)
        records[name] = rec
        assert stream[0] == stream[-1] == tok.sep == tok.n_vocab
        docs, tail = job_mod.split_docs(stream, tok.sep)
        assert [tok.decode(d) for d in docs] == s.val and list(tail) == [tok.sep]
        assert (stream == tok.sep).sum() == len(s.val) + 1  # no document token collides with the separator
        assert rec["targets"] == len(stream) - 1 and rec["bytes"] == sum(len(d.encode()) for d in s.val)
    assert len({r["docs_sha256"] for r in records.values()}) == 1 and len({r["bytes"] for r in records.values()}) == 1
    assert len({r["targets"] for r in records.values()}) == 3  # same bytes, different token counts


def test_lossy_tokenizer_is_rejected(toks, shards):
    root, specs = toks
    tok = job_mod.load_tokenizer(specs["lp"], root)
    tok.decode = lambda ids: job_mod.TokenisersTok.decode(tok, ids).lower()
    with pytest.raises(RuntimeError, match=r"decode\(encode\(doc\)\) != doc for \d+ of 100 val documents"):
        job_mod.encode_val(tok, _source(toks, shards).val)


def test_train_stream_is_one_pass_over_the_shared_document_order(toks, shards):
    root, specs = toks
    s = _source(toks, shards)
    tok = job_mod.load_tokenizer(specs["uni"], root)
    stream, rec = job_mod.encode_train(tok, s, target=3000)
    assert rec["tokens"] == len(stream) >= 3000 and s.shards_used == ["t1.bin"]
    docs, tail = job_mod.split_docs(stream, tok.sep)
    assert [tok.decode(d) for d in docs] == s.train[:rec["docs"]] and list(tail) == [tok.sep]
    more = len(stream) + 50  # more than shard 1 holds: shard 2 is decoded on demand
    stream, rec = job_mod.encode_train(tok, _source(toks, shards), target=more)
    assert len(stream) >= more and rec["docs"] > len(docs)
    with pytest.raises(RuntimeError, match="hold only"):
        job_mod.encode_train(tok, _source(toks, shards), target=10**7)


# ----------------------------------------------------------------------------
# BPB accounting
# ----------------------------------------------------------------------------

class TargetSum(torch.nn.Module):
    """'Loss' = mean of the non-ignored target ids: summed back by count, BPB's numerator is the
    sum of all target ids, which shows that every target is counted once and padding never."""

    def forward(self, x, y, return_logits=False):
        mask = y != -1
        return None, (y * mask).sum().double() / mask.sum()


@pytest.mark.parametrize("n_tokens", [129, 130, 64 * 3 + 1])
def test_bpb_counts_every_val_target_once_over_the_document_bytes(tmp_path, n_tokens):
    mod = _trainer_module()
    tokens = np.random.default_rng(0).integers(0, 500, n_tokens).astype(np.uint16)
    job_mod.write_bin(tmp_path / "val.bin", tokens)
    mod.SWAP = {"val_bytes": 1000, "result_path": str(tmp_path / "r.json")}
    loss = mod.swap_val(TargetSum(), "cpu", 4, 16, 7, str(tmp_path / "val.bin"))
    total = float(tokens[1:].astype(np.int64).sum())
    rec = json.loads((tmp_path / "r.json").read_text())["evals"][0]
    assert rec["val_targets"] == n_tokens - 1 and rec["step"] == 7
    assert rec["val_nll"] == pytest.approx(total) and loss == pytest.approx(total / (n_tokens - 1))
    assert rec["val_bpb"] == pytest.approx(total / math.log(2) / 1000)


def test_speedrun_target_in_bits_per_byte():
    assert swap.target_bpb({"bytes_per_token": 4.436}) == pytest.approx(3.28 / math.log(2) / 4.436)


# ----------------------------------------------------------------------------
# The trainer: micro-batch accumulation; the Colab job
# ----------------------------------------------------------------------------

def _trainer_job(tmp_path: Path, toks, shards, steps: int, eval_every: int) -> tuple:
    root, specs = toks
    job = tmp_path / "trainer"
    run = job / "run"
    run.mkdir(parents=True)
    _cpu_trainer(job / "train_gpt2_swap.py")
    s = _source(toks, shards)
    tok = job_mod.load_tokenizer(specs["lp"], root)
    val, rec = job_mod.encode_val(tok, s.val)
    train, _ = job_mod.encode_train(tok, s, target=steps * 128 * 2)
    job_mod.write_bin(run / "train.bin", train)
    job_mod.write_bin(run / "val.bin", val)
    cfg = {"trainer_args": {**swap.recipe_args(steps, eval_every), **TINY_ARGS},
           "trainer_env": {"CUDA_VISIBLE_DEVICES": ""}}
    return job, run, cfg, tok.n_vocab + 1, rec["bytes"]


def test_micro_batch_accumulation_gives_the_same_training(tmp_path, toks, shards):
    """The micro-batch only changes how the 128-token step is split (llm.c scales each micro-batch's
    mean loss by 1/accumulation): losses, gradient norms and val BPB agree for 4x32, 2x(2x32), 4x(1x32)."""
    job, run, cfg, vocab, val_bytes = _trainer_job(tmp_path, toks, shards, steps=3, eval_every=3)
    out = {}
    for mbs in (4, 2, 1):
        out[mbs] = job_mod.run_trainer(job, {**cfg, "mbs": mbs}, run, vocab, val_bytes)
        assert out[mbs]["grad_accum_steps"] == 4 // mbs and out[mbs]["mbs"] == mbs
    for mbs in (2, 1):
        for a, b in zip(out[4]["steps"], out[mbs]["steps"]):
            assert b["train_loss"] == pytest.approx(a["train_loss"], rel=1e-5)
            assert b["norm"] == pytest.approx(a["norm"], rel=1e-4)
        assert out[mbs]["evals"][-1]["val_bpb"] == pytest.approx(out[4]["evals"][-1]["val_bpb"], rel=1e-5)
    r = out[4]
    assert [e["step"] for e in r["evals"]] == [0, 3] and len(r["steps"]) == 3
    assert r["params"] == sum(p.numel() for p in _tiny_model(vocab).parameters())
    assert r["evals"][0]["val_loss"] == pytest.approx(math.log(vocab), rel=0.02)  # ~uniform over the real ids


def _tiny_model(vocab: int):
    mod = _trainer_module()
    return mod.GPT(mod.GPTConfig(block_size=1024, vocab_size=job_mod.padded_vocab(vocab), n_layer=1, n_head=2, n_embd=32))


def _colab_job(tmp_path: Path, toks, shards, mode: str = "full") -> tuple:
    """A job dir as the CLI writes it, with the tiny CPU trainer and the stand-in source."""
    root, specs = toks
    d, window, *_ = shards
    job = tmp_path / "job"
    job.mkdir()
    shutil.copytree(root / "tokenizers", job / "tokenizers")
    with tarfile.open(job / "tokenizers.tar.gz", "w:gz") as tar:
        tar.add(job / "tokenizers", arcname="tokenizers")
    shutil.rmtree(job / "tokenizers")
    _cpu_trainer(job / "train_gpt2_swap.py")
    s = _source(toks, shards)
    s.extend()
    tt = s.train[:20]
    gpt2 = job_mod.load_tokenizer(specs["gpt2"], root)
    cfg = {"source": {"local_dir": str(d), "val_shard": "val.bin", "val_tokens": window,
                      "train_shards": ["t1.bin", "t2.bin"], "tokenizer": specs["gpt2"]},
           "val_docs_sha256": job_mod.docs_sha256(s.val),
           "tokenizer_text": {"docs": len(tt), "bytes": sum(len(x.encode()) for x in tt),
                              "docs_sha256": job_mod.docs_sha256(tt)},
           "headroom": 0.01,
           "tokenizers": [{**specs["gpt2"], "ids_docs": 5, "ids_sha256": job_mod.ids_sha256(gpt2.encode_batch(tt[:5]))},
                          specs["lp"]],
           "prepared": {"tokenmonster_why": "n/a"},
           "train": ["gpt2"] if mode == "pilot" else ["gpt2", "lp"],
           "trainer_args": {**swap.recipe_args(4, 2), **TINY_ARGS}, "trainer_env": {"CUDA_VISIBLE_DEVICES": ""},
           "mbs": 2, "mode": mode, "gpu": "L4", "eval_docs": 30 if mode == "pilot" else None}
    (job / "config.json").write_text(json.dumps(cfg))
    files = [job / "config.json", job / "train_gpt2_swap.py", job / "tokenizers.tar.gz"]
    return job, cfg, files


def _run(tmp_path, job, files):
    return colab.ColabOrchestrator(colab_bin=str(_fake_colab(tmp_path))).run_job(
        job, gpu="L4", poll_interval=0.2, script=swap.JOB_SCRIPT, files=files)


def test_colab_job_runs_every_tokenizer_downloads_each_run_and_resumes(tmp_path, monkeypatch, toks, shards):
    job, cfg, files = _colab_job(tmp_path, toks, shards)
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm1"))
    result = _run(tmp_path, job, files)
    assert result["runs"] == ["gpt2", "lp"] and result["backend"]["kind"] == "colab"
    runs = {p.stem: json.loads(p.read_text()) for p in (job / "runs").glob("*.json")}
    assert set(runs) == {"gpt2", "lp"}  # downloaded during the polls
    assert runs["gpt2"]["val"]["docs_sha256"] == runs["lp"]["val"]["docs_sha256"]
    assert runs["gpt2"]["val"]["bytes"] == runs["lp"]["val"]["bytes"] == result["data"]["val_bytes"]
    for r in runs.values():
        assert r["vocab_size"] == r["n_vocab"] + 1 == r["separator_id"] + 1
        assert r["padded_vocab_size"] % 128 == 0 and r["train"]["tokens"] >= 4 * 128
        assert [e["step"] for e in r["trainer"]["evals"]] == [0, 2, 4]
    assert result["data"]["train_docs_dropped_as_val"] == 1

    doc = swap.collect(job, cfg, result, "L4", {"git_commit": "abc", "git_dirty": False})
    m = doc["models"]["lp"]
    assert m["curve"][-1][1] == m["final_bpb"] and m["tokens_per_s"] == pytest.approx(128 / m["ms_per_step"] * 1000)
    assert doc["meta"]["target"]["bpb"] == pytest.approx(3.28 / math.log(2) / runs["gpt2"]["val"]["bytes_per_token"])
    md = render_nanogpt_swap(doc)
    assert "| `lp` |" in md and ("not reached" in md or str(m["reached_target_step"]) in md)

    # Session dropped after gpt2: a new session gets runs/gpt2.json uploaded and trains only lp
    (job / "runs" / "lp.json").unlink()
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm2"))
    _run(tmp_path, job, files)
    log = (tmp_path / "vm2" / "content" / "job.log").read_text()
    assert "gpt2: finished in an earlier session, skipped" in log and "lp: val" in log
    assert log.count("val_bpb") == 1 and (job / "runs" / "lp.json").exists()
    calls = [json.loads(line) for line in (tmp_path / "vm2" / "calls.log").read_text().splitlines()]
    assert any(c[0] == "upload" and c[-1].endswith("/runs/gpt2.json") for c in calls) and calls[-1][0] == "stop"


def test_pilot_trains_some_tokenizers_and_projects_the_full_run(tmp_path, monkeypatch, toks, shards):
    job, cfg, files = _colab_job(tmp_path, toks, shards, mode="pilot")
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm"))
    result = _run(tmp_path, job, files)
    doc = swap.collect(job, cfg, result, "L4", {"git_commit": "abc", "git_dirty": False})
    assert "ms_per_step" in doc["models"]["gpt2"] and "ms_per_step" not in doc["models"]["lp"]
    run = json.loads((job / "runs" / "gpt2.json").read_text())
    ev = run["trainer"]["evals"][-1]  # timed on the first 30 val documents only
    assert ev["val_targets"] < run["val"]["targets"] and ev["val_targets"] == doc["models"]["gpt2"]["eval_targets"]
    p = doc["projection"] = swap.project(doc)
    assert p["per_tokenizer"]["lp"]["estimated_from"] == "gpt2" and p["evals_per_run"] == 13
    assert p["total_hours"] > p["per_tokenizer"]["gpt2"]["train_hours"] > 0
    md = render_nanogpt_swap(doc)
    assert "PILOT" in md and "projection, one session" in md


def test_flops_scale_with_the_vocabulary():
    small, big = swap.flops_per_token(32128), swap.flops_per_token(50304)
    assert 0.85 < small / big < 0.95  # the 32k TokenMonster vocabulary saves ~10% of the step


def test_cli_refuses_a_t4_and_unflagged_resume(tmp_path, capsys):
    with pytest.raises(SystemExit):
        swap.main(["--job-dir", str(tmp_path / "j"), "--gpu", "T4"])
    assert "a T4 has no bf16" in capsys.readouterr().err
    job = tmp_path / "j"
    (job / "runs").mkdir(parents=True)
    (job / "runs" / "gpt2.json").write_text("{}")
    (job / "config.json").write_text(json.dumps({"tokenizers": [{"name": "gpt2", "kind": "tiktoken"}]}))
    with pytest.raises(SystemExit, match="pass --resume"):
        swap.main(["--job-dir", str(job), "--gpu", "L4"])
