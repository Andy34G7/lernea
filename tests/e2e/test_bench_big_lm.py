"""Big proxy-LM tier (--preset big): config resolution, size, init, per-run files, concurrency, resume.
Everything runs on CPU; only the size/init checks build the full-width model."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import pytest

torch = pytest.importorskip("torch")

from tokenisers.benchmarks.colab import lm  # noqa: E402
from tokenisers.benchmarks.colab.__main__ import BIG_PRESET, parse_args  # noqa: E402

from test_bench_proxy_lm import _job  # noqa: E402

SMALL = {"n_layers": 2, "d_model": 128, "n_heads": 4, "d_ffn": 512, "seq_len": 128, "batch_size": 16,
         "lr": 3e-3, "byte_budget": 10_000_000, "seeds": [0, 1, 2], "gpu": "T4", "concurrency": 1,
         "weight_decay": 0.01, "warmup_frac": None, "bf16": False, "output_dir": "benchmarks/results/colab"}


def test_small_preset_is_unchanged():
    args = parse_args([])
    assert {k: getattr(args, k) for k in SMALL} == SMALL
    assert args.preset == "small" and "wordpiece_pmi" in args.variants


def test_big_preset_resolves_and_flags_override():
    args = parse_args(["--preset", "big"])
    assert {k: getattr(args, k) for k in BIG_PRESET} == BIG_PRESET
    assert args.batch_size * args.seq_len == 32_768 and len(args.variants) + 2 == 12  # + bpe_dropout_p0.1 + unigram_sampling
    assert not {"wordpiece_pmi", "wp_bpe_base", "wp_bpe_ft"} & set(args.variants)
    assert (args.gpu, args.concurrency) == ("L4", 2)  # grad_accum_steps=4 reduces peak memory enough for 2
    over = parse_args(["--preset", "big", "--gpu", "A100", "--concurrency", "2", "--lr", "1e-3"])
    assert (over.gpu, over.concurrency, over.lr, over.d_model) == ("A100", 2, 1e-3, 512)


def _big_model(vocab=16384):
    torch.manual_seed(0)
    p = BIG_PRESET
    return lm.ProxyLanguageModel(vocab_size=vocab, d_model=p["d_model"], n_layers=p["n_layers"],
                                 n_heads=p["n_heads"], d_ffn=p["d_ffn"], max_seq_len=p["seq_len"])


def test_big_parameter_count_and_initial_loss():
    model = _big_model()
    emb = model.embedding.weight.numel()
    non_emb = sum(p.numel() for p in model.parameters()) - emb  # lm_head is tied to the embedding
    assert emb == 16384 * 512 and abs(emb - 8.4e6) < 0.05e6
    assert 24.5e6 < non_emb < 25.5e6
    ids = torch.randint(0, 16384, (2, 65))
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(model(ids[:, :-1]).reshape(-1, 16384), ids[:, 1:].reshape(-1))
    assert abs(float(loss) - math.log(16384)) < 0.1


def test_concurrent_runs_aggregate_like_sequential(tmp_path):
    seq = lm.run_job(str(_job(tmp_path / "a", seeds=(0, 1))), device="cpu")
    job = _job(tmp_path / "b", seeds=(0, 1))
    cfg = json.loads((job / "config.json").read_text())
    cfg["concurrency"] = 2
    (job / "config.json").write_text(json.dumps(cfg))
    par = lm.run_job(str(job), device="cpu")
    assert sorted(p.name for p in (job / "runs").iterdir()) == ["tok__s0.json", "tok__s1.json"]
    a, b = seq["models"]["tok"], par["models"]["tok"]
    assert b["macro_val_bpb_by_seed"] == pytest.approx(a["macro_val_bpb_by_seed"], rel=1e-5)
    assert b["steps"] == a["steps"] and b["initial_loss_nats"] == pytest.approx(a["initial_loss_nats"], rel=1e-5)


def test_job_resumes_from_finished_run_files(tmp_path, monkeypatch):
    """A dropped job loses only the runs in flight: finished run files are reused, not retrained."""
    job = _job(tmp_path, seeds=(0, 1))
    first = lm.run_job(str(job), device="cpu")
    (job / "runs" / "tok__s1.json").unlink()  # s1 was in flight when the session dropped
    calls = []
    real = lm._run_one
    monkeypatch.setattr(lm, "_run_one", lambda d, n, s, dev=None: calls.append((n, s)) or real(d, n, s, dev))
    again = lm.run_job(str(job), device="cpu")
    assert calls == [("tok", 1)]
    assert again["models"]["tok"]["macro_val_bpb_by_seed"] == first["models"]["tok"]["macro_val_bpb_by_seed"]


def test_trainer_warmup_fraction_and_weight_decay():
    model = lm.ProxyLanguageModel(vocab_size=40, d_model=16, n_layers=1, n_heads=2, d_ffn=32)
    tr = lm.ByteBudgetTrainer(model, byte_budget=4000, lr=1e-3, device="cpu", weight_decay=0.1, warmup_frac=0.5)
    curve = tr.train_on_token_stream(list(range(40)) * 50, seq_len=8, batch_size=2,
                                     token_bytes=__import__("numpy").full(40, 2), log_every=1)
    assert tr.optimizer.param_groups[0]["weight_decay"] == 0.1
    lrs = [c["lr"] for c in curve]
    peak = lrs.index(max(lrs))
    assert abs(peak + 1 - round(len(lrs) * 0.5)) <= 1 and lrs[-1] < max(lrs)


def test_big_preset_fails_loudly_without_enough_lm_text(monkeypatch):
    from tokenisers.benchmarks.colab import __main__ as cli

    assert parse_args(["--preset", "big"]).lm_text_bytes >= 1.1 * BIG_PRESET["byte_budget"]
    monkeypatch.setattr(cli, "load_dataset", lambda *a: pytest.fail("loaded data before the LM-text check"))
    with pytest.raises(SystemExit, match="set --lm-text-bytes"):
        cli.main(["--preset", "big", "--offline", "--lm-text-bytes", "50000000"])


@pytest.fixture
def big_args(tmp_path):
    """--preset big with toy sizes: offline data, 30 KB train split + LM text pool up to 80 KB."""
    return ["--preset", "big", "--offline", "--cache-dir", str(tmp_path / "cache"), "--train-bytes", "30000",
            "--lm-text-bytes", "80000", "--byte-budget", "50000", "--vocab-size", "280",
            "--variants", "bpe_frequency", "--seeds", "0", "1", "--seq-len", "16", "--batch-size", "4",
            "--d-model", "16", "--n-layers", "1", "--n-heads", "2", "--d-ffn", "32", "--concurrency", "1",
            "--job-dir", str(tmp_path / "job"), "--output-dir", str(tmp_path / "out"), "--poll-interval", "0.05",
            "--skip-memory-check"]


def test_big_cli_refuses_lm_text_that_overlaps_val_or_test(monkeypatch, big_args, require_tokenisers):
    from tokenisers.benchmarks.colab import __main__ as cli
    from tokenisers.data import DataPipeline

    real = DataPipeline.lm_pool
    monkeypatch.setattr(DataPipeline, "lm_pool", lambda self, base, n: (
        real(self, base, n)[0] + base.texts("test")[:1], {}))  # one leaked test document
    monkeypatch.setattr(cli, "train_suite", lambda *a, **k: pytest.fail("trained before the overlap check"))
    with pytest.raises(RuntimeError, match="1 validation/test documents also occur"):
        cli.main(big_args + ["--prepare-only"])


def test_failed_cli_run_resumes_from_its_job_dir(tmp_path, monkeypatch, big_args, require_tokenisers):
    """--prepare-only builds the job (train split + pool, disjoint from val/test, >= the budget per
    variant); a run that fails keeps the finished run files in --job-dir; --resume reuses the job
    without retraining tokenizers and the (Colab CPU, --gpu none) VM trains only the missing run."""
    from tokenisers.benchmarks.colab import __main__ as cli
    from test_bench_proxy_lm import _fake_colab

    job = tmp_path / "job"
    cli.main(big_args + ["--prepare-only"])
    cfg = json.loads((job / "config.json").read_text())
    text = cfg["prepared"]["lm_text"]
    assert text["bytes"] >= 55_000 and 0 < text["pool_bytes"] < text["bytes"] and text["pool"]["docs"] > 0
    assert sorted(p.name for p in job.glob("*.npz")) == ["bpe_dropout_p0.1.npz", "bpe_frequency.npz"]
    assert not (job / "runs").exists()
    with pytest.raises(SystemExit, match="already holds a prepared job"):
        cli.main(big_args)
    with pytest.raises(SystemExit, match=r"different \['train'\]"):
        cli.main(big_args + ["--resume", "--lr", "1e-2"])

    real = lm._run_one

    def dies_on_the_last_run(d, n, s, dev=None):
        if (n, s) == ("bpe_dropout_p0.1", 1):
            raise KeyboardInterrupt  # the command is interrupted
        return real(d, n, s, dev)

    monkeypatch.setattr(lm, "_run_one", dies_on_the_last_run)
    with pytest.raises(KeyboardInterrupt):
        cli.main(big_args + ["--resume", "--local"])
    assert len(list((job / "runs").glob("*.json"))) == 3

    monkeypatch.setattr(cli, "train_suite", lambda *a, **k: pytest.fail("--resume retrained the tokenizers"))
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm"))
    fake = _fake_colab(tmp_path)
    cli.main(big_args + ["--resume", "--gpu", "none", "--concurrency", "2", "--colab-bin", str(fake)])
    log = (tmp_path / "vm" / "content" / "job.log").read_text()
    assert "bpe_dropout_p0.1 seed=1" in log and "seed=0" not in log and "bpe_frequency" not in log, log
    calls = [json.loads(line) for line in (tmp_path / "vm" / "calls.log").read_text().splitlines()]
    assert ["new", "-s", calls[1][2]] == calls[1]  # no --gpu: a CPU runtime
    first_run_upload = next(i for i, c in enumerate(calls) if c[0] == "upload" and "/runs/" in c[-1])
    assert calls[first_run_upload - 1][0] == "exec"  # the remote runs/ dir is created first
    doc = json.loads((tmp_path / "out" / "extrinsic_bpb_results.json").read_text())
    assert set(doc["models"]) == {"bpe_frequency", "bpe_dropout_p0.1"}
    assert doc["meta"]["backend"]["requested_gpu"] is None
    assert doc["meta"]["lm_text"] == text and all(m["epochs"] < 1 for m in doc["models"].values())

    src = tmp_path / "f.txt"
    src.write_text("x")
    up = subprocess.run([str(fake), "upload", "-s", "s", str(src), "/content/missing/f.txt"], capture_output=True)
    assert up.returncode != 0  # like colab 0.7.5: upload does not create remote directories


def test_finished_runs_are_downloaded_and_reused_after_a_drop(tmp_path, monkeypatch):
    """Each finished run file is downloaded as the job runs; a new session after a drop uploads the
    held run files, and the VM job trains only the missing runs."""
    from tokenisers.benchmarks import colab
    from test_bench_proxy_lm import _fake_colab

    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm1"))
    job = _job(tmp_path, seeds=(0, 1))
    first = colab.ColabOrchestrator(colab_bin=str(_fake_colab(tmp_path))).run_job(job, poll_interval=0.05)
    assert sorted(p.name for p in (job / "runs").iterdir()) == ["tok__s0.json", "tok__s1.json"]
    (job / "runs" / "tok__s1.json").unlink()  # s1 was in flight when the session dropped
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm2"))  # a fresh VM
    again = colab.ColabOrchestrator(colab_bin=str(_fake_colab(tmp_path))).run_job(job, poll_interval=0.05)
    log = (tmp_path / "vm2" / "content" / "job.log").read_text()
    assert "tok seed=1" in log and "tok seed=0" not in log, log
    assert again["models"]["tok"]["macro_val_bpb_by_seed"] == first["models"]["tok"]["macro_val_bpb_by_seed"]
