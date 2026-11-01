"""
Proxy-LM benchmark core: model, byte-budget trainer, BPB evaluator and the job runner.

This file imports only torch, numpy and the standard library. It is imported for local runs and
uploaded unchanged to the Colab VM for remote runs, so both execute the same implementation.

Job directory layout (written by ``tokenisers.benchmarks.colab.__main__``):
- ``config.json``: {"lm": {...model kwargs}, "train": {byte_budget, seq_len, batch_size, lr,
  eval_stride}, "seeds": [...], "domains": [...], "tokenizers": {name: {"vocab_size": int}}}
- ``<name>.npz``: ``train`` and ``val_<domain>`` id arrays plus ``token_bytes`` (UTF-8 length of
  every vocabulary id), so consumed and scored bytes are exact.
``run_job`` writes ``result.json``; ``python lm.py <job_dir>`` also writes ``status.json`` (see ``main``).
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INIT_STD = 0.02
# Recorded with every result: this is not a Llama/GPT block (no positional encoding, ReLU, not SwiGLU).
ARCHITECTURE = (
    "pre-LayerNorm causal Transformer (torch TransformerEncoderLayer: ReLU FFN, biases), no positional "
    "encoding of any kind (NoPE: order comes only from the causal mask), tied input/output embeddings"
)


class ProxyLanguageModel(nn.Module):
    """
    Small pre-LayerNorm causal Transformer with tied input/output embeddings and no positional
    encoding (see ``ARCHITECTURE``). Every weight matrix (embeddings included) starts at
    N(0, INIT_STD), biases at 0, so the initial loss is ~ln V.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 2,
        d_ffn: int = 256,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ffn,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=INIT_STD)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids[:, : self.max_seq_len]
        seq_len = input_ids.shape[1]
        h = self.embedding(input_ids)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        return self.lm_head(self.norm(h))


def _token_byte_counts(ids: np.ndarray, token_bytes: Optional[np.ndarray], total_bytes: Optional[int]) -> np.ndarray:
    """Per-position UTF-8 byte counts: exact from ``token_bytes``, else the stream's average."""
    if token_bytes is not None:
        return token_bytes[ids].astype(np.float64)
    avg = (total_bytes or len(ids)) / max(1, len(ids))
    return np.full(len(ids), avg, dtype=np.float64)


class ByteBudgetTrainer:
    """
    Trains on consecutive, non-overlapping windows of a token stream until the target tokens of
    the steps taken cover ``byte_budget`` UTF-8 bytes. AdamW with linear warmup (``warmup_steps``,
    default 10% of the estimated steps, at most 200) and cosine decay to 10% of ``lr``. Passing
    over the stream more than once is counted in ``epochs``.
    """

    def __init__(
        self,
        model: nn.Module,
        byte_budget: int = 10_000,
        lr: float = 3e-3,
        device: Optional[str] = None,
        warmup_steps: Optional[int] = None,
        weight_decay: float = 0.01,
        warmup_frac: Optional[float] = None,
        bf16: bool = False,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.byte_budget = byte_budget
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.warmup_frac = warmup_frac
        self.bf16 = bf16  # bf16 autocast for the forward pass (CUDA only)
        self.bytes_consumed = 0
        self.tokens_consumed = 0
        self.epochs = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def step_batch(self, text: str) -> None:
        self.bytes_consumed += len(text.encode("utf-8"))

    def step_loss(self, loss_val: float) -> None:
        if math.isnan(loss_val) or math.isinf(loss_val):
            raise ValueError(f"Encountered NaN/Inf loss during training step: {loss_val}")

    def _lr_at(self, step: int, total: int, warmup: int) -> float:
        if step < warmup:
            return self.lr * (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return self.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    def train_on_token_stream(
        self,
        token_ids: Sequence[int],
        total_utf8_bytes: Optional[int] = None,
        seq_len: int = 64,
        batch_size: int = 8,
        token_bytes: Optional[np.ndarray] = None,
        log_every: int = 10,
    ) -> List[Dict[str, float]]:
        """Returns the training curve: one entry per ``log_every`` steps plus the first and last."""
        ids = np.asarray(token_ids, dtype=np.int64)
        if len(ids) < seq_len + 1:
            ids = np.resize(ids, seq_len + 1)
        nbytes = _token_byte_counts(ids, token_bytes, total_utf8_bytes)
        per_step = batch_size * seq_len * float(nbytes.mean())
        total_steps = max(1, math.ceil((self.byte_budget - self.bytes_consumed) / max(per_step, 1e-9)))
        if self.warmup_steps is not None:
            warmup = self.warmup_steps
        elif self.warmup_frac is not None:
            warmup = max(1, round(total_steps * self.warmup_frac))
        else:
            warmup = max(1, min(200, total_steps // 10))

        self.model.train()
        history: List[Dict[str, float]] = []
        n_windows = (len(ids) - 1) // seq_len
        window = 0
        step = 0
        while self.bytes_consumed < self.byte_budget:
            starts = []
            for _ in range(batch_size):
                if window >= n_windows:
                    window = 0
                    self.epochs = float(math.floor(self.epochs) + 1)
                starts.append(window * seq_len)
                window += 1
            batch = np.stack([ids[s : s + seq_len + 1] for s in starts])
            t = torch.from_numpy(batch).to(self.device)
            lr = self._lr_at(step, total_steps, warmup)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):
                logits = self.model(t[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), t[:, 1:].reshape(-1))
            loss_val = float(loss.item())
            self.step_loss(loss_val)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            step_bytes = float(sum(nbytes[s + 1 : s + seq_len + 1].sum() for s in starts))
            self.tokens_consumed += batch_size * seq_len
            self.bytes_consumed += max(1, int(round(step_bytes)))
            step += 1
            if step == 1 or step % log_every == 0 or self.bytes_consumed >= self.byte_budget:
                history.append({
                    "step": step,
                    "bytes_consumed": self.bytes_consumed,
                    "tokens_consumed": self.tokens_consumed,
                    "lr": lr,
                    "train_loss_nats": loss_val,
                    "train_bpb": loss_val * batch_size * seq_len / (math.log(2.0) * max(step_bytes, 1e-9)),
                })
        self.epochs = math.floor(self.epochs) + (window * seq_len) / max(1, len(ids) - 1)
        return history


def _check_stride(stride: int, seq_len: int) -> None:
    if not 1 <= stride <= seq_len:
        raise ValueError(f"eval stride must be in 1..seq_len ({seq_len}), got {stride}: a larger stride "
                         "leaves tokens between windows unscored")


class BpbEvaluator:
    """Bits per byte: total NLL of the scored tokens in bits / UTF-8 bytes of those tokens."""

    def compute_bpb(self, loss_nats: float, bytes_per_token: float) -> float:
        if bytes_per_token <= 0.0:
            return 0.0
        return loss_nats / (math.log(2.0) * bytes_per_token)

    @torch.no_grad()
    def evaluate_validation_bpb(
        self,
        model: nn.Module,
        token_ids: Sequence[int],
        utf8_bytes: int,
        seq_len: int = 64,
        stride: Optional[int] = None,
        token_bytes: Optional[np.ndarray] = None,
        batch_size: int = 16,
    ) -> Dict[str, float]:
        """
        Sliding-window evaluation: windows of ``seq_len`` inputs advance by ``stride`` (default
        seq_len // 2) and each window scores only targets no earlier window scored, so every
        token after the first is scored exactly once with at least ``seq_len - stride`` tokens of
        context (except near the start). The first token has no context and is excluded from both
        the NLL and the byte count. Bytes come from ``token_bytes`` when given, else the average
        ``utf8_bytes / len(token_ids)``. Fewer than two tokens (nothing to score) is an error, not
        a BPB of 0; so is a stride outside 1..seq_len (it would skip tokens between windows).
        """
        ids = np.asarray(token_ids, dtype=np.int64)
        if len(ids) < 2 or utf8_bytes <= 0:
            raise ValueError(f"validation text too short to score: {len(ids)} tokens, {utf8_bytes} bytes")
        stride = max(1, seq_len // 2) if stride is None else stride
        _check_stride(stride, seq_len)
        device = next(model.parameters()).device
        model.eval()

        # (input start, number of inputs, first target position to score)
        windows = []
        scored_to = 1
        start = 0
        n = len(ids)
        while scored_to < n:
            length = min(seq_len, n - 1 - start)
            windows.append((start, length, scored_to))
            scored_to = start + length + 1
            start += stride
        total_nll = 0.0
        scored = np.zeros(n, dtype=bool)
        for length in sorted({w[1] for w in windows}):
            group = [w for w in windows if w[1] == length]
            for i in range(0, len(group), batch_size):
                chunk = group[i : i + batch_size]
                batch = torch.from_numpy(np.stack([ids[s : s + length + 1] for s, _, _ in chunk])).to(device)
                logits = model(batch[:, :-1])
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(), batch[:, 1:].reshape(-1), reduction="none"
                ).view(len(chunk), length)
                for row, (s, _, first) in enumerate(chunk):
                    offset = first - (s + 1)
                    total_nll += float(nll[row, offset:].sum())
                    scored[first : s + length + 1] = True
        n_scored = int(scored.sum())
        nbytes = _token_byte_counts(ids, token_bytes, utf8_bytes)
        scored_bytes = float(nbytes[scored].sum())
        loss = total_nll / max(1, n_scored)
        return {
            "loss_nats": loss,
            "bytes_per_token": utf8_bytes / n,
            "bpb": total_nll / (math.log(2.0) * scored_bytes) if scored_bytes > 0 else 0.0,
            "scored_tokens": n_scored,
        }


def _mean_sd(xs: Sequence[float]) -> tuple:
    """(mean, sample sd); the sd is None (undefined) for a single seed."""
    return (statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else None)


def backend_info(device: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "device": device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "colab_runtime": "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ,
    }
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        info["cpu"] = platform.processor()
    return info


def _run_one(job_dir: str, name: str, seed: int, device: Optional[str] = None) -> Dict[str, Any]:
    """Train and evaluate one (tokenizer, seed) proxy LM; writes ``runs/<name>__s<seed>.json``."""
    job = Path(job_dir)
    cfg = json.loads((job / "config.json").read_text())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr = cfg["train"]
    arrays = np.load(job / f"{name}.npz")
    token_bytes = arrays["token_bytes"]
    train_ids = arrays["train"].astype(np.int64)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ProxyLanguageModel(vocab_size=cfg["tokenizers"][name]["vocab_size"], **cfg["lm"])
    bf16 = bool(tr.get("bf16")) and device == "cuda" and torch.cuda.is_bf16_supported()
    trainer = ByteBudgetTrainer(model, byte_budget=tr["byte_budget"], lr=tr["lr"], device=device,
                                weight_decay=tr.get("weight_decay", 0.01), warmup_frac=tr.get("warmup_frac"),
                                bf16=bf16)
    curve = trainer.train_on_token_stream(
        train_ids, seq_len=tr["seq_len"], batch_size=tr["batch_size"], token_bytes=token_bytes
    )
    ev = BpbEvaluator()
    val = {
        d: ev.evaluate_validation_bpb(
            trainer.model, arrays[f"val_{d}"], int(token_bytes[arrays[f"val_{d}"]].sum()),
            seq_len=tr["seq_len"], stride=tr.get("eval_stride"), token_bytes=token_bytes,
        )
        for d in cfg["domains"]
    }
    run = {"seed": seed, "curve": curve, "val": val, "steps": curve[-1]["step"], "bf16": bf16,
           "bytes_consumed": trainer.bytes_consumed, "tokens_consumed": trainer.tokens_consumed,
           "epochs": trainer.epochs, "train_bytes_per_token": float(token_bytes[train_ids].sum()) / max(1, len(train_ids))}
    out = job / "runs" / f"{name}__s{seed}.json"
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(run))
    os.replace(tmp, out)  # atomic: a partially written run file never looks finished
    return run


def run_job(job_dir: str, device: Optional[str] = None, log=print) -> Dict[str, Any]:
    """Train and evaluate one proxy LM per tokenizer and seed; writes and returns result.json.
    Each run writes its own ``runs/<name>__s<seed>.json``; runs whose file exists are not repeated, so
    a restarted job resumes. ``config["concurrency"]`` (default 1) runs that many at once in separate
    processes on the one device; the aggregation reads the run files, so it does not depend on order."""
    job = Path(job_dir)
    cfg = json.loads((job / "config.json").read_text())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr = cfg["train"]
    models: Dict[str, Any] = {}
    t_start = time.time()
    for name in cfg["tokenizers"]:
        with np.load(job / f"{name}.npz") as arrays:
            short = [d for d in cfg["domains"] if len(arrays[f"val_{d}"]) < 2]
        if short:  # checked before any training: an empty domain would otherwise fail hours later
            raise ValueError(f"{name}: validation domain(s) {short} have fewer than 2 tokens")
    if tr.get("eval_stride") is not None:  # likewise
        _check_stride(tr["eval_stride"], tr["seq_len"])
    todo = [(n, s) for n in cfg["tokenizers"] for s in cfg["seeds"]
            if not (job / "runs" / f"{n}__s{s}.json").exists()]
    workers = max(1, int(cfg.get("concurrency", 1)))
    if workers == 1:
        for name, seed in todo:
            run = _run_one(job_dir, name, seed, device)
            log(f"  {name} seed={seed}: steps={run['steps']} initial_loss={run['curve'][0]['train_loss_nats']:.3f} "
                f"macro_bpb={statistics.fmean(v['bpb'] for v in run['val'].values()):.4f} "
                f"({time.time() - t_start:.0f}s)")
    elif todo:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            futs = {pool.submit(_run_one, job_dir, n, s, device): (n, s) for n, s in todo}
            for fut in cf.as_completed(futs):
                run = fut.result()
                log(f"  {futs[fut][0]} seed={futs[fut][1]}: steps={run['steps']} "
                    f"({time.time() - t_start:.0f}s)")
    for name, tcfg in cfg["tokenizers"].items():
        runs = [json.loads((job / "runs" / f"{name}__s{s}.json").read_text()) for s in cfg["seeds"]]
        val_domains = {}
        for d in cfg["domains"]:
            mean, sd = _mean_sd([r["val"][d]["bpb"] for r in runs])
            val_domains[d] = {"bpb": mean, "bpb_sd": sd,
                              "loss_nats": statistics.fmean(r["val"][d]["loss_nats"] for r in runs),
                              "bytes_per_token": runs[0]["val"][d]["bytes_per_token"],
                              "scored_tokens": runs[0]["val"][d]["scored_tokens"]}
        macro = [statistics.fmean(r["val"][d]["bpb"] for d in cfg["domains"]) for r in runs]
        mean, sd = _mean_sd(macro)
        models[name] = {
            "vocab_size": tcfg["vocab_size"],
            "train_bytes_per_token": runs[0]["train_bytes_per_token"],
            "steps": runs[0]["steps"],
            "bytes_consumed": runs[0]["bytes_consumed"],
            "tokens_consumed": runs[0]["tokens_consumed"],
            "epochs": runs[0]["epochs"],
            "initial_loss_nats": [r["curve"][0]["train_loss_nats"] for r in runs],
            "ln_vocab": math.log(tcfg["vocab_size"]),
            "val_domains": val_domains,
            "macro_val_bpb": mean,
            "macro_val_bpb_sd": sd,
            "macro_val_bpb_by_seed": dict(zip(map(str, cfg["seeds"]), macro)),
            "curve": runs[0]["curve"],
        }
    result = {"backend": backend_info(device), "architecture": ARCHITECTURE,
              "seconds": round(time.time() - t_start, 1), "models": models}
    (job / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main(job_dir: str) -> int:
    """Entry point of the detached job on the Colab VM. Writes ``status.json`` last, atomically:
    {"state": "ok"} or {"state": "error", "traceback": ...}; the orchestrator polls for it."""
    try:
        run_job(job_dir)
        status: Dict[str, Any] = {"state": "ok"}
    except BaseException:
        status = {"state": "error", "traceback": traceback.format_exc()}
    tmp = Path(job_dir) / "status.json.tmp"
    tmp.write_text(json.dumps(status))
    os.replace(tmp, Path(job_dir) / "status.json")
    return 0 if status["state"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
