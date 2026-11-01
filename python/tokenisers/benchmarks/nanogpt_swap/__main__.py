"""
CLI: tokenizer-swap benchmark on llm.c's GPT-2 (124M) recipe, run on one Colab GPU session.

    uv run python -m tokenisers.benchmarks.nanogpt_swap --job-dir DIR --prepare-only   # local: tokenizers
    uv run python -m tokenisers.benchmarks.nanogpt_swap --job-dir DIR --pilot --gpu L4  # ~20 timed steps
    uv run python -m tokenisers.benchmarks.nanogpt_swap --job-dir DIR --gpu L4          # 12 x 600 steps
    uv run python -m tokenisers.benchmarks.nanogpt_swap --job-dir DIR --gpu L4 --resume # after a drop

``--tokenizers`` picks what is prepared (default: GPT-2, the two TokenMonster vocabularies and our
9); a run trains every tokenizer of the job's config.json unless ``--tokenizers`` narrows it. The
``_sp`` names (SP_VARIANTS) are our variants retrained with ``pretokenizer="space_prefix"``, the
``_g2`` names (G2_VARIANTS) with ``pretokenizer="gpt2"``: prepare them in a job dir of their own,
e.g. ``--prepare-only --tokenizers bpe_frequency_sp ...``. ``--from-pool DIR ...`` prepares a job
from tokenizers already prepared in other job dirs (no training): e.g. a few ``_g2`` ones and the
``_sp300`` ones that ``python -m tokenisers.benchmarks.nanogpt_swap.train_tokenizers`` trains on
300 MB on a Colab VM.

Preparation (local, CPU): the first ``VAL_TOKENS`` GPT-2 tokens of the speedrun's val shard and the
head of train shard 1 are read by HTTP range requests and decoded; our 9 tokenizers are trained on
the first ~50 MB of train documents (``train_suite``, default parameters) at 50,256 tokens, so with
the separator the vocabulary is 50,257 like GPT-2's. On the VM, ``swap_job.py`` decodes the full
shards and verifies that its val documents and first train documents hash to the same values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tarfile
import time
from typing import Any, Dict, List, Optional

import numpy as np

from tokenisers.benchmarks.nanogpt_swap import swap_job

REPO = Path(__file__).resolve().parents[4]
VENDOR = REPO / "vendor" / "llmc"
TRAINER = VENDOR / "train_gpt2_swap.py"
UPSTREAM_TRAINER = VENDOR / "train_gpt2.py"
UPSTREAM_SCRIPT = VENDOR / "pyrun_gpt2_124M.sh"  # llm.c's PyTorch GPT-2 124M run: the recipe's hyperparameters
LLMC_COMMIT = "f1e2ace651495b74ae22d45d1723443fd00ecd3a"
JOB_SCRIPT = Path(swap_job.__file__)

SOURCE_REPO = "kjj0/fineweb10B-gpt2"  # modded-nanogpt data/cached_fineweb10B.py
VAL_SHARD = "fineweb_val_000000.bin"
TRAIN_SHARDS = [f"fineweb_train_{i:06d}.bin" for i in range(1, 104)]
VAL_TOKENS = 20 * 524288  # the speedrun's val window (llm.c's record uses the same first ~10M tokens)
SPEEDRUN_TARGET_NATS = 3.28  # modded-nanogpt's target: llm.c's GPT-2 (124M) val loss on GPT-2 tokens
SOURCE = {"repo": SOURCE_REPO, "val_shard": VAL_SHARD, "val_tokens": VAL_TOKENS, "train_shards": TRAIN_SHARDS,
          "tokenizer": {"kind": "tiktoken", "encoding": "gpt2"}}  # the text every tokenizer is scored on

OUR_VARIANTS = ("bpe_frequency", "wp_bpe_pig", "wp_bpe_ppig", "wp_bpe_npmi", "super_bpe", "unigram_lm",
                "length_penalized", "morpheme_bpe", "mdl_bits")
OUR_VOCAB = 50_256  # + 1 separator = 50,257, GPT-2's size
# Every prebuilt englishcode TokenMonster vocabulary NFD-normalizes its input (header byte 2), so
# decode(encode(x)) != x on ~7% of FineWeb val documents; capcode (the non-"nocapcode" vocabularies)
# also folds U+0130 "İ" to "I". The closest lossless variant: the nocapcode vocabulary of the same
# size and mode with the normalization flag cleared (swap_job.TokenMonsterTok), lossless on every
# val document (checked again on the VM before each run).
TOKENMONSTER = {
    "tm_englishcode_50256": {"requested": "englishcode-50256-consistent-v1",
                             "vocab": "englishcode-50256-consistent-nocapcode-v1",
                             "vocab_sha256": "875b8dae84e79909f51f8e6299d55701a8c938ec74f5e44b3e6ca7182ba658d8"},
    "tm_englishcode_32000": {"requested": "englishcode-32000-consistent-v1",
                             "vocab": "englishcode-32000-consistent-nocapcode-v1",
                             "vocab_sha256": "adf653f934fea8d41fc3f3059e73d8157c5738db685c6e2d073f050d66a9b801"},
}
TOKENMONSTER_WHY = ("every prebuilt englishcode vocabulary NFD-normalizes its input (lossy on 1,053 of 14,982 val "
                    "documents); capcode also folds U+0130 to I (2 documents). Used: the nocapcode vocabulary of the "
                    "same size and mode, normalization flag cleared (same tokens and ids), lossless on all val documents")
TOKENIZERS = ("gpt2", *TOKENMONSTER, *OUR_VARIANTS)
# Our variants that pre-split text, retrained with pretokenizer="space_prefix" / "gpt2"
# (common.variant_config).
SP_VARIANTS = tuple(f"{v}_sp" for v in OUR_VARIANTS if v != "super_bpe")
G2_VARIANTS = tuple(f"{v}_g2" for v in OUR_VARIANTS if v != "super_bpe")
PIP = ["tiktoken==0.14.0", "tokenmonster==1.1.12"]
GPUS = ("L4", "A100", "H100")  # bf16; a T4 has none
DEFAULT_MBS = {"L4": 16, "A100": 32, "H100": 32}  # sequences per micro-batch; halved on the VM on OOM
# Fields of the llm.c run script this benchmark may change (identically for every tokenizer):
# schedule length (warmup scaled with it), micro-batch, val cadence, data and log paths.
ALLOWED_OVERRIDES = {"num_iterations", "warmup_iters", "batch_size", "val_loss_every", "input_bin",
                     "input_val_bin", "output_dir"}
# Opt-in only (--lr): the same peak learning rate for every tokenizer; recorded in trainer_args.
LR_OVERRIDE = "learning_rate"
IDS_DOCS = 200  # documents whose ids are hashed at preparation and checked on the VM


# ----------------------------------------------------------------------------
# Recipe
# ----------------------------------------------------------------------------

def upstream_args() -> Dict[str, str]:
    """``--flag value`` pairs passed to train_gpt2.py by llm.c's scripts/pyrun_gpt2_124M.sh."""
    script_args = UPSTREAM_SCRIPT.read_text().rsplit("train_gpt2.py", 1)[1]  # the torchrun line
    return {k: v.strip('"') for k, v in re.findall(r"--(\w+) (\S+)", script_args)}


def recipe_args(steps: int, eval_every: int, lr: Optional[float] = None) -> Dict[str, str]:
    """The llm.c run's arguments with the allowed changes: ``steps`` iterations, warmup scaled by the
    same fraction (700 of 18,865 -> 22 of 600), val every ``eval_every`` steps. Batch size and data
    paths are set per run on the VM. Cosine decay to lr x learning_rate_decay_frac is unchanged.
    ``lr`` is the one opt-in deviation (LR_OVERRIDE): a peak learning rate shared by every tokenizer,
    e.g. 1.8e-3, the 3x AdamW lr of modded-nanogpt's early records; None keeps llm.c's 6e-4."""
    up = upstream_args()
    out = dict(up, num_iterations=str(steps), val_loss_every=str(eval_every),
               warmup_iters=str(max(1, round(int(up["warmup_iters"]) * steps / int(up["num_iterations"])))))
    if lr is not None:
        out[LR_OVERRIDE] = repr(float(lr))
    for k in ("batch_size", "input_bin", "input_val_bin", "output_dir"):
        del out[k]
    return out


# ----------------------------------------------------------------------------
# Local preparation
# ----------------------------------------------------------------------------

def fetch_head(name: str, ntok: int, path: Path) -> None:
    """The first ``ntok`` tokens of a source shard by HTTP range read (anonymous), saved as a .bin."""
    from huggingface_hub import HfFileSystem

    with HfFileSystem(token=False).open(f"datasets/{SOURCE_REPO}/{name}", "rb", block_size=8 << 20) as f:
        raw = f.read(1024 + 2 * ntok)
    path.parent.mkdir(parents=True, exist_ok=True)
    swap_job.write_bin(path, np.frombuffer(raw[1024:], dtype=np.uint16))


def prepare(args: argparse.Namespace, job: Path, start: Dict[str, Any]) -> None:
    """Decodes the val window and the tokenizer text locally, trains, saves and verifies our
    tokenizers, and writes config.json and tokenizers.tar.gz into ``job``."""
    try:
        import tiktoken
    except ImportError as e:
        raise SystemExit("preparation decodes the GPT-2 shards with tiktoken: uv sync --extra dev") from e
    from tokenisers.benchmarks.common import (DEFAULT_MORPH_LANGS, check_memory, find_duplicates,
                                              load_morphology, parse_variant, train_suite)

    wanted = args.tokenizers or TOKENIZERS
    scaled = [n for n in wanted if n not in TOKENIZERS and parse_variant(n)[2] is not None]
    if scaled:
        raise SystemExit(f"{', '.join(scaled)}: trained on more text than this machine holds; train them with "
                         "`python -m tokenisers.benchmarks.nanogpt_swap.train_tokenizers` and prepare the job "
                         "with --from-pool")
    variants = [v for v in (*OUR_VARIANTS, *SP_VARIANTS, *G2_VARIANTS) if v in wanted]
    if variants and not args.skip_memory_check:
        check_memory(variants, args.tokenizer_train_bytes)
    enc = tiktoken.get_encoding("gpt2")
    cache = job / "source"
    head_tokens = int(args.tokenizer_train_bytes / 3.5) + 1_000_000  # GPT-2 gives ~4.4 bytes/token on FineWeb
    for name, ntok in ((VAL_SHARD, VAL_TOKENS), (TRAIN_SHARDS[0], head_tokens)):
        if not (cache / name).exists():
            print(f"reading the first {ntok:,} tokens of {name}")
            fetch_head(name, ntok, cache / name)
    gpt2 = swap_job.TiktokenTok({"encoding": "gpt2"}, job)
    source = swap_job.SourceDocs(lambda n: cache / n, gpt2.decode, enc.eot_token, VAL_SHARD, VAL_TOKENS,
                                 [TRAIN_SHARDS[0]])
    docs, nbytes = source.first_docs(args.tokenizer_train_bytes)
    if nbytes < 0.9 * args.tokenizer_train_bytes:
        raise SystemExit(f"the shard head holds only {nbytes:,} bytes of documents")
    source.assert_disjoint()
    val_bytes = sum(len(d.encode()) for d in source.val)
    print(f"val: {len(source.val):,} docs, {val_bytes:,} bytes; tokenizer text: {len(docs):,} train docs, "
          f"{nbytes:,} bytes ({source.dropped_as_val} train docs dropped as val duplicates)")

    specs: List[Dict[str, Any]] = []
    for name in TOKENIZERS:
        if name == "gpt2" and name in wanted:
            specs.append({"name": name, "kind": "tiktoken", "encoding": "gpt2"})
        elif name in TOKENMONSTER and name in wanted:
            specs.append({"name": name, "kind": "tokenmonster", "clear_normalization": True, **TOKENMONSTER[name]})
    records: Dict[str, Any] = {}
    morph = None
    duplicates: Dict[str, str] = {}
    if variants:
        lexicon: Dict[str, List[str]] = {}
        if any(parse_variant(v)[0] == "morpheme_bpe" for v in variants):  # SIGMORPHON train split, as elsewhere
            margs = argparse.Namespace(cache_dir=args.cache_dir, offline=False)
            lexicon, _, morph = load_morphology(margs, docs, DEFAULT_MORPH_LANGS)
        print(f"training {len(variants)} tokenizers at V={OUR_VOCAB} on {nbytes / 1e6:.1f} MB")
        models, records = train_suite(variants, docs, OUR_VOCAB, lexicon)
        duplicates = find_duplicates(models)
        for name, tok in models.items():
            path = f"tokenizers/{name}"
            tok.save(str(job / path))
            digest = swap_job.ids_sha256(tok.encode(d) for d in docs[:IDS_DOCS])
            again = type(tok).from_pretrained(str(job / path))
            if swap_job.ids_sha256(again.encode(d) for d in docs[:IDS_DOCS]) != digest:
                raise SystemExit(f"{name}: save -> from_pretrained changes the ids")
            specs.append({"name": name, "kind": "tokenisers", "class": type(tok).__name__, "path": path,
                          "pretokenizer": tok.pretokenizer, "ids_sha256": digest, "ids_docs": IDS_DOCS})
        with tarfile.open(job / "tokenizers.tar.gz", "w:gz") as tar:
            tar.add(job / "tokenizers", arcname="tokenizers")
    cfg = {
        "source": SOURCE,
        "val_docs_sha256": swap_job.docs_sha256(source.val),
        "tokenizer_text": {"docs": len(docs), "bytes": nbytes, "docs_sha256": swap_job.docs_sha256(docs),
                           "shard": TRAIN_SHARDS[0]},
        "pip": PIP, "headroom": 0.01, "tokenizers": specs,
        "prepared": {**{k: start[k] for k in ("command", "git_commit", "git_dirty", "native_extension")},
                     "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "val_docs": len(source.val), "val_bytes": val_bytes,
                     "train_docs_dropped_as_val": source.dropped_as_val, "variants": records,
                     "duplicates": duplicates, "morphology": morph, "tokenmonster_why": TOKENMONSTER_WHY},
    }
    (job / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def assemble(pools: List[Path], names: List[str], job: Path, start: Dict[str, Any]) -> None:
    """Prepares ``job`` from tokenizers already prepared in the ``pools`` (job dirs written by
    ``prepare`` or by train_tokenizers.py), without training or reading any data: their files are
    copied, their specs and training records kept, and each spec records the text it was trained on
    (``tokenizer_text``, checked again on the VM). The pools must share the source and val
    documents; the job's own ``tokenizer_text`` is the first pool's."""
    cfgs = [(p, json.loads((p / "config.json").read_text(encoding="utf-8"))) for p in pools]
    first = cfgs[0][1]
    for p, c in cfgs[1:]:
        if c["source"] != first["source"] or c["val_docs_sha256"] != first["val_docs_sha256"]:
            raise SystemExit(f"{p}: another source or other val documents than {pools[0]}")
    specs: List[Dict[str, Any]] = []
    records: Dict[str, Any] = {}
    morph = None
    for name in names:
        found = [(p, c, s) for p, c in cfgs for s in c["tokenizers"] if s["name"] == name]
        if not found:
            raise SystemExit(f"{name}: in none of {', '.join(map(str, pools))}")
        pool, c, spec = found[0]
        spec = dict(spec)
        if spec["kind"] == "tokenisers":
            if not (pool / spec["path"]).is_dir():
                with tarfile.open(pool / "tokenizers.tar.gz") as tar:
                    tar.extractall(pool, filter="data")
            shutil.copytree(pool / spec["path"], job / spec["path"], dirs_exist_ok=True)
            spec.setdefault("tokenizer_text", c["tokenizer_text"])
        specs.append(spec)
        if name in (c["prepared"].get("variants") or {}):
            records[name] = dict(c["prepared"]["variants"][name])
        if name.startswith("morpheme_bpe") and c["prepared"].get("morphology"):
            morph = morph or c["prepared"]["morphology"]  # the lexicon depends on the training text:
            records.setdefault(name, {})["morphology"] = c["prepared"]["morphology"]  # each keeps its own
    if any(s["kind"] == "tokenisers" for s in specs):
        with tarfile.open(job / "tokenizers.tar.gz", "w:gz") as tar:
            tar.add(job / "tokenizers", arcname="tokenizers")
    seen: Dict[str, str] = {}
    duplicates: Dict[str, str] = {}
    for s in specs:
        if s["kind"] == "tokenisers":
            digest = hashlib.sha256((job / s["path"] / "tokenizer.json").read_bytes()).hexdigest()
            if digest in seen:
                duplicates[s["name"]] = seen[digest]
            seen.setdefault(digest, s["name"])
    # Each pool's own preparation (command, commit, where it trained: e.g. `trained_on` a Colab VM)
    # stays under `pools`; the job level keeps only what every pool shares (e.g. the val documents).
    per_tokenizer = ("variants", "duplicates", "morphology", "pools")  # rebuilt above, per tokenizer
    pool_prep = {str(p): {k: v for k, v in c["prepared"].items() if k not in per_tokenizer} for p, c in cfgs}
    shared = {k: v for k, v in pool_prep[str(pools[0])].items() if all(pp.get(k) == v for pp in pool_prep.values())}
    prepared = {"tokenmonster_why": TOKENMONSTER_WHY, **shared,
                **{k: start[k] for k in ("command", "git_commit", "git_dirty", "native_extension")},
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "variants": records,
                "duplicates": duplicates, "morphology": morph, "pools": pool_prep}
    cfg = {k: first[k] for k in ("source", "val_docs_sha256", "tokenizer_text", "pip", "headroom")}
    cfg.update(tokenizers=specs, prepared=prepared)
    (job / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------

def target_bpb(gpt2_val: Dict[str, Any]) -> float:
    """The speedrun's 3.28 nats per GPT-2 token, in bits per byte of this val text."""
    return SPEEDRUN_TARGET_NATS / math.log(2) / gpt2_val["bytes_per_token"]


def step_ms(trainer: Dict[str, Any]) -> Optional[float]:
    """Median wall time of a training step: steps without a val pass, the first (compile) excluded."""
    evals = {e["step"] for e in trainer["evals"]}
    times = [s["seconds"] for s in trainer["steps"] if s["step"] > 0 and s["step"] not in evals]
    return 1000 * statistics.median(times) if times else None


def summarize(run: Dict[str, Any], target: Optional[float]) -> Dict[str, Any]:
    """One tokenizer's row: curve, final BPB, first step at the target, data and speed."""
    tr = run["trainer"]
    out = {"vocab_size": run["vocab_size"], "padded_vocab_size": run["padded_vocab_size"],
           "val_bytes_per_token": run["val"]["bytes_per_token"], "val_targets": run["val"]["targets"],
           "val_docs_sha256": run["val"]["docs_sha256"], "encode_mb_per_s": run["val"]["encode_mb_per_s"],
           "spec": run["spec"]}
    if not tr:
        return out
    curve = [[e["step"], e["val_bpb"], e["val_loss"]] for e in tr["evals"]]
    ms = step_ms(tr)
    batch = run["train"]["tokens_per_step"]
    out.update(
        curve=curve, final_bpb=curve[-1][1], final_nats_per_token=curve[-1][2],
        reached_target_step=None if target is None else next((s for s, b, _ in curve if b <= target), None),
        train_bytes_per_token=run["train"]["bytes_per_token"], train_tokens=run["train"]["tokens"],
        train_docs=run["train"]["docs"], params=tr["params"], mbs=tr["mbs"], grad_accum_steps=tr["grad_accum_steps"],
        ms_per_step=ms, tokens_per_s=batch / ms * 1000 if ms else None,
        eval_seconds=[round(e["seconds"], 1) for e in tr["evals"]], eval_targets=tr["evals"][-1]["val_targets"],
        peak_memory_bytes=tr["peak_memory_bytes"],
        trainer_seconds=tr["process_seconds"], wall_seconds=run["seconds"], gpu=tr["gpu"], torch=tr["torch"],
    )
    return out


def collect(job: Path, cfg: Dict[str, Any], result: Dict[str, Any], gpu: str, start: Dict[str, Any]) -> Dict[str, Any]:
    runs = {p.stem: json.loads(p.read_text()) for p in sorted((job / "runs").glob("*.json"))}
    missing = sorted(set(result.get("runs") or []) - set(runs))
    if missing:  # finished on the VM, never downloaded: the results would silently leave them out
        raise SystemExit(f"result.json lists runs not in {job / 'runs'}: {', '.join(missing)}; rerun with --resume")
    gpt2 = runs.get("gpt2")
    target = target_bpb(gpt2["val"]) if gpt2 else None
    order = [s["name"] for s in cfg["tokenizers"] if s["name"] in runs]
    meta = {
        "benchmark": "nanogpt_swap", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **start,
        "mode": cfg["mode"], "requested_gpu": gpu, "backend": result.get("backend"), "timings": result.get("timings"),
        "data": result.get("data"), "recipe": {"upstream": f"karpathy/llm.c@{LLMC_COMMIT}",
                                               "trainer": "vendor/llmc/train_gpt2_swap.py",
                                               "args": cfg["trainer_args"], "mbs_requested": cfg["mbs"]},
        "target": {"nats_per_gpt2_token": SPEEDRUN_TARGET_NATS, "bpb": target},
        "prepared": cfg["prepared"], "tokenizer_text": cfg["tokenizer_text"], "source": cfg["source"],
    }
    doc = {"meta": meta, "models": {n: summarize(runs[n], target) for n in order}}
    if gpt2 and gpt2["trainer"] and cfg["mode"] == "full":
        doc["gpt2_equivalent"] = gpt2_equivalent(gpt2, doc["models"])
    return doc


EQUIV_WINDOW = 21  # steps: centred rolling mean over GPT-2's per-step train loss (one 524,288-token batch each)


def gpt2_equivalent(gpt2: Dict[str, Any], models: Dict[str, Any], window: int = EQUIV_WINDOW) -> Dict[str, Any]:
    """GPT-2-equivalent steps: per tokenizer, the first step at which GPT-2's curve reaches that
    tokenizer's final val BPB, i.e. how many of GPT-2's steps buy the same quality (GPT-2 itself: its
    step count, by definition). GPT-2's per-step curve is its train loss in bits per byte of its train
    text (nats / ln 2 / train bytes per token), smoothed by a centred rolling mean over ``window``
    steps (fewer at both ends); the train text is one pass over unseen documents, so it tracks val.
    ``train_minus_val_bpb`` checks that against every val pass after step 0 (the last val pass,
    after the last step, against the last step's train loss)."""
    steps = [s["step"] for s in gpt2["trainer"]["steps"]]
    x = np.array([s["train_loss"] for s in gpt2["trainer"]["steps"]]) / math.log(2) / gpt2["train"]["bytes_per_token"]
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    lo, hi = np.maximum(i - window // 2, 0), np.minimum(i + window // 2 + 1, len(x))
    curve = (c[hi] - c[lo]) / (hi - lo)
    out: Dict[str, Optional[int]] = {}
    for name, m in models.items():
        hit = np.flatnonzero(curve <= m["final_bpb"]) if "final_bpb" in m else []
        out[name] = steps[-1] + 1 if name == "gpt2" else (steps[hit[0]] if len(hit) else None)
    at = dict(zip(steps, curve.tolist()))
    check = [[e["step"], at[min(e["step"], steps[-1])] - e["val_bpb"]] for e in gpt2["trainer"]["evals"] if e["step"] > 0]
    return {"window": window, "steps": out, "train_minus_val_bpb": check}


def train_steps_csv(runs_dir: Path) -> str:
    """The runs' step curves in one table: step, lr (the same for every tokenizer), and each
    tokenizer's train loss in nats per its own token."""
    runs = {p.stem: json.loads(p.read_text())["trainer"]["steps"] for p in sorted(runs_dir.glob("*.json"))}
    first = next(iter(runs.values()))
    if any([(s["step"], s["lr"]) for s in r] != [(s["step"], s["lr"]) for s in first] for r in runs.values()):
        raise ValueError("the runs differ in their steps or learning rates")
    rows = [",".join(["step", "lr", *runs])]
    rows += [",".join(map(str, [s["step"], s["lr"], *(r[k]["train_loss"] for r in runs.values())]))
             for k, s in enumerate(first)]
    return "\n".join(rows) + "\n"


def flops_per_token(padded_vocab: int, n_layer: int = 12, d: int = 768, seq: int = 1024) -> float:
    """Training FLOPs per token of the GPT-2 (124M) model: 6 N + 12 L d T (nanoGPT's estimate)."""
    params = n_layer * 12 * d * d + padded_vocab * d + seq * d  # blocks (no biases/norms) + tied wte + wpe
    return 6 * params + 12 * n_layer * d * seq


def project(doc: Dict[str, Any], steps: int = 600, eval_every: int = 50, n_tokenizers: int = 12) -> Dict[str, Any]:
    """Full-run wall time from a pilot: per tokenizer, steps x measured ms/step (scaled by FLOPs per
    token for a vocabulary that was not trained in the pilot), evals x the warm eval time per val
    token x its full val tokens, the compile/start overhead of a trainer process and the train
    encoding; plus the one-time setup and the decoding of the source shards the most compact
    tokenizer needs. The overhead of an untimed tokenizer is that of the last timed run with the
    same padded vocabulary (compiled kernels are cached on the VM), else of the first (cold)."""
    models = doc["models"]
    trained = {n: m for n, m in models.items() if m.get("ms_per_step")}
    ref_name, ref = next(iter(trained.items()))

    def overhead(m: Dict[str, Any]) -> float:  # trainer time beyond steps and warm evals: start, data, compile
        return max(0.0, m["trainer_seconds"] - m["curve"][-1][0] * m["ms_per_step"] / 1000
                   - len(m["eval_seconds"]) * m["eval_seconds"][-1])

    warm = {m["padded_vocab_size"]: overhead(m) for m in trained.values()}
    n_evals = steps // eval_every + (1 if steps % eval_every else 0) + 1
    batch = 524288
    rows = {}
    for name, m in models.items():
        base = trained.get(name, ref)
        scale = flops_per_token(m["padded_vocab_size"]) / flops_per_token(base["padded_vocab_size"])
        ms = base["ms_per_step"] * scale
        warm_eval = base["eval_seconds"][-1] * scale * m["val_targets"] / base["eval_targets"]
        start_s = overhead(m) if name in trained else warm.get(m["padded_vocab_size"], overhead(ref))
        train_bytes = steps * batch * 1.01 * m["val_bytes_per_token"]
        encode_s = train_bytes / (m["encode_mb_per_s"] * 1e6)
        rows[name] = {"ms_per_step": ms, "estimated_from": None if name in trained else ref_name,
                      "train_hours": steps * ms / 3.6e6, "eval_hours": n_evals * warm_eval / 3600,
                      "compile_and_start_hours": start_s / 3600, "encode_hours": encode_s / 3600}
        rows[name]["total_hours"] = sum(v for k, v in rows[name].items() if k.endswith("_hours"))
    t = doc["meta"]["timings"] or {}
    setup_s = (t.get("setup", {}).get("pip_seconds", 0) + (t.get("setup", {}).get("build") or {}).get("seconds", 0))
    max_bpt = max(m["val_bytes_per_token"] for m in models.values())
    data = doc["meta"]["data"] or {}
    shard_s = (t.get("decode", {}).get("train_decode", 0) + sum(v for k, v in t.get("download", {}).items()
                                                                  if k.startswith("fineweb_train")))
    shard_gb = data.get("train_bytes_decoded", 0) / 1e9 or float("nan")
    need_gb = steps * batch * 1.01 * max_bpt / 1e9
    source_s = shard_s / shard_gb * need_gb
    total = sum(r["total_hours"] for r in rows.values()) * n_tokenizers / len(rows) + (setup_s + source_s) / 3600
    return {"steps": steps, "evals_per_run": n_evals, "per_tokenizer": rows,
            "setup_hours": setup_s / 3600, "source_text_gb": need_gb, "source_decode_hours": source_s / 3600,
            "total_hours": total, "tokenizers_measured": len(rows), "tokenizers_projected": n_tokenizers}


def write_outputs(doc: Dict[str, Any], out: Path, stem: str) -> None:
    from tokenisers.benchmarks.common import write_json
    from tokenisers.benchmarks.report import render_nanogpt_swap

    write_json(out / f"{stem}.json", doc)
    md = render_nanogpt_swap(doc)
    (out / f"{stem}.md").write_text(md + "\n", encoding="utf-8")
    print(md)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def tokenizer_name(name: str) -> str:
    """A --tokenizers name: GPT-2, TokenMonster, one of our variants or its ``_sp`` / ``_g2`` retrain,
    optionally on more text (``_sp300``: 300 MB, see common.parse_variant)."""
    from tokenisers.benchmarks.common import parse_variant

    try:
        if name in TOKENIZERS or parse_variant(name)[0] in OUR_VARIANTS:
            return name
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"unknown tokenizer {name!r}")


def main(argv: Optional[List[str]] = None) -> None:
    from tokenisers.benchmarks.common import start_provenance

    parser = argparse.ArgumentParser(description="Tokenizer swap on llm.c's GPT-2 (124M) recipe (Colab GPU)")
    parser.add_argument("--job-dir", type=Path, required=True,
                        help="Prepared tokenizers, config and finished run files (prepared here if it has no config.json)")
    parser.add_argument("--gpu", default="L4", help=f"Colab GPU: {', '.join(GPUS)} (bf16 needed; not T4)")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=50, help="Val BPB every N steps, and after the last")
    parser.add_argument("--lr", type=float, default=None,
                        help="Peak learning rate for every tokenizer (default: llm.c's 6e-4, unchanged)")
    parser.add_argument("--tokenizers", nargs="+", default=None, type=tokenizer_name, metavar="NAME",
                        help="Prepare these (default: the original 12, without the _sp / _g2 retrains); run these "
                             "of the job's config (default: all of them)")
    parser.add_argument("--from-pool", nargs="+", type=Path, default=None, metavar="DIR",
                        help="Prepare the job from the --tokenizers prepared in these job dirs (no training)")
    parser.add_argument("--mbs", type=int, default=None,
                        help="Sequences per micro-batch (default by GPU; halved on the VM when out of memory). "
                             "The step stays 524,288 tokens through gradient accumulation")
    parser.add_argument("--resume", action="store_true",
                        help="Continue a run whose session dropped: finished tokenizers (job-dir/runs) are kept")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare the job dir locally; no Colab")
    parser.add_argument("--pilot", action="store_true",
                        help="Time the recipe: --pilot-steps steps for --pilot-train, val encoding for every tokenizer")
    parser.add_argument("--pilot-steps", type=int, default=20)
    parser.add_argument("--pilot-train", nargs="+", default=["gpt2", "bpe_frequency"], choices=TOKENIZERS)
    parser.add_argument("--pilot-eval-docs", type=int, default=2000,
                        help="The pilot times its val passes on this many val documents (scaled to all in the projection)")
    parser.add_argument("--tokenizer-train-bytes", type=int, default=50_000_000,
                        help="Train-document bytes our tokenizers are trained on")
    parser.add_argument("--cache-dir", default=None, help="SIGMORPHON cache (default: the data pipeline's)")
    parser.add_argument("--skip-memory-check", action="store_true")
    parser.add_argument("--timeout", type=float, default=30 * 3600, help="Colab job wall-clock limit (s)")
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--colab-bin", default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO / "benchmarks" / "results" / "nanogpt_swap")
    args = parser.parse_args(argv)
    if args.gpu not in GPUS:
        parser.error(f"--gpu {args.gpu}: the recipe trains in bf16; use one of {', '.join(GPUS)} (a T4 has no bf16)")
    start = start_provenance(__package__, argv)

    job = args.job_dir
    job.mkdir(parents=True, exist_ok=True)
    if not (job / "config.json").is_file():
        if args.from_pool:
            if not args.tokenizers:
                parser.error("--from-pool needs --tokenizers")
            assemble(args.from_pool, args.tokenizers, job, start)
        else:
            prepare(args, job, start)
    if args.prepare_only:
        return
    cfg = json.loads((job / "config.json").read_text(encoding="utf-8"))
    names = [s["name"] for s in cfg["tokenizers"] if args.tokenizers is None or s["name"] in args.tokenizers]
    if args.pilot:
        run_dir = job / f"pilot-{args.gpu}"
        if not args.resume:
            shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(exist_ok=True)
        steps, eval_every, train = args.pilot_steps, args.pilot_steps, [n for n in args.pilot_train if n in names]
    else:
        run_dir, steps, eval_every, train = job, args.steps, args.eval_every, names
        held = sorted((job / "runs").glob("*.json"))
        if held and not args.resume:
            raise SystemExit(f"{job / 'runs'} holds {len(held)} finished runs: pass --resume to keep them")
        if held and cfg.get("trainer_args", {}).get("num_iterations") not in (None, str(steps)):
            raise SystemExit("--resume with a different --steps would mix schedules")
        if held and cfg.get("trainer_args", {}).get(LR_OVERRIDE) != recipe_args(steps, eval_every, args.lr).get(LR_OVERRIDE):
            raise SystemExit("--resume with a different --lr would mix learning rates")
    cfg.update(train=train, trainer_args=recipe_args(steps, eval_every, args.lr), mbs=args.mbs or DEFAULT_MBS[args.gpu],
               mode="pilot" if args.pilot else "full", gpu=args.gpu,
               eval_docs=args.pilot_eval_docs if args.pilot else None)
    # The job's config.json keeps every prepared tokenizer (a later --resume runs them all); the VM
    # gets its own config.json with only this run's.
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    vm_cfg = dict(cfg, tokenizers=[s for s in cfg["tokenizers"] if s["name"] in names])
    (run_dir / "vm").mkdir(exist_ok=True)
    (run_dir / "vm" / "config.json").write_text(json.dumps(vm_cfg, indent=1), encoding="utf-8")
    files = [run_dir / "vm" / "config.json", TRAINER, REPO / "python" / "tokenisers" / "benchmarks" / "throughput.py"]
    if any(s["kind"] == "tokenisers" for s in vm_cfg["tokenizers"]):
        if start["git_dirty"] is not False:
            raise SystemExit("the VM builds the committed code (`git archive HEAD`): commit code changes first")
        subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(run_dir / "src.tar.gz"), start["git_commit"]],
                       cwd=REPO, check=True)
        files += [job / "tokenizers.tar.gz", run_dir / "src.tar.gz"]

    from tokenisers.benchmarks.colab import ColabError, ColabOrchestrator

    try:
        result = ColabOrchestrator(colab_bin=args.colab_bin).run_job(
            run_dir, gpu=args.gpu, timeout=args.timeout, poll_interval=args.poll_interval, script=JOB_SCRIPT,
            files=files)
    except ColabError as e:
        held = len(list((run_dir / "runs").glob("*.json")))
        raise SystemExit(f"Colab run failed ({held} finished runs kept in {run_dir / 'runs'}; "
                         f"rerun with --resume): {e}") from e
    doc = collect(run_dir, cfg, result, args.gpu, start)
    if args.pilot:
        doc["projection"] = project(doc)
        write_outputs(doc, args.output_dir, f"pilot_{args.gpu.lower()}")
    else:
        write_outputs(doc, args.output_dir, "nanogpt_swap")
        (args.output_dir / "nanogpt_swap_steps.csv").write_text(train_steps_csv(run_dir / "runs"), encoding="utf-8")


if __name__ == "__main__":
    main()
