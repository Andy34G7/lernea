"""CLI: extrinsic proxy-LM bits-per-byte benchmark (Colab GPU, or local with --local)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Dict, List

import numpy as np

from tokenisers.benchmarks.colab import ColabError, ColabOrchestrator, run_local
from tokenisers.benchmarks.common import (
    DEFAULT_MORPH_LANGS,
    add_common_args,
    assert_disjoint,
    check_memory,
    find_duplicates,
    start_provenance,
    load_dataset,
    load_morphology,
    run_metadata,
    train_suite,
    write_json,
)
from tokenisers.benchmarks.report import render_colab
from tokenisers.data import DataPipeline

DOMAINS = ("english", "multilingual", "code")
DROPOUT_VARIANT = "bpe_dropout_p0.1"
SAMPLING_VARIANT = "unigram_sampling"
# --preset big: defaults only, so any explicit flag still wins. The small tier is the parser's defaults.
# 64 x 512 = 32,768 tokens per step; warmup 2% of steps, cosine decay (ByteBudgetTrainer), bf16 on CUDA.
BIG_PRESET: Dict[str, Any] = {
    "n_layers": 8, "d_model": 512, "n_heads": 8, "d_ffn": 2048, "seq_len": 512, "batch_size": 64,
    "lr": 6e-4, "weight_decay": 0.1, "warmup_frac": 0.02, "bf16": True, "byte_budget": 100_000_000,
    "grad_accum_steps": 4,  # micro_batch=16: logits 16x512x16384 = 0.5 GiB instead of 2 GiB
    "gpu": "L4", "concurrency": 2, "seeds": [0, 1, 2], "output_dir": "benchmarks/results/colab_big",
    # LM text: the train split plus an LM text pool up to 120 MB in all (budget + 10% needs 110 MB).
    # The job (~0.9 GB of ids) is kept on disk, not in a tmpfs /tmp, so a failed run can be resumed.
    "lm_text_bytes": 120_000_000, "job_dir": Path("benchmarks/results/colab_big/job"),
    # bpe_dropout_p0.1 is derived from bpe_frequency (kept on by default); wordpiece_pmi, wp_bpe_base and
    # wp_bpe_ft are left out.
    "variants": ["bpe_frequency", "wp_bpe_pig", "wp_bpe_scp", "wp_bpe_npmi", "wp_bpe_ppig", "mdl_bits",
                 "length_penalized", "super_bpe", "morpheme_bpe", "unigram_lm"],
}


def _encode_docs(tok: Any, docs: List[str], dtype: Any, dropout: float | None = None,
                  sample: bool = False) -> np.ndarray:
    """Ids of every document, concatenated. With ``dropout`` or ``sample``, document i uses seed i."""
    if dropout:
        kw = lambda i: {"dropout": dropout, "seed": i}
    elif sample:
        kw = lambda i: {"sample": True, "seed": i}
    else:
        kw = lambda i: {}
    parts = [np.asarray(tok.encode(d, **kw(i)), dtype=dtype) for i, d in enumerate(docs)]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)


def _token_bytes(tok: Any) -> np.ndarray:
    return np.array([len(tok.id_to_bytes(i) or b"") for i in range(tok.vocab_size)], dtype=np.int32)


def _lm_train_docs(train: List[str], byte_budget: int, seed: int) -> List[str]:
    """Shuffled LM-text documents (the train split the tokenizers were trained on, plus the LM text
    pool if any) covering ``byte_budget`` bytes plus 10%, each used once (no repetition). Only val
    and test are held out."""
    docs = list(train)
    random.Random(seed).shuffle(docs)
    out: List[str] = []
    n = 0
    for d in docs:
        if n >= byte_budget * 1.1:
            break
        out.append(d)
        n += len(d.encode("utf-8"))
    if n < byte_budget * 1.1:
        raise SystemExit(
            f"--byte-budget {byte_budget:,} needs {int(byte_budget * 1.1):,} bytes of LM text but there are "
            f"{n:,}; raise --lm-text-bytes or lower the budget"
        )
    return out


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrinsic proxy-LM bits-per-byte benchmark")
    add_common_args(parser, "benchmarks/results/colab")
    parser.add_argument("--byte-budget", type=int, default=10_000_000,
                        help="UTF-8 bytes each proxy LM trains on: train-split documents (the tokenizers' "
                             "training text), each used at most once")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="LM seeds (mean ± sd reported)")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ffn", type=int, default=512)
    parser.add_argument("--gpu", default="T4", help="Colab accelerator (T4, L4, A100, H100; none: a CPU runtime)")
    parser.add_argument("--timeout", type=float, default=6 * 3600, help="Colab job wall-clock limit (s)")
    parser.add_argument("--poll-interval", type=float, default=60.0,
                        help="Seconds between status polls of the detached Colab job")
    parser.add_argument("--colab-bin", default=None, help="colab CLI executable (default ~/.local/bin/colab)")
    parser.add_argument("--local", action="store_true", help="Run the proxy LMs locally instead of on Colab")
    parser.add_argument("--allow-local-fallback", action="store_true",
                        help="If Colab fails, run locally and record the fallback in the results")
    parser.add_argument("--preset", choices=["small", "big"], default="small",
                        help="small: the default tier above; big: 8x512 LM, 100 MB budget, L4, 12 variants, "
                             "concurrency 1, output benchmarks/results/colab_big (explicit flags override)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="(variant, seed) runs trained at once on the one GPU, as separate processes")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-frac", type=float, default=None,
                        help="Warmup as a fraction of steps (default: 10%% of steps, at most 200)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps: splits batch_size into micro-batches to reduce "
                             "peak GPU memory (batch_size must be divisible by this value)")
    parser.add_argument("--bf16", action="store_true", help="bf16 autocast when the GPU supports it")
    parser.add_argument("--no-dropout-variant", action="store_true",
                        help=f"Skip {DROPOUT_VARIANT} (bpe_frequency trained on BPE-dropout encodings)")
    parser.add_argument("--no-sampling-variant", action="store_true",
                        help=f"Skip {SAMPLING_VARIANT} (unigram_lm trained on sampled encodings)")
    parser.add_argument("--lm-text-bytes", type=int, default=None,
                        help="LM text = the train split plus an LM text pool (same sources and mix, disjoint "
                             "from val/test; DataPipeline.lm_pool) up to about this many bytes. Default: the "
                             "train split only")
    parser.add_argument("--job-dir", type=Path, default=None,
                        help="Keep the job (config, encoded ids, finished run files) here (default: a temp dir, "
                             "deleted afterwards)")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse the prepared --job-dir (no tokenizer training or encoding); finished "
                             "(variant, seed) runs are not retrained")
    parser.add_argument("--prepare-only", action="store_true", help="Build --job-dir, train no LM")
    if parser.parse_known_args(argv)[0].preset == "big":
        parser.set_defaults(**BIG_PRESET)
    return parser.parse_args(argv)


def _prepare(args: argparse.Namespace, ds: Any, job_dir: Path, spec: Dict[str, Any],
             start: Dict[str, Any]) -> Dict[str, Any]:
    """Trains the tokenizers on the train split, writes one <name>.npz of ids per tokenizer into
    ``job_dir`` and returns the job's config: ``spec``, the tokenizers and what the results
    metadata needs."""
    train = ds.texts("train")
    val = {d: ds.texts("val", domain=d) for d in DOMAINS}
    pool, pool_prov = [], None
    if args.lm_text_bytes:
        pool, pool_prov = DataPipeline(cache_dir=args.cache_dir, force_offline=args.offline).lm_pool(
            ds, args.lm_text_bytes)
    assert_disjoint(train + pool, ds.texts("val") + ds.texts("test"), "validation/test")
    lm_docs = _lm_train_docs(train + pool, args.byte_budget, args.data_seed)
    in_pool = set(pool)
    pool_bytes = sum(len(d.encode("utf-8")) for d in lm_docs if d in in_pool)
    lm_text = {"docs": len(lm_docs), "bytes": sum(len(d.encode("utf-8")) for d in lm_docs),
               "pool_docs": sum(d in in_pool for d in lm_docs), "pool_bytes": pool_bytes, "pool": pool_prov}
    del pool, in_pool

    lexicon, _, morph_prov = load_morphology(args, train, DEFAULT_MORPH_LANGS) if "morpheme_bpe" in args.variants \
        else ({}, [], None)
    print(f"Training {len(args.variants)} tokenizers at V={args.vocab_size} on {len(train)} train docs")
    models, records = train_suite(args.variants, train, args.vocab_size, lexicon)
    duplicates = find_duplicates(models)
    tokenizers: Dict[str, Any] = {}
    unique = {n: t for n, t in models.items() if n not in duplicates}
    if not args.no_dropout_variant and "bpe_frequency" in unique:
        unique[DROPOUT_VARIANT] = unique["bpe_frequency"]
    if not args.no_sampling_variant and "unigram_lm" in unique:
        unique[SAMPLING_VARIANT] = unique["unigram_lm"]
    print(f"Encoding {lm_text['bytes']:,} bytes of LM text ({lm_text['pool_bytes']:,} from the pool) per tokenizer")
    for name, tok in unique.items():
        dtype = np.uint16 if tok.vocab_size <= 65536 else np.uint32
        dropout = 0.1 if name == DROPOUT_VARIANT else None
        sample = name == SAMPLING_VARIANT
        token_bytes = _token_bytes(tok)
        arrays = {"train": _encode_docs(tok, lm_docs, dtype, dropout, sample=sample), "token_bytes": token_bytes}
        for d, texts in val.items():
            arrays[f"val_{d}"] = _encode_docs(tok, texts, dtype)
        lm_bytes = int(token_bytes[arrays["train"]].sum())
        if lm_bytes < args.byte_budget:  # the LM would pass over its text twice
            sys.exit(f"{name}: the encoded LM text covers {lm_bytes:,} bytes < --byte-budget {args.byte_budget:,}")
        np.savez_compressed(job_dir / f"{name}.npz", **arrays)  # ~2x smaller: the upload dominates
        tokenizers[name] = {"vocab_size": tok.vocab_size}
        val_bpt = {d: round(int(token_bytes[arrays[f"val_{d}"]].sum()) / len(arrays[f"val_{d}"]), 4) for d in DOMAINS}
        print(f"  {name}: {len(arrays['train']):,} LM tokens ({lm_bytes:,} bytes); val bytes/token {val_bpt}")
    prepared = {"provenance": {k: start[k] for k in ("command", "git_commit", "git_dirty", "native_extension")},
                "records": records, "duplicates": duplicates, "morphology": morph_prov, "lm_text": lm_text}
    return {**spec, "tokenizers": tokenizers, "prepared": prepared}


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    start = start_provenance(__package__, argv)  # before the tree can change during a long run

    if max(args.train_bytes, args.lm_text_bytes or 0) < args.byte_budget * 1.1:  # before any work
        sys.exit(f"--byte-budget {args.byte_budget:,} needs {int(args.byte_budget * 1.1):,} bytes of LM text (each "
                 f"document used once) but the train split is {args.train_bytes:,} bytes: set --lm-text-bytes to "
                 "add an LM text pool (same sources, disjoint from val/test). Raising --train-bytes would also "
                 "change the tokenizers.")
    if (args.resume or args.prepare_only) and args.job_dir is None:
        sys.exit("--resume and --prepare-only need --job-dir")
    lm = {"d_model": args.d_model, "n_layers": args.n_layers, "n_heads": args.n_heads,
          "d_ffn": args.d_ffn, "max_seq_len": args.seq_len}
    train_cfg = {"byte_budget": args.byte_budget, "seq_len": args.seq_len, "batch_size": args.batch_size,
                 "lr": args.lr, "eval_stride": max(1, args.seq_len // 2),
                 "grad_accum_steps": args.grad_accum_steps}
    if args.preset != "small" or args.weight_decay != 0.01 or args.warmup_frac is not None or args.bf16:
        train_cfg.update(weight_decay=args.weight_decay, warmup_frac=args.warmup_frac, bf16=args.bf16)
    # Everything the prepared job depends on; --resume refuses a job prepared with other values.
    spec = {"lm": lm, "train": train_cfg, "seeds": args.seeds, "domains": list(DOMAINS),
            "data": {"vocab_size": args.vocab_size, "train_bytes": args.train_bytes, "data_seed": args.data_seed,
                     "offline": args.offline, "lm_text_bytes": args.lm_text_bytes, "variants": args.variants,
                     "dropout_variant": not args.no_dropout_variant,
                     "sampling_variant": not args.no_sampling_variant}}

    job_dir = args.job_dir or Path(tempfile.mkdtemp(prefix="tokenisers-bpb-"))
    config_path = job_dir / "config.json"
    try:
        if args.resume:
            if not config_path.is_file():
                sys.exit(f"--resume: {config_path} does not exist")
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            changed = [k for k in spec if cfg.get(k) != json.loads(json.dumps(spec[k]))]
            if changed:
                sys.exit(f"--resume: {job_dir} was prepared with different {changed}; pass the same arguments, "
                         "or prepare a new job dir")
            print(f"Resuming {job_dir}: {len(list(job_dir.glob('runs/*.json')))} finished run file(s)")
            ds = load_dataset(args)  # cached; for the results metadata
        else:
            if config_path.exists():
                sys.exit(f"{job_dir} already holds a prepared job: pass --resume to reuse it, or delete it")
            if not args.skip_memory_check:  # before downloading/preparing anything
                check_memory(args.variants, args.train_bytes)
            ds = load_dataset(args)
            job_dir.mkdir(parents=True, exist_ok=True)
            cfg = _prepare(args, ds, job_dir, spec, start)
        cfg["concurrency"] = args.concurrency  # may change on resume (e.g. lower after an out-of-memory)
        write_json(config_path, cfg)
        if args.prepare_only:
            print(f"Prepared {job_dir}; run without --prepare-only and with --resume to train the LMs")
            return

        fallback = None
        gpu = None if args.gpu.lower() == "none" else args.gpu
        if args.local:
            result = run_local(job_dir)
        else:
            try:
                result = ColabOrchestrator(colab_bin=args.colab_bin).run_job(
                    job_dir, gpu=gpu, timeout=args.timeout, poll_interval=args.poll_interval)
            except ColabError as e:
                if not args.allow_local_fallback:
                    sys.exit(f"Colab run failed (pass --local or --allow-local-fallback to run locally): {e}"
                             + ("\nFinished runs are kept in --job-dir: re-run with --resume" if args.job_dir else ""))
                print(f"Colab run failed, running locally: {e}", file=sys.stderr)
                fallback = {"from": "colab", "requested_gpu": gpu, "error": str(e)}
                result = run_local(job_dir)
    finally:
        if args.job_dir is None:
            shutil.rmtree(job_dir, ignore_errors=True)

    prep = cfg["prepared"]
    meta = run_metadata(
        args, ds, "proxy_lm_bpb", "val", prep["records"], prep["duplicates"], start=start,
        backend=result["backend"], runs_by_hardware=result["runs_by_hardware"], fallback=fallback, lm=lm,
        lm_architecture=result["architecture"],
        train=train_cfg, byte_budget=args.byte_budget,
        seeds=args.seeds, val_domains=list(DOMAINS), lm_train_docs=prep["lm_text"]["docs"], lm_text=prep["lm_text"],
        lm_seconds=result["seconds"], morphology=prep["morphology"], job_prepared=prep["provenance"],
        variants_extra={
            **(
                {DROPOUT_VARIANT: "bpe_frequency tokenizer; LM train ids encoded with BPE-dropout p=0.1, "
                                  "validation ids deterministic"}
                if DROPOUT_VARIANT in cfg["tokenizers"] else {}
            ),
            **(
                {SAMPLING_VARIANT: "unigram_lm tokenizer; LM train ids encoded with FFBS sampling, "
                                   "validation ids deterministic (greedy Viterbi)"}
                if SAMPLING_VARIANT in cfg["tokenizers"] else {}
            ),
        },
    )
    doc = {"meta": meta, "models": result["models"]}
    out = Path(args.output_dir)
    write_json(out / "extrinsic_bpb_results.json", doc)
    md = render_colab(doc)
    (out / "extrinsic_bpb_leaderboard.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
