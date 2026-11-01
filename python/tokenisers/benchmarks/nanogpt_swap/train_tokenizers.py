"""
CLI: train tokenizer-swap variants on more text than this machine can hold, on a Colab VM, into a
prepared job dir (a pool) that the swap CLI assembles jobs from (``--from-pool``).

    uv run python -m tokenisers.benchmarks.nanogpt_swap.train_tokenizers --job-dir POOL \\
        --variants bpe_frequency wp_bpe_pig mdl_bits             # 300 MB, space_prefix, on an L4
    ... --resume                                                 # after a dropped session
    ... --local --source-dir DIR --train-bytes 5000000           # dry run here on cached shards

Training needs about 60-75 bytes of RAM per byte of text, about 20 GB at 300 MB, so the default
runtime is the L4 (53 GB RAM, the cheapest with enough). On the VM (``tokenizer_job.py``): build the
extension from ``git archive HEAD``, fetch the speedrun's shards, decode the first train documents
up to ``--train-bytes`` (extending the ordered stream the swap's 50 MB comes from, which is checked
to be its start), check them disjoint from the val documents by sha256, train each variant with
``--pretokenizer`` at 50,256 tokens, and write each tokenizer to its own run file, which is
downloaded as soon as it exists (a dropped session keeps the finished ones; ``--resume`` skips
them). The session is always stopped. The names are ``<variant>_sp300`` (``_g2300`` with gpt2): in
the pool's config.json they carry their training text (bytes, documents, sha256) like any locally
prepared tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional

from tokenisers.benchmarks.nanogpt_swap import __main__ as swap
from tokenisers.benchmarks.nanogpt_swap import swap_job

VM_SCRIPT = Path(__file__).with_name("tokenizer_job.py")
SUFFIX = {"space_prefix": "sp", "gpt2": "g2"}  # common.PRETOKENIZER_SUFFIXES inverted
VARIANTS = [v for v in swap.OUR_VARIANTS if v != "super_bpe"]
REFERENCE = swap.REPO / "benchmarks" / "results" / "nanogpt_swap" / "nanogpt_swap.json"


def names(variants: List[str], pretokenizer: str, train_bytes: int) -> List[str]:
    return [f"{v}_{SUFFIX[pretokenizer]}{train_bytes // 1_000_000}" for v in variants]


def reference(path: Path) -> Dict[str, Any]:
    """Val documents and tokenizer text of a swap result JSON (meta) or a prepared job's config.json."""
    d = json.loads(path.read_text(encoding="utf-8"))
    if "meta" in d:
        return {"val_docs_sha256": next(iter(d["models"].values()))["val_docs_sha256"],
                "tokenizer_text": d["meta"]["tokenizer_text"]}
    return {"val_docs_sha256": d["val_docs_sha256"], "tokenizer_text": d["tokenizer_text"]}


def vm_config(names_: List[str], train_bytes: int, ref: Dict[str, Any], source_dir: Optional[Path] = None,
              vocab_size: int = swap.OUR_VOCAB) -> Dict[str, Any]:
    source = dict(swap.SOURCE, **({"local_dir": str(source_dir.resolve())} if source_dir else {}))
    return {"source": source, "val_docs_sha256": ref["val_docs_sha256"],
            "reference_text": {k: ref["tokenizer_text"][k] for k in ("docs", "docs_sha256")},
            "train_bytes": train_bytes, "vocab_size": vocab_size, "variants": names_, "ids_docs": swap.IDS_DOCS,
            "pip": ["tiktoken==0.14.0"]}


def finish(job: Path, cfg: Dict[str, Any], result: Dict[str, Any], start: Dict[str, Any]) -> Dict[str, Any]:
    """Writes the pool from the downloaded run files: ``tokenizers/<name>/`` (the files exactly as the
    VM saved them, by sha256), tokenizers.tar.gz and a swap job's config.json."""
    from tokenisers import models

    remote = job / "remote"
    specs: List[Dict[str, Any]] = []
    records: Dict[str, Any] = {}
    seen: Dict[str, str] = {}
    duplicates: Dict[str, str] = {}
    for name in cfg["variants"]:
        run = json.loads((remote / "runs" / f"{name}.json").read_text(encoding="utf-8"))
        if run["tokenizer_text"] != result["data"]["tokenizer_text"]:  # a resumed run's earlier session
            raise RuntimeError(f"{name}: trained on another text than this session decoded")
        out = job / "tokenizers" / name
        out.mkdir(parents=True, exist_ok=True)
        for fname, text in run["files"].items():
            data = text.encode("utf-8")
            if hashlib.sha256(data).hexdigest() != run["files_sha256"][fname]:
                raise RuntimeError(f"{name}/{fname}: not the file the VM saved")
            (out / fname).write_bytes(data)
        tok = getattr(models, run["class"]).from_pretrained(str(out))
        if tok.pretokenizer != run["pretokenizer"] or tok.vocab_size != run["record"]["vocab_size"]:
            raise RuntimeError(f"{name}: reloads as another tokenizer")
        digest = run["files_sha256"]["tokenizer.json"]
        if digest in seen:
            duplicates[name] = seen[digest]
        seen.setdefault(digest, name)
        specs.append({"name": name, "kind": "tokenisers", "class": run["class"], "path": f"tokenizers/{name}",
                      "pretokenizer": run["pretokenizer"], "ids_sha256": run["ids_sha256"], "ids_docs": run["ids_docs"],
                      "tokenizer_text": run["tokenizer_text"]})
        records[name] = run["record"]
    with tarfile.open(job / "tokenizers.tar.gz", "w:gz") as tar:
        tar.add(job / "tokenizers", arcname="tokenizers")
    data = result["data"]
    pool = {"source": swap.SOURCE, "val_docs_sha256": cfg["val_docs_sha256"], "tokenizer_text": data["tokenizer_text"],
            "pip": swap.PIP, "headroom": 0.01, "tokenizers": specs,
            "prepared": {**{k: start[k] for k in ("command", "git_commit", "git_dirty", "native_extension")},
                         "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "val_docs": data["val_docs"], "val_bytes": data["val_bytes"],
                         "train_docs_dropped_as_val": data["train_docs_dropped_as_val"], "variants": records,
                         "duplicates": duplicates, "morphology": result.get("morphology"),
                         "tokenmonster_why": swap.TOKENMONSTER_WHY, "trained_on": result["backend"],
                         "timings": result.get("timings")}}
    (job / "config.json").write_text(json.dumps(pool, indent=1), encoding="utf-8")
    return pool


def run_local(remote: Path, files: List[Path]) -> Dict[str, Any]:
    """The VM job on this machine (dry runs): the installed extension, no pip, no build."""
    for f in files:
        if f.resolve() != (remote / f.name).resolve():
            shutil.copy(f, remote / f.name)
    (remote / "status.json").unlink(missing_ok=True)
    subprocess.run([sys.executable, "-u", str(remote / VM_SCRIPT.name), str(remote)], check=True)
    status = json.loads((remote / "status.json").read_text())
    if status["state"] != "ok":
        raise SystemExit(f"the job failed:\n{status.get('traceback')}")
    result = json.loads((remote / "result.json").read_text())
    result["backend"]["kind"] = "local"
    return result


def main(argv: Optional[List[str]] = None) -> None:
    from tokenisers.benchmarks.common import check_memory, start_provenance

    parser = argparse.ArgumentParser(description="Train tokenizer-swap variants on a Colab VM (more text than here)")
    parser.add_argument("--job-dir", type=Path, required=True, help="The pool: tokenizers, config.json, remote/ job")
    parser.add_argument("--variants", nargs="+", required=True, choices=VARIANTS, metavar="VARIANT")
    parser.add_argument("--pretokenizer", default="space_prefix", choices=sorted(SUFFIX))
    parser.add_argument("--train-bytes", type=int, default=300_000_000,
                        help="Bytes of train documents (a whole number of MB; it names the tokenizers)")
    parser.add_argument("--gpu", default="L4", help="Colab runtime (L4: 53 GB RAM, the cheapest that fits 300 MB)")
    parser.add_argument("--resume", action="store_true", help="Keep the tokenizers a dropped session finished")
    parser.add_argument("--reference", type=Path, default=REFERENCE,
                        help="Swap result JSON or prepared job config.json whose val documents and tokenizer text "
                             "(the start of the training text) the VM checks against")
    parser.add_argument("--local", action="store_true", help="Run the VM job on this machine instead (dry run)")
    parser.add_argument("--source-dir", type=Path, default=None,
                        help="Directory holding the source shards (e.g. a prepared job's source/); default: download")
    parser.add_argument("--skip-memory-check", action="store_true")
    parser.add_argument("--timeout", type=float, default=4 * 3600, help="Colab job wall-clock limit (s)")
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--colab-bin", default=None)
    args = parser.parse_args(argv)
    if args.train_bytes < 1_000_000 or args.train_bytes % 1_000_000:
        parser.error("--train-bytes must be a whole number of MB (it names the tokenizers: _sp300 = 300 MB)")
    start = start_provenance("tokenisers.benchmarks.nanogpt_swap.train_tokenizers", argv)
    job, remote = args.job_dir, args.job_dir / "remote"
    if (job / "config.json").is_file():
        raise SystemExit(f"{job} is already a finished pool (config.json)")
    held = sorted((remote / "runs").glob("*.json"))
    if held and not args.resume:
        raise SystemExit(f"{remote / 'runs'} holds {len(held)} finished tokenizers: pass --resume to keep them")
    remote.mkdir(parents=True, exist_ok=True)
    wanted = names(args.variants, args.pretokenizer, args.train_bytes)
    cfg = vm_config(wanted, args.train_bytes, reference(args.reference), args.source_dir)
    (remote / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    files = [remote / "config.json", swap.JOB_SCRIPT, swap.REPO / "python" / "tokenisers" / "benchmarks" / "throughput.py"]
    print(f"training {', '.join(wanted)} on {args.train_bytes / 1e6:.0f} MB "
          f"({'here' if args.local else f'Colab {args.gpu}'})")
    if args.local:
        if not args.skip_memory_check:
            check_memory(args.variants, args.train_bytes)
        cfg["pip"] = []
        (remote / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        result = run_local(remote, [VM_SCRIPT, *files[1:]])
    else:
        if start["git_dirty"] is not False:
            raise SystemExit("the VM builds the committed code (`git archive HEAD`): commit code changes first")
        subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(remote / "src.tar.gz"), start["git_commit"]],
                       cwd=swap.REPO, check=True)
        from tokenisers.benchmarks.colab import ColabError, ColabOrchestrator

        try:
            result = ColabOrchestrator(colab_bin=args.colab_bin).run_job(
                remote, gpu=args.gpu, timeout=args.timeout, poll_interval=args.poll_interval, script=VM_SCRIPT,
                files=[*files, remote / "src.tar.gz"])
        except ColabError as e:
            held = len(list((remote / "runs").glob("*.json")))
            raise SystemExit(f"Colab run failed ({held} finished tokenizers kept in {remote / 'runs'}; "
                             f"rerun with --resume): {e}") from e
    pool = finish(job, cfg, result, start)
    tt = pool["tokenizer_text"]
    print(f"pool {job}: {', '.join(s['name'] for s in pool['tokenizers'])} on {tt['docs']:,} train docs "
          f"({tt['bytes']:,} bytes); assemble a swap job with --from-pool {job}")


if __name__ == "__main__":
    main()
