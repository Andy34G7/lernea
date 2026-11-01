"""
Encode/decode throughput of saved tokenizers, re-measured locally or on a Colab CPU runtime.

The CLI (``python -m tokenisers.benchmarks.throughput --job-dir DIR``) trains the suite variants at
each vocabulary size exactly as the other benchmarks do, saves them, checks that ``save`` ->
``from_pretrained`` gives the same ids on the throughput documents and records a sha256 of those
ids (``prepare``). ``run_job`` then loads the saved tokenizers, fails if any id hash differs, and
times ``IntrinsicEvaluator.benchmark_throughput`` (the intrinsic suite's timing) in ``passes``
full passes over all tokenizers, so the run-to-run spread is measured. With ``--colab`` the job
runs on a Colab CPU VM through ``ColabOrchestrator``; the native extension is built there from
``git archive`` of the commit (``build_extension``), because a locally built wheel is linked
against this machine's glibc.

This file's top level imports only the standard library: it is uploaded to the VM and started
there as ``python throughput.py <job_dir>`` before the extension exists.

Job directory: ``config.json`` (settings, provenance of the preparation, every tokenizer with its
class, path and id hash), ``docs.json`` (the documents), ``tokenizers.tar.gz`` (the saved
tokenizers) and, for Colab only, ``src.tar.gz``. The job writes ``result.json``; ``main`` also
writes ``status.json`` last, which the orchestrator polls for.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

LSCPU_KEYS = ("Model name", "CPU(s)", "Thread(s) per core", "Core(s) per socket", "Socket(s)",
              "Hypervisor vendor", "L3 cache")


def ids_sha256(tok: Any, docs: Sequence[str]) -> Tuple[str, int]:
    """(sha256 of the ids of every document, total token count). One encode call per document."""
    h = hashlib.sha256()
    n = 0
    for d in docs:
        ids = tok.encode(d)
        n += len(ids)
        h.update(json.dumps(ids).encode("ascii"))  # the brackets keep document boundaries
    return h.hexdigest(), n


def _load() -> Dict[str, Any]:
    """1/5/15-min load average and MemAvailable (bytes) now."""
    mem = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                mem = int(line.split()[1]) * 1024
    except OSError:
        pass
    return {"loadavg": [round(x, 2) for x in os.getloadavg()], "mem_available": mem}


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def host_info() -> Dict[str, Any]:
    """CPU (lscpu), the CPUs this process may run on with their max frequency and governor (e.g. a
    ``taskset`` pin), RAM, Python and C library."""
    lscpu: Dict[str, str] = {}
    try:
        for line in subprocess.run(["lscpu"], capture_output=True, text=True, timeout=30).stdout.splitlines():
            key, _, value = line.partition(":")
            if key.strip() in LSCPU_KEYS:
                lscpu[key.strip()] = value.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    cpus = []
    for c in sorted(os.sched_getaffinity(0)):
        sysfs = Path(f"/sys/devices/system/cpu/cpu{c}")
        max_khz = _read(sysfs / "cpufreq" / "cpuinfo_max_freq")
        cpus.append({"cpu": c, "max_mhz": int(max_khz) // 1000 if max_khz else None,
                     "governor": _read(sysfs / "cpufreq" / "scaling_governor"),
                     "thread_siblings": _read(sysfs / "topology" / "thread_siblings_list")})
    mem_total = next((int(l.split()[1]) * 1024 for l in (_read(Path("/proc/meminfo")) or "").splitlines()
                      if l.startswith("MemTotal:")), None)
    return {"cpu": lscpu.get("Model name"), "lscpu": lscpu, "cpu_count": os.cpu_count(), "affinity": cpus,
            "mem_total": mem_total, "python": platform.python_version(), "platform": platform.platform(),
            "libc": " ".join(platform.libc_ver())}


def build_extension(job: Path) -> Dict[str, Any]:
    """Builds the native extension from ``src.tar.gz`` on this machine and installs it into this
    interpreter (rustup's minimal toolchain is installed first when cargo is missing). Release
    profile from the repository; no target-cpu flags, so the code is the same as a local build."""
    t0 = time.time()
    src = job / "src"
    with tarfile.open(job / "src.tar.gz") as tar:
        tar.extractall(src, filter="data")
    env = {**os.environ, "PATH": f"{Path.home() / '.cargo' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"}

    def run(cmd: Any, **kw: Any) -> str:
        return subprocess.run(cmd, check=True, env=env, cwd=src, **kw).stdout or ""

    if shutil.which("cargo", path=env["PATH"]) is None:
        run("curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal", shell=True)
    run([sys.executable, "-m", "pip", "install", "-q", "maturin>=1.15,<2"])
    wheels = job / "wheels"
    run([sys.executable, "-m", "maturin", "build", "--release", "--locked", "-i", sys.executable, "-o", str(wheels)])
    wheel = next(wheels.glob("*.whl"))
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", str(wheel)])
    importlib.invalidate_caches()
    rustc = run(["rustc", "-V"], capture_output=True, text=True).strip()
    return {"wheel": wheel.name, "rustc": rustc, "seconds": round(time.time() - t0, 1)}


def run_job(job_dir: str, build: Optional[Dict[str, Any]] = None, log=print) -> Dict[str, Any]:
    """Loads every saved tokenizer, raises if its ids on the documents differ from the recorded
    hash, then times them all in ``passes`` full passes; writes and returns ``result.json``."""
    from tokenisers import models
    from tokenisers.benchmarks.intrinsic import IntrinsicEvaluator

    job = Path(job_dir)
    cfg = json.loads((job / "config.json").read_text(encoding="utf-8"))
    docs = json.loads((job / "docs.json").read_text(encoding="utf-8"))
    if not (job / "tokenizers").is_dir():
        with tarfile.open(job / "tokenizers.tar.gz") as tar:
            tar.extractall(job, filter="data")
    toks: Dict[Tuple[int, str], Any] = {}
    for t in cfg["tokenizers"]:
        tok = getattr(models, t["class"]).from_pretrained(str(job / t["path"]))
        digest, _ = ids_sha256(tok, docs)
        if digest != t["ids_sha256"]:
            raise RuntimeError(f"{t['name']} at V={t['vocab_size']}: ids on the throughput documents differ from "
                               f"the prepared tokenizer's (sha256 {digest} != {t['ids_sha256']}); not timing it")
        toks[(t["vocab_size"], t["name"])] = tok
    log(f"loaded {len(toks)} tokenizers; every id hash matches")

    evaluator = IntrinsicEvaluator()
    out: Dict[str, Dict[str, Any]] = {}
    passes: List[Dict[str, Any]] = []
    for p in range(cfg["passes"]):
        before, t0 = _load(), time.perf_counter()
        for (v, name), tok in toks.items():
            r = evaluator.benchmark_throughput(tok, docs, num_warmup=cfg["warmup"], repeats=cfg["repeats"])
            m = out.setdefault(str(v), {}).setdefault(
                name, {"vocab_size": tok.vocab_size, "encode_mb_per_sec": [], "decode_mb_per_sec": []})
            m["encode_mb_per_sec"].append(r["encode_mb_per_sec"])
            m["decode_mb_per_sec"].append(r["decode_mb_per_sec"])
        passes.append({"before": before, "after": _load(), "seconds": round(time.perf_counter() - t0, 1)})
        log(f"pass {p + 1}/{cfg['passes']}: {passes[-1]['seconds']} s")
    result = {"backend": {"kind": "local", **host_info(), "build": build}, "passes": passes, "models": out}
    (job / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(job_dir: str) -> int:
    """Entry point of the detached job on the Colab VM: builds the extension when ``src.tar.gz`` is
    in the job, runs it, and writes ``status.json`` last, atomically: {"state": "ok"} or
    {"state": "error", "traceback": ...}."""
    job = Path(job_dir)
    try:
        run_job(job_dir, build_extension(job) if (job / "src.tar.gz").exists() else None)
        status: Dict[str, Any] = {"state": "ok"}
    except BaseException:
        status = {"state": "error", "traceback": traceback.format_exc()}
    tmp = job / "status.json.tmp"
    tmp.write_text(json.dumps(status))
    os.replace(tmp, job / "status.json")
    return 0 if status["state"] == "ok" else 1


# ----------------------------------------------------------------------------
# Local side: prepare the job directory, run it here or on Colab, write the results
# ----------------------------------------------------------------------------

def prepare(args: Any, job: Path, start: Dict[str, Any]) -> None:
    """Trains, saves and verifies every (vocabulary size, variant) into ``job``; ``start`` is the
    run's ``start_provenance``."""
    from tokenisers.benchmarks.common import DEFAULT_MORPH_LANGS, check_memory, load_dataset, load_morphology, train_suite
    from tokenisers.benchmarks.intrinsic.__main__ import throughput_docs

    if not args.skip_memory_check:
        check_memory(args.variants, args.train_bytes)
    ds = load_dataset(args)
    train = ds.texts("train")
    docs = throughput_docs(ds)
    lexicon = load_morphology(args, train, DEFAULT_MORPH_LANGS)[0] if "morpheme_bpe" in args.variants else {}
    tokenizers: List[Dict[str, Any]] = []
    variants: Dict[str, Any] = {}
    for v in args.vocab_sizes:
        print(f"Training {len(args.variants)} tokenizers at V={v} on {len(train)} train docs")
        trained, variants[str(v)] = train_suite(args.variants, train, v, lexicon)
        for name, tok in trained.items():
            path = f"tokenizers/v{v}/{name}"
            tok.save(str(job / path))
            digest, n = ids_sha256(tok, docs)
            if ids_sha256(type(tok).from_pretrained(str(job / path)), docs)[0] != digest:
                raise SystemExit(f"{name} at V={v}: save -> from_pretrained changes the ids")
            tokenizers.append({"vocab_size": v, "name": name, "class": type(tok).__name__, "path": path,
                               "ids_sha256": digest, "tokens": n})
    (job / "docs.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    with tarfile.open(job / "tokenizers.tar.gz", "w:gz") as tar:
        tar.add(job / "tokenizers", arcname="tokenizers")
    manifest = ds.manifest
    prepared = {
        **{k: start[k] for k in ("command", "git_commit", "git_dirty", "native_extension")},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "synthetic_data": ds.synthetic,
        "dataset": {"root": ds.root.name, "created_utc": manifest.get("created_utc"),
                    "manifest_sha256": hashlib.sha256((ds.root / "manifest.json").read_bytes()).hexdigest()
                    if (ds.root / "manifest.json").is_file() else None},
        "train_bytes": manifest["splits"]["train"]["text_bytes"],
        "throughput_docs": len(docs),
        "throughput_bytes": sum(len(d.encode("utf-8")) for d in docs),
        "variants": variants,
    }
    (job / "config.json").write_text(json.dumps({"prepared": prepared, "tokenizers": tokenizers}, indent=2),
                                     encoding="utf-8")


def cli(argv: Optional[List[str]] = None) -> None:
    import argparse

    from tokenisers.benchmarks.common import STANDARD_VOCAB_SIZES, VARIANTS, start_provenance, write_json
    from tokenisers.benchmarks.report import render_throughput
    from tokenisers.data.pipeline import DEFAULT_TRAIN_BYTES

    parser = argparse.ArgumentParser(description="Encode/decode throughput, locally or on a Colab CPU runtime")
    parser.add_argument("--job-dir", type=Path, required=True,
                        help="Saved tokenizers, documents and id hashes: prepared (trained) here if it has no "
                             "config.json, reused otherwise, so local and Colab runs time the same tokenizers")
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=list(STANDARD_VOCAB_SIZES))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS, metavar="VARIANT")
    parser.add_argument("--train-bytes", type=int, default=DEFAULT_TRAIN_BYTES)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true", help="Synthetic offline corpus (labelled synthetic)")
    parser.add_argument("--skip-memory-check", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Train, save and verify; time nothing")
    parser.add_argument("--repeats", type=int, default=5, help="Timed passes per measurement (median reported)")
    parser.add_argument("--passes", type=int, default=2, help="Full passes over all tokenizers (run-to-run spread)")
    parser.add_argument("--colab", action="store_true",
                        help="Time on a Colab CPU runtime; the extension is built there from `git archive HEAD`")
    parser.add_argument("--timeout", type=float, default=3600, help="Colab job wall-clock limit (s)")
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--colab-bin", default=None)
    parser.add_argument("--output-dir", default=None,
                        help="Default benchmarks/results/throughput/colab_cpu (--colab) or .../local")
    args = parser.parse_args(argv)
    start = start_provenance("tokenisers.benchmarks.throughput", argv)

    job = args.job_dir
    job.mkdir(parents=True, exist_ok=True)
    if not (job / "config.json").is_file():
        prepare(args, job, start)
    if args.prepare_only:
        return
    cfg = json.loads((job / "config.json").read_text(encoding="utf-8"))
    cfg.update(repeats=args.repeats, warmup=1, passes=args.passes)
    (job / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if args.colab:
        from tokenisers.benchmarks.colab import ColabError, ColabOrchestrator

        if start["git_dirty"] is not False:
            raise SystemExit("--colab builds the committed code (`git archive HEAD`) on the VM: run it from a "
                             "git checkout without uncommitted code changes")
        top = Path(__file__).resolve().parents[3]
        subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(job / "src.tar.gz"), start["git_commit"]],
                       cwd=top, check=True)
        files = [job / f for f in ("config.json", "docs.json", "tokenizers.tar.gz", "src.tar.gz")]
        try:
            result = ColabOrchestrator(colab_bin=args.colab_bin).run_job(
                job, gpu=None, timeout=args.timeout, poll_interval=args.poll_interval,
                script=Path(__file__), files=files)
        except ColabError as e:
            raise SystemExit(f"Colab run failed: {e}") from e
    else:
        result = run_job(str(job))

    meta = {"benchmark": "throughput", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **start, "prepared": cfg["prepared"], "repeats": args.repeats, "warmup": 1, "passes": args.passes,
            "vocab_sizes": sorted({t["vocab_size"] for t in cfg["tokenizers"]}),
            "ids_sha256": {f"{t['vocab_size']}/{t['name']}": t["ids_sha256"] for t in cfg["tokenizers"]},
            "backend": result["backend"], "pass_load": result["passes"]}
    doc = {"meta": meta, "models": result["models"]}
    out = Path(args.output_dir or f"benchmarks/results/throughput/{'colab_cpu' if args.colab else 'local'}")
    write_json(out / "throughput.json", doc)
    md = render_throughput(doc)
    (out / "throughput.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    # `python -m tokenisers.benchmarks.throughput ...` is the CLI; the copy that ColabOrchestrator
    # uploads runs as a plain script (`python throughput.py <job_dir>`), i.e. without a package.
    sys.exit(cli() if __package__ else main(sys.argv[1]))
