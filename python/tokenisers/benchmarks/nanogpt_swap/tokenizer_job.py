"""
VM side of ``train_tokenizers.py``: uploaded with ``swap_job.py`` and ``throughput.py`` and started
as ``python tokenizer_job.py <job_dir>``. Top-level imports are the standard library and the
uploaded ``swap_job`` (this script's directory is on ``sys.path``); the native extension is built
by the job itself.

Builds the extension (``src.tar.gz``), fetches the speedrun's shards, decodes the first train
documents up to ``train_bytes`` (``SourceDocs.first_docs``: the stream the tokenizer swap's local
preparation takes its 50 MB from, whose documents are checked to be its first ones), checks them
disjoint from the val documents by sha256, and trains every variant of ``config["variants"]`` in
turn. Each tokenizer is saved, reloaded (same ids on the first ``ids_docs`` documents) and written
with its files and training record as ``runs/<name>.json``, which the orchestrator downloads on its
next poll; a variant whose run file exists is skipped (resume).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional
import warnings

import swap_job


def peak_rss() -> Optional[int]:
    """Peak resident memory of this process since the last ``reset_peak_rss`` (VmHWM)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def reset_peak_rss() -> None:
    try:
        Path("/proc/self/clear_refs").write_text("5")  # Linux: resets VmHWM to the current RSS
    except OSError:
        pass


def run_job(job_dir: str) -> Dict[str, Any]:
    job = Path(job_dir)
    t_job = time.time()
    cfg = json.loads((job / "config.json").read_text())
    backend = swap_job.backend_info()
    swap_job.log(f"backend: {backend}")
    timings: Dict[str, Any] = {"setup": swap_job.setup(job, cfg), "download": {}}
    from tokenisers.benchmarks.common import DEFAULT_MORPH_LANGS, load_morphology, train_variant, variant_config

    src = cfg["source"]
    dec = swap_job.load_tokenizer(src["tokenizer"], job)
    t0 = time.time()
    source = swap_job.SourceDocs(swap_job.make_fetch(job, src, timings["download"]), dec.decode, dec.sep,
                                 src["val_shard"], src["val_tokens"], src["train_shards"])
    if swap_job.docs_sha256(source.val) != cfg["val_docs_sha256"]:
        raise RuntimeError("the val documents differ from the tokenizer swap's")
    budget = cfg["train_bytes"]
    docs, nbytes = source.first_docs(budget)
    if nbytes < 0.9 * budget:
        raise RuntimeError(f"the source shards hold only {nbytes:,} bytes of train documents; {budget:,} requested")
    shards = list(source.shards_used)
    ref = cfg["reference_text"]  # the swap's 50 MB: the first documents of the same stream
    while len(source.train) < ref["docs"] and source.extend():
        pass
    if swap_job.docs_sha256(source.train[:ref["docs"]]) != ref["docs_sha256"]:
        raise RuntimeError(f"the first {ref['docs']:,} train documents differ from the swap's tokenizer text")
    overlap = sum(swap_job.doc_hash(d) in source.val_hashes for d in docs)
    if overlap:
        raise RuntimeError(f"{overlap} training documents also occur in val (sha256)")
    tt = {"docs": len(docs), "bytes": nbytes, "docs_sha256": swap_job.docs_sha256(docs), "shards": shards,
          "starts_with": {"docs": ref["docs"], "docs_sha256": ref["docs_sha256"]} if len(docs) >= ref["docs"] else None,
          "val_disjoint_sha256": True}
    timings["source_seconds"] = round(time.time() - t0, 1)
    swap_job.log(f"tokenizer text: {len(docs):,} train docs, {nbytes:,} bytes from {', '.join(shards)}; "
                 f"disjoint from {len(source.val):,} val docs")

    lexicon: Dict[str, Any] = {}
    morph = None
    if any(n.startswith("morpheme_bpe") for n in cfg["variants"]):  # SIGMORPHON train split, as locally
        lexicon, _, morph = load_morphology(argparse.Namespace(cache_dir=None, offline=False), docs,
                                            DEFAULT_MORPH_LANGS)
    runs = job / "runs"
    vocab = cfg["vocab_size"]
    for name in cfg["variants"]:
        if (runs / f"{name}.json").exists():
            swap_job.log(f"{name}: finished in an earlier session, skipped")
            continue
        swap_job.log(f"{name}: training at V={vocab} on {nbytes / 1e6:.1f} MB")
        gc.collect()
        reset_peak_rss()
        t = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tok = train_variant(name, docs, vocab, lexicon)
        seconds, peak = time.time() - t, peak_rss()
        out = job / "work" / name
        tok.save(str(out))
        probe = source.train[:cfg["ids_docs"]]  # the documents the swap VM checks the ids on
        digest = swap_job.ids_sha256(tok.encode(d) for d in probe)
        again = type(tok).from_pretrained(str(out))
        if swap_job.ids_sha256(again.encode(d) for d in probe) != digest:
            raise RuntimeError(f"{name}: save -> from_pretrained changes the ids")
        files = {f.name: f.read_bytes() for f in sorted(out.iterdir())}
        record = {"name": name, "class": type(tok).__name__, "pretokenizer": tok.pretokenizer, "ids_sha256": digest,
                  "ids_docs": cfg["ids_docs"], "tokenizer_text": tt,
                  "record": {**variant_config(name, vocab), "requested_vocab_size": vocab, "vocab_size": tok.vocab_size,
                             "train_seconds": round(seconds, 2), "warnings": [str(w.message) for w in caught],
                             "peak_rss_bytes": peak, "train_docs": len(docs), "train_bytes": nbytes},
                  "files": {k: v.decode("utf-8") for k, v in files.items()},
                  "files_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
        swap_job.write_json_atomic(runs / f"{name}.json", record)
        swap_job.log(f"{name}: |V|={tok.vocab_size} in {seconds:.0f} s, peak RSS "
                     f"{(peak or 0) / 2**30:.1f} GiB")
        del tok, again
    timings["decode"] = source.seconds
    data = {"val_docs": len(source.val), "val_bytes": sum(len(d.encode()) for d in source.val),
            "train_docs_decoded": len(source.train), "train_bytes_decoded": sum(source.train_bytes),
            "train_docs_dropped_as_val": source.dropped_as_val, "shards": [src["val_shard"], *source.shards_used],
            "tokenizer_text": tt}
    result = {"backend": backend, "timings": timings, "data": data, "morphology": morph,
              "seconds": round(time.time() - t_job, 1), "runs": sorted(p.stem for p in runs.glob("*.json"))}
    swap_job.write_json_atomic(job / "result.json", result)
    return result


if __name__ == "__main__":
    sys.exit(swap_job.main(sys.argv[1], run=run_job))
