"""
VM side of the tokenizer-swap benchmark: ``ColabOrchestrator`` uploads this file with the job and
starts it there as ``python swap_job.py <job_dir>``. Top-level imports are the standard library and
numpy only (the native extension is built by the job itself).

The text is the same for every tokenizer: the GPT-2 FineWeb shards of the NanoGPT speedrun, decoded
back to documents. Val = every complete document in the speedrun's val window (the first
``val_tokens`` GPT-2 tokens of the val shard). Train = the documents of the train shards in order,
minus any whose sha256 equals a val document's. Each tokenizer's token stream is
``[sep] doc_1 [sep] doc_2 [sep] ... doc_n [sep]``: one separator per document boundary, as GPT-2's
<|endoftext|> in the original shards.

Per tokenizer, in config order: encode the val documents (``decode(encode(doc)) == doc`` is checked
for every one), encode the train stream until it holds ``steps x batch_tokens`` plus headroom, write
llm.c ``.bin`` files, run the patched llm.c trainer (``train_gpt2_swap.py``) and write
``runs/<name>.json``, which the orchestrator downloads on its next poll. A run whose file exists is
not repeated (resume). Tokenizers not in ``config["train"]`` (pilot) are only val-encoded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

MAGIC, VERSION = 20240520, 1  # llm.c / modded-nanogpt .bin header: 256 int32, then uint16 tokens


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# .bin files and documents
# ----------------------------------------------------------------------------

def write_bin(path: Path, tokens: np.ndarray) -> None:
    """llm.c token file: header [20240520, 1, n, 0...] (256 int32), then n uint16 tokens."""
    tokens = np.asarray(tokens)
    if tokens.size and (int(tokens.min()) < 0 or int(tokens.max()) >= 1 << 16):
        raise ValueError("token ids must fit in uint16")
    header = np.zeros(256, dtype=np.int32)
    header[:3] = (MAGIC, VERSION, len(tokens))
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


def read_bin(path: Path, max_tokens: Optional[int] = None) -> np.ndarray:
    """Tokens of an llm.c .bin file (the first ``max_tokens`` only, if given)."""
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(1024), dtype=np.int32)
        if len(header) != 256 or header[0] != MAGIC or header[1] != VERSION:
            raise ValueError(f"{path}: not an llm.c .bin file (magic/version)")
        n = int(header[2]) if max_tokens is None else min(int(header[2]), max_tokens)
        tokens = np.frombuffer(f.read(2 * n), dtype=np.uint16)
    if len(tokens) != n:
        raise ValueError(f"{path}: header claims {n} tokens, file holds {len(tokens)}")
    return tokens


def split_docs(tokens: np.ndarray, sep: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """(complete documents, tail). ``tokens`` is a separator-delimited stream; a document is the run
    between two separators. Tokens before the first separator are dropped (a document cut by the
    previous shard); the tail from the last separator on is returned for the next shard."""
    cuts = np.flatnonzero(tokens == sep)
    if len(cuts) == 0:
        return [], tokens[:0]
    return [tokens[a + 1:b] for a, b in zip(cuts[:-1], cuts[1:])], tokens[cuts[-1]:]


def docs_sha256(docs: Iterable[str]) -> str:
    """sha256 over the documents' UTF-8 bytes, each prefixed with its length (so boundaries count)."""
    h = hashlib.sha256()
    for d in docs:
        b = d.encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def doc_hash(doc: str) -> bytes:
    return hashlib.sha256(doc.encode("utf-8")).digest()


def token_stream(id_lists: Iterable[Sequence[int]], sep: int) -> np.ndarray:
    """[sep] + doc_1 + [sep] + ... + doc_n + [sep], as uint16."""
    parts: List[np.ndarray] = [np.array([sep], dtype=np.int64)]
    for ids in id_lists:
        parts += [np.asarray(ids, dtype=np.int64), np.array([sep], dtype=np.int64)]
    out = np.concatenate(parts)
    if int(out.max()) >= 1 << 16 or int(out.min()) < 0:
        raise ValueError("token ids must fit in uint16")
    return out.astype(np.uint16)


def padded_vocab(n: int, multiple: int = 128) -> int:
    return -(-n // multiple) * multiple


class SourceDocs:
    """Documents of the GPT-2 source shards. ``val`` is decoded at construction; train documents are
    decoded shard by shard as tokenizers need more text (``extend``) and kept for the next one."""

    def __init__(self, fetch: Callable[[str], Path], decode: Callable[[List[int]], str], sep: int,
                 val_shard: str, val_tokens: int, train_shards: Sequence[str]) -> None:
        self.fetch, self.decode, self.sep = fetch, decode, sep
        t0 = time.time()
        val_tok = read_bin(fetch(val_shard), max_tokens=val_tokens)
        if val_tok[0] != sep:
            raise ValueError("the val shard does not start with a separator")
        docs, _ = split_docs(val_tok, sep)  # the document cut by the window end is not a val document
        self.val = [self._decode(d) for d in docs]
        self.val_hashes = {doc_hash(d) for d in self.val}
        self.train: List[str] = []
        self.train_bytes: List[int] = []
        self.dropped_as_val = 0
        self.shards_used: List[str] = []
        self._pending = list(train_shards)
        self._tail: Optional[np.ndarray] = None
        self.seconds = {"val_decode": round(time.time() - t0, 1), "train_decode": 0.0}

    def _decode(self, ids: np.ndarray) -> str:
        return self.decode(ids.tolist())  # strict UTF-8: a document that does not decode is an error

    def extend(self) -> bool:
        """Decodes the next train shard; False when none is left."""
        if not self._pending:
            return False
        name = self._pending.pop(0)
        tok = read_bin(self.fetch(name))
        t0 = time.time()
        if self._tail is not None:
            tok = np.concatenate([self._tail, tok])
        docs, self._tail = split_docs(tok, self.sep)  # before the first shard's first sep: a val-shard doc
        for d in docs:
            text = self._decode(d)
            if doc_hash(text) in self.val_hashes:
                self.dropped_as_val += 1
                continue
            self.train.append(text)
            self.train_bytes.append(len(text.encode("utf-8")))
        self.shards_used.append(name)
        self.seconds["train_decode"] = round(self.seconds["train_decode"] + time.time() - t0, 1)
        log(f"decoded {name}: {len(self.train):,} train docs, {sum(self.train_bytes) / 1e9:.2f} GB")
        return True

    def first_docs(self, budget: int) -> Tuple[List[str], int]:
        """The longest prefix of the train documents with at most ``budget`` UTF-8 bytes, decoding
        more shards while the decoded ones hold no more than that (so a larger budget extends the
        same ordered stream: a smaller budget's documents are its prefix). Returns (docs, bytes)."""
        while sum(self.train_bytes) <= budget and self.extend():
            pass
        docs: List[str] = []
        nbytes = 0
        for d, b in zip(self.train, self.train_bytes):
            if nbytes + b > budget:
                break
            docs.append(d)
            nbytes += b
        return docs, nbytes

    def assert_disjoint(self) -> None:
        overlap = sum(doc_hash(d) in self.val_hashes for d in self.train)
        if overlap:
            raise RuntimeError(f"{overlap} train documents also occur in val (sha256)")


# ----------------------------------------------------------------------------
# Tokenizers: encode_batch(docs) -> id lists, decode(ids) -> str, n_vocab, sep = n_vocab
# ----------------------------------------------------------------------------

class TiktokenTok:
    """tiktoken's GPT-2: 50,256 tokens plus <|endoftext|> (50256), the separator of the source shards."""

    def __init__(self, spec: Dict[str, Any], job: Path) -> None:
        import tiktoken

        self.enc = tiktoken.get_encoding(spec["encoding"])
        self.sep = self.n_vocab = self.enc.eot_token
        if self.enc.n_vocab != self.sep + 1:
            raise ValueError(f"{spec['encoding']}: <|endoftext|> is not the last id")

    def encode_batch(self, docs: List[str]) -> List[List[int]]:
        return self.enc.encode_ordinary_batch(docs, num_threads=os.cpu_count() or 1)

    def decode(self, ids: Sequence[int]) -> str:
        return self.enc.decode_bytes(list(ids)).decode("utf-8")


class TokenMonsterTok:
    """A prebuilt TokenMonster vocabulary, checked against its sha256. ``clear_normalization`` writes
    a copy whose header byte 2 (normalization flags, NFD in every prebuilt englishcode vocabulary)
    is 0: same tokens and ids, but the text is no longer NFD-normalized, so decode(encode(x)) == x."""

    def __init__(self, spec: Dict[str, Any], job: Path) -> None:
        import tokenmonster

        vocab = tokenmonster.load(spec["vocab"])
        raw = bytearray(Path(vocab.fname).read_bytes())
        digest = hashlib.sha256(raw).hexdigest()
        if digest != spec["vocab_sha256"]:
            raise ValueError(f"{spec['vocab']}: sha256 {digest} != {spec['vocab_sha256']}")
        if spec.get("clear_normalization"):
            raw[2] = 0
            path = job / "tokenmonster" / f"{spec['vocab']}-nonorm.vocab"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(raw)
            vocab = tokenmonster.load(str(path.resolve()))
        self.vocab = vocab
        self.sep = self.n_vocab = vocab.vocab_size

    def encode_batch(self, docs: List[str]) -> List[Any]:
        return self.vocab.tokenize(docs)  # tokenized in parallel by the tokenmonster server

    def decode(self, ids: Sequence[int]) -> str:
        return self.vocab.decode(list(ids))


class TokenisersTok:
    """A tokenizer of this repository, saved with ``save`` and loaded with its class."""

    def __init__(self, spec: Dict[str, Any], job: Path) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from tokenisers import models

        self.tok = getattr(models, spec["class"]).from_pretrained(str(job / spec["path"]))
        self.sep = self.n_vocab = self.tok.vocab_size
        self.pool = ThreadPoolExecutor(os.cpu_count() or 1)  # encode releases the GIL

    def encode_batch(self, docs: List[str]) -> List[List[int]]:
        return list(self.pool.map(self.tok.encode, docs))

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode_bytes(list(ids)).decode("utf-8")


KINDS = {"tiktoken": TiktokenTok, "tokenmonster": TokenMonsterTok, "tokenisers": TokenisersTok}


def load_tokenizer(spec: Dict[str, Any], job: Path) -> Any:
    return KINDS[spec["kind"]](spec, job)


def ids_sha256(id_lists: Iterable[Sequence[int]]) -> str:
    h = hashlib.sha256()
    for ids in id_lists:
        h.update(json.dumps([int(i) for i in ids]).encode("ascii"))
    return h.hexdigest()


def encode_chunks(tok: Any, docs: Sequence[str], start: int = 0, chunk: int = 2000) -> Iterator[Tuple[int, List[Any]]]:
    for i in range(start, len(docs), chunk):
        yield i, tok.encode_batch(list(docs[i:i + chunk]))


def encode_val(tok: Any, docs: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Val stream and its record; raises unless decode(encode(doc)) == doc for every document."""
    t0 = time.time()
    ids = [x for _, part in encode_chunks(tok, docs) for x in part]
    seconds = time.time() - t0
    lossy = []
    for k, (d, x) in enumerate(zip(docs, ids)):
        try:
            ok = tok.decode(x) == d
        except (UnicodeDecodeError, ValueError):
            ok = False
        if not ok:
            lossy.append(k)
    if lossy:
        raise RuntimeError(f"decode(encode(doc)) != doc for {len(lossy)} of {len(docs)} val documents "
                           f"(first: #{lossy[0]}, {docs[lossy[0]][:80]!r})")
    stream = token_stream(ids, tok.sep)
    nbytes = sum(len(d.encode("utf-8")) for d in docs)
    targets = len(stream) - 1
    return stream, {"docs": len(docs), "bytes": nbytes, "targets": targets, "bytes_per_token": nbytes / targets,
                    "docs_sha256": docs_sha256(docs), "lossless_docs": len(docs), "encode_seconds": round(seconds, 2),
                    "encode_mb_per_s": round(nbytes / max(seconds, 1e-9) / 1e6, 2)}


def encode_train(tok: Any, source: SourceDocs, target: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The train stream over source.train[0:n] for the smallest chunk-aligned n with >= ``target``
    tokens (one pass, no repeats; more shards are decoded when the documents run out)."""
    t0 = time.time()
    parts = [np.array([tok.sep], dtype=np.uint16)]
    total, ndocs = 1, 0
    while total < target:
        if ndocs >= len(source.train) and not source.extend():
            raise RuntimeError(f"the source shards hold only {total:,} tokens for this tokenizer; {target:,} needed")
        for i, ids in encode_chunks(tok, source.train, start=ndocs):
            part = token_stream(ids, tok.sep)[1:]  # each doc + its trailing separator
            parts.append(part)
            total += len(part)
            ndocs = i + len(ids)
            if total >= target:
                break
    seconds = time.time() - t0
    nbytes = sum(source.train_bytes[:ndocs])
    return np.concatenate(parts), {"docs": ndocs, "bytes": nbytes, "tokens": total, "bytes_per_token": nbytes / total,
                                   "target_tokens": target, "encode_seconds": round(seconds, 2),
                                   "encode_mb_per_s": round(nbytes / max(seconds, 1e-9) / 1e6, 2)}


# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------

def trainer_argv(cfg: Dict[str, Any], mbs: int, run: Path) -> List[str]:
    args = {**cfg["trainer_args"], "batch_size": mbs, "input_bin": str(run / "train.bin"),
            "input_val_bin": str(run / "val.bin"), "output_dir": str(run / "log")}
    return [a for k, v in args.items() for a in (f"--{k}", str(v))]


def run_trainer(job: Path, cfg: Dict[str, Any], run: Path, vocab_size: int, val_bytes: int) -> Dict[str, Any]:
    """Runs the patched llm.c trainer in its own process. When it runs out of GPU memory the
    micro-batch is halved (gradient accumulation keeps the step at the same tokens) and the run
    restarts."""
    settings = {"vocab_size": vocab_size, "padded_vocab_size": padded_vocab(vocab_size), "val_bytes": val_bytes,
                "result_path": str(run / "trainer.json")}
    if cfg.get("save_checkpoint"):
        settings["checkpoint_path"] = str(run / "checkpoint.pt")
    (run / "swap.json").write_text(json.dumps(settings))
    env = {**os.environ, "SWAP_CONFIG": str(run / "swap.json"), **cfg.get("trainer_env", {})}
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):  # one process, no DDP
        env.pop(k, None)
    mbs, attempts = int(cfg["mbs"]), []
    while True:
        t0 = time.time()
        with open(run / "train.log", "w") as f:
            proc = subprocess.run([sys.executable, "-u", str(job / "train_gpt2_swap.py"), *trainer_argv(cfg, mbs, run)],
                                  cwd=run, env=env, stdout=f, stderr=subprocess.STDOUT)
        seconds = time.time() - t0
        tail = (run / "train.log").read_text(errors="replace")[-3000:]
        attempts.append({"mbs": mbs, "returncode": proc.returncode, "seconds": round(seconds, 1)})
        if proc.returncode == 0:
            break
        oom = "OutOfMemoryError" in tail or "CUDA out of memory" in tail
        if not oom or mbs == 1:
            raise RuntimeError(f"trainer failed (exit {proc.returncode}, micro-batch {mbs}):\n{tail}")
        log(f"  out of GPU memory at micro-batch {mbs}; retrying with {mbs // 2}")
        mbs //= 2
    result = json.loads((run / "trainer.json").read_text())
    if not result.get("done"):
        raise RuntimeError("the trainer exited without its final record")
    return {**result, "settings": settings, "mbs": mbs, "attempts": attempts, "process_seconds": round(seconds, 1)}


# ----------------------------------------------------------------------------
# Job
# ----------------------------------------------------------------------------

def backend_info() -> Dict[str, Any]:
    gpu = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        gpu = out or None
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import importlib.metadata as md

        torch_version = md.version("torch")
    except Exception:
        torch_version = None
    return {"gpu": gpu, "torch": torch_version, "python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count()}


def setup(job: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """pip installs and the native extension (built from src.tar.gz, as the throughput benchmark does)."""
    out: Dict[str, Any] = {}
    if cfg.get("pip"):
        t0 = time.time()
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *cfg["pip"]], check=True)
        out["pip_seconds"] = round(time.time() - t0, 1)
    if (job / "src.tar.gz").exists():
        sys.path.insert(0, str(job))
        from throughput import build_extension  # uploaded next to this file

        out["build"] = build_extension(job)
    if (job / "tokenizers.tar.gz").exists() and not (job / "tokenizers").is_dir():
        with tarfile.open(job / "tokenizers.tar.gz") as tar:
            tar.extractall(job, filter="data")
    return out


def make_fetch(job: Path, source: Dict[str, Any], seconds: Dict[str, float]) -> Callable[[str], Path]:
    """Shard name -> local path: hf_hub_download from the speedrun's dataset repo, anonymously (no
    token), or a local directory (tests)."""
    def fetch(name: str) -> Path:
        if "local_dir" in source:
            return Path(source["local_dir"]) / name
        from huggingface_hub import hf_hub_download

        t0 = time.time()
        path = Path(hf_hub_download(repo_id=source["repo"], filename=name, repo_type="dataset",
                                    local_dir=str(job / "shards"), token=False))
        seconds[name] = round(time.time() - t0, 1)
        log(f"downloaded {name} in {seconds[name]} s")
        return path
    return fetch


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)  # a partially written run file never looks finished


def run_job(job_dir: str) -> Dict[str, Any]:
    job = Path(job_dir)
    t_job = time.time()
    cfg = json.loads((job / "config.json").read_text())
    backend = backend_info()
    log(f"backend: {backend}")
    timings: Dict[str, Any] = {"setup": setup(job, cfg), "download": {}}
    src = cfg["source"]
    source_tok = load_tokenizer(src["tokenizer"], job)
    t0 = time.time()
    source = SourceDocs(make_fetch(job, src, timings["download"]), source_tok.decode, source_tok.sep,
                        src["val_shard"], src["val_tokens"], src["train_shards"])
    log(f"val: {len(source.val):,} documents, {sum(len(d.encode()) for d in source.val):,} bytes")
    if docs_sha256(source.val) != cfg["val_docs_sha256"]:
        raise RuntimeError("the val documents differ from those decoded at preparation")
    # The first train documents trained our tokenizers (the job's text, and each tokenizer's own
    # when it was trained on more, e.g. on a Colab VM): the same decode here?
    for tt in [cfg["tokenizer_text"], *(s["tokenizer_text"] for s in cfg["tokenizers"] if "tokenizer_text" in s)]:
        while len(source.train) < tt["docs"] and source.extend():
            pass
        if docs_sha256(source.train[:tt["docs"]]) != tt["docs_sha256"]:
            raise RuntimeError(f"the first {tt['docs']:,} train documents differ from a tokenizer training text "
                               "decoded at preparation")
    timings["source_seconds"] = round(time.time() - t0, 1)

    steps, batch = int(cfg["trainer_args"]["num_iterations"]), int(cfg["trainer_args"]["total_batch_size"])
    target = math.ceil(steps * batch * (1 + cfg["headroom"])) + 1
    runs = job / "runs"
    for spec in cfg["tokenizers"]:
        name = spec["name"]
        if (runs / f"{name}.json").exists():
            log(f"{name}: finished in an earlier session, skipped")
            continue
        t_run = time.time()
        log(f"{name}: loading")
        tok = load_tokenizer(spec, job)
        record: Dict[str, Any] = {"name": name, "spec": spec, "launched": cfg.get("launched"),
                                  "n_vocab": tok.n_vocab, "separator_id": tok.sep,
                                  "vocab_size": tok.n_vocab + 1, "padded_vocab_size": padded_vocab(tok.n_vocab + 1)}
        if spec.get("ids_sha256"):  # same tokenizer as at preparation (a reload or version mismatch shows here)
            got = ids_sha256(tok.encode_batch(source.train[:spec["ids_docs"]]))
            if got != spec["ids_sha256"]:
                raise RuntimeError(f"{name}: ids on the first {spec['ids_docs']} train docs differ from preparation")
        val, record["val"] = encode_val(tok, source.val)
        log(f"{name}: val {record['val']['targets']:,} targets, {record['val']['bytes_per_token']:.4f} B/token, "
            f"lossless on {len(source.val):,} docs, {record['val']['encode_mb_per_s']} MB/s")
        record["train"] = record["trainer"] = None
        if name in cfg["train"]:
            run = job / "work" / name
            run.mkdir(parents=True, exist_ok=True)
            stream, record["train"] = encode_train(tok, source, target)
            record["train"]["tokens_per_step"] = batch
            source.assert_disjoint()
            log(f"{name}: train {record['train']['tokens']:,} tokens from {record['train']['docs']:,} docs, "
                f"{record['train']['bytes_per_token']:.4f} B/token, {record['train']['encode_mb_per_s']} MB/s")
            write_bin(run / "train.bin", stream)
            del stream
            val_bytes = record["val"]["bytes"]
            if cfg.get("eval_docs"):  # pilot: time the val pass on a prefix of the val documents
                val = val[:np.flatnonzero(val == tok.sep)[cfg["eval_docs"]] + 1]
                val_bytes = sum(len(d.encode("utf-8")) for d in source.val[:cfg["eval_docs"]])
            write_bin(run / "val.bin", val)
            record["trainer"] = run_trainer(job, cfg, run, tok.n_vocab + 1, val_bytes)
            ev = record["trainer"]["evals"][-1]
            log(f"{name}: step {ev['step']} val_bpb {ev['val_bpb']:.4f} ({ev['val_loss']:.4f} nats/token)")
            ckpt = run / "checkpoint.pt"
            if ckpt.exists():
                shutil.move(str(ckpt), str(runs / f"{name}.pt"))
            shutil.rmtree(run / "log", ignore_errors=True)
            for f in ("train.bin", "val.bin"):
                (run / f).unlink()
        record["seconds"] = round(time.time() - t_run, 1)
        write_json_atomic(runs / f"{name}.json", record)
    timings["decode"] = source.seconds
    data = {"val_docs": len(source.val), "val_bytes": sum(len(d.encode()) for d in source.val),
            "train_docs_decoded": len(source.train), "train_bytes_decoded": sum(source.train_bytes),
            "train_docs_dropped_as_val": source.dropped_as_val, "shards": [src["val_shard"], *source.shards_used]}
    result = {"backend": backend, "timings": timings, "data": data, "seconds": round(time.time() - t_job, 1),
              "runs": sorted(p.name for p in runs.iterdir() if p.suffix in (".json", ".pt"))}
    write_json_atomic(job / "result.json", result)
    return result


def main(job_dir: str, run: Optional[Callable[[str], Any]] = None) -> int:
    """Runs the job (``run``, default ``run_job``) and writes ``status.json`` last, atomically:
    {"state": "ok"} or {"state": "error", "traceback": ...}; the orchestrator polls for it."""
    job = Path(job_dir)
    try:
        (run or run_job)(job_dir)
        status: Dict[str, Any] = {"state": "ok"}
    except BaseException:
        status = {"state": "error", "traceback": traceback.format_exc()}
        print(status["traceback"], flush=True)
    tmp = job / "status.json.tmp"
    tmp.write_text(json.dumps(status))
    os.replace(tmp, job / "status.json")
    return 0 if status["state"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
