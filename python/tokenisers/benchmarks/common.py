"""
Shared plumbing for the three benchmark CLIs: data loading with held-out splits, the tokenizer
suite, duplicate-vocabulary detection and run provenance.

Every CLI trains tokenizers on the prepared dataset's TRAIN split only and evaluates on a held-out
split (tokka and intrinsic: TEST; proxy LM: VAL). ``assert_disjoint`` checks that at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from tokenisers.data.pipeline import (
    DEFAULT_TRAIN_BYTES,
    DataPipeline,
    PreparedDataset,
    SigmorphonLoader,
)

# git runs here. Only when this file is tracked by the enclosing repository is that repository the
# source checkout of this code; an installed copy (a venv inside some other repository) is not.
_HERE = Path(__file__).resolve().parent

DEFAULT_VOCAB_SIZE = 16_384
# 16k is the default; 32k and 64k are the other sizes the suite is meant to be run at.
STANDARD_VOCAB_SIZES = (16_384, 32_768, 65_536)

# SIGMORPHON 2022 languages that also occur in the prepared corpus.
DEFAULT_MORPH_LANGS = ("eng", "fra", "hun", "rus", "spa")

MORPHEME_LAMBDA = 2.0

VARIANTS: Tuple[str, ...] = (
    "bpe_frequency",
    "wordpiece_pmi",
    "wp_bpe_base",
    "wp_bpe_ft",
    "wp_bpe_pig",
    "wp_bpe_scp",
    "wp_bpe_npmi",
    "wp_bpe_ppig",
    "mdl_bits",
    "length_penalized",
    "super_bpe",
    "morpheme_bpe",
    "unigram_lm",
)

# "<variant>_sp": the variant retrained with pretokenizer="space_prefix" (a word takes the one space
# before it, GPT-2-style) and otherwise identically; "<variant>_g2": with pretokenizer="gpt2" (GPT-2's
# exact regex pre-split). A number after the suffix ("_sp300") is the size of the training text in
# MB, which the benchmark that trains it records; the parameters are the same. SuperBPE has no
# pre-tokenizer modes.
PRETOKENIZER_SUFFIXES: Dict[str, str] = {"sp": "space_prefix", "g2": "gpt2"}
SPACE_PREFIX_VARIANTS: Dict[str, str] = {f"{v}_sp": v for v in VARIANTS if v != "super_bpe"}


def parse_variant(name: str) -> Tuple[str, str, Optional[int]]:
    """(base variant, pretokenizer, training MB or None) of a variant name: ``"mdl_bits"`` ->
    ("mdl_bits", "whitespace", None), ``"mdl_bits_g2"`` -> (..., "gpt2", None), ``"mdl_bits_sp300"``
    -> (..., "space_prefix", 300). Raises ValueError for an unknown name."""
    m = re.fullmatch(r"(.+)_(sp|g2)([1-9][0-9]*)?", name)
    if m and m[1] in VARIANTS and m[1] != "super_bpe":
        return m[1], PRETOKENIZER_SUFFIXES[m[2]], int(m[3]) if m[3] else None
    if name in VARIANTS:
        return name, "whitespace", None
    raise ValueError(f"unknown variant {name!r}")

# Scorer name and every parameter of each BpeTokenizer.train variant. Nothing is left to a library
# default, so the training record in each result file (meta.variants) is exactly what ran. The
# values are the scorers' defaults (README formula table). min_frequency: pairs seen fewer times
# are never merged; alpha is the scp discount d and the npmi support exponent; penalty is the ppig
# length exponent eta and the mdl cost in bits per vocabulary entry.
_BPE_SCORERS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "bpe_frequency": ("frequency", {"min_frequency": 1}),
    "wordpiece_pmi": ("wordpiece", {"min_frequency": 2, "alpha": 1.0}),
    "wp_bpe_base": ("wp-bpe-base", {"min_frequency": 1}),
    "wp_bpe_ft": ("wp-bpe-ft", {"min_frequency": 2}),
    "wp_bpe_pig": ("wp-bpe-pig", {"min_frequency": 1}),
    "wp_bpe_scp": ("wp-bpe-scp", {"min_frequency": 2, "alpha": 0.75}),
    "wp_bpe_npmi": ("wp-bpe-npmi", {"min_frequency": 2, "alpha": 0.5}),
    "wp_bpe_ppig": ("wp-bpe-ppig", {"min_frequency": 2, "penalty": 0.35}),
    "mdl_bits": ("mdl", {"min_frequency": 2, "penalty": 24.0}),
    "length_penalized": ("length_penalized", {"min_frequency": 1, "beta": 0.5}),
}


# ----------------------------------------------------------------------------
# CLI arguments and data
# ----------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser, output_dir: str) -> None:
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE,
                        help=f"Total vocabulary size (default {DEFAULT_VOCAB_SIZE}; also run at 32768 / 65536)")
    parser.add_argument("--train-bytes", type=int, default=DEFAULT_TRAIN_BYTES,
                        help="Train-split size of the prepared dataset (python -m tokenisers.data)")
    parser.add_argument("--data-seed", type=int, default=0, help="Dataset split seed")
    parser.add_argument("--cache-dir", default=None, help="Dataset cache dir (default: the data pipeline's)")
    parser.add_argument("--offline", action="store_true",
                        help="Use the synthetic offline corpus and the 12-word morpheme fixture (results are labelled synthetic)")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS, metavar="VARIANT",
                        help="Tokenizer variants to train (default: all)")
    parser.add_argument("--output-dir", default=output_dir, help="Directory for result JSON/Markdown")
    parser.add_argument("--skip-memory-check", action="store_true",
                        help="Train even if the estimated peak memory exceeds what is available")


def load_dataset(args: argparse.Namespace) -> PreparedDataset:
    pipeline = DataPipeline(cache_dir=args.cache_dir, force_offline=args.offline)
    return pipeline.prepare(train_bytes=args.train_bytes, seed=args.data_seed)


# Peak memory of training one variant ~ BASE_MEMORY + per-byte cost x training text bytes. Measured
# whole-process peak RSS at V=65536 (the largest supported size) on the real corpus, commit 1e3b4da
# (see MEASURED_TRAIN_PEAKS in tests/e2e/test_bench_reporting.py): super_bpe 0.42 / 0.79 / 1.53 /
# 2.92 GiB at 4.3 / 10.1 / 25.1 / 52.5 MB of text, the largest plain BPE variant (wp_bpe_pig) 1.25
# GiB and unigram_lm 0.52 GiB at 52.5 MB. Growth per byte falls with size (super_bpe ~58 B/B at
# 52.5 MB); the costs below cover every measurement with >= 20% margin. Variants train one after
# another, so the run's peak is the largest variant's.
TRAIN_MEMORY_PER_BYTE: Dict[str, int] = {"super_bpe": 75}
DEFAULT_TRAIN_MEMORY_PER_BYTE = 60
BASE_MEMORY = 600 * 2**20  # interpreter, torch, corpus copies, SIGMORPHON gold


def usable_system_memory(meminfo: str) -> Optional[int]:
    """MemAvailable minus a reserve for the rest of the machine (desktop session, other jobs):
    max(2 GiB, 15% of MemTotal). A benchmark may not take all free memory of a shared machine."""
    kb = {}
    for line in meminfo.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            kb[key] = int(rest.split()[0]) * 1024
    if "MemAvailable" not in kb:
        return None
    reserve = max(2 * 2**30, int(0.15 * kb.get("MemTotal", 0)))
    return max(0, kb["MemAvailable"] - reserve)


def available_memory() -> Optional[int]:
    """Bytes this process may allocate: min(usable_system_memory, the free part of every cgroup v2
    memory.max above it). The desktop reserve applies inside a cgroup too: a limit such as
    systemd-run -p MemoryMax= can lower the allowance, never raise it."""
    avail: Optional[int] = None
    try:
        avail = usable_system_memory(Path("/proc/meminfo").read_text())
        rel = next(l[3:] for l in Path("/proc/self/cgroup").read_text().splitlines() if l.startswith("0::"))
        cg = Path("/sys/fs/cgroup") / rel.strip().lstrip("/")
        for d in [cg, *cg.parents]:
            limit = d / "memory.max"
            if limit.is_file() and limit.read_text().strip() != "max":
                free = int(limit.read_text()) - int((d / "memory.current").read_text())
                avail = free if avail is None else min(avail, free)
            if d == Path("/sys/fs/cgroup"):
                break
    except (OSError, StopIteration, ValueError):
        pass
    return avail


def estimate_train_memory(variants: Sequence[str], train_bytes: int) -> int:
    per_byte = max(TRAIN_MEMORY_PER_BYTE.get(v, DEFAULT_TRAIN_MEMORY_PER_BYTE) for v in variants)
    return BASE_MEMORY + per_byte * train_bytes


def own_rss() -> int:
    """Resident memory of this process now (already counted as used by ``available_memory``)."""
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def check_memory(variants: Sequence[str], train_bytes: int) -> None:
    """Exit before training if the estimated peak exceeds the memory this process may use, instead
    of letting the kernel OOM-kill it (or the desktop session) halfway through. The estimate is the
    process's total peak; what the process already holds (interpreter, torch) is part of it and is
    not available memory any more, so only the difference must fit."""
    need = estimate_train_memory(variants, train_bytes)
    avail = available_memory()
    if avail is not None and need - own_rss() > avail:
        worst = max(variants, key=lambda v: TRAIN_MEMORY_PER_BYTE.get(v, DEFAULT_TRAIN_MEMORY_PER_BYTE))
        raise SystemExit(
            f"Training {worst} on {train_bytes / 1e6:.0f} MB needs ~{need / 2**30:.1f} GiB but only "
            f"{avail / 2**30:.1f} GiB is available. Use a smaller --train-bytes, drop {worst} from "
            "--variants, or pass --skip-memory-check."
        )


def assert_disjoint(train_texts: Iterable[str], eval_texts: Iterable[str], what: str = "evaluation") -> None:
    """Raise if any evaluation document is also a training document."""
    overlap = set(train_texts) & set(eval_texts)
    if overlap:
        raise RuntimeError(f"{len(overlap)} {what} documents also occur in the training split")


def load_morphology(
    args: argparse.Namespace, train_texts: Sequence[str], langs: Sequence[str]
) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]], Dict[str, Any]]:
    """(lexicon for morpheme_bpe, gold annotations, provenance).

    Gold = SIGMORPHON TEST split (surface entries). Lexicon = SIGMORPHON TRAIN split surface entries
    whose word occurs (whitespace-delimited) in the training text, minus every gold word, so the
    evaluated words never reach the tokenizer through the lexicon. Offline runs use the 12-word
    fixture as gold and an empty lexicon.
    """
    loader = SigmorphonLoader(cache_dir=args.cache_dir, offline=args.offline)
    if args.offline:
        gold = loader.load_offline_fixture()
        for g in gold:
            g["lang"] = "fixture"
        src = "offline test fixture (12 hand-written words, NOT SIGMORPHON)"
        return {}, gold, {"gold_source": src, "gold_entries": len(gold), "lexicon": None}

    gold: List[Dict[str, Any]] = []
    lexicon: Dict[str, List[str]] = {}
    prov: Dict[str, Any] = {"gold_source": f"SIGMORPHON 2022 test split, surface entries ({', '.join(langs)})",
                            "gold": [], "lexicon": []}
    words = {w for text in train_texts for w in text.split()}
    for lang in langs:
        gold.extend(loader.load(lang, "test"))
        prov["gold"].append(loader.provenance(lang, "test"))
        # Streamed: the train files hold up to ~750k words per language, only corpus words are kept.
        with open(loader.path(lang, "train"), encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 2 or parts[0] not in words:
                    continue
                segments = [s for s in parts[1].split(" @@") if s]
                if "".join(segments) == parts[0]:  # surface segmentations only, as in SigmorphonLoader
                    lexicon.setdefault(parts[0], segments)
        prov["lexicon"].append(loader.provenance(lang, "train"))
    gold_words = {g["word"] for g in gold}
    dropped = sorted(gold_words & set(lexicon))
    for w in dropped:
        del lexicon[w]
    prov["gold_entries"] = len(gold)
    prov["lexicon_entries"] = len(lexicon)
    prov["lexicon_words_dropped_as_gold"] = len(dropped)
    return lexicon, gold, prov


# ----------------------------------------------------------------------------
# Tokenizer suite
# ----------------------------------------------------------------------------

def hf_pretokenizer_label(config: Any) -> str:
    """``"hf:<sha>"``: the first 12 hex digits of the SHA-256 of a Hugging Face ``pre_tokenizer``
    config (dict or JSON string) as canonical JSON (sorted keys, no spaces, UTF-8)."""
    config = json.loads(config) if isinstance(config, str) else config
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "hf:" + hashlib.sha256(canonical.encode()).hexdigest()[:12]


def variant_config(name: str, vocab_size: int, pretokenizer: Any = None) -> Dict[str, Any]:
    """Class and every training parameter of a suite variant (or of its ``_sp`` / ``_g2`` retrain,
    see ``parse_variant``) besides corpus and vocab_size (the morpheme lexicon is in
    meta.morphology), recorded in every result file. ``pretokenizer`` (a mode name or
    ``{"hf": config}``) replaces the name's mode; an HF config is recorded as
    ``"pretokenizer": hf_pretokenizer_label(config)`` plus ``"pretokenizer_config": config``."""
    base, mode, _ = parse_variant(name)
    if isinstance(pretokenizer, dict):
        config = pretokenizer["hf"]
        config = json.loads(config) if isinstance(config, str) else config
        pretok = {"pretokenizer": hf_pretokenizer_label(config), "pretokenizer_config": config}
    else:
        pretok = {"pretokenizer": pretokenizer or mode}
    if base in _BPE_SCORERS:
        scorer, params = _BPE_SCORERS[base]
        return {"class": "BpeTokenizer", "scorer": scorer, **params, **pretok}
    if base == "super_bpe":
        if pretokenizer is not None:
            raise ValueError("super_bpe has no pre-tokenizer modes")
        return {"class": "SuperBpeTokenizer", "v_switch": 256 + (max(vocab_size, 256) - 256) * 9 // 10}
    if base == "morpheme_bpe":
        return {"class": "MorphemeBpeTokenizer", "scorer": "frequency", "min_frequency": 1,
                "lambda_weight": MORPHEME_LAMBDA, **pretok}
    if base == "unigram_lm":
        return {"class": "UnigramTokenizer", "num_iterations": 10, "shrinking_factor": 0.75, **pretok}
    raise ValueError(f"unknown variant {name!r}")


def train_variant(name: str, corpus: List[str], vocab_size: int, lexicon: Dict[str, List[str]],
                  pretokenizer: Any = None) -> Any:
    """Trains ``name`` with exactly the parameters ``variant_config`` records."""
    from tokenisers import models

    params = variant_config(name, vocab_size, pretokenizer)
    if "pretokenizer_config" in params:
        params["pretokenizer"] = {"hf": params.pop("pretokenizer_config")}
    cls = getattr(models, params.pop("class"))
    if parse_variant(name)[0] == "morpheme_bpe":
        params["morpheme_lexicon"] = lexicon
    return cls.train(corpus=corpus, vocab_size=vocab_size, **params)


def train_suite(
    variants: Sequence[str],
    corpus: List[str],
    vocab_size: int,
    lexicon: Dict[str, List[str]],
    log: Callable[[str], None] = print,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Train each variant sequentially. Returns (tokenizers, per-variant training record)."""
    models: Dict[str, Any] = {}
    records: Dict[str, Dict[str, Any]] = {}
    for name in variants:
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tok = train_variant(name, corpus, vocab_size, lexicon)
        secs = time.perf_counter() - t0
        models[name] = tok
        records[name] = {
            **variant_config(name, vocab_size),
            "requested_vocab_size": vocab_size,
            "vocab_size": tok.vocab_size,
            "train_seconds": round(secs, 2),
            "warnings": [str(w.message) for w in caught],
        }
        log(f"  trained {name}: |V|={tok.vocab_size} in {secs:.1f}s")
    return models, records


def tokenizer_fingerprint(tok: Any) -> str:
    """sha256 of the serialized tokenizer (vocab, merges/scores, pre-tokenizer): equal iff it encodes identically."""
    if hasattr(tok, "export_hf_json"):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tokenizer.json"
            tok.export_hf_json(str(path))
            return hashlib.sha256(path.read_bytes()).hexdigest()
    vocab = sorted(tok.get_vocab().items())
    return hashlib.sha256(json.dumps(vocab).encode("utf-8")).hexdigest()


def find_duplicates(models: Dict[str, Any]) -> Dict[str, str]:
    """{duplicate_name: first_name_with_the_same_tokenizer}, in ``models`` order."""
    first: Dict[str, str] = {}
    dups: Dict[str, str] = {}
    for name, tok in models.items():
        fp = tokenizer_fingerprint(tok)
        if fp in first:
            dups[name] = first[fp]
        else:
            first[fp] = name
    return dups


# ----------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------

def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=_HERE, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


# Tracked inputs of the native extension (tokenisers._core), which builds and runs every tokenizer.
_NATIVE_SOURCES = (":(top)crates", ":(top)Cargo.toml", ":(top)Cargo.lock")


def native_extension(in_checkout: bool) -> Optional[Dict[str, Any]]:
    """File name, sha256 and build time of the loaded native extension. The extension is untracked
    and only rebuilt by ``maturin develop``: after a pull or merge without a rebuild, old Rust code
    runs under the new commit. In a source checkout ``stale_sources`` lists the tracked Rust
    sources modified after the extension was written (git checkout/merge touch the files they
    change); non-empty means the extension is stale or may be. None outside a checkout."""
    try:
        from tokenisers import _core
    except ImportError:
        return None
    so = Path(_core.__file__)
    built = so.stat().st_mtime
    stale: Optional[List[str]] = None
    if in_checkout:
        top = Path(_git("rev-parse", "--show-toplevel") or ".")
        files = (_git("ls-files", "--full-name", "--", *_NATIVE_SOURCES) or "").splitlines()
        stale = [f for f in files if (top / f).is_file() and (top / f).stat().st_mtime > built]
    return {"file": so.name, "sha256": hashlib.sha256(so.read_bytes()).hexdigest(),
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(built)), "stale_sources": stale}


def start_provenance(module: str, argv: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Command line, package version, commit and dirty flag of the source checkout (None for an
    installed copy, which no commit identifies), and the native extension that produces the
    tokenizers. CLIs call this when they start: a run can take hours, and the tree may change
    before its results are written. Dirty means modified tracked code: result files and README.md
    (e.g. written by an earlier benchmark run) do not count."""
    in_checkout = _git("ls-files", "--error-unmatch", "--", Path(__file__).name) is not None
    status = _git("status", "--porcelain", "--untracked-files=no", "--", ":(top)",
                  ":(top,exclude)benchmarks/results", ":(top,exclude)README.md") if in_checkout else None
    try:
        version: Optional[str] = importlib.metadata.version("tokenisers")
    except importlib.metadata.PackageNotFoundError:
        version = None
    native = native_extension(in_checkout)
    if native and native["stale_sources"]:
        print(f"warning: the native extension {native['file']} is older than {len(native['stale_sources'])} "
              f"Rust source file(s) (e.g. {native['stale_sources'][0]}): results may come from older code "
              "than the recorded commit. Rebuild with `uv run maturin develop --release`.", file=sys.stderr)
    return {
        "command": ["python", "-m", module, *(sys.argv[1:] if argv is None else argv)],
        "tokenisers_version": version,
        "git_commit": _git("rev-parse", "HEAD") if in_checkout else None,
        "git_dirty": None if status is None else bool(status),
        "native_extension": native,
    }


def run_metadata(
    args: argparse.Namespace,
    dataset: PreparedDataset,
    benchmark: str,
    eval_split: str,
    train_records: Dict[str, Dict[str, Any]],
    duplicates: Dict[str, str],
    *,
    start: Dict[str, Any],
    **extra: Any,
) -> Dict[str, Any]:
    """``start`` is ``start_provenance()`` taken when the run started."""
    manifest = dataset.manifest
    return {
        "benchmark": benchmark,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **start,
        "host": {
            "platform": platform.platform(),
            "cpu": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "requested_vocab_size": args.vocab_size,
        "synthetic_data": dataset.synthetic,
        "train_split": "train",
        "eval_split": eval_split,
        "train_bytes": manifest["splits"]["train"]["text_bytes"],
        "train_docs": manifest["splits"]["train"]["docs"],
        "eval_bytes": manifest["splits"][eval_split]["text_bytes"],
        "eval_docs": manifest["splits"][eval_split]["docs"],
        "dataset": {"root": dataset.root.name, **manifest},
        "variants": train_records,
        "duplicates": duplicates,
        **extra,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_VOCAB_SIZE",
    "STANDARD_VOCAB_SIZES",
    "PRETOKENIZER_SUFFIXES",
    "SPACE_PREFIX_VARIANTS",
    "VARIANTS",
    "add_common_args",
    "assert_disjoint",
    "available_memory",
    "usable_system_memory",
    "check_memory",
    "estimate_train_memory",
    "find_duplicates",
    "start_provenance",
    "load_dataset",
    "load_morphology",
    "native_extension",
    "run_metadata",
    "tokenizer_fingerprint",
    "train_suite",
    "parse_variant",
    "train_variant",
    "variant_config",
    "write_json",
]
