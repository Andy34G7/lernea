"""
Cross-lingual tokenization parity on FLORES+: tokens per parallel sentence and each language's
premium over English, tokens(lang) / tokens(English) on the same sentences (Petrov et al. 2023,
"Language Model Tokenizers Introduce Unfairness Between Languages"). 1.0 is parity.

Data: ``openlanguagedata/flores_plus`` (CC BY-SA 4.0, gated: accept its terms on Hugging Face) at
a pinned revision, DEVTEST split only, never used for training. Each sentence is encoded on its own
with no separator.

Tokenizers:
  intrinsic  the 13 suite variants at V = 16,384 / 32,768 / 65,536, retrained exactly as the
             committed intrinsic results (same dataset, parameters and SIGMORPHON lexicon); each
             one's test-split bytes/token is checked against the committed leaderboard.
  swap       the GPT-2-swap tokenizers (English FineWeb, ~50k), loaded from prepared job
             directories (config.json) exactly as the swap job loads them.

    python -m tokenisers.benchmarks.flores --skip-memory-check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from tokenisers.benchmarks.common import (
    DEFAULT_MORPH_LANGS,
    STANDARD_VOCAB_SIZES,
    VARIANTS,
    _cpu_model,
    assert_disjoint,
    check_memory,
    load_dataset,
    load_morphology,
    start_provenance,
    train_variant,
    variant_config,
    write_json,
)
from tokenisers.data.pipeline import DEFAULT_TRAIN_BYTES, FINEWEB2_LANGUAGES, _default_cache_dir

FLORES_REPO = "openlanguagedata/flores_plus"
FLORES_REVISION = "1147e2dc59f37fbb5c22fd8478770de02cd399f9"
FLORES_SPLIT = "devtest"  # the only split read: dev is left alone, nothing is tuned on FLORES+
FLORES_LICENSE = "CC BY-SA 4.0"
ENGLISH = "eng_Latn"
# FineWeb-2 config of the training mix (data.pipeline.FINEWEB2_LANGUAGES) -> FLORES+ file. FineWeb-2's
# cmn_Hani does not separate Simplified from Traditional Han; FLORES+ does, cmn_Hans is used.
IN_MIX: Dict[str, str] = {
    "deu_Latn": "deu_Latn",
    "fra_Latn": "fra_Latn",
    "spa_Latn": "spa_Latn",
    "rus_Cyrl": "rus_Cyrl",
    "cmn_Hani": "cmn_Hans",
    "jpn_Jpan": "jpn_Jpan",
    "arb_Arab": "arb_Arab",
    "hin_Deva": "hin_Deva",
    "hun_Latn": "hun_Latn",
    "fin_Latn": "fin_Latn",
}
# In no tokenizer's training mix: scripts the mix lacks, plus Greek and Hebrew.
OUT_OF_MIX: Tuple[str, ...] = ("kor_Hang", "tha_Thai", "tam_Taml", "amh_Ethi", "ell_Grek", "heb_Hebr", "ben_Beng")
LANGUAGES: Tuple[str, ...] = (ENGLISH, *IN_MIX.values(), *OUT_OF_MIX)
GROUPS = {"in_mix": tuple(IN_MIX.values()), "out_of_mix": OUT_OF_MIX}
PARITY_THRESHOLD = 1.5

_CACHE = Path.home() / ".cache" / "tokenisers"
SWAP_JOBS: Tuple[Path, ...] = (
    _CACHE / "nanogpt-swap-job",  # gpt2, TokenMonster 50k/32k, our 9 whitespace variants
    _CACHE / "nanogpt-swap-job-sp",  # <variant>_sp
    _CACHE / "nanogpt-swap-pool-g2",  # <variant>_g2
    _CACHE / "nanogpt-swap-pool-sp300",  # <variant>_sp300
)

UNAVAILABLE = (
    "FLORES+ ({repo}@{rev}) is unavailable: {exc}. The dataset is gated: accept its terms at "
    "https://huggingface.co/datasets/{repo} while logged in, and make that account's token available "
    "(`hf auth login` or HF_TOKEN). Or pass --flores-dir DIR with DIR/devtest/<lang>.jsonl files."
)


class FloresUnavailable(RuntimeError):
    """FLORES+ could not be downloaded (terms not accepted, no token, no network)."""


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def flores_root(cache_dir: Optional[str]) -> Path:
    """Download directory under the data pipeline's cache dir, one per pinned revision."""
    base = Path(cache_dir) if cache_dir else _default_cache_dir()
    return base / f"flores_plus-{FLORES_REVISION[:12]}"


def flores_file(lang: str, root: Path, download: bool = True) -> Path:
    """``root/devtest/<lang>.jsonl``, downloaded from the pinned revision on first use."""
    path = root / FLORES_SPLIT / f"{lang}.jsonl"
    if path.is_file() or not download:
        return path
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(FLORES_REPO, f"{FLORES_SPLIT}/{lang}.jsonl", repo_type="dataset",
                                    revision=FLORES_REVISION, local_dir=root))
    except Exception as exc:  # gated (401/403), no token, network: one actionable message
        raise FloresUnavailable(UNAVAILABLE.format(repo=FLORES_REPO, rev=FLORES_REVISION[:12], exc=exc)) from exc


def load_flores(langs: Sequence[str], root: Path, download: bool = True) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """({lang: devtest sentences in id order}, per-file provenance). Every row must be from the
    devtest split and every language must have exactly the first language's sentence ids."""
    texts: Dict[str, List[str]] = {}
    files: Dict[str, Any] = {}
    ids0: Optional[List[int]] = None
    for lang in langs:
        path = flores_file(lang, root, download)
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing")
        raw = path.read_bytes()
        rows = sorted((json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()),
                      key=lambda r: r["id"])
        splits = {r.get("split") for r in rows}
        if splits != {FLORES_SPLIT}:
            raise ValueError(f"{path}: rows from split(s) {sorted(map(str, splits))}; only {FLORES_SPLIT} is used")
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate sentence ids")
        if ids0 is None:
            ids0 = ids
        elif ids != ids0:
            raise ValueError(f"{lang}: sentence ids differ from {langs[0]}'s, the files are not parallel")
        texts[lang] = [r["text"] for r in rows]
        files[lang] = {"file": f"{FLORES_SPLIT}/{lang}.jsonl", "sha256": hashlib.sha256(raw).hexdigest(),
                       "sentences": len(rows)}
    return texts, files


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def byte_token_ids(tok: Any) -> Tuple[Set[int], Set[int]]:
    """(ids of one-byte tokens, ids of one-byte tokens >= 0x80, i.e. a raw piece of a multi-byte
    UTF-8 character: byte fallback). ``tok`` is a model of this repo or a swap_job wrapper."""
    if hasattr(tok, "enc"):  # swap_job.TiktokenTok
        pairs = ((i, tok.enc.decode_single_token_bytes(i)) for i in range(tok.enc.n_vocab))
    elif hasattr(tok, "vocab"):  # swap_job.TokenMonsterTok: type "single" is a one-byte token; the
        d = tok.vocab.get_dictionary()  # library decodes token bytes with errors="replace", so >= 0x80 is U+FFFD
        single = {i for i, e in d.items() if e["type"] == "single"}
        return single, {i for i in single if d[i]["token"] == "�"}
    else:
        model = getattr(tok, "tok", tok)  # swap_job.TokenisersTok wraps one of ours
        pairs = ((i, model.id_to_bytes(i) or b"") for i in range(model.vocab_size))
    one = {i: b[0] for i, b in pairs if len(b) == 1}
    return set(one), {i for i, b in one.items() if b >= 0x80}


def count_tokens(encode: Callable[[List[str]], Sequence[Sequence[int]]], single: Set[int], fallback: Set[int],
                 sentences: List[str]) -> Dict[str, int]:
    """Tokens, one-byte tokens and byte-fallback tokens over ``sentences``, each encoded on its own."""
    ids = [int(t) for seq in encode(sentences) for t in seq]
    return {"tokens": len(ids), "single_byte_tokens": sum(t in single for t in ids),
            "byte_fallback_tokens": sum(t in fallback for t in ids)}


def language_metrics(counts: Dict[str, Dict[str, int]], text: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
    """Per-language metrics from token counts and text sizes ({"sentences", "bytes", "chars"});
    premium = tokens / English tokens on the same parallel sentences."""
    eng = counts[ENGLISH]["tokens"]
    out: Dict[str, Dict[str, Any]] = {}
    for lang, c in counts.items():
        n, t = c["tokens"], text[lang]
        out[lang] = {
            **c,
            "tokens_per_sentence": n / t["sentences"],
            "premium": n / eng,
            "bytes_per_token": t["bytes"] / n,
            "chars_per_token": t["chars"] / n,
            "single_byte_share": c["single_byte_tokens"] / n,
            "byte_fallback_rate": c["byte_fallback_tokens"] / n,
        }
    return out


def aggregate(langs: Dict[str, Dict[str, Any]], group: Sequence[str]) -> Dict[str, Any]:
    """Mean / max premium, worst language, std of ln(premium) across the group (0 = every language
    equally far from English), share with premium <= 1.5, pooled byte-fallback rate."""
    prem = {lang: langs[lang]["premium"] for lang in group}
    worst = max(prem, key=prem.__getitem__)
    return {
        "languages": len(prem),
        "mean_premium": statistics.fmean(prem.values()),
        "max_premium": prem[worst],
        "worst_language": worst,
        "std_log_premium": statistics.pstdev(math.log(p) for p in prem.values()),
        "share_premium_le_1_5": sum(p <= PARITY_THRESHOLD for p in prem.values()) / len(prem),
        "byte_fallback_rate": sum(langs[g]["byte_fallback_tokens"] for g in group)
        / sum(langs[g]["tokens"] for g in group),
    }


def score(encode: Callable[[List[str]], Sequence[Sequence[int]]], tok: Any, flores: Dict[str, List[str]],
          text: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    single, fallback = byte_token_ids(tok)
    counts = {lang: count_tokens(encode, single, fallback, sents) for lang, sents in flores.items()}
    langs = language_metrics(counts, text)
    return {"languages": langs, "aggregates": {g: aggregate(langs, members) for g, members in GROUPS.items()
                                               if all(m in langs for m in members)}}


# ----------------------------------------------------------------------------
# Tokenizers
# ----------------------------------------------------------------------------

def committed_intrinsic(reference_dir: Path, vocab_size: int) -> Path:
    sub = "intrinsic" if vocab_size == 16_384 else f"v{vocab_size}/intrinsic"
    return reference_dir / sub / "intrinsic_leaderboard.json"


def intrinsic_tier(args: argparse.Namespace, flores: Dict[str, List[str]], text: Dict[str, Dict[str, int]],
                   models: Dict[str, Any], log: Callable[[str], None]) -> Dict[str, Any]:
    """Retrain each variant at each size as the intrinsic CLI does (train split, variant_config,
    SIGMORPHON lexicon), check test-split bytes/token against the committed leaderboard, score it."""
    ds = load_dataset(args)
    train, test = ds.texts("train"), ds.texts("test")
    assert_disjoint(train, test, "test")
    lexicon, _, morph = load_morphology(args, train, DEFAULT_MORPH_LANGS)
    test_bytes = sum(len(d.encode("utf-8")) for d in test)
    refs: Dict[str, Any] = {}
    for v in args.vocab_sizes:
        ref_path = committed_intrinsic(Path(args.reference_dir), v)
        ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.is_file() else None
        if ref and ref["meta"]["dataset"]["root"] != ds.root.name:
            ref = None  # a different dataset: nothing to compare against
        refs[str(v)] = str(ref_path) if ref else None
        for name in args.variants:
            t0 = time.perf_counter()
            tok = train_variant(name, train, v, lexicon)
            secs = time.perf_counter() - t0
            bpt = test_bytes / max(1, sum(len(tok.encode(d)) for d in test))
            want = ref["models"][name]["bytes_per_token"] if ref and name in ref["models"] else None
            training = {**variant_config(name, v), "requested_vocab_size": v, "vocab_size": tok.vocab_size,
                        "train_seconds": round(secs, 2), "test_bytes_per_token": bpt,
                        "committed_test_bytes_per_token": want,
                        "reproduces_committed": None if want is None else bpt == want}
            if want is not None and bpt != want:
                print(f"warning: {name} at V={v}: test bytes/token {bpt!r} != committed {want!r}", file=sys.stderr)
            models[f"{name}@{v}"] = {"tier": "intrinsic", "base": name, "vocab_size": tok.vocab_size,
                                     "training": training, **score(lambda s, t=tok: [t.encode(x) for x in s], tok,
                                                                   flores, text)}
            log(f"  {name}@{v}: |V|={tok.vocab_size}, trained in {secs:.1f}s, test bytes/token {bpt:.4f}"
                + ("" if want is None else f" ({'=' if bpt == want else '!='} committed)"))
            del tok
    m = ds.manifest
    return {"dataset": {"root": ds.root.name, "synthetic": ds.synthetic, "train_bytes": m["splits"]["train"]["text_bytes"],
                        "train_docs": m["splits"]["train"]["docs"], "test_bytes": test_bytes, "test_docs": len(test),
                        "config": m.get("config")},
            "morphology": morph, "vocab_sizes": list(args.vocab_sizes), "variants": list(args.variants),
            "committed_reference": refs}


def _home(path: Path) -> str:
    return str(path).replace(str(Path.home()), "~", 1)


def swap_tier(jobs: Sequence[Path], flores: Dict[str, List[str]], text: Dict[str, Dict[str, int]],
              models: Dict[str, Any], log: Callable[[str], None]) -> Dict[str, Any]:
    """Every tokenizer of each prepared job's config.json, loaded with swap_job.load_tokenizer (read
    only: TokenMonster's normalization-cleared copy goes to a temporary directory)."""
    from tokenisers.benchmarks.nanogpt_swap import swap_job

    out: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="flores-tokenmonster-") as tm_dir:
        for job in map(Path, jobs):
            cfg_raw = (job / "config.json").read_bytes()
            cfg = json.loads(cfg_raw)
            prepared = cfg.get("prepared") or {}
            out[job.name] = {"path": _home(job), "config_sha256": hashlib.sha256(cfg_raw).hexdigest(),
                             "prepared_git_commit": prepared.get("git_commit"),
                             "prepared_utc": prepared.get("created_utc"),
                             "tokenizer_text": {k: v for k, v in (cfg.get("tokenizer_text") or {}).items()
                                                if k in ("docs", "bytes", "docs_sha256")}}
            for spec in cfg["tokenizers"]:
                name = spec["name"]
                if name in models:
                    raise ValueError(f"{name} occurs in more than one swap job")
                if spec["kind"] == "tokenmonster":
                    try:
                        import tokenmonster  # noqa: F401
                    except ImportError as e:
                        raise SystemExit("TokenMonster tokenizers need the tokenmonster package: "
                                         "uv run --with tokenmonster==1.1.12 python -m tokenisers.benchmarks.flores") from e
                tok = swap_job.load_tokenizer(spec, Path(tm_dir) if spec["kind"] == "tokenmonster" else job)
                source = {"job": job.name, **{k: v for k, v in spec.items() if k not in ("name", "ids_docs")}}
                if spec["kind"] == "tokenisers":
                    source["tokenizer_json_sha256"] = hashlib.sha256(
                        (job / spec["path"] / "tokenizer.json").read_bytes()).hexdigest()
                models[name] = {"tier": "swap", "vocab_size": tok.n_vocab, "source": source,
                                **score(tok.encode_batch, tok, flores, text)}
                log(f"  {name} ({job.name}): |V|={tok.n_vocab}")
                del tok
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

CSV_FIELDS = ("tokens", "tokens_per_sentence", "premium", "bytes_per_token", "chars_per_token",
              "single_byte_share", "byte_fallback_rate")


def group_of(lang: str) -> str:
    return "reference" if lang == ENGLISH else next(g for g, m in GROUPS.items() if lang in m)


def write_results(out_dir: Path, doc: Dict[str, Any]) -> None:
    from tokenisers.benchmarks.report import render_flores

    write_json(out_dir / "flores.json", doc)
    with open(out_dir / "flores_by_language.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tokenizer", "tier", "vocab_size", "language", "group", "sentences", *CSV_FIELDS])
        for name, m in doc["models"].items():
            for lang, r in m["languages"].items():
                w.writerow([name, m["tier"], m["vocab_size"], lang, group_of(lang),
                            doc["meta"]["text"][lang]["sentences"], *(r[k] for k in CSV_FIELDS)])
    (out_dir / "flores.md").write_text(render_flores(doc) + "\n", encoding="utf-8")


def _version(pkg: str) -> Optional[str]:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="FLORES+ cross-lingual tokenization premium / parity")
    parser.add_argument("--vocab-sizes", type=int, nargs="*", default=list(STANDARD_VOCAB_SIZES),
                        help="Intrinsic tier sizes (none: skip the tier)")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS, metavar="VARIANT")
    parser.add_argument("--swap-jobs", type=Path, nargs="*", default=list(SWAP_JOBS),
                        help="Prepared GPT-2-swap job directories (none: skip the tier)")
    parser.add_argument("--train-bytes", type=int, default=DEFAULT_TRAIN_BYTES)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=None, help="Dataset cache dir (default: the data pipeline's)")
    parser.add_argument("--offline", action="store_true", help="Synthetic training corpus (tests); labelled synthetic")
    parser.add_argument("--flores-dir", type=Path, default=None,
                        help="Local FLORES+ copy with devtest/<lang>.jsonl (default: download the pinned revision)")
    parser.add_argument("--reference-dir", default="benchmarks/results",
                        help="Committed results holding the intrinsic leaderboards to reproduce")
    parser.add_argument("--output-dir", default="benchmarks/results/flores")
    parser.add_argument("--skip-memory-check", action="store_true")
    args = parser.parse_args(argv)
    start = start_provenance("tokenisers.benchmarks.flores", argv)

    if args.vocab_sizes and not args.skip_memory_check:
        check_memory(args.variants, args.train_bytes)
    root = args.flores_dir or flores_root(args.cache_dir)
    try:
        flores, files = load_flores(LANGUAGES, root, download=args.flores_dir is None)
    except FloresUnavailable as e:
        raise SystemExit(str(e)) from e
    text = {lang: {"sentences": len(s), "bytes": sum(len(x.encode("utf-8")) for x in s), "chars": sum(map(len, s))}
            for lang, s in flores.items()}

    models: Dict[str, Any] = {}
    intrinsic = None
    if args.vocab_sizes:
        print(f"Intrinsic tier: {len(args.variants)} variants x V = {args.vocab_sizes}")
        intrinsic = intrinsic_tier(args, flores, text, models, print)
    swap = None
    if args.swap_jobs:
        print(f"Swap tier: {len(args.swap_jobs)} prepared jobs")
        swap = swap_tier(args.swap_jobs, flores, text, models, print)

    doc = {"meta": {
        "benchmark": "flores",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **start,
        "host": {"platform": platform.platform(), "cpu": _cpu_model(), "cpu_count": os.cpu_count(),
                 "python": platform.python_version()},
        "flores": {"repo": FLORES_REPO, "revision": None if args.flores_dir else FLORES_REVISION,
                   "local_dir": str(args.flores_dir) if args.flores_dir else None, "license": FLORES_LICENSE,
                   "split": FLORES_SPLIT, "sentences": text[ENGLISH]["sentences"], "files": files},
        "languages": {"english": ENGLISH, "in_mix": IN_MIX, "out_of_mix": list(OUT_OF_MIX)},
        "text": text,
        "encoding": "each sentence on its own, no separator or special tokens",
        "premium": "total tokens over the language's devtest sentences / total over the same English sentences",
        "parity_threshold": PARITY_THRESHOLD,
        "intrinsic": intrinsic,
        "swap_jobs": swap,
        "versions": {p: _version(p) for p in ("tiktoken", "tokenmonster", "huggingface_hub")},
    }, "models": models}
    write_results(Path(args.output_dir), doc)
    print((Path(args.output_dir) / "flores.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
