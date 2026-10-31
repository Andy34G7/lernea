"""
Markdown leaderboards rendered from benchmark result JSON ({"meta": ..., "models": ...}).

The CLIs write these next to their JSON, and scripts/gen_readme_tables.py puts the same text into
README.md, so README numbers always come from a result file. Variants whose tokenizer is
byte-identical to another (``meta["duplicates"]``) share one row.
"""

from __future__ import annotations

import math
from pathlib import Path
import statistics
from typing import Any, Callable, Dict, List, Sequence, Tuple


def _mb(n: Any) -> str:
    return f"{n / 1e6:.1f} MB" if isinstance(n, (int, float)) else "?"


def _summary(meta: Dict[str, Any]) -> str:
    data = "SYNTHETIC offline corpus" if meta.get("synthetic_data") else "real corpus"
    commit = (meta.get("git_commit") or "unknown")[:10] + (" (dirty)" if meta.get("git_dirty") else "")
    stale = (meta.get("native_extension") or {}).get("stale_sources")
    if stale:
        commit += f" (STALE native extension: built before {len(stale)} Rust source file(s) changed)"
    if meta.get("tokenizer_path"):  # tokka --tokenizer-path: nothing was trained by this run
        parts = [f"{data}: external tokenizer `{Path(meta['tokenizer_path']).name}`, not trained by this run "
                 "(its training data is unknown and was not checked against the evaluation split)"]
    else:
        parts = [f"requested V = {meta.get('requested_vocab_size', '?')}",
                 f"{data}: tokenizers trained on the train split ({_mb(meta.get('train_bytes'))})"]
    parts += [
        f"evaluated on the {'' if meta.get('tokenizer_path') else 'held-out '}{meta.get('eval_split', '?')} "
        f"split ({_mb(meta.get('eval_bytes'))})",
        f"commit `{commit}`",
        f"generated {meta.get('created_utc', '?')}",
    ]
    args = " ".join(meta.get("command", [])[3:])
    return "- " + "; ".join(parts) + f"\n- arguments: {f'`{args}`' if args else '(defaults)'}"


def _label(name: str, duplicates: Dict[str, str]) -> str:
    aliases = [d for d, c in duplicates.items() if c == name]
    return " = ".join(f"`{n}`" for n in [name, *aliases])


def _table(
    doc: Dict[str, Any],
    columns: Sequence[Tuple[str, Callable[[Dict[str, Any]], str]]],
    sort_key: Callable[[Dict[str, Any]], float],
    note: bool = True,
) -> List[str]:
    dups = doc["meta"].get("duplicates", {})
    rows = sorted(((n, m) for n, m in doc["models"].items() if n not in dups), key=lambda kv: sort_key(kv[1]))
    lines = [
        "| Tokenizer | " + " | ".join(h for h, _ in columns) + " |",
        "| :--- |" + " ---: |" * len(columns),
    ]
    for name, m in rows:
        lines.append(f"| {_label(name, dups)} | " + " | ".join(f(m) for _, f in columns) + " |")
    if dups and note:
        lines.append("")
        lines.append("Variants joined with `=` produced byte-identical tokenizers and are one result.")
    return lines


def render_tokka(doc: Dict[str, Any]) -> str:
    meta = doc["meta"]
    langs = [k for k in next(iter(doc["models"].values()), {}) if isinstance(k, str)
             and k not in ("macro_avg", "micro_avg", "unique_tokens", "vocab_size")]
    cols = [
        ("\\|V\\|", lambda m: str(m["vocab_size"])),
        ("Macro bytes/token (↑)", lambda m: f"{m['macro_avg']['bytes_per_token']:.3f}"),
        ("Micro bytes/token (↑)", lambda m: f"{m['micro_avg']['bytes_per_token']:.3f}"),
        ("Macro fertility (↓)", lambda m: f"{m['macro_avg']['subword_fertility']:.3f}"),
        ("Macro continued-token % (↓)", lambda m: f"{m['macro_avg']['continued_word_rate']:.2f}"),
        ("Unique tokens used", lambda m: str(m["unique_tokens"])),
    ]
    macro = meta.get("macro_languages", langs)
    rule = (f" with >= {meta['macro_min_docs']} held-out documents and >= {meta['macro_min_bytes']:,} bytes"
            if "macro_min_docs" in meta else "")
    head = [
        _summary(meta),
        f"- macro = unweighted mean over the {len(macro)} languages{rule} ({', '.join(macro)}); "
        f"micro = totals summed over all {len(langs)} languages (token-weighted); unique tokens = union over languages",
        "",
    ]
    sizes = ["", "Held-out text per language:", "",
             "| Language | Docs | Bytes | In macro |", "| :--- | ---: | ---: | :---: |"]
    for lang in langs:
        sizes.append(f"| {lang} | {meta.get('language_docs', {}).get(lang, '?')} | "
                     f"{meta.get('language_bytes', {}).get(lang, 0):,} | {'yes' if lang in macro else 'no'} |")
    return "\n".join(head + _table(doc, cols, lambda m: -m["macro_avg"]["bytes_per_token"]) + sizes)


def _f3(x: Any) -> str:
    """Three decimals; "n/a" for an undefined score (nothing was scored), never a fake 0.000."""
    return "n/a" if x is None else f"{x:.3f}"


def render_intrinsic(doc: Dict[str, Any]) -> str:
    meta = doc["meta"]
    cols = [
        ("\\|V\\|", lambda m: str(m["vocab_size"])),
        ("Bytes/token (↑)", lambda m: f"{m['bytes_per_token']:.3f}"),
        ("Fertility (↓)", lambda m: f"{m['fertility']:.3f}"),
        ("Rényi E1", lambda m: f"{m['renyi_e_1_0']:.3f}"),
        ("Rényi E2.5", lambda m: f"{m['renyi_e_2_5']:.3f}"),
        ("Morph F1 (↑)", lambda m: _f3(m["morph_f1"])),
        ("Ident F1 (↑)", lambda m: _f3(m["identifier_f1"])),
        ("Encode MB/s", lambda m: f"{m['encode_mb_per_sec']:.2f}"),
        ("Decode MB/s", lambda m: f"{m['decode_mb_per_sec']:.2f}"),
    ]
    first = next(iter(doc["models"].values()), {})
    morph = meta.get("morphology") or {}
    words = first.get("morph_words_by_lang", {})
    head = [
        _summary(meta),
        f"- morph F1: per-word boundary F1 on {first.get('morph_words', '?')} gold words "
        f"({morph.get('gold_source', 'SIGMORPHON 2022 test split')}), averaged per language, then over the "
        f"{len(words) or '?'} languages unweighted (per language below); identifier F1: "
        f"{first.get('identifiers', '?')} identifiers mined from held-out code; throughput: median of "
        f"{first.get('throughput_repeats', '?')} passes over {_mb(first.get('throughput_bytes'))} "
        f"({first.get('throughput_docs', '?')} documents, one encode call each) on {meta.get('host', {}).get('cpu', '?')}",
        "",
    ]
    lines = head + _table(doc, cols, lambda m: -m["renyi_e_2_5"])
    if words:
        by_lang = [(f"{lang} ({n:,} words)", (lambda lang: lambda m: _f3(m["morph_f1_by_lang"][lang]))(lang))
                   for lang, n in words.items()]
        by_lang.append(("Word-weighted", lambda m: _f3(m["morph_f1_word_weighted"])))
        lines += ["", "Morph F1 by language:", "", *_table(doc, by_lang, lambda m: -m["renyi_e_2_5"], note=False)]
    return "\n".join(lines)


def _pm(mean: float, sd: Any) -> str:
    """mean ± sd; just the mean when sd is undefined (a single seed)."""
    return f"{mean:.3f}" if sd is None else f"{mean:.3f} ± {sd:.3f}"


def render_colab(doc: Dict[str, Any]) -> str:
    meta = doc["meta"]
    domains = list(meta.get("val_domains", []))
    backend = meta.get("backend", {})
    if backend.get("device") == "cuda":
        where = f"Colab GPU `{backend.get('gpu')}`" if backend.get("kind") == "colab" else f"local GPU `{backend.get('gpu')}`"
    else:
        where = f"{backend.get('kind', '?')} CPU `{backend.get('cpu', '?')}`"
    fb = meta.get("fallback")
    cols = [("\\|V\\|", lambda m: str(m["vocab_size"])),
            ("Train bytes/token", lambda m: f"{m['train_bytes_per_token']:.3f}"),
            ("Tokens trained", lambda m: f"{m['tokens_consumed']:,}")]
    cols += [(f"{d} BPB (↓)", (lambda d: lambda m: _pm(m["val_domains"][d]["bpb"], m["val_domains"][d]["bpb_sd"]))(d))
             for d in domains]
    cols.append(("Macro BPB (↓)", lambda m: f"**{_pm(m['macro_val_bpb'], m['macro_val_bpb_sd'])}**"))
    seeds = meta.get("seeds") or []
    lm_text = meta.get("lm_text") or {}
    pool = lm_text.get("pool_bytes")
    source = ("the train split the tokenizers were trained on plus an LM text pool with no val or test document"
              if pool else "the same train split the tokenizers were trained on")
    used = (f"each document used once; {_mb(pool)} of the {_mb(lm_text['bytes'])} of LM text came from the pool, "
            "`meta.lm_text`" if pool else "each document used once")
    head = [
        _summary(meta),
        f"- backend: {where}" + (f" — FALLBACK after Colab failed: {fb.get('error', '')}" if fb else ""),
        f"- proxy LM: {meta.get('lm_architecture', 'architecture not recorded')}; "
        f"{', '.join(f'{k}={v}' for k, v in meta.get('lm', {}).items())}",
        f"- fixed budget {meta.get('byte_budget', 0):,} UTF-8 bytes of LM training text per tokenizer, drawn "
        f"from {source} ({used}); seeds "
        f"{seeds} " + ("(mean ± sample sd)" if len(seeds) > 1 else "(single seed: no sd)"),
        "",
    ]
    return "\n".join(head + _table(doc, cols, lambda m: m["macro_val_bpb"]))


def render_colab_compare(small: Dict[str, Any], big: Dict[str, Any], ref: str = "bpe_frequency") -> str:
    """Proxy-LM ranking of the tokenizers both tiers trained, small LM vs big LM: rank by mean macro val BPB
    and the difference to ``ref`` in the same tier, ± its standard error over the seeds."""
    tiers = {"small LM": small, "big LM": big}
    dups = {**small["meta"].get("duplicates", {}), **big["meta"].get("duplicates", {})}
    names = [n for n in big["models"] if n in small["models"] and n not in dups]
    stats: Dict[str, Dict[str, Tuple[int, float, float]]] = {}
    for tier, doc in tiers.items():
        ms, r = doc["models"], doc["models"][ref]
        order = sorted(names, key=lambda n: ms[n]["macro_val_bpb"])
        stats[tier] = {n: (order.index(n) + 1, ms[n]["macro_val_bpb"] - r["macro_val_bpb"],
                           math.sqrt(ms[n]["macro_val_bpb_sd"] ** 2 / len(ms[n]["macro_val_bpb_by_seed"])
                                     + r["macro_val_bpb_sd"] ** 2 / len(r["macro_val_bpb_by_seed"])))
                       for n in names}

    def setup(doc: Dict[str, Any]) -> str:
        meta, lm = doc["meta"], doc["meta"].get("lm", {})
        return (f"d_model={lm.get('d_model')}, n_layers={lm.get('n_layers')}, {meta.get('byte_budget', 0):,}-byte "
                f"budget, seeds {meta.get('seeds')}, `{meta.get('backend', {}).get('gpu')}`")

    head = [f"- the {len(names)} tokenizers both proxy-LM tiers trained; small LM: {setup(small)}; "
            f"big LM: {setup(big)}",
            f"- rank: by mean macro val BPB (1 = best); Δ: macro val BPB minus `{ref}`'s in the same tier "
            "(negative = better) ± its standard error √(sd²/n + sd_ref²/n) over the n seeds (sample sd)"]
    for tier, st in stats.items():
        better = [st[n] for n in names if st[n][1] < 0]
        if better:
            head.append(f"- {tier}: {len(better)} tokenizers beat `{ref}`, by {min(-d for _, d, _ in better):.3f}–"
                        f"{max(-d for _, d, _ in better):.3f} BPB, i.e. {min(-d / se for _, d, se in better):.1f}–"
                        f"{max(-d / se for _, d, se in better):.1f} standard errors")
    lines = head + ["", "| Tokenizer | Small LM rank | Small LM Δ BPB | Big LM rank | Big LM Δ BPB |",
                    "| :--- | ---: | ---: | ---: | ---: |"]
    for n in sorted(names, key=lambda n: stats["big LM"][n][0]):
        cells = [f"{st[n][0]} | " + ("reference" if n == ref else f"{st[n][1]:+.3f} ± {st[n][2]:.3f}")
                 for st in stats.values()]
        lines.append(f"| `{n}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _gib(n: Any) -> str:
    return f"{n / 2**30:.1f} GiB" if isinstance(n, (int, float)) else "?"


def render_throughput(doc: Dict[str, Any]) -> str:
    """Throughput re-measurement (``tokenisers.benchmarks.throughput``): every pass per cell, the
    run-to-run spread (max/min over the passes) and the load during each pass."""
    meta = doc["meta"]
    b = meta.get("backend", {})
    prep = meta.get("prepared", {})
    cpus = b.get("affinity") or []
    lscpu = b.get("lscpu", {})
    if b.get("kind") == "colab":
        where = (f"Colab CPU runtime (no accelerator requested), session `{b.get('session', '?')}`; "
                 f"{len(cpus)} vCPUs ({lscpu.get('Core(s) per socket', '?')} core(s) x {lscpu.get('Thread(s) per core', '?')} "
                 f"threads, {lscpu.get('Hypervisor vendor', '?')} VM)")
    else:
        pin = ", ".join(f"CPU {c['cpu']} (max {c['max_mhz']} MHz, governor {c['governor']}, "
                        f"SMT sibling(s) {c['thread_siblings']})" for c in cpus)
        where = (f"local, pinned to {pin}" if len(cpus) == 1 else f"local, {len(cpus)} CPUs")
    build = b.get("build")
    built = (f"; extension built on the VM from commit `{(meta.get('git_commit') or '?')[:10]}` "
             f"({build.get('rustc')}, {build.get('seconds')} s)") if build else ""
    commit = (meta.get("git_commit") or "unknown")[:10] + (" (dirty)" if meta.get("git_dirty") else "")
    ds = prep.get("dataset", {})
    n_tok = sum(len(v) for v in doc["models"].values())
    passes = meta.get("passes", 0)
    spread = {}
    for key in ("encode_mb_per_sec", "decode_mb_per_sec"):
        ratios = sorted(max(m[key]) / min(m[key]) for v in doc["models"].values() for m in v.values())
        spread[key] = (statistics.median(ratios), ratios[-1]) if ratios else (math.nan, math.nan)
    load = "; ".join(f"run {i + 1}: {p['before']['loadavg'][0]:.2f} -> {p['after']['loadavg'][0]:.2f}, "
                     f"{_gib(p['before']['mem_available'])} -> {_gib(p['after']['mem_available'])} "
                     f"({p['seconds']:.0f} s)" for i, p in enumerate(meta.get("pass_load", [])))
    head = [
        f"- {where}: {b.get('cpu', '?')}, {_gib(b.get('mem_total'))} RAM, Python {b.get('python', '?')}, "
        f"{b.get('libc', '?')}{built}; commit `{commit}`; generated {meta.get('created_utc', '?')}",
        f"- {n_tok} tokenizers: the suite's variants at V = {', '.join(map(str, meta.get('vocab_sizes', [])))}, "
        f"retrained from the {'SYNTHETIC offline' if prep.get('synthetic_data') else 'real'} train split "
        f"({_mb(prep.get('train_bytes'))}, dataset `{ds.get('root', '?')}`), saved, reloaded on the measuring "
        "machine and checked there: identical token ids (sha256) on every throughput document",
        f"- {prep.get('throughput_docs', '?')} documents, {_mb(prep.get('throughput_bytes'))}, one encode/decode "
        f"call per document; each cell is the median of {meta.get('repeats', '?')} timed passes after "
        f"{meta.get('warmup', '?')} warm-up (`IntrinsicEvaluator.benchmark_throughput`), measured in {passes} "
        f"full runs over all {n_tok} tokenizers: run 1 / run 2" + (" / ..." if passes > 2 else ""),
        f"- run-to-run spread (max/min over the runs, per tokenizer): encode median "
        f"{spread['encode_mb_per_sec'][0]:.2f}x, max {spread['encode_mb_per_sec'][1]:.2f}x; decode median "
        f"{spread['decode_mb_per_sec'][0]:.2f}x, max {spread['decode_mb_per_sec'][1]:.2f}x",
        f"- 1-min load average (the benchmark's own process adds about 1) and MemAvailable before -> after "
        f"each run: {load}",
        "",
        "| Tokenizer | \\|V\\| | Encode MB/s | Decode MB/s |",
        "| :--- | ---: | ---: | ---: |",
    ]
    rows = [f"| `{name}` | {m['vocab_size']} | " + " | ".join(
        " / ".join(f"{x:.2f}" for x in m[key]) for key in ("encode_mb_per_sec", "decode_mb_per_sec")) + " |"
        for v in doc["models"].values() for name, m in v.items()]
    return "\n".join(head + rows)


__all__ = ["render_tokka", "render_intrinsic", "render_colab", "render_throughput"]
