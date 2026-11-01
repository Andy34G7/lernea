"""
Markdown leaderboards rendered from benchmark result JSON ({"meta": ..., "models": ...}).

The CLIs write these next to their JSON, and scripts/gen_readme_tables.py puts the same text into
README.md, so README numbers always come from a result file. Variants whose tokenizer is
byte-identical to another (``meta["duplicates"]``) share one row.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _mb(n: Any) -> str:
    return f"{n / 1e6:.1f} MB" if isinstance(n, (int, float)) else "?"


def _commit(p: Dict[str, Any]) -> str:
    """A provenance record's short commit, flagged when its tree was dirty."""
    return (p.get("git_commit") or "unknown")[:10] + (" (dirty)" if p.get("git_dirty") else "")


def _swap_provenance(meta: Dict[str, Any], models: Dict[str, Any]) -> str:
    """What a swap header adds to the run's own commit: the commits that prepared its tokenizers (an
    earlier ``--prepare-only``, and each ``--from-pool`` pool that trained them) and that ran a run kept
    from an earlier session (``--resume``), wherever they differ from the run's."""
    own, prep = _commit(meta), meta.get("prepared") or {}
    # Results written before pools carried provenance store each pool as its command list: no commit to report.
    pools = [p for p in (prep.get("pools") or {}).values() if isinstance(p, dict)]
    made = sorted({_commit(p) for p in [prep, *pools]} - {own})
    ran: Dict[str, List[str]] = {}
    for n, m in models.items():
        if m.get("launched") and _commit(m["launched"]) != own:
            ran.setdefault(_commit(m["launched"]), []).append(f"`{n}`")
    out = (f", tokenizers prepared at commit {' / '.join(f'`{c}`' for c in made)}" if made else "") + "".join(
        f", {', '.join(ns)} run in an earlier session at commit `{c}`" for c, ns in sorted(ran.items()))
    return out + (" (dirty = uncommitted changes, not reproducible from any commit)" if "(dirty)" in out else "")


def _summary(meta: Dict[str, Any]) -> str:
    data = "SYNTHETIC offline corpus" if meta.get("synthetic_data") else "real corpus"
    commit = _commit(meta)
    stale = (meta.get("native_extension") or {}).get("stale_sources")
    if stale:
        commit += f" (STALE native extension: built before {len(stale)} Rust source file(s) changed)"
    commit = f"commit `{commit}`"
    prep = meta.get("job_prepared") or {}  # --resume: the job (tokenizers, data, ids) was prepared by another run
    if prep and ((prep.get("git_commit"), bool(prep.get("git_dirty")))
                 != (meta.get("git_commit"), bool(meta.get("git_dirty")))):
        commit += (f", resuming a job (tokenizers, LM text and its ids) prepared at commit "
                   f"`{(prep.get('git_commit') or 'unknown')[:10]}`"
                   + (" from a dirty tree (uncommitted changes: not reproducible from any commit)"
                      if prep.get("git_dirty") else ""))
    if meta.get("tokenizer_path"):  # tokka --tokenizer-path: nothing was trained by this run
        parts = [f"{data}: external tokenizer `{Path(meta['tokenizer_path']).name}`, not trained by this run "
                 "(its training data is unknown and was not checked against the evaluation split)"]
    else:
        parts = [f"requested V = {meta.get('requested_vocab_size', '?')}",
                 f"{data}: tokenizers trained on the train split ({_mb(meta.get('train_bytes'))})"]
    parts += [
        f"evaluated on the {'' if meta.get('tokenizer_path') else 'held-out '}{meta.get('eval_split', '?')} "
        f"split ({_mb(meta.get('eval_bytes'))})",
        commit,
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
        f"{seeds} " + ("(mean ± sample sd; the seed sets only the LM's initialization: every seed trains on the "
                       "same LM text in the same order, so the sd leaves out data-sampling variance)"
                       if len(seeds) > 1 else "(single seed: no sd)"),
        "",
    ]
    return "\n".join(head + _table(doc, cols, lambda m: m["macro_val_bpb"]))


def colab_compare_stats(small: Dict[str, Any], big: Dict[str, Any], ref: str = "bpe_frequency"
                        ) -> Tuple[List[str], Dict[str, Dict[str, Tuple[int, float, float]]]]:
    """The tokenizers both proxy-LM tiers trained, and per tier {name: (rank by mean macro val BPB, macro val
    BPB minus ``ref``'s in the same tier, its standard error over the seeds)}."""
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
    return names, stats


def render_colab_compare(small: Dict[str, Any], big: Optional[Dict[str, Any]], ref: str = "bpe_frequency") -> str:
    """Proxy-LM ranking of the tokenizers both tiers trained, small LM vs big LM: rank by mean macro val BPB
    and the difference to ``ref`` in the same tier, ± its standard error over the seeds. ``big`` is None until
    the big tier has been run."""
    if big is None:
        return "_The big proxy LM has not been run yet, so there is nothing to compare._"
    names, stats = colab_compare_stats(small, big, ref)

    def setup(doc: Dict[str, Any]) -> str:
        meta, lm = doc["meta"], doc["meta"].get("lm", {})
        return (f"d_model={lm.get('d_model')}, n_layers={lm.get('n_layers')}, {meta.get('byte_budget', 0):,}-byte "
                f"budget, seeds {meta.get('seeds')}, `{meta.get('backend', {}).get('gpu')}`")

    head = [f"- the {len(names)} tokenizers both proxy-LM tiers trained; small LM: {setup(small)}; "
            f"big LM: {setup(big)}",
            f"- rank: by mean macro val BPB (1 = best); Δ: macro val BPB minus `{ref}`'s in the same tier "
            "(negative = better) ± its standard error √(sd²/n + sd_ref²/n) over the n seeds (sample sd). The seeds "
            "vary only the LM's initialization (same LM text, same order), so the standard errors measure "
            "initialization noise on this one text sample, not data-sampling variance"]
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
    commit = _commit(meta)
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


def _swap_findings(doc: Dict[str, Any], models: Dict[str, Any]) -> List[str]:
    """Method notes and findings under a full tokenizer-swap table (``models`` sorted by final BPB);
    every number comes from the JSON."""
    eq, g = doc.get("gpt2_equivalent"), models.get("gpt2")
    if not eq or not g:
        return []
    meta = doc["meta"]
    steps, batch = int(meta["recipe"]["args"]["num_iterations"]), int(meta["recipe"]["args"]["total_batch_size"])
    check = eq["train_minus_val_bpb"]
    worst = max(check, key=lambda c: abs(c[1]))
    best, second = list(models)[:2]
    more = {n: m for n, m in models.items() if n != "gpt2" and m["val_bytes_per_token"] > g["val_bytes_per_token"]}
    more_bpb = [m["final_bpb"] for m in more.values()]
    # Ours with the "whitespace" pre-tokenizer (the default: specs written before they recorded it), which
    # isolates every whitespace run as its own token; super_bpe merges across whitespace
    ws = {n: m for n, m in models.items() if m["spec"]["kind"] == "tokenisers" and n != "super_bpe"
          and m["spec"].get("pretokenizer", "whitespace") == "whitespace"}
    ws_bpt = [m["val_bytes_per_token"] for m in ws.values()]
    less = [100 * (1 - m["train_bytes_per_token"] / g["train_bytes_per_token"]) for m in ws.values()]
    nats = meta["target"]["nats_per_gpt2_token"]
    lines = [
        "",
        f"- GPT-2-equivalent steps: the first step at which GPT-2's train loss, in bits per byte of its train text "
        f"(nats / ln 2 / {g['train_bytes_per_token']:.3f} bytes per token) and smoothed by a centred "
        f"{eq['window']}-step rolling mean, reaches the tokenizer's final val BPB: how many of GPT-2's {steps} steps "
        f"buy the same quality (GPT-2: {eq['steps']['gpt2']} by definition). The train text is one pass over unseen "
        f"documents; at GPT-2's val passes (steps {check[0][0]}-{check[-1][0]}) the smoothed train BPB is within "
        f"{abs(worst[1]):.4f} of val BPB (largest at step {worst[0]}).",
        f"- train: each tokenizer's first {steps} x {batch:,} tokens of the same FineWeb train documents in the same "
        "order, one pass; one run per tokenizer, llm.c's fixed seed",
        "",
        "Findings (one seed per tokenizer: the seed-to-seed spread is not measured here):",
        "",
        f"- Lowest final BPB: `{best}` ({models[best]['final_bpb']:.4f})"
        + (", GPT-2's own tokenizer" if best == "gpt2" else "")
        + f"; next `{second}` ({models[second]['final_bpb']:.4f}, +"
        f"{models[second]['final_bpb'] - models[best]['final_bpb']:.4f}).",
    ]
    if more:
        lines.append(
            f"- More val bytes per token than GPT-2's {g['val_bytes_per_token']:.3f}: "
            + ", ".join(f"`{n}` {m['val_bytes_per_token']:.3f}" for n, m in more.items())
            + f". Their final BPB is {min(more_bpb):.4f}-{max(more_bpb):.4f}, "
            + ("all above GPT-2's: more text per step did not give a lower BPB."
               if min(more_bpb) > g["final_bpb"] else f"against GPT-2's {g['final_bpb']:.4f}."))
    if ws:
        lines.append(
            f"- Our {len(ws)} whitespace-isolating tokenizers (the repo's pre-tokenizer makes every whitespace run its "
            f"own token) get {min(ws_bpt):.2f}-{max(ws_bpt):.2f} val bytes per token, so each {batch:,}-token step "
            f"holds {min(less):.0f}-{max(less):.0f}% less text than GPT-2's (train bytes per token). The comparison "
            "is confounded by that design choice: they train on less text, not only on different tokens.")
    lines.append(
        f"- Early training: after {steps} steps GPT-2 is at {g['final_nats_per_token']:.2f} nats per token on val, "
        f"far from the speedrun's target of {nats}; "
        + ("no tokenizer reaches the target. " if all(m["reached_target_step"] is None for m in models.values())
           else "")
        + "The ranking at this budget need not hold for longer training.")
    return lines


def render_nanogpt_swap(doc: Dict[str, Any]) -> str:
    """Tokenizer swap on llm.c's GPT-2 (124M) recipe: final val BPB and speed per tokenizer (a pilot
    adds the projected full-run time)."""
    meta, models = doc["meta"], doc["models"]
    b = meta.get("backend") or {}
    rec = meta["recipe"]
    a = rec["args"]
    target = meta["target"]["bpb"]
    commit = _commit(meta)
    data = meta.get("data") or {}
    tt = meta["tokenizer_text"]
    # Each of our tokenizers' training text (its spec's, e.g. 300 MB from a Colab VM, else the job's).
    texts = sorted({(t["docs"], t["bytes"]) for t in (m["spec"].get("tokenizer_text") or tt for m in models.values()
                                                     if m["spec"].get("kind") == "tokenisers")}
                   or {(tt["docs"], tt["bytes"])}, key=lambda x: x[1])
    pilot = meta["mode"] == "pilot"
    pretoks = sorted({v["pretokenizer"] for v in (meta["prepared"].get("variants") or {}).values() if "pretokenizer" in v})
    head = [
        f"- {'PILOT (timing only, not a result): ' if pilot else ''}{b.get('gpu') or meta['requested_gpu']}, "
        f"torch {b.get('torch', '?')}, Colab session `{b.get('session', '?')}`; commit `{commit}`"
        f"{_swap_provenance(meta, models)}; "
        f"generated {meta['created_utc']}",
        f"- recipe: `{rec['upstream']}` scripts/pyrun_gpt2_124M.sh (GPT-2 124M, tied embeddings, AdamW lr "
        f"{a['learning_rate']}, wd {a['weight_decay']}, {a['total_batch_size']} tokens/step, bf16, compiled) with "
        f"{a['num_iterations']} steps, warmup {a['warmup_iters']}, val every {a['val_loss_every']} steps",
        f"- val: {data.get('val_docs', '?')} FineWeb documents ({_mb(data.get('val_bytes'))}, the speedrun's first "
        f"{meta['source']['val_tokens']:,} GPT-2 val tokens), the same bytes for every tokenizer; BPB = NLL bits "
        "over every val token (one separator per document included) / UTF-8 bytes",
        f"- our tokenizers: trained on the first {' / '.join(f'{d:,} train documents ({_mb(n)})' for d, n in texts)} at 50,256 "
        "tokens + 1 separator" + (f", pre-tokenizer {' / '.join(pretoks)}" if pretoks else "")
        + "; TokenMonster: " + meta["prepared"]["tokenmonster_why"],
        f"- target: the speedrun's {meta['target']['nats_per_gpt2_token']} nats per GPT-2 token = "
        + (f"{target:.4f} BPB on this val text" if target else "? (GPT-2 did not run)"),
        *([f"- note: {meta['note']}"] if meta.get("note") else []),
        "",
    ]
    if not pilot:
        eq = (doc.get("gpt2_equivalent") or {}).get("steps", {})
        cols = ("| Tokenizer | \\|V\\| | Params | Val B/token | Train B/token | Final BPB | GPT-2-equivalent steps "
                "| Step at target | ms/step | tok/s |")
        lines = [cols, "| :--- |" + " ---: |" * 9]
        done = dict(sorted(((n, m) for n, m in models.items() if "final_bpb" in m), key=lambda kv: kv[1]["final_bpb"]))
        for name, m in done.items():
            step, eq_step = m["reached_target_step"], eq.get(name, "?")
            lines.append(f"| `{name}` | {m['vocab_size']:,} | {m['params'] / 1e6:.1f} M | {m['val_bytes_per_token']:.3f} | "
                         f"{m['train_bytes_per_token']:.3f} | **{m['final_bpb']:.4f}** | "
                         f"{eq_step if eq_step is not None else 'not reached'} | "
                         f"{step if step is not None else 'not reached'} | {m['ms_per_step']:.0f} | "
                         f"{m['tokens_per_s']:,.0f} |")
        return "\n".join(head + lines + _swap_findings(doc, done))
    p = doc["projection"]
    lines = ["| Tokenizer | \\|V\\| | Val B/token | Encode MB/s | ms/step | tok/s | Peak GiB | Warm eval s "
             "| Projected h |", "| :--- |" + " ---: |" * 8]
    for name, m in models.items():
        r = p["per_tokenizer"][name]
        timed = m.get("ms_per_step") is not None
        lines.append(
            f"| `{name}` | {m['vocab_size']:,} | {m['val_bytes_per_token']:.3f} | {m['encode_mb_per_s']:.1f} | "
            + (f"{m['ms_per_step']:.0f} | {m['tokens_per_s']:,.0f} | {(m['peak_memory_bytes'] or 0) / 2**30:.1f} | "
               f"{m['eval_seconds'][-1]:.0f} | " if timed else f"({r['ms_per_step']:.0f}) | | | | ")
            + f"{r['total_hours']:.2f} |")
    t = meta.get("timings") or {}
    build = (t.get("setup") or {}).get("build") or {}
    lines += ["", f"- setup: pip {(t.get('setup') or {}).get('pip_seconds', '?')} s, extension build "
              f"{build.get('seconds', '?')} s, downloads (s) {t.get('download')}, decode (s) {t.get('decode')}",
              f"- projection, one session, sequential: {p['total_hours']:.1f} h for {p['tokenizers_projected']} "
              f"tokenizers x {p['steps']} steps ({p['evals_per_run']} evals each, source text "
              f"{p['source_text_gb']:.2f} GB); ms/step in parentheses: estimated from a timed run by FLOPs per token"]
    return "\n".join(head + lines)


SPACE_PREFIX_SWAP = "benchmarks/results/nanogpt_swap_sp/nanogpt_swap.json"


def render_nanogpt_swap_space_prefix(base: Dict[str, Any], sp: Optional[Dict[str, Any]]) -> str:
    """Per variant, the whitespace-isolating run (``base``, the committed tokenizer swap) against its
    ``_sp`` space-prefix retrain (``sp``, a separate run of the same recipe; None until it exists),
    with GPT-2's row from ``base`` for reference."""
    if sp is None:
        return (f"_The space-prefix retrain has not been run yet: no `{SPACE_PREFIX_SWAP}` is committed, so "
                "there is nothing to compare._")
    bm, sm = base["models"], sp["models"]
    g = bm.get("gpt2", {})
    bmeta, smeta = base["meta"], sp["meta"]
    checks = {
        "recipe": bmeta["recipe"]["args"] == smeta["recipe"]["args"],
        "val documents": {m["val_docs_sha256"] for m in bm.values()} == {m["val_docs_sha256"] for m in sm.values()},
        "tokenizer training text": bmeta["tokenizer_text"]["docs_sha256"] == smeta["tokenizer_text"]["docs_sha256"],
    }
    differ = [k for k, ok in checks.items() if not ok]
    gpus = [(x.get("backend") or {}).get("gpu") or x.get("requested_gpu") for x in (bmeta, smeta)]
    commit = _commit(smeta)

    def num(m: Optional[Dict[str, Any]], key: str, fmt: str) -> str:
        return format(m[key], fmt) if m and m.get(key) is not None else "-"

    lines = [
        f"- space-prefix run: {gpus[1]}, commit `{commit}`{_swap_provenance(smeta, sm)}, generated "
        f"{smeta['created_utc']}; whitespace run: {gpus[0]}",
        "- " + (f"DIFFERENT {', '.join(differ)} from the whitespace run: the BPB columns are not comparable"
                if differ else "same recipe, val documents and tokenizer training text as the whitespace run"),
        "",
        "| Variant | Val B/token whitespace | Val B/token space_prefix | Final BPB whitespace | "
        "Final BPB space_prefix | Change |",
        "| :--- |" + " ---: |" * 5,
    ]
    for name, m in sm.items():
        old = bm.get(name[:-3] if name.endswith("_sp") else name)
        change = (f"{m['final_bpb'] - old['final_bpb']:+.4f}"
                  if old and "final_bpb" in old and "final_bpb" in m else "-")
        lines.append(f"| `{name[:-3] if name.endswith('_sp') else name}` | {num(old, 'val_bytes_per_token', '.3f')} | "
                     f"{num(m, 'val_bytes_per_token', '.3f')} | {num(old, 'final_bpb', '.4f')} | "
                     f"{num(m, 'final_bpb', '.4f')} | {change} |")
    if g:
        lines.append(f"| `gpt2` (reference) | {num(g, 'val_bytes_per_token', '.3f')} | | "
                     f"{num(g, 'final_bpb', '.4f')} | | |")
    return "\n".join(lines)


ABLATION_SWAP = "benchmarks/results/nanogpt_swap_ablation/nanogpt_swap.json"


def render_nanogpt_swap_ablation(base: Dict[str, Any], sp: Optional[Dict[str, Any]],
                                 abl: Optional[Dict[str, Any]]) -> str:
    """The pre-split and data-scaling ablation (``abl``: ``<variant>_g2`` = GPT-2's regex pre-split
    and ``<variant>_sp300`` = space-prefix on 300 MB, one run of the same recipe; None until it
    exists) per variant against its whitespace run (``base``) and its 50 MB space-prefix run (``sp``,
    None until committed), with GPT-2's row from ``base`` for reference. Cells: final val BPB (val
    bytes per token)."""
    if abl is None:
        return (f"_Not yet run: no `{ABLATION_SWAP}` is committed, so there is nothing to compare._")
    import re

    bm, am = base["models"], abl["models"]
    bmeta, ameta = base["meta"], abl["meta"]
    parsed = {n: re.fullmatch(r"(.+)_(sp|g2)([0-9]*)", n) for n in am}
    suffixes = list(dict.fromkeys(f"_{m[2]}{m[3]}" for m in parsed.values() if m))
    variants = list(dict.fromkeys(m[1] for m in parsed.values() if m))
    tt = bmeta["tokenizer_text"]

    def text(name: str) -> str:
        t = am[name]["spec"].get("tokenizer_text") or ameta["tokenizer_text"]
        same = t["docs_sha256"] == tt["docs_sha256"] or (t.get("starts_with") or {}).get("docs_sha256") == tt["docs_sha256"]
        return f"{t['docs']:,} docs, {_mb(t['bytes'])}" + ("" if same else " (NOT starting with the 50 MB text)")

    checks = {
        "recipe": bmeta["recipe"]["args"] == ameta["recipe"]["args"],
        "val documents": {m["val_docs_sha256"] for m in bm.values()} == {m["val_docs_sha256"] for m in am.values()},
    }
    differ = [k for k, ok in checks.items() if not ok]
    commit = _commit(ameta)
    gpu = (ameta.get("backend") or {}).get("gpu") or ameta.get("requested_gpu")
    cols = [("whitespace, 50 MB", bm, ""), ("space_prefix, 50 MB", (sp or {}).get("models") or {}, "_sp")]
    for suf in suffixes:
        mode = "gpt2" if suf.startswith("_g2") else "space_prefix"
        size = f"{suf[3:]} MB" if suf[3:] else "50 MB"
        cols.append((f"{mode}, {size}", am, suf))

    def cell(models: Dict[str, Any], name: str) -> str:
        m = models.get(name)
        if not m:
            return "-"
        bpb = f"**{m['final_bpb']:.4f}**" if m.get("final_bpb") is not None else "-"
        return f"{bpb} ({m['val_bytes_per_token']:.2f})"

    lines = [
        f"- ablation run: {gpu}, commit `{commit}`{_swap_provenance(ameta, am)}, generated {ameta['created_utc']}"
        + ("" if sp else "; the 50 MB space-prefix run is not committed yet (its column is empty)"),
        "- " + (f"DIFFERENT {', '.join(differ)} from the whitespace run: the BPB columns are not comparable"
                if differ else "same recipe and val documents as the whitespace run"),
        "- training text: " + "; ".join(f"`{n}` {text(n)}" for n in am if parsed.get(n)),
        "- cells: final val BPB (val bytes per token)",
        "",
        "| Variant | " + " | ".join(c[0] for c in cols) + " |",
        "| :--- |" + " ---: |" * len(cols),
    ]
    for v in variants:
        lines.append(f"| `{v}` | " + " | ".join(cell(models, v + suf) for _, models, suf in cols) + " |")
    if "gpt2" in bm:
        lines.append(f"| `gpt2` (reference) | {cell(bm, 'gpt2')} |" + " |" * (len(cols) - 1))
    return "\n".join(lines)


_TIERS = {"intrinsic": "intrinsic tier (30% multilingual mix)", "swap": "GPT-2-swap tier (English-centric)"}


def _swap_sources(ms: Dict[str, Any]) -> str:
    """What the swap-tier tokenizers are, from each one's source spec: GPT-2's and TokenMonster's
    were not trained by this repo (nor on FineWeb); ours were trained on English FineWeb text."""
    by_kind: Dict[str, List[str]] = {}
    for n, m in ms.items():
        by_kind.setdefault(m["source"]["kind"], []).append(n)
    out = []
    for n in by_kind.pop("tiktoken", []):
        enc = ms[n]["source"].get("encoding")
        out.append(f"`{n}` (tiktoken's `{enc}` encoding" + (": GPT-2's own BPE, trained by OpenAI on WebText"
                                                             if enc == "gpt2" else "") + ")")
    out += [f"`{n}` (TokenMonster's prebuilt `{ms[n]['source'].get('vocab')}` vocabulary, "
            f"{ms[n]['vocab_size']:,} tokens)" for n in by_kind.pop("tokenmonster", [])]
    ours = by_kind.pop("tokenisers", [])
    if ours:
        sizes = " / ".join(f"{v:,}" for v in sorted({ms[n]["vocab_size"] for n in ours}))
        out.append(f"{len(ours)} of ours, trained on English FineWeb text at {sizes} tokens")
    out += [", ".join(f"`{n}`" for n in names) + f" ({kind})" for kind, names in by_kind.items()]
    return "; ".join(out)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _flores_findings(doc: Dict[str, Any], tiers: Dict[str, Dict[str, Any]]) -> List[str]:
    """Findings under the FLORES+ tables; every number comes from the JSON."""
    meta = doc["meta"]
    in_mix = list(meta["languages"]["in_mix"].values())
    latin = [lang for lang in in_mix if lang.endswith("_Latn")]
    other = [lang for lang in in_mix if lang not in latin]
    eng = meta["languages"]["english"]
    lines = ["", "Findings:", ""]
    for g, label in (("in_mix", "in-mix"), ("out_of_mix", "out-of-mix")):
        rng = {t: sorted(m["aggregates"][g]["mean_premium"] for m in ms.values()) for t, ms in tiers.items()}
        line = f"- Mean {label} premium: " + "; ".join(f"{_TIERS[t]} {v[0]:.2f}-{v[-1]:.2f} (median "
                                                       f"{statistics.median(v):.2f})" for t, v in rng.items())
        if len(rng) == 2:
            below = sum(x < rng["swap"][0] for x in rng["intrinsic"])
            line += (f". {below} of {len(rng['intrinsic'])} intrinsic-tier tokenizers are below the lowest "
                     "swap-tier tokenizer")
        lines.append(line + ".")
    def cost(ms: Dict[str, Any], members: List[str]) -> float:
        return statistics.median(statistics.fmean(m["languages"][lang]["tokens_per_sentence"] for lang in members)
                                 for m in ms.values())

    groups = {"in-mix": in_mix, "out-of-mix": meta["languages"]["out_of_mix"]}
    lines.append("- Absolute cost, mean tokens per sentence over the group's languages (English plays no part), "
                 "median over each tier: " + "; ".join(
                     f"{label} " + " vs ".join(f"{t} tier {cost(ms, members):.1f}" for t, ms in tiers.items())
                     for label, members in groups.items()) + ".")
    if latin and other:
        def mean_over(ms: Dict[str, Any], langs: List[str]) -> float:
            return statistics.fmean(m["languages"][lang]["premium"] for m in ms.values() for lang in langs)

        lines.append("- Latin vs non-Latin in-mix languages (premium averaged over the tier's tokenizers): " + "; ".join(
            f"{_TIERS[t]} {mean_over(ms, latin):.2f} on {', '.join(latin)} vs {mean_over(ms, other):.2f} on "
            f"{', '.join(other)}" for t, ms in tiers.items()) + ".")
    for t, ms in tiers.items():
        p, lang, n = max((m["languages"][lang]["premium"], lang, n) for n, m in ms.items() for lang in m["languages"])
        worst = "; ".join(f"worst {label} language " + ", ".join(
            f"{w} for {c}" for w, c in Counter(m["aggregates"][g]["worst_language"] for m in ms.values()).most_common())
            for g, label in (("in_mix", "in-mix"), ("out_of_mix", "out-of-mix")))
        fb = [m["aggregates"]["out_of_mix"]["byte_fallback_rate"] for m in ms.values()]
        best = min(ms, key=lambda k: (ms[k]["aggregates"]["in_mix"]["mean_premium"], k))
        en = statistics.median(m["languages"][eng]["tokens_per_sentence"] for m in ms.values())
        lines.append(f"- {_cap(_TIERS[t])}, {len(ms)} tokenizers: highest premium {p:.2f} (`{n}` on {lang}); {worst}; "
                     f"byte-fallback tokens are {100 * min(fb):.1f}-{100 * max(fb):.1f}% of out-of-mix tokens. Its "
                     f"lowest in-mix premium, `{best}` ({ms[best]['aggregates']['in_mix']['mean_premium']:.2f}), comes "
                     f"with {ms[best]['languages'][eng]['tokens_per_sentence']:.1f} tokens per English sentence against "
                     f"the tier's median {en:.1f}: the premium is relative to each tokenizer's own English cost.")
    raw = {lang: t["bytes"] / meta["text"][eng]["bytes"] for lang, t in meta["text"].items()}
    near = {n: m for ms in tiers.values() for n, m in ms.items() if m["languages"][eng]["bytes_per_token"] < 1.5}
    if near:
        gap = statistics.fmean(abs(m["languages"][lang]["premium"] - raw[lang]) for m in near.values()
                               for lang in m["languages"] if lang != eng)
        lines.append(f"- {len(near)} tokenizers encode English at under 1.5 bytes per token, close to raw bytes ("
                     + ", ".join(f"`{n}`" for n in near) + f"): their premium is on average {gap:.2f} from the UTF-8 "
                     "byte ratio (column \"UTF-8 bytes\"), so it reflects the script's byte length, not fair "
                     "compression.")
    return lines


def render_flores(doc: Dict[str, Any]) -> str:
    """FLORES+ tokenization premium: per-tokenizer aggregates per tier, premium by language for the
    best and worst tokenizer of each tier (and GPT-2), and findings."""
    meta, models = doc["meta"], doc["models"]
    fl, langs = meta["flores"], meta["languages"]
    eng = langs["english"]
    tiers = {t: {n: m for n, m in models.items() if m["tier"] == t} for t in _TIERS}
    tiers = {t: ms for t, ms in tiers.items() if ms}
    commit = _commit(meta)
    mapping = ", ".join(k if k == v else f"{k} -> {v}" for k, v in langs["in_mix"].items())
    head = [
        f"- FLORES+ `{fl['repo']}` at `{(fl['revision'] or 'a local copy')[:12]}` ({fl['license']}), {fl['split']} "
        f"split only: {fl['sentences']:,} parallel sentences per language, each encoded on its own with no "
        "separator; never used for training",
        f"- in-mix: the {len(langs['in_mix'])} FineWeb-2 languages of the data pipeline's training mix (FineWeb-2 "
        f"config -> FLORES+ file): {mapping}; out-of-mix (in no tokenizer's training mix): "
        + ", ".join(langs["out_of_mix"]),
    ]
    intr = meta.get("intrinsic")
    if intr:
        checks = [m["training"]["reproduces_committed"] for m in tiers.get("intrinsic", {}).values()]
        ds = intr["dataset"]
        head.append(
            f"- intrinsic tier: {len(intr['variants'])} variants x V = "
            + " / ".join(f"{v:,}" for v in intr["vocab_sizes"])
            + f", retrained as the committed intrinsic results ({'SYNTHETIC' if ds['synthetic'] else 'real'} "
            f"dataset `{ds['root']}`, {_mb(ds['train_bytes'])} train split); "
            + (f"{sum(c is True for c in checks)} of {len(checks)} reproduce the committed test-split bytes/token "
               "exactly" if any(c is not None for c in checks) else "no committed leaderboard to check against"))
    if meta.get("swap_jobs"):
        dirty = [f"`{j}`" for j, v in meta["swap_jobs"].items() if v.get("prepared_git_dirty")]
        head.append("- swap tier: the GPT-2-swap tokenizers from the prepared jobs "
                    + ", ".join(f"`{j}`" for j in meta["swap_jobs"]) + ": " + _swap_sources(tiers.get("swap", {}))
                    + (f"; prepared from a dirty tree (uncommitted changes: not reproducible from any commit): "
                       f"{', '.join(dirty)}" if dirty else ""))
    head += [
        f"- premium = tokens(language) / tokens(English) on the same sentences; σ ln = standard deviation of "
        f"ln(premium) over the group's languages; <= {meta['parity_threshold']} = share of them with premium "
        f"<= {meta['parity_threshold']}; byte fallback = share of tokens that are one byte >= 0x80, a piece of a "
        "multi-byte character",
        f"- commit `{commit}`; generated {meta['created_utc']}; per-language tokens, bytes/token, characters/token "
        "and one-byte shares of every tokenizer are in `flores_by_language.csv`",
        "",
        "How to read: premium 1.0 is parity with English; 2.0 means the same content costs twice the tokens (so half "
        "the context and twice the price). Bytes or characters per token are not comparable across scripts: UTF-8 "
        "spends 1 byte on a Latin letter, 2 on Cyrillic, Greek, Hebrew or Arabic, 3 on Devanagari, Han, Kana, "
        "Hangul or Thai, so a script's bytes per token can rise while its text still costs more tokens. The premium "
        "counts tokens for the same parallel sentences, which is comparable.",
    ]

    def cell(a: Dict[str, Any]) -> str:
        return (f"{a['mean_premium']:.2f} | {a['max_premium']:.2f} {a['worst_language']} | {a['std_log_premium']:.2f} "
                f"| {100 * a['share_premium_le_1_5']:.0f}%")

    lines = list(head)
    for t, ms in tiers.items():
        lines += ["", f"{_cap(_TIERS[t])}, sorted by mean in-mix premium:", "",
                  "| Tokenizer | \\|V\\| | EN tokens/sentence | In-mix mean | In-mix max | In-mix σ ln | In-mix <= 1.5 "
                  "| Out-of-mix mean | Out-of-mix max | Out-of-mix σ ln | Out-of-mix <= 1.5 | Out-of-mix byte fallback |",
                  "| :--- |" + " ---: |" * 11]
        rows = sorted(ms.items(), key=lambda kv: (kv[1]["aggregates"]["in_mix"]["mean_premium"], kv[0]))
        for n, m in rows:
            a = m["aggregates"]
            lines.append(f"| `{n}` | {m['vocab_size']:,} | {m['languages'][eng]['tokens_per_sentence']:.1f} | "
                         f"{cell(a['in_mix'])} | {cell(a['out_of_mix'])} | "
                         f"{100 * a['out_of_mix']['byte_fallback_rate']:.1f}% |")

    picks: List[str] = []
    for ms in tiers.values():
        order = sorted(ms, key=lambda n: (ms[n]["aggregates"]["in_mix"]["mean_premium"], n))
        picks += [order[0], order[-1]]
    if "gpt2" in models:
        picks.append("gpt2")
    picks = list(dict.fromkeys(picks))
    text = meta["text"]
    lines += ["", "Premium by language: the lowest and highest mean in-mix premium of each tier"
              + (" and GPT-2" if "gpt2" in models else "") + "; \"UTF-8 bytes\" is the byte ratio, the premium of "
              "a tokenizer with one token per byte. The English row gives tokens (bytes) per sentence.", "",
              "| Language | Group | UTF-8 bytes | " + " | ".join(f"`{n}`" for n in picks) + " |",
              "| :--- | :--- |" + " ---: |" * (1 + len(picks))]
    for lang in text:
        group = "reference" if lang == eng else "in-mix" if lang in langs["in_mix"].values() else "out-of-mix"
        if lang == eng:
            vals = [f"{models[n]['languages'][lang]['tokens_per_sentence']:.1f}" for n in picks]
            raw = f"({text[lang]['bytes'] / text[lang]['sentences']:.1f})"
        else:
            vals = [f"{models[n]['languages'][lang]['premium']:.2f}" for n in picks]
            raw = f"{text[lang]['bytes'] / text[eng]['bytes']:.2f}"
        lines.append(f"| {lang} | {group} | {raw} | " + " | ".join(vals) + " |")
    return "\n".join(lines + _flores_findings(doc, tiers))


__all__ = ["render_tokka", "render_intrinsic", "render_colab", "render_colab_compare", "colab_compare_stats",
           "render_throughput", "render_nanogpt_swap", "render_nanogpt_swap_space_prefix",
           "render_nanogpt_swap_ablation", "render_flores"]
