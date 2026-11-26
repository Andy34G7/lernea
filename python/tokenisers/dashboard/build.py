"""
Static site build for the Lernea benchmark page: collects every result set under benchmarks/results,
precomputes the playground tokenisations, and writes site/data.js. No server: site/ is plain files.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tokenisers.benchmarks.report import (colab_compare_stats, render_colab, render_colab_compare, render_flores,
                                          render_intrinsic, render_nanogpt_swap, render_nanogpt_swap_ablation,
                                          render_nanogpt_swap_space_prefix, render_throughput, render_tokka)

logger = logging.getLogger("tokenisers.dashboard")

REPO_ROOT = Path(os.environ["REPO_ROOT"]) if "REPO_ROOT" in os.environ else Path(__file__).resolve().parents[3]
CORPUS_TRAIN_BYTES = 100_000   # playground training corpus (train split of the data pipeline)
VOCAB_DIRS = {16384: "", 32768: "v32768/", 65536: "v65536/"}  # suite vocabulary size -> results subdirectory

# Metadata for all algorithm architectures and theoretical objectives. Formulas state what the
# scorers in crates/tokenisers_core/src/scoring and curricula/morpheme.rs compute, with the
# defaults the playground trains with (README "Implemented Tokenizer Algorithms" table); the
# formulas on static/index.html are copies of these strings.
ALGORITHM_METADATA: Dict[str, Dict[str, Any]] = {
    "bpe_frequency": {
        "display_name": "BPE (Frequency)",
        "family": "Frequency BPE",
        "description": "Standard byte-level byte-pair encoding merging most frequent adjacent byte pairs.",
        "formula": r"\mathrm{score}(u, v) = \mathrm{count}(uv)",
        "objective": "Pair Frequency",
        "generation": "Baseline",
        "pros": ["Fastest training", "High in-domain compression", "Industry standard baseline"],
        "cons": ["Blind to word boundaries", "Over-merges frequent affixes into root words", "Poor morphological alignment"],
    },
    "wordpiece_pmi": {
        "display_name": "WordPiece (PMI)",
        "family": "Information-Theoretic",
        "description": "WordPiece likelihood score: pair count over the product of Laplace-smoothed marginal counts (alpha = 1), for pairs seen at least twice.",
        "formula": r"\mathrm{score}(u, v) = \frac{\mathrm{count}(uv)}{(\mathrm{count}(u) + \alpha)(\mathrm{count}(v) + \alpha)}, \quad \alpha = 1,\ \mathrm{count}(uv) \ge 2",
        "objective": "Smoothed PMI (WordPiece likelihood)",
        "generation": "Baseline",
        "pros": ["Favors statistically coherent morphemes", "Smoothing damps pairs whose parts are rare"],
        "cons": ["Ranks rare exclusive pairs above frequent ones (no frequency weighting)", "Higher vocabulary fragmentation"],
    },
    "wp_bpe_base": {
        "display_name": "WP-BPE (Base Uncalibrated PMI)",
        "family": "Information-Theoretic",
        "description": "BPE merges driven by raw uncalibrated PMI. Highlights theoretical vulnerability of naive PMI.",
        "formula": r"\mathrm{score}(u, v) = \frac{\mathrm{freq}(uv)}{\mathrm{freq}(u) \cdot \mathrm{freq}(v)}",
        "objective": "Uncalibrated PMI",
        "generation": "WP-BPE Gen 1",
        "pros": ["Exposes true statistical dependencies across tokens"],
        "cons": ["A pair seen once whose parts occur nowhere else gets the maximum score (1)", "Severe corpus overfitting without thresholding"],
    },
    "wp_bpe_ft": {
        "display_name": "WP-BPE (Frequency-Thresholded)",
        "family": "Information-Theoretic",
        "description": "The uncalibrated PMI ratio of WP-BPE Base, restricted to pairs seen at least theta = 2 times.",
        "formula": r"\mathrm{score}(u, v) = \begin{cases} \frac{\mathrm{freq}(uv)}{\mathrm{freq}(u) \cdot \mathrm{freq}(v)} & \text{if } \mathrm{freq}(uv) \ge \theta \\ -\infty & \text{otherwise} \end{cases}, \quad \theta = 2",
        "objective": "Frequency-Thresholded PMI",
        "generation": "WP-BPE Gen 1",
        "pros": ["Filters out singleton noise", "Stable vocabulary distribution"],
        "cons": ["Requires tuning threshold theta", "Slightly lower compression ratio on rare words"],
    },
    "wp_bpe_pig": {
        "display_name": "WP-BPE (Pointwise Information Gain)",
        "family": "Information-Theoretic",
        "description": "Directly weights PMI by candidate pair frequency, measuring total bits of information saved across the corpus.",
        "formula": r"\mathrm{score}(u, v) = \mathrm{freq}(uv) \cdot \log_2 \frac{\mathrm{freq}(uv) \cdot N}{\mathrm{freq}(u) \cdot \mathrm{freq}(v)}",
        "objective": "Corpus Information Gain",
        "generation": "WP-BPE Gen 1",
        "pros": ["Entropy-reduction grounded in information theory", "Balances frequency against statistical cohesion", "Superior downstream LM perplexity"],
        "cons": ["Slightly slower merge selection queue", "Sensitive to corpus size normalization"],
    },
    "wp_bpe_scp": {
        "display_name": "WP-BPE (Symmetric Conditional Probability)",
        "family": "Information-Theoretic (Improved)",
        "description": "Geometric-mean calibrated PMI with a Kneser-Ney style discount d = 0.75 on the pair count, for pairs seen at least twice. With d = 0 it is the square root of the symmetric conditional probability P(u|v) P(v|u).",
        "formula": r"\mathrm{score}(u, v) = \frac{(\mathrm{freq}(uv) - d)_+}{\sqrt{\mathrm{freq}(u)\,\mathrm{freq}(v)}}, \quad d = 0.75",
        "objective": "Symmetric Conditional Probability",
        "generation": "WP-BPE Gen 2 (Improved)",
        "pros": ["Self-bounding cohesion index", "Equalizes prefix-root and root-suffix merge biases", "Robust across agglutinative morphologies"],
        "cons": ["Discount d is a tuning parameter; pairs with freq(uv) <= d score 0"],
    },
    "wp_bpe_npmi": {
        "display_name": "WP-BPE (Normalized PMI)",
        "family": "Information-Theoretic (Improved)",
        "description": "Bouma normalized PMI (in [-1, +1]) weighted by sub-linear frequency support freq(uv)^alpha (alpha = 0.5), for pairs seen at least twice. Negatively associated pairs keep a negative score: eligible, ranked last.",
        "formula": r"\mathrm{score}(u, v) = \mathrm{freq}(uv)^{\alpha} \cdot \frac{\ln \frac{P(uv)}{P(u)P(v)}}{-\ln P(uv)}, \quad P(x) = \frac{\mathrm{freq}(x)}{N},\ \alpha = 0.5",
        "objective": "Frequency-Weighted Normalized PMI",
        "generation": "WP-BPE Gen 2 (Improved)",
        "pros": ["NPMI factor is bounded in [-1, +1] regardless of frequency scale", "NPMI = +1 exactly when u and v only ever occur together"],
        "cons": ["Log denominator requires eps guard against zero probabilities", "Support exponent alpha trades association against frequency"],
    },
    "wp_bpe_ppig": {
        "display_name": "WP-BPE (Penalized PIG)",
        "family": "Information-Theoretic (Improved)",
        "description": "Pointwise information gain counting only positive association, divided by a polynomial merged-length penalty (eta = 0.35), for pairs seen at least twice.",
        "formula": r"\mathrm{score}(u, v) = \frac{\mathrm{freq}(uv) \cdot \max(0, \mathrm{PMI}(u, v))}{(|u| + |v|)^{\eta}}, \quad \mathrm{PMI}(u, v) = \log_2 \frac{\mathrm{freq}(uv) \cdot N}{\mathrm{freq}(u) \cdot \mathrm{freq}(v)},\ \eta = 0.35",
        "objective": "Penalized Information Gain",
        "generation": "WP-BPE Gen 2 (Improved)",
        "pros": ["Prevents vocabulary bloating with overly long collocations", "Negatively associated pairs score 0 instead of a negative gain"],
        "cons": ["Requires length-penalty eta selection"],
    },
    "mdl_bits": {
        "display_name": "MDL (Delta-Bits)",
        "family": "Information-Theoretic",
        "description": "Minimum Description Length: merges maximize the corpus's unigram code length saved minus the new entry's cost (24 bits plus 8 bits per byte).",
        "formula": r"\Delta L = \big[L(T) - L(T')\big] - (c_{\mathrm{entry}} + 8\,|uv|), \quad L(T) = N \log_2 N - \sum_t c_t \log_2 c_t,\ c_{\mathrm{entry}} = 24",
        "objective": "Minimum Description Length",
        "generation": "Baseline",
        "pros": ["Rigorous MDL foundation", "Resistant to vocabulary bloat", "Guaranteed theoretical compressibility"],
        "cons": ["Higher computational cost during candidate selection", "Sensitive to table overhead estimation"],
    },
    "length_penalized": {
        "display_name": "Length-Penalized BPE",
        "family": "Frequency BPE",
        "description": "Frequency scoring divided by token length power beta to discourage greedy concatenation of long subwords.",
        "formula": r"\mathrm{score}(u, v) = \frac{\mathrm{freq}(uv)}{(|u| + |v|)^\beta}",
        "objective": "Length-Penalized Frequency",
        "generation": "Baseline",
        "pros": ["Prevents extreme long-tail token dominance", "Balances subword fertility across long words", "Simple and predictable"],
        "cons": ["Beta parameter is empirical", "Does not consider morphological boundaries"],
    },
    "super_bpe": {
        "display_name": "SuperBPE (Two-Stage)",
        "family": "Structural / Multi-Stage",
        "description": "Constrains merges strictly within word boundaries up to v_switch, then unlocks cross-word collocation merges.",
        "formula": r"\begin{cases} \text{Word-internal BPE} & |V| \le V_{\mathrm{switch}} \\ \text{Cross-word Collocations} & V_{\mathrm{switch}} < |V| \le V_{\mathrm{final}} \end{cases}",
        "objective": "Two-Stage Structural BPE",
        "generation": "Baseline",
        "pros": ["Guarantees morphological integrity for core vocabulary", "Captures frequent phrases (e.g. 'in the', 'et al.') efficiently", "Outstanding code compression"],
        "cons": ["Requires dual-pass vocabulary boundary tracking", "Slightly larger initial graph memory footprint"],
    },
    "morpheme_bpe": {
        "display_name": "Morpheme-Guided BPE",
        "family": "Morphology / Guided",
        "description": "Frequency BPE whose candidate scores are scaled by exp(lambda M): M is the share of the pair's current occurrences in lexicon words whose merged token stays inside one morpheme minus the share that would span a boundary. Pairs with count(uv) exp(lambda M) < 1 are never merged. The playground uses lambda = 2.5 and SIGMORPHON 2022 English train words as the lexicon (offline: a 12-word test fixture).",
        "formula": r"\mathrm{score}(u, v) = \mathrm{count}(uv) \cdot e^{\lambda M(u, v)}, \quad M = \frac{n_{\mathrm{intra}} - n_{\mathrm{cross}}}{n_{\mathrm{intra}} + n_{\mathrm{cross}}} \in [-1, 1],\ \lambda = 2.5",
        "objective": "Morpheme Boundary Regularization",
        "generation": "Baseline",
        "pros": ["Penalizes merges across known morpheme boundaries", "Linguistically aligned token pieces"],
        "cons": ["Requires supervised or semi-supervised morphological dictionary", "Lambda weight hyperparameter tuning"],
    },
    "unigram_lm": {
        "display_name": "Unigram Language Model",
        "family": "EM / Probabilistic",
        "description": "Top-down vocabulary pruning using Expectation-Maximization and Viterbi / stochastic lattice decoding.",
        "formula": r"\mathcal{L} = \sum_{i=1}^N \log \left( \sum_{x \in \mathcal{S}(X_i)} P(x) \right)",
        "objective": "Maximum Likelihood Unigram",
        "generation": "Baseline",
        "pros": ["Probabilistic foundation with lattice sampling (subword regularization)", "High entropy retention (Rényi E_2.0)", "Smooth token distributions"],
        "cons": ["Slower vocabulary training phase due to iterative EM loops", "Higher Viterbi decoding overhead compared to deterministic BPE"],
    },
    "bpe_dropout_p0.1": {
        "display_name": "BPE (Merge Dropout p=0.1)",
        "family": "Regularized BPE",
        "description": "Stochastic BPE segmenter that randomly skips merges with probability p=0.1 during tokenization.",
        "formula": r"P(\text{apply merge } (u,v)) = 1 - p",
        "objective": "Subword Regularization Dropout",
        "generation": "Baseline",
        "pros": ["Data augmentation for downstream LM robustness", "Prevents memorization of rigid subword boundaries"],
        "cons": ["Non-deterministic segmentation", "Requires seed control for reproducibility"],
    },
}


PLAYGROUND_VOCABS = (300, 500, 1000)
PLAYGROUND_SAMPLES = [
    ("prose", "English prose", "The quick brown fox jumps over the lazy dog, and nobody was surprised by the outcome."),
    ("morphology", "English morphology", "The internationalization of unhappily disproportionate misunderstandings requires robust subword tokenization."),
    ("code", "Rust code", """pub fn compute_pmi(freq_uv: u64, freq_u: u64, freq_v: u64, total: u64) -> f64 {
    let p_uv = (freq_uv as f64) / (total as f64);
    let p_u = (freq_u as f64) / (total as f64);
    let p_v = (freq_v as f64) / (total as f64);
    (p_uv / (p_u * p_v)).log2()
}"""),
    ("multilingual", "Multilingual", "Tokenizers in Deutsch: Unabhängigkeitserklärung; Français: internationalisation; 汉语: 自然语言处理; 日本語: 機械学習モデル; Русский: сверхпроводимость; العربية: معالجة اللغات الطبيعية."),
    ("agglutinative", "Agglutinative", "megszentségteleníthetetlenségeskedéseitekért epäjärjestelmällistyttämättömyydelläänsäkäänkö unconstitutionality hypercholesterolemia"),
]


def _playground_corpus(offline: bool) -> Tuple[List[str], Dict[str, List[str]], Dict[str, Any]]:
    from tokenisers.data.pipeline import DataPipeline, SigmorphonLoader

    ds = DataPipeline(force_offline=offline).prepare(train_bytes=CORPUS_TRAIN_BYTES)
    corpus = ds.texts("train")
    loader = SigmorphonLoader(offline=offline)
    if offline:
        gold = loader.load_offline_fixture()
        lexicon_source = "offline test fixture (12 hand-written words, not SIGMORPHON)"
    else:
        words = {w for text in corpus for w in text.split()}
        gold = [e for e in loader.load("eng", "train") if e["word"] in words]
        lexicon_source = "SIGMORPHON 2022 eng train (surface entries found in the corpus)"
    lexicon = {e["word"]: e["segments"] for e in gold}
    info = {"synthetic": ds.synthetic, "train_docs": len(corpus),
            "train_bytes": ds.manifest["splits"]["train"]["text_bytes"],
            "lexicon_source": lexicon_source, "lexicon_words": len(lexicon)}
    return corpus, lexicon, info


def _train_playground(corpus: List[str], lexicon: Dict[str, List[str]], vocab_size: int) -> Dict[str, Any]:
    from tokenisers.models import (BpeTokenizer, LengthPenalizedTokenizer, MdlTokenizer, MorphemeBpeTokenizer,
                                   SuperBpeTokenizer, UnigramTokenizer, WordPieceTokenizer, WpBpeBaseTokenizer,
                                   WpBpeFtTokenizer, WpBpeNpmiTokenizer, WpBpePigTokenizer, WpBpePpigTokenizer,
                                   WpBpeScpTokenizer)

    v_switch = max(256, int(vocab_size * 0.75))
    return {
        "bpe_frequency": BpeTokenizer.train(corpus=corpus, vocab_size=vocab_size, scorer="frequency"),
        "wordpiece_pmi": WordPieceTokenizer.train(corpus=corpus, vocab_size=vocab_size),
        "wp_bpe_base": WpBpeBaseTokenizer.train(corpus=corpus, vocab_size=vocab_size),
        "wp_bpe_ft": WpBpeFtTokenizer.train(corpus=corpus, vocab_size=vocab_size, min_frequency=2),
        "wp_bpe_pig": WpBpePigTokenizer.train(corpus=corpus, vocab_size=vocab_size, min_frequency=1),
        "wp_bpe_scp": WpBpeScpTokenizer.train(corpus=corpus, vocab_size=vocab_size),
        "wp_bpe_npmi": WpBpeNpmiTokenizer.train(corpus=corpus, vocab_size=vocab_size),
        "wp_bpe_ppig": WpBpePpigTokenizer.train(corpus=corpus, vocab_size=vocab_size),
        "mdl_bits": MdlTokenizer.train(corpus=corpus, vocab_size=vocab_size),
        "length_penalized": LengthPenalizedTokenizer.train(corpus=corpus, vocab_size=vocab_size, beta=0.5),
        "super_bpe": SuperBpeTokenizer.train(corpus=corpus, v_switch=v_switch, v_final=vocab_size),
        "morpheme_bpe": MorphemeBpeTokenizer.train(corpus=corpus, vocab_size=vocab_size, morpheme_lexicon=lexicon, lambda_weight=2.5),
        "unigram_lm": UnigramTokenizer.train(corpus=corpus, vocab_size=vocab_size),
    }


def build_playground(offline: bool = False, vocabs: Tuple[int, ...] = PLAYGROUND_VOCABS) -> Dict[str, Any]:
    """Every sample text encoded by every tokenizer at each vocabulary size. A token is [id, byte length]:
    the page cuts the sample's UTF-8 bytes by the lengths, so the lengths of one row sum to the text's bytes."""
    corpus, lexicon, info = _playground_corpus(offline)
    runs: Dict[str, Any] = {}
    for v in vocabs:
        toks = _train_playground(corpus, lexicon, v)
        runs[str(v)] = {sid: {name: [[tid, len(tok.id_to_bytes(tid) or b"")] for tid in tok.encode(text)]
                              for name, tok in toks.items()}
                        for sid, _, text in PLAYGROUND_SAMPLES}
    return {"corpus": info, "vocabs": list(vocabs),
            "samples": [{"id": sid, "label": label, "text": text} for sid, label, text in PLAYGROUND_SAMPLES],
            "runs": runs}



# Every committed result set: key -> (path under benchmarks/results, the README renderer whose header lines
# are the set's provenance and whose bullets after the table are its notes; None: no renderer).
RESULT_SETS: Dict[str, Tuple[str, Optional[Callable[[Dict[str, Any]], str]]]] = {
    **{f"tokka_{v}": (f"{d}tokka/tokka_matrix.json", render_tokka) for v, d in VOCAB_DIRS.items()},
    **{f"intrinsic_{v}": (f"{d}intrinsic/intrinsic_leaderboard.json", render_intrinsic) for v, d in VOCAB_DIRS.items()},
    "throughput_local": ("throughput/local/throughput.json", render_throughput),
    "throughput_colab_cpu": ("throughput/colab_cpu/throughput.json", render_throughput),
    "proxy_small": ("colab/extrinsic_bpb_results.json", render_colab),
    "proxy_big": ("colab_big/extrinsic_bpb_results.json", render_colab),
    "swap": ("nanogpt_swap/nanogpt_swap.json", render_nanogpt_swap),
    "swap_lr1p8e-3": ("nanogpt_swap_lr1p8e-3/gpt2_run.json", None),
    # GPT-2-style space-prefix rerun of our tokenizers.
    "swap_space_prefix": ("nanogpt_swap_sp/nanogpt_swap.json", render_nanogpt_swap),
    # _g2 / _sp300 retrains: shown as soon as this file is committed (provenance from
    # report.render_nanogpt_swap_ablation, which needs the other swap runs too: see load_benchmark_data).
    "swap_ablation": ("nanogpt_swap_ablation/nanogpt_swap.json", None),
    "flores": ("flores/flores.json", render_flores),
}

# Ablation arms: the column name and the tokenizer-name suffix of a variant's run in it.
ABLATION_ARMS = {"whitespace": "", "space_prefix": "_sp", "gpt2_split": "_g2", "space_prefix_300mb": "_sp300"}


def _report(render: Callable[..., str], *docs: Dict[str, Any]) -> Dict[str, List[str]]:
    """The README rendering of a result split into its provenance (bullets before the first table) and its
    notes (bullets after it); empty when the file lacks a field the renderer needs."""
    try:
        md = render(*docs).splitlines()
    except Exception as e:  # an older or partial file: its numbers still show, its summary does not
        logger.warning(f"{render.__name__} failed: {e!r}")
        return {"provenance": [], "notes": []}
    first = next((i for i, line in enumerate(md) if line.startswith("|")), len(md))
    return {"provenance": [line[2:] for line in md[:first] if line.startswith("- ")],
            "notes": [line[2:] for line in md[first:] if line.startswith("- ")]}


def _load_result(rel: str, render: Optional[Callable[..., str]]) -> Dict[str, Any]:
    path = REPO_ROOT / "benchmarks" / "results" / rel
    entry: Dict[str, Any] = {"path": f"benchmarks/results/{rel}", "available": False, "data": None,
                             "provenance": [], "notes": []}
    if not path.is_file():
        return entry
    try:
        entry["data"] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning(f"Failed reading {path}: {e}")
        entry["error"] = f"unreadable: {e}"
        return entry
    entry["available"] = True
    if render:
        entry.update(_report(render, entry["data"]))
    return entry


def _train_bpb(csv_path: Path, models: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Per-step train BPB of each tokenizer swap run from nanogpt_swap_steps.csv (train loss in nats per
    token / ln 2 / that tokenizer's train bytes per token)."""
    if not csv_path.is_file():
        return None
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    names = [n for n in (rows[0] if rows else {}) if n not in ("step", "lr") and models.get(n, {}).get("train_bytes_per_token")]
    return {"steps": [int(r["step"]) for r in rows],
            "bpb": {n: [round(float(r[n]) / math.log(2) / models[n]["train_bytes_per_token"], 4) for r in rows]
                    for n in names}}


# The README's checks that a swap rerun is comparable with the whitespace run (report.render_nanogpt_swap_space_prefix
# and render_nanogpt_swap_ablation): what of a run each one compares.
COMPARABLE: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "recipe": lambda d: d["meta"]["recipe"]["args"],
    "val documents": lambda d: {m["val_docs_sha256"] for m in d["models"].values()},
    "tokenizer training text": lambda d: d["meta"]["tokenizer_text"]["docs_sha256"],
}


def not_comparable(base: Dict[str, Any], run: Dict[str, Any], checks=tuple(COMPARABLE)) -> List[str]:
    """The ``checks`` (COMPARABLE keys) on which ``run`` differs from the whitespace run ``base``. Fails closed: a
    check that a file lacks the fields for counts too, so an older or partial file is not shown as comparable."""
    out = []
    for k in checks:
        try:
            if COMPARABLE[k](base) != COMPARABLE[k](run):
                out.append(f"different {k}")
        except (KeyError, TypeError, AttributeError):
            out.append(f"{k} unknown (missing from a file)")
    return out


def sp_gpt2_equivalent(base: Dict[str, Any], sp: Dict[str, Any], csv_path: Path) -> Dict[str, Optional[int]]:
    """GPT-2-equivalent steps of a space-prefix rerun that has no GPT-2 run of its own, read off GPT-2's train
    curve in the original swap run (nanogpt_swap's gpt2_equivalent on GPT-2's column of that run's steps CSV).
    Only when both runs have the same recipe and val documents, so that GPT-2's curve and the rerun's final val
    BPB measure the same training on the same text; else {}."""
    bm, sm = base.get("models", {}), sp.get("models", {})
    eq, g = base.get("gpt2_equivalent"), bm.get("gpt2")
    same = not not_comparable(base, sp, ("recipe", "val documents"))
    if "gpt2" in sm or not (eq and g and same and csv_path.is_file()):
        return {}
    from tokenisers.benchmarks.nanogpt_swap.__main__ import gpt2_equivalent

    with csv_path.open(encoding="utf-8", newline="") as f:
        steps = [{"step": int(r["step"]), "train_loss": float(r["gpt2"])} for r in csv.DictReader(f)]
    run = {"trainer": {"steps": steps, "evals": []}, "train": {"bytes_per_token": g["train_bytes_per_token"]}}
    return gpt2_equivalent(run, sm, eq["window"])["steps"]


def space_prefix_pairs(old: Dict[str, Any], sp: Dict[str, Any],
                       sp_steps: Optional[Dict[str, Optional[int]]] = None) -> Dict[str, Any]:
    """Each of our tokenizers in the space-prefix swap run next to its run in the original swap (matched by
    name, a trailing ``_sp`` / ``_space_prefix`` dropped), plus GPT-2 as the reference (from the space-prefix
    run when it has one, else from the original run). The space-prefix run's GPT-2-equivalent steps are its
    own, else ``sp_steps`` (sp_gpt2_equivalent)."""
    om, sm = old.get("models", {}), sp.get("models", {})
    eq = {"old": (old.get("gpt2_equivalent") or {}).get("steps", {}),
          "sp": (sp.get("gpt2_equivalent") or {}).get("steps") or sp_steps or {}}

    def pick(name: str, m: Dict[str, Any], run: str) -> Dict[str, Any]:
        return {"name": name, "final_bpb": m.get("final_bpb"), "val_bytes_per_token": m.get("val_bytes_per_token"),
                "train_bytes_per_token": m.get("train_bytes_per_token"), "gpt2_equivalent_steps": eq[run].get(name)}

    rows = []
    for name, m in sm.items():
        base = re.sub(r"_(sp|space_prefix)$", "", name)
        if m.get("spec", {}).get("kind", "tokenisers") != "tokenisers" or base not in om or "final_bpb" not in m:
            continue
        o, n = pick(base, om[base], "old"), pick(name, m, "sp")
        rows.append({"variant": base, "old": o, "sp": n,
                     "delta_bpb": n["final_bpb"] - o["final_bpb"] if o["final_bpb"] is not None else None})
    run = "sp" if "gpt2" in sm else "old"
    g = (sm if run == "sp" else om).get("gpt2")
    return {"rows": rows, "gpt2": {**pick("gpt2", g, run), "run": run} if g else None}


def ablation_arms(base: Dict[str, Any], sp: Optional[Dict[str, Any]], abl: Dict[str, Any]) -> Dict[str, Any]:
    """Per variant of the ablation run (it has ``<variant>_g2`` or ``<variant>_sp300``), its final val BPB and
    val bytes per token in each of ABLATION_ARMS: whitespace from ``base`` (the original swap), space_prefix
    from ``sp`` (the 50 MB rerun, None until committed) else the ablation run, the rest from ``abl``; None where
    that run is missing. GPT-2 from ``base`` is the reference."""
    bm, am = base.get("models", {}), abl.get("models", {})
    pool = {**am, **(sp or {}).get("models", {}), **bm}

    def pick(m: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return {"final_bpb": m.get("final_bpb"), "val_bytes_per_token": m.get("val_bytes_per_token")} if m else None

    variants = dict.fromkeys(m[1] for n in am if (m := re.fullmatch(r"(.+)_(?:g2|sp300)", n)))
    return {"arms": list(ABLATION_ARMS), "gpt2": pick(bm.get("gpt2")),
            "rows": [{"variant": v, **{arm: pick(pool.get(v + suf)) for arm, suf in ABLATION_ARMS.items()}}
                     for v in variants]}


def _derived(entry: Dict[str, Any], what: str, compute: Callable[[], Any]) -> Any:
    """A view computed from result files (train BPB, small vs big, the swap pairings). An odd or partial file
    (e.g. a single seed: no sd) must not turn /api/benchmarks into a 500 that blanks the whole dashboard: on a
    failure the files' own numbers still show, ``entry["error"]`` says why, and the view is None."""
    try:
        return compute()
    except Exception as e:
        logger.warning(f"{what} failed: {e!r}")
        entry["error"] = f"{what} failed: {type(e).__name__}: {e}"
        return None


def load_benchmark_data() -> Dict[str, Any]:
    """Every result set under benchmarks/results (with an availability flag), plus the unified V = 16,384
    per-tokenizer view (tokka, intrinsic, small and big proxy LM) the overview charts use."""
    results = {key: _load_result(rel, render) for key, (rel, render) in RESULT_SETS.items()}

    def steps_csv(key: str) -> Path:
        return REPO_ROOT / "benchmarks" / "results" / Path(RESULT_SETS[key][0]).parent / "nanogpt_swap_steps.csv"

    for key in ("swap", "swap_space_prefix"):
        e = results[key]
        if e["available"]:
            e["train_bpb"] = _derived(e, "train BPB from the steps CSV",
                                      lambda: _train_bpb(steps_csv(key), e["data"].get("models", {})))
    small, big = results["proxy_small"], results["proxy_big"]
    cmp = results["proxy_compare"] = {"path": f"{small['path']} + {big['path']}", "available": False, "data": None,
                                      "provenance": [], "notes": []}
    if small["available"] and big["available"]:
        stats = _derived(cmp, "small vs big proxy-LM comparison",
                         lambda: colab_compare_stats(small["data"], big["data"]))
        if stats:
            cmp.update(available=True, data={"ref": "bpe_frequency", "names": stats[0], "tiers": stats[1]},
                       **_report(render_colab_compare, small["data"], big["data"]))
    swap, sp = results["swap"], results["swap_space_prefix"]
    if sp["available"] and swap["available"]:
        # The README's check that the rerun is comparable (recipe, val documents, tokenizer training text) first
        check = _report(render_nanogpt_swap_space_prefix, swap["data"], sp["data"])["provenance"]
        sp["provenance"] = check + sp["provenance"]
        sp["not_comparable"] = not_comparable(swap["data"], sp["data"])
        sp["pairs"] = _derived(sp, "pairing with the original swap run", lambda: space_prefix_pairs(
            swap["data"], sp["data"], sp_gpt2_equivalent(swap["data"], sp["data"], steps_csv("swap"))))
    abl = results["swap_ablation"]
    if abl["available"] and swap["available"]:
        docs = (swap["data"], sp["data"] if sp["available"] else None, abl["data"])
        # The 300 MB arm changes the tokenizer training text on purpose: recipe and val documents only
        abl.update(ablation=_derived(abl, "ablation arms", lambda: ablation_arms(*docs)),
                   not_comparable=not_comparable(swap["data"], abl["data"], ("recipe", "val documents")),
                   **_report(render_nanogpt_swap_ablation, *docs))

    def doc(key: str) -> Dict[str, Any]:
        d = results[key]["data"]
        return d if isinstance(d, dict) else {}

    # Heatmap columns come from the data, not a fixed list: per-language cells are dicts; the
    # aggregates are not languages and scalars (vocab_size, unique_tokens) are not cells. The union
    # over every vocabulary size's matrix (the same held-out text, so the same languages).
    tokka_languages = list(dict.fromkeys(
        lang
        for key in RESULT_SETS if key.startswith("tokka_")
        for row in doc(key).get("models", doc(key)).values() if isinstance(row, dict)
        for lang, cell in row.items() if isinstance(cell, dict) and lang not in ("macro_avg", "micro_avg")
    ))
    return {
        "tokka_languages": tokka_languages,
        "results": results,
        "metadata": ALGORITHM_METADATA,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


# Result-file blocks the page never reads (file hashes, per-document lists): left out of site/data.js.
_META_DROP = {"dataset", "morphology", "lm_text", "ids_sha256", "source", "timings", "job_prepared", "intrinsic",
              "swap_jobs", "pass_load", "command", "host", "native_extension"}
_PROXY_KEEP = {"vocab_size", "train_bytes_per_token", "tokens_consumed", "macro_val_bpb", "macro_val_bpb_sd",
               "macro_val_bpb_by_seed", "val_domains", "curve"}
CURVE_POINTS = 120


def _rolling(a: List[float], w: int) -> List[float]:
    h = w // 2
    return [sum(a[max(0, i - h):i + h + 1]) / len(a[max(0, i - h):i + h + 1]) for i in range(len(a))]


def _thin(n: int, k: int = CURVE_POINTS) -> List[int]:
    return list(range(n)) if n <= k else sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return x if not math.isfinite(x) else float(f"{x:.5g}")
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def compact(data: Dict[str, Any]) -> Dict[str, Any]:
    """load_benchmark_data() cut down to what the page reads: curves smoothed and thinned to CURVE_POINTS,
    floats to 5 significant digits, bulky provenance blocks dropped. Numbers shown in tables keep >= 4 decimals."""
    R = data["results"]
    for e in R.values():
        d = e.get("data")
        meta = d.get("meta") if isinstance(d, dict) else None
        if isinstance(meta, dict):
            if isinstance(meta.get("prepared"), dict):
                meta["prepared"] = {"train_bytes": meta["prepared"].get("train_bytes")}
            for k in _META_DROP & set(meta):
                del meta[k]
    for key in ("proxy_small", "proxy_big"):
        for name, m in ((R[key]["data"] or {}).get("models") or {}).items():
            curve = m.get("curve") or []
            # A window of about a tenth of the run: 14 raw step curves on one chart are not readable
            smooth = _rolling([p["train_bpb"] for p in curve], max(9, len(curve) // 10 | 1))
            m["curve"] = [[curve[i]["bytes_consumed"] / 1e6, smooth[i]] for i in _thin(len(curve))]
            R[key]["data"]["models"][name] = {k: v for k, v in m.items() if k in _PROXY_KEEP}
    for key in ("swap", "swap_space_prefix"):
        e = R[key]
        tb = e.get("train_bpb")
        if tb:
            w = ((e["data"].get("gpt2_equivalent") or {}).get("window")) or 21
            idx = _thin(len(tb["steps"]), 200)
            e["train_bpb"] = {"window": w, "steps": [tb["steps"][i] for i in idx],
                              "bpb": {n: [s[i] for i in idx] for n, s in ((n, _rolling(v, w)) for n, v in tb["bpb"].items())}}
    lr = R["swap_lr1p8e-3"]
    if lr["available"]:
        tr = lr["data"].get("trainer") or {}
        steps = tr.get("steps") or []
        warm = int(((((R["swap"]["data"] or {}).get("meta") or {}).get("recipe") or {}).get("args") or {}).get("warmup_iters") or 0)
        norms = [s["norm"] for s in steps if s["step"] >= warm and isinstance(s.get("norm"), (int, float))]
        lr["data"] = {"train": {"tokens_per_step": (lr["data"].get("train") or {}).get("tokens_per_step")},
                      "val": {k: (lr["data"].get("val") or {}).get(k) for k in ("docs", "bytes")},
                      "trainer": {"evals": tr.get("evals"), "gpu": tr.get("gpu"), "torch": tr.get("torch"),
                                  "n_steps": len(steps), "peak_lr": max((s["lr"] for s in steps), default=None),
                                  "warmup": warm, "max_norm_after_warmup": max(norms, default=None)}}
    fl = R["flores"]
    if fl["available"]:
        for m in fl["data"]["models"].values():
            m.pop("source", None)
            m.pop("training", None)
            m["languages"] = {l: {k: c.get(k) for k in ("premium", "tokens_per_sentence")}
                              for l, c in (m.get("languages") or {}).items()}
    meta = {k: {a: b for a, b in v.items() if a not in ("pros", "cons", "generation", "objective")}
            for k, v in data["metadata"].items()}
    return _round({**data, "metadata": meta})


def build_site(out: Path, playground: bool = True, offline: bool = False) -> Path:
    """Write <out>/data.js (window.LERNEA_DATA = {...}). Without ``playground`` the previous file's playground
    block is kept, so the benchmark numbers can be refreshed on a machine without the playground corpus."""
    data = compact(load_benchmark_data())
    target = out / "data.js"
    if playground:
        data["playground"] = build_playground(offline)
    elif target.is_file():
        data["playground"] = json.loads(target.read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";")).get("playground")
    out.mkdir(parents=True, exist_ok=True)
    target.write_text("window.LERNEA_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
                      encoding="utf-8")
    return target
