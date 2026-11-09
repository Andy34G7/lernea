"""
HTTP & JSON API Server for Tokenisers Lab Benchmark Intelligence Dashboard.
Inspired by Artificial Analysis (artificialanalysis.ai).
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import math
import mimetypes
import os
import re
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from tokenisers.benchmarks.report import (colab_compare_stats, render_colab, render_colab_compare, render_flores,
                                          render_intrinsic, render_nanogpt_swap, render_nanogpt_swap_ablation,
                                          render_nanogpt_swap_space_prefix, render_throughput, render_tokka)

logger = logging.getLogger("tokenisers.dashboard")

# Determine repository root
REPO_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "static"

LOOPBACK_HOSTS = ("127.0.0.1", "localhost")
PUBLIC_HOSTS = ("0.0.0.0",)
MIN_VOCAB_SIZE = 260
MAX_VOCAB_SIZE = 1000          # matches the playground's number input
CACHE_ENTRIES = 4              # trained tokenizer sets kept (LRU)
MAX_BODY_BYTES = 64 * 1024
MAX_TEXT_CHARS = 16_384
CORPUS_TRAIN_BYTES = 100_000   # playground training corpus (train split of the data pipeline)
VOCAB_DIRS = {16384: "", 32768: "v32768/", 65536: "v65536/"}  # suite vocabulary size -> results subdirectory
DEFAULT_TEXT = "The internationalization of unhappily disproportionate misunderstandings requires robust subword tokenization."

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


class TokenizerCache:
    """Thread-safe LRU cache of trained tokenizer sets, keyed by vocab size."""

    def __init__(self, offline: bool = False, max_entries: int = CACHE_ENTRIES) -> None:
        self.offline = offline
        self.max_entries = max_entries
        # ponytail: one lock serialises training; fine for a single-user localhost tool.
        self._lock = threading.Lock()
        self._cache: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        self._corpus: Optional[List[str]] = None
        self._lexicon: Dict[str, List[str]] = {}
        self.corpus_info: Dict[str, Any] = {}

    def _prepare_corpus(self) -> Tuple[List[str], Dict[str, List[str]]]:
        if self._corpus is None:
            from tokenisers.data.pipeline import DataPipeline, SigmorphonLoader

            ds = DataPipeline(force_offline=self.offline).prepare(train_bytes=CORPUS_TRAIN_BYTES)
            corpus = ds.texts("train")
            loader = SigmorphonLoader(offline=self.offline)
            if self.offline:
                gold = loader.load_offline_fixture()
                lexicon_source = "offline test fixture (12 hand-written words, not SIGMORPHON)"
            else:
                words = {w for text in corpus for w in text.split()}
                gold = [e for e in loader.load("eng", "train") if e["word"] in words]
                lexicon_source = "SIGMORPHON 2022 eng train (surface entries found in the corpus)"
            self._lexicon = {e["word"]: e["segments"] for e in gold}
            self.corpus_info = {
                "synthetic": ds.synthetic,
                "dataset": str(ds.root),
                "train_docs": len(corpus),
                "train_bytes": ds.manifest["splits"]["train"]["text_bytes"],
                "fallbacks": ds.manifest["fallbacks"],
                "lexicon_source": lexicon_source,
                "lexicon_words": len(self._lexicon),
            }
            self._corpus = corpus
        return self._corpus, self._lexicon

    def get_tokenizers(self, vocab_size: int = 400) -> Dict[str, Any]:
        with self._lock:
            if vocab_size in self._cache:
                self._cache.move_to_end(vocab_size)
                return self._cache[vocab_size]
            toks = self._train(vocab_size)
            self._cache[vocab_size] = toks
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
            return toks

    def _train(self, vocab_size: int) -> Dict[str, Any]:
        corpus, lexicon = self._prepare_corpus()
        from tokenisers.models import (
            BpeTokenizer,
            LengthPenalizedTokenizer,
            MdlTokenizer,
            MorphemeBpeTokenizer,
            SuperBpeTokenizer,
            UnigramTokenizer,
            WordPieceTokenizer,
            WpBpeBaseTokenizer,
            WpBpeFtTokenizer,
            WpBpeNpmiTokenizer,
            WpBpePigTokenizer,
            WpBpePpigTokenizer,
            WpBpeScpTokenizer,
        )

        v_switch = max(256, int(vocab_size * 0.75))
        factories = [
            ("bpe_frequency", lambda: BpeTokenizer.train(corpus=corpus, vocab_size=vocab_size, scorer="frequency")),
            ("wordpiece_pmi", lambda: WordPieceTokenizer.train(corpus=corpus, vocab_size=vocab_size)),
            ("wp_bpe_base", lambda: WpBpeBaseTokenizer.train(corpus=corpus, vocab_size=vocab_size)),
            ("wp_bpe_ft", lambda: WpBpeFtTokenizer.train(corpus=corpus, vocab_size=vocab_size, min_frequency=2)),
            ("wp_bpe_pig", lambda: WpBpePigTokenizer.train(corpus=corpus, vocab_size=vocab_size, min_frequency=1)),
            ("wp_bpe_scp", lambda: WpBpeScpTokenizer.train(corpus=corpus, vocab_size=vocab_size)),
            ("wp_bpe_npmi", lambda: WpBpeNpmiTokenizer.train(corpus=corpus, vocab_size=vocab_size)),
            ("wp_bpe_ppig", lambda: WpBpePpigTokenizer.train(corpus=corpus, vocab_size=vocab_size)),
            ("mdl_bits", lambda: MdlTokenizer.train(corpus=corpus, vocab_size=vocab_size)),
            ("length_penalized", lambda: LengthPenalizedTokenizer.train(corpus=corpus, vocab_size=vocab_size, beta=0.5)),
            ("super_bpe", lambda: SuperBpeTokenizer.train(corpus=corpus, v_switch=v_switch, v_final=vocab_size)),
            ("morpheme_bpe", lambda: MorphemeBpeTokenizer.train(corpus=corpus, vocab_size=vocab_size, morpheme_lexicon=lexicon, lambda_weight=2.5)),
            ("unigram_lm", lambda: UnigramTokenizer.train(corpus=corpus, vocab_size=vocab_size)),
        ]
        toks: Dict[str, Any] = {}
        for name, factory in factories:
            try:
                toks[name] = factory()
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
        return toks


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

    tokka_data, intrinsic_data, colab_data = doc("tokka_16384"), doc("intrinsic_16384"), doc("proxy_small")
    colab_models = colab_data.get("models", {})
    big_models = doc("proxy_big").get("models", {})
    # Benchmark JSON is {"meta": ..., "models": {...}}; older files were the bare model map (no meta).
    # meta is each run's provenance (data, budgets, backend, commit, synthetic flag): pass it through.
    provenance = {
        "tokka": tokka_data.get("meta"),
        "intrinsic": intrinsic_data.get("meta"),
        "proxy_lm": colab_data.get("meta"),
    }
    tokka_data = tokka_data.get("models", tokka_data)
    intrinsic_data = intrinsic_data.get("models", intrinsic_data)
    # Heatmap columns come from the data, not a fixed list: per-language cells are dicts; the
    # aggregates are not languages and scalars (vocab_size, unique_tokens) are not cells. The union
    # over every vocabulary size's matrix (the same held-out text, so the same languages).
    tokka_languages = list(dict.fromkeys(
        lang
        for key in RESULT_SETS if key.startswith("tokka_")
        for row in doc(key).get("models", doc(key)).values() if isinstance(row, dict)
        for lang, cell in row.items() if isinstance(cell, dict) and lang not in ("macro_avg", "micro_avg")
    ))

    # Consolidate all model identifiers
    all_keys = set(tokka_data.keys()) | set(intrinsic_data.keys()) | set(colab_models.keys()) | set(ALGORITHM_METADATA.keys())

    # Build unified model list
    unified_models: Dict[str, Dict[str, Any]] = {}
    for key in sorted(all_keys):
        meta = ALGORITHM_METADATA.get(
            key,
            {
                "display_name": key.replace("_", " ").title(),
                "family": "Custom",
                "description": f"Tokenizer implementation {key}",
                "formula": r"\mathrm{score}(u,v)",
                "objective": "Heuristic",
                "generation": "Other",
                "pros": [],
                "cons": [],
            },
        )

        intr = intrinsic_data.get(key, {})
        tok = tokka_data.get(key, {})
        col = colab_models.get(key, {})
        big_bpb = big_models.get(key, {}).get("macro_val_bpb")

        macro_bpb = col.get("macro_val_bpb")
        val_domains = col.get("val_domains", {})

        # Intrinsic metrics
        bytes_per_token = intr.get("bytes_per_token")
        if bytes_per_token is None and "macro_avg" in tok:
            bytes_per_token = tok["macro_avg"].get("bytes_per_token")

        unified_models[key] = {
            "key": key,
            "display_name": meta.get("display_name", key),
            "family": meta.get("family", "Unknown"),
            "generation": meta.get("generation", "Baseline"),
            "description": meta.get("description", ""),
            "formula": meta.get("formula", ""),
            "objective": meta.get("objective", ""),
            "pros": meta.get("pros", []),
            "cons": meta.get("cons", []),
            "macro_val_bpb": macro_bpb,
            "big_macro_val_bpb": big_bpb,
            "bytes_per_token": bytes_per_token,
            "renyi_e_2_0": intr.get("renyi_e_2_0"),
            "morph_f1": intr.get("morph_f1"),
            "identifier_f1": intr.get("identifier_f1"),
            "decode_mb_per_sec": intr.get("decode_mb_per_sec"),
            "encode_mb_per_sec": intr.get("encode_mb_per_sec"),
            "renyi_e_2_5": intr.get("renyi_e_2_5"),
            "val_domains": val_domains,
        }

    # Summary KPI Calculation
    valid_bpbs = [(k, v["macro_val_bpb"]) for k, v in unified_models.items() if v["macro_val_bpb"] is not None]
    valid_renyi = [(k, v["renyi_e_2_0"]) for k, v in unified_models.items() if v["renyi_e_2_0"] is not None]
    valid_morph = [(k, v["morph_f1"]) for k, v in unified_models.items() if v["morph_f1"] is not None]
    valid_bpt = [(k, v["bytes_per_token"]) for k, v in unified_models.items() if v["bytes_per_token"] is not None]
    valid_speed = [(k, v["decode_mb_per_sec"]) for k, v in unified_models.items() if v["decode_mb_per_sec"] is not None]

    best_bpb = min(valid_bpbs, key=lambda x: x[1]) if valid_bpbs else ("none", 0.0)
    best_renyi = max(valid_renyi, key=lambda x: x[1]) if valid_renyi else ("none", 0.0)
    best_morph = max(valid_morph, key=lambda x: x[1]) if valid_morph else ("none", 0.0)
    best_bpt = max(valid_bpt, key=lambda x: x[1]) if valid_bpt else ("none", 0.0)
    best_speed = max(valid_speed, key=lambda x: x[1]) if valid_speed else ("none", 0.0)

    summary = {
        "best_downstream_quality": {
            "model": best_bpb[0],
            "display_name": unified_models.get(best_bpb[0], {}).get("display_name", best_bpb[0]),
            "value": best_bpb[1],
            "metric": "Macro Val BPB (Lower is Better)",
        },
        "highest_information_efficiency": {
            "model": best_renyi[0],
            "display_name": unified_models.get(best_renyi[0], {}).get("display_name", best_renyi[0]),
            "value": best_renyi[1],
            "metric": "Rényi E_2.0",
        },
        "best_morphological_f1": {
            "model": best_morph[0],
            "display_name": unified_models.get(best_morph[0], {}).get("display_name", best_morph[0]),
            "value": best_morph[1],
            "metric": "Morph Boundary F1",
        },
        "best_compression": {
            "model": best_bpt[0],
            "display_name": unified_models.get(best_bpt[0], {}).get("display_name", best_bpt[0]),
            "value": best_bpt[1],
            "metric": "Bytes / Token",
        },
        "fastest_throughput": {
            "model": best_speed[0],
            "display_name": unified_models.get(best_speed[0], {}).get("display_name", best_speed[0]),
            "value": best_speed[1],
            "metric": "Decode MB/s",
        },
    }

    return {
        "summary": summary,
        "models": unified_models,
        "tokka_matrix": tokka_data,
        "tokka_languages": tokka_languages,
        "meta": provenance,
        "results": results,
        "metadata": ALGORITHM_METADATA,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class DashboardHandler(BaseHTTPRequestHandler):
    """Static assets plus a JSON API. Same-origin only: no CORS, Host/Origin pinned to loopback."""

    def _send(self, status: int, body: Any, content_type: str = "application/json; charset=utf-8") -> None:
        if not isinstance(body, bytes):
            body = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _origin_ok(self) -> bool:
        """On loopback: reject DNS-rebinding and cross-site requests. In public mode: allow all origins."""
        if getattr(self.server, "public", False):
            return True
        port = self.server.server_address[1]
        allowed = {f"{h}:{port}" for h in LOOPBACK_HOSTS}
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in allowed or (origin is not None and origin not in {f"http://{h}" for h in allowed}):
            self._send(HTTPStatus.FORBIDDEN, {"error": "forbidden host or origin"})
            return False
        return True

    def do_GET(self) -> None:
        if not self._origin_ok():
            return
        path = self.path.split("?")[0]

        if path == "/healthz":
            self._send(HTTPStatus.OK, b"ok", "text/plain")
            return

        if path == "/api/benchmarks":
            try:
                self._send(HTTPStatus.OK, load_benchmark_data())
            except Exception as e:
                logger.exception("Error loading benchmark data")
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})
            return

        static_root = STATIC_DIR.resolve()
        if path == "/" or path == "/index.html":
            file_path = static_root / "index.html"
        elif path.startswith("/static/"):
            file_path = (static_root / path[len("/static/"):]).resolve()
            if not file_path.is_relative_to(static_root):
                self._send(HTTPStatus.FORBIDDEN, b"Forbidden", "text/plain")
                return
        else:
            self._send(HTTPStatus.NOT_FOUND, b"Not Found", "text/plain")
            return

        if not file_path.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"File Not Found", "text/plain")
            return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or {".js": "application/javascript", ".css": "text/css", ".json": "application/json"}.get(
            file_path.suffix, "application/octet-stream"
        )
        if "text" in mime_type or mime_type in ("application/javascript", "application/json"):
            mime_type += "; charset=utf-8"
        self._send(HTTPStatus.OK, file_path.read_bytes(), mime_type)

    def do_POST(self) -> None:
        if not self._origin_ok():
            return
        if self.path.split("?")[0] != "/api/tokenize":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            text, vocab_size = self._parse_tokenize_request()
        except ApiError as e:
            self._send(e.status, {"error": str(e)})
            return
        try:
            self._send(HTTPStatus.OK, self._tokenize(text, vocab_size))
        except Exception as e:
            logger.exception("Error in tokenize endpoint")
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})

    def _parse_tokenize_request(self) -> Tuple[str, int]:
        content_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type != "application/json":
            raise ApiError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise ApiError(HTTPStatus.LENGTH_REQUIRED, "Content-Length required")
        if length < 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
        if length > MAX_BODY_BYTES:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body exceeds {MAX_BODY_BYTES} bytes")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError) as e:  # RecursionError: nesting too deep
            raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid JSON payload: {e}")
        if not isinstance(payload, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "JSON payload must be an object")

        text = payload.get("text") or DEFAULT_TEXT
        if not isinstance(text, str):
            raise ApiError(HTTPStatus.BAD_REQUEST, "text must be a string")
        if len(text) > MAX_TEXT_CHARS:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"text exceeds {MAX_TEXT_CHARS} characters")
        vocab_size = payload.get("vocab_size", 400)
        if type(vocab_size) is not int or not MIN_VOCAB_SIZE <= vocab_size <= MAX_VOCAB_SIZE:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"vocab_size must be an integer in [{MIN_VOCAB_SIZE}, {MAX_VOCAB_SIZE}], got {vocab_size!r}",
            )
        return text, vocab_size

    def _tokenize(self, text: str, vocab_size: int) -> Dict[str, Any]:
        cache: TokenizerCache = self.server.tokenizer_cache  # type: ignore[attr-defined]
        tokenizers = cache.get_tokenizers(vocab_size)
        raw_bytes = len(text.encode("utf-8"))
        results: List[Dict[str, Any]] = []
        for name, tok in tokenizers.items():
            meta = ALGORITHM_METADATA.get(name, {})
            t0 = time.perf_counter()
            token_ids = tok.encode(text)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

            token_details: List[Dict[str, Any]] = []
            tokens: List[str] = []
            for tid in token_ids:
                # byte_len counts the token's own bytes: a partial-UTF-8 token is 1 byte even
                # though its display piece is U+FFFD (3 bytes), so byte_lens sum to raw_bytes.
                b_val = tok.id_to_bytes(tid) or b""
                piece = b_val.decode("utf-8", errors="replace") if b_val else f"<{tid}>"
                tokens.append(piece)
                token_details.append({"id": tid, "piece": piece, "byte_len": len(b_val)})

            num_tokens = len(token_ids)
            results.append(
                {
                    "name": name,
                    "display_name": meta.get("display_name", name),
                    "family": meta.get("family", "Unknown"),
                    "num_tokens": num_tokens,
                    "raw_bytes": raw_bytes,
                    "bytes_per_token": round(raw_bytes / num_tokens, 3) if num_tokens > 0 else 0.0,
                    "encode_ms": elapsed_ms,
                    "token_ids": token_ids,
                    "tokens": tokens,
                    "token_details": token_details,
                }
            )
        return {
            "text": text,
            "vocab_size": vocab_size,
            "raw_bytes": raw_bytes,
            "corpus": cache.corpus_info,
            "results": results,
        }

    def log_message(self, format: str, *args: Any) -> None:
        if os.environ.get("TOKENISERS_DASHBOARD_VERBOSE"):
            super().log_message(format, *args)


class DashboardServer:
    """ThreadingHTTPServer with lifecycle controls. Loopback by default; ``0.0.0.0`` for public deploy."""

    ALLOWED_HOSTS = LOOPBACK_HOSTS + PUBLIC_HOSTS

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, offline: bool = False) -> None:
        if host not in self.ALLOWED_HOSTS:
            raise ValueError(f"host must be one of {self.ALLOWED_HOSTS}, got {host!r}")
        self.host = host
        self.port = port
        self.server = ThreadingHTTPServer((host, port), DashboardHandler)
        self.server.tokenizer_cache = TokenizerCache(offline=offline)  # type: ignore[attr-defined]
        self.server.public = host in PUBLIC_HOSTS  # type: ignore[attr-defined]
        self._thread: Optional[threading.Thread] = None

    def start(self, blocking: bool = True) -> None:
        print(f"Tokenisers Lab Benchmark Dashboard running at http://{self.host}:{self.port}/")
        if blocking:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("Server shutdown completed.")


def create_server(host: str = "127.0.0.1", port: int = 8765, offline: bool = False) -> DashboardServer:
    """Creates a configured DashboardServer instance."""
    return DashboardServer(host=host, port=port, offline=offline)


def start_server(host: str = "127.0.0.1", port: int = 8765, blocking: bool = True, offline: bool = False) -> DashboardServer:
    """Factory creating and running the dashboard server."""
    server = create_server(host, port, offline=offline)
    server.start(blocking=blocking)
    return server
