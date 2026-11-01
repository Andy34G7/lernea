"""
Data pipeline: real tokenizer-training corpora with a deterministic, exact-hash
de-duplicated, document-disjoint train/val/test split, plus SIGMORPHON 2022
morpheme segmentation gold data. Only exact duplicate documents are removed:
near-duplicates and boilerplate lines shared across documents (and splits) stay.

Real sources are read as parquet straight from pinned Hugging Face dataset
revisions (huggingface_hub picks HF_TOKEN up from the environment):

  english       HuggingFaceFW/fineweb-edu    sample/10BT
  multilingual  HuggingFaceFW/fineweb-2      data/<lang>/train for each of FINEWEB2_LANGUAGES
  code          bigcode/starcoderdata        <lang>/ for each of CODE_LANGUAGES (gated: needs
                HF_TOKEN with the terms accepted). If access is denied the code domain is
                read from codeparrot/github-code-clean (ungated) instead, with the same
                per-language budgets, and the substitution is recorded in the manifest.

Synthetic text exists only for unit tests and explicit offline use
(``DataPipeline(force_offline=True)``). It is never mixed into a real dataset and is
labelled in the directory name, the manifest and every document.

On-disk layout (one directory per parameter set, written atomically)::

    <cache_dir>/<hf|synthetic>-<key>/
        manifest.json    provenance: config, sources + revisions, per-source doc/byte
                         counts, fallbacks, errors, split file checksums
        train.jsonl      one JSON object per line:
        val.jsonl          {"text", "domain", "lang", "source", "sha256"}
        test.jsonl
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import random
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)

# 2: the github-code-clean fallback fills per-language budgets (v1 filled one mixed budget in file order).
FORMAT_VERSION = 2
SPLITS: Tuple[str, ...] = ("train", "val", "test")
DEFAULT_TRAIN_BYTES = 50 * 1024 * 1024

PRESET_BYTE_SIZES: Dict[str, int] = {
    "smoke": 100_000,
    "small": DEFAULT_TRAIN_BYTES,
    "medium": 100 * 1024 * 1024,
    "large": 200 * 1024 * 1024,
}

DOMAIN_WEIGHTS: Dict[str, float] = {
    "english": 0.50,
    "multilingual": 0.30,
    "code": 0.20,
}

FINEWEB2_LANGUAGES: Tuple[str, ...] = (
    "deu_Latn",
    "fra_Latn",
    "spa_Latn",
    "rus_Cyrl",
    "cmn_Hani",
    "jpn_Jpan",
    "arb_Arab",
    "hin_Deva",
    "hun_Latn",
    "fin_Latn",
)

CODE_LANGUAGES: Tuple[str, ...] = ("python", "javascript", "java", "cpp", "go", "rust")

# Documents larger than this are skipped (minified/generated files would dominate small budgets).
MAX_DOC_BYTES = 100_000

# Pinned dataset revisions: the same parameters always read the same bytes.
REVISIONS: Dict[str, str] = {
    "HuggingFaceFW/fineweb-edu": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "HuggingFaceFW/fineweb-2": "af9c13333eb981300149d5ca60a8e9d659b276b9",
    "bigcode/starcoderdata": "9fc30b578cedaec69e47302df72cf00feed7c8c4",
    "codeparrot/github-code-clean": "c48d40f9e70f0196f8236901ee35807f7d6c44c0",
}

# github-code-clean "language" labels for CODE_LANGUAGES; rows with other labels are skipped.
GITHUB_CODE_LANGUAGES: Tuple[Tuple[str, str], ...] = (
    ("Python", "python"),
    ("JavaScript", "javascript"),
    ("Java", "java"),
    ("C++", "cpp"),
    ("GO", "go"),
    ("Rust", "rust"),
)

SIGMORPHON_COMMIT = "ac161e1107e423577e922b05f8c43c6ebad6722a"
SIGMORPHON_LANGUAGES: Tuple[str, ...] = ("ces", "eng", "fra", "hun", "ita", "lat", "mon", "rus", "spa")
_SIGMORPHON_URL = "https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/{commit}/data/{path}"
_SIGMORPHON_SUFFIX = {"train": "train.tsv", "dev": "dev.tsv", "test": "test.gold.tsv"}

# Hand-written segmentations. NOT SIGMORPHON data: an offline test fixture only.
OFFLINE_MORPHEME_FIXTURE: List[Dict[str, Any]] = [
    {"word": "unhappiness", "segments": ["un", "happi", "ness"]},
    {"word": "internationalization", "segments": ["inter", "nation", "al", "iz", "ation"]},
    {"word": "reconfigurations", "segments": ["re", "configur", "ation", "s"]},
    {"word": "disproportionately", "segments": ["dis", "proportion", "ate", "ly"]},
    {"word": "misunderstandings", "segments": ["mis", "under", "stand", "ing", "s"]},
    {"word": "decentralization", "segments": ["de", "central", "iz", "ation"]},
    {"word": "unquestionably", "segments": ["un", "question", "ab", "ly"]},
    {"word": "counterproductive", "segments": ["counter", "product", "ive"]},
    {"word": "overgeneralization", "segments": ["over", "general", "iz", "ation"]},
    {"word": "antidisestablishmentarianism", "segments": ["anti", "dis", "establish", "ment", "arian", "ism"]},
    {"word": "megszentségteleníthetetlenségeskedéseitekért", "segments": ["meg", "szent", "ség", "telen", "ít", "hetetlen", "ség", "es", "ked", "és", "eitek", "ért"]},
    {"word": "epäjärjestelmällistyttämättömyydelläänsäkään", "segments": ["epä", "järjestelmä", "llis", "tyttä", "mättö", "myy", "dellä", "än", "sä", "kään"]},
]


class IncompleteDatasetError(RuntimeError):
    """A source failed while streaming. The partial dataset was still written; see ``.dataset``."""

    def __init__(self, message: str, dataset: "PreparedDataset") -> None:
        super().__init__(message)
        self.dataset = dataset


def _default_cache_dir() -> Path:
    env = os.environ.get("TOKENISERS_CACHE_DIR")
    return Path(env) if env else Path.home() / ".cache" / "tokenisers" / "data"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _byte_cuts(segments: Sequence[str]) -> Set[int]:
    cuts: Set[int] = set()
    offset = 0
    for seg in segments[:-1]:
        offset += len(seg.encode("utf-8"))
        cuts.add(offset)
    return cuts


# ----------------------------------------------------------------------------
# Synthetic text (offline / unit tests only)
# ----------------------------------------------------------------------------

class OfflineFallbackGenerator:
    """Seeded synthetic text for unit tests and explicit offline runs. Not real data."""

    SAMPLE_PROSE = [
        "Tokenization is the foundational step in modern natural language processing systems.",
        "High quality subword algorithms balance fertility, compression efficiency, and morphology.",
        "Neural language models represent vocabulary entries as learned high dimensional vectors.",
        "SuperBPE curates subword vocabularies by respecting whitespace boundaries before collocation merging.",
        "Information theoretic scoring objectives measure description length reduction in bits per byte.",
        "Pointwise mutual information rewards strongly associated lexical stems and compound constructions.",
        "Unigram language models prune candidate subword vocabularies via expectation maximization and Viterbi lattices.",
        "Stochastic regularization during language model training improves robustness to rare morphological inflections.",
    ]

    SAMPLE_MULTILINGUAL = [
        "Falsches Üben von Xylophonmusik quält jeden größeren Zwerg.",
        "Portez ce vieux whisky au juge blond qui fume.",
        "Съешь же ещё этих мягких французских булок, да выпей чаю.",
        "道可道，非常道；名可名，非常名。无名天地之始，有名万物之母。",
        "いろはにほへと ちりぬるを わかよたれそ つねならむ うゐのおくやま けふこえて",
        "نص حكيم له سر قاطع وذو شأن عظيم مكتوب على ثوب أخضر ومغلف بجلد غزال.",
        "ऋषियों को ज्ञान और शांति की खोज में हिमालय के शांत वातावरण में तपस्या करते देखा गया है।",
        "Egy hűtlen feleség gyönyörű kalandja a varázslatos éjszakában.",
        "Viekas kettu hypähti laiskan koiran ylitse ja juoksi metsän siimekseen.",
        "El veloz murciélago hindú comía feliz cardillo y kiwi mientras la cigüeña tocaba el saxofón.",
    ]

    SAMPLE_CODE = [
        "def compute_renyi_entropy(probabilities: list[float], alpha: float = 1.0) -> float:\n    total = sum(probabilities)\n    return total\n",
        "pub struct TokenizerModel<S: PairScorer> {\n    vocab: HashMap<TokenId, Vec<u8>>,\n    merges: Vec<(Pair, TokenId)>,\n}\n",
        "template <typename T>\nclass VectorBuffer {\n    std::vector<T> buffer_;\n};\n",
        "export const tokenizeStream = async (stream) => {\n  const chunks = [];\n  return chunks;\n};\n",
        "fn calculate_bits_per_byte(loss_nats: f64, bytes_per_token: f64) -> f64 {\n    if bytes_per_token <= 0.0 { return 0.0; }\n    loss_nats / (std::f64::consts::LN_2 * bytes_per_token)\n}\n",
        "public final class MergeTable {\n    private final Map<Long, Integer> ranks = new HashMap<>();\n    public int rank(long pair) { return ranks.getOrDefault(pair, -1); }\n}\n",
        "func CountPairs(ids []int) map[[2]int]int {\n\tcounts := make(map[[2]int]int)\n\tfor i := 0; i+1 < len(ids); i++ {\n\t\tcounts[[2]int{ids[i], ids[i+1]}]++\n\t}\n\treturn counts\n}\n",
    ]

    _LANG_SAMPLE: Dict[str, str] = {
        "eng_Latn": " ".join(SAMPLE_PROSE),
        "deu_Latn": SAMPLE_MULTILINGUAL[0],
        "fra_Latn": SAMPLE_MULTILINGUAL[1],
        "rus_Cyrl": SAMPLE_MULTILINGUAL[2],
        "cmn_Hani": SAMPLE_MULTILINGUAL[3],
        "jpn_Jpan": SAMPLE_MULTILINGUAL[4],
        "arb_Arab": SAMPLE_MULTILINGUAL[5],
        "hin_Deva": SAMPLE_MULTILINGUAL[6],
        "hun_Latn": SAMPLE_MULTILINGUAL[7],
        "fin_Latn": SAMPLE_MULTILINGUAL[8],
        "spa_Latn": SAMPLE_MULTILINGUAL[9],
        "python": SAMPLE_CODE[0],
        "rust": SAMPLE_CODE[1] + SAMPLE_CODE[4],
        "cpp": SAMPLE_CODE[2],
        "javascript": SAMPLE_CODE[3],
        "java": SAMPLE_CODE[5],
        "go": SAMPLE_CODE[6],
    }

    def iter_docs(self, lang: str, seed: int = 0) -> Iterator[str]:
        """Endless seeded stream of word-salad documents built from ``lang``'s sample text."""
        if lang not in self._LANG_SAMPLE:
            raise ValueError(f"no synthetic sample text for language {lang!r}")
        sample = self._LANG_SAMPLE[lang]
        words = sample.split()
        sep = " "
        if len(words) < 4:  # unspaced script: sample characters instead
            words, sep = [c for c in sample if not c.isspace()], ""
        rng = random.Random(f"{seed}:{lang}")
        while True:
            yield sep.join(rng.choice(words) for _ in range(rng.randint(30, 120))) + "\n"

    def generate_synthetic_corpus(self, target_bytes: int = 50_000, seed: int = 0) -> List[str]:
        if target_bytes <= 0:
            raise ValueError(f"target_bytes must be strictly positive, got {target_bytes}")
        streams = [self.iter_docs(lang, seed) for lang in self._LANG_SAMPLE]
        docs: List[str] = []
        total = 0
        for stream in itertools.cycle(streams):
            doc = next(stream)
            docs.append(doc)
            total += len(doc.encode("utf-8"))
            if total >= target_bytes:
                break
        return docs


# ----------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------

class DeterministicSplitter:
    """Content-hash train/val/test assignment: a document's split depends only on (seed, text)."""

    def __init__(self, val_fraction: float = 0.05, test_fraction: float = 0.05, seed: int = 0) -> None:
        if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
            raise ValueError(
                f"need val_fraction > 0, test_fraction > 0 and val+test < 1, got {val_fraction}, {test_fraction}"
            )
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.seed = int(seed)

    def hash_document(self, doc: str) -> str:
        """Unsalted SHA-256 of the UTF-8 text: the exact-duplicate key."""
        return hashlib.sha256(doc.encode("utf-8")).hexdigest()

    def assign_split(self, doc: str) -> str:
        digest = hashlib.sha256(f"{self.seed}:".encode("utf-8") + doc.encode("utf-8")).digest()
        u = int.from_bytes(digest[:8], "big") / 2**64
        if u < self.val_fraction:
            return "val"
        if u < self.val_fraction + self.test_fraction:
            return "test"
        return "train"

    def fraction(self, split: str) -> float:
        return {
            "train": 1.0 - self.val_fraction - self.test_fraction,
            "val": self.val_fraction,
            "test": self.test_fraction,
        }[split]


# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Source:
    domain: str
    lang: str                  # language / programming-language label ("mixed" when per-row)
    repo: str                  # HF dataset repo id, or "synthetic"
    path: str = ""             # directory of parquet files inside the repo
    text_field: str = "text"
    lang_field: Optional[str] = None                 # per-row label column
    row_langs: Tuple[Tuple[str, str], ...] = ()      # label -> lang; unmapped rows skipped


_CODE_FALLBACK = Source(
    "code", "mixed", "codeparrot/github-code-clean", "data", "code", "language", GITHUB_CODE_LANGUAGES
)


def _plan(offline: bool, languages: Sequence[str]) -> Dict[str, List[Source]]:
    if offline:
        return {
            "english": [Source("english", "eng_Latn", "synthetic")],
            "multilingual": [Source("multilingual", lang, "synthetic") for lang in languages],
            "code": [Source("code", lang, "synthetic") for lang in CODE_LANGUAGES],
        }
    return {
        "english": [Source("english", "eng_Latn", "HuggingFaceFW/fineweb-edu", "sample/10BT")],
        "multilingual": [
            Source("multilingual", lang, "HuggingFaceFW/fineweb-2", f"data/{lang}/train") for lang in languages
        ],
        "code": [Source("code", lang, "bigcode/starcoderdata", lang, "content") for lang in CODE_LANGUAGES],
    }


def _iter_hf_rows(src: Source) -> Iterator[Tuple[str, str]]:
    """Yield (text, lang) rows of ``src`` in file order, reading parquet row groups lazily."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f"datasets/{src.repo}@{REVISIONS[src.repo]}/{src.path}"
    labels = dict(src.row_langs)
    columns = [src.text_field] + ([src.lang_field] if src.lang_field else [])
    files = sorted(f for f in fs.ls(base, detail=False) if f.endswith(".parquet"))
    for file in files:
        with fs.open(file, "rb") as fh:
            for batch in pq.ParquetFile(fh).iter_batches(batch_size=256, columns=columns):
                texts = batch.column(0).to_pylist()
                langs = batch.column(1).to_pylist() if src.lang_field else [src.lang] * len(texts)
                for text, label in zip(texts, langs):
                    lang = labels.get(label) if labels else label
                    if text and lang:
                        yield text, lang


def _iter_rows(src: Source, seed: int) -> Iterator[Tuple[str, str]]:
    if src.repo == "synthetic":
        return ((doc, src.lang) for doc in OfflineFallbackGenerator().iter_docs(src.lang, seed))
    return _iter_hf_rows(src)


def _is_access_denied(exc: BaseException) -> bool:
    try:
        from huggingface_hub.errors import GatedRepoError
    except Exception:  # pragma: no cover - very old huggingface_hub
        GatedRepoError = ()  # type: ignore
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return isinstance(exc, (GatedRepoError, PermissionError)) or status in (401, 403)


def _hf_token_present() -> bool:
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return bool(os.environ.get("HF_TOKEN"))


# ----------------------------------------------------------------------------
# Prepared dataset
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PreparedDataset:
    root: Path
    manifest: Dict[str, Any]

    @property
    def synthetic(self) -> bool:
        return bool(self.manifest["synthetic"])

    def path(self, split: str) -> Path:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        return self.root / self.manifest["splits"][split]["file"]

    def docs(self, split: str) -> List[Dict[str, Any]]:
        with open(self.path(split), encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def texts(self, split: str, domain: Optional[str] = None, lang: Optional[str] = None) -> List[str]:
        return [
            d["text"]
            for d in self.docs(split)
            if (domain is None or d["domain"] == domain) and (lang is None or d["lang"] == lang)
        ]


class DataPipeline:
    """Builds, caches and loads prepared datasets (see module docstring)."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None, force_offline: bool = False) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_offline = force_offline

    # -- public API ---------------------------------------------------------

    def prepare(
        self,
        train_bytes: int = DEFAULT_TRAIN_BYTES,
        *,
        seed: int = 0,
        val_fraction: float = 0.05,
        test_fraction: float = 0.05,
        languages: Sequence[str] = FINEWEB2_LANGUAGES,
        allow_partial: bool = False,
    ) -> PreparedDataset:
        """Return the dataset for these parameters, building it if it is not cached.

        Raises IncompleteDatasetError (after writing what was streamed) when a source
        fails, unless ``allow_partial`` is set.
        """
        if not isinstance(train_bytes, int) or train_bytes <= 0:
            raise ValueError(f"train_bytes must be a positive int, got {train_bytes!r}")
        splitter = DeterministicSplitter(val_fraction, test_fraction, seed)
        offline = self.force_offline
        if offline:
            unknown = [lang for lang in languages if lang not in OfflineFallbackGenerator._LANG_SAMPLE]
            if unknown:
                raise ValueError(f"no synthetic sample text for languages {unknown}")
        config: Dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "train_bytes": train_bytes,
            "val_fraction": splitter.val_fraction,
            "test_fraction": splitter.test_fraction,
            "seed": splitter.seed,
            "offline": offline,
            "languages": list(languages),
            "code_languages": list(CODE_LANGUAGES),
            "domain_weights": DOMAIN_WEIGHTS,
            "max_doc_bytes": MAX_DOC_BYTES,
            "revisions": None if offline else REVISIONS,
            # A token changes which code source is reachable, so it is part of the key.
            "hf_token_present": None if offline else _hf_token_present(),
        }
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        root = self.cache_dir / f"{'synthetic' if offline else 'hf'}-{key}"

        cached = self._load(root)
        if cached is not None and cached.manifest["complete"]:
            return cached

        manifest, splits = self._build(config, splitter)
        manifest["key"] = key
        dataset = self._write(root, manifest, splits)
        if not manifest["complete"]:
            msg = f"dataset {root} is incomplete: " + "; ".join(e["error"] for e in manifest["errors"])
            logger.warning(msg)
            if not allow_partial:
                raise IncompleteDatasetError(msg + " (pass allow_partial=True to use it)", dataset)
        return dataset

    def sample_preset(self, preset: str = "smoke", max_bytes: Optional[int] = None, split: str = "train") -> List[str]:
        """Texts of one split of ``prepare(train_bytes=max_bytes or PRESET_BYTE_SIZES[preset])``."""
        if max_bytes is None:
            if preset not in PRESET_BYTE_SIZES:
                raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESET_BYTE_SIZES)}")
            max_bytes = PRESET_BYTE_SIZES[preset]
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be strictly positive, got {max_bytes}")
        return self.prepare(train_bytes=max_bytes).texts(split)

    def lm_pool(self, base: PreparedDataset, total_bytes: int) -> Tuple[List[str], Dict[str, Any]]:
        """Extra LM-training text for ``base``: (texts, provenance).

        The pool is the train split of the dataset with ``base``'s parameters but
        ``train_bytes=total_bytes`` (built and cached under that dataset's own key, so ``base`` and
        its cache are untouched), minus every document of base's train/val/test by sha256. Same
        sources, revisions and per-source byte shares as ``base``; each source is read further in
        file order, so base train + pool holds about ``total_bytes``. Pool documents land in train
        under the split rule, so none could be a val/test document of ``base`` either.
        """
        cfg = base.manifest["config"]
        if total_bytes <= cfg["train_bytes"]:
            raise ValueError(f"total_bytes ({total_bytes:,}) must exceed the base train budget "
                             f"({cfg['train_bytes']:,})")
        if self.force_offline != cfg["offline"]:  # before building anything
            raise ValueError(f"{base.root.name} and this pipeline do not read the same sources (offline differs)")
        big = self.prepare(train_bytes=total_bytes, seed=cfg["seed"], val_fraction=cfg["val_fraction"],
                           test_fraction=cfg["test_fraction"], languages=cfg["languages"])
        used = lambda m: [f["used"] for f in m["fallbacks"]]  # noqa: E731
        same_config = {**big.manifest["config"], "train_bytes": cfg["train_bytes"]} == cfg
        if not same_config or used(big.manifest) != used(base.manifest):
            raise ValueError(f"{big.root.name} does not read the same sources as {base.root.name}")
        held = {d["sha256"] for s in SPLITS for d in base.docs(s)}
        train = big.docs("train")
        docs = [d for d in train if d["sha256"] not in held]
        prov = {"rule": "train split of the dataset below minus every base train/val/test document (sha256)",
                "base": base.root.name, "docs": len(docs),
                "text_bytes": sum(len(d["text"].encode("utf-8")) for d in docs),
                "excluded_base_docs": len(train) - len(docs), "dataset": {"root": big.root.name, **big.manifest}}
        return [d["text"] for d in docs], prov

    # -- internals ----------------------------------------------------------

    def _build(
        self, config: Dict[str, Any], splitter: DeterministicSplitter
    ) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
        seen: Set[str] = set()
        # ponytail: all splits are held in memory until _write (measured ~1.7x the dataset's text
        # bytes; ~100 MB at the 50 MB default). Stream docs into the temp split files if presets reach GBs.
        splits: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SPLITS}
        records: List[Dict[str, Any]] = []
        fallbacks: List[Dict[str, Any]] = []
        plan = _plan(config["offline"], config["languages"])

        for domain, weight in DOMAIN_WEIGHTS.items():
            sources = plan[domain]
            domain_records: List[Dict[str, Any]] = []
            for i, src in enumerate(sources):
                (rec,) = self._fill(src, config["train_bytes"] * weight / len(sources), splitter, seen, splits)
                domain_records.append(rec)
                if i == 0 and rec["error_kind"] == "access_denied" and domain == "code" and not config["offline"]:
                    rec["replaced_by_fallback"] = True
                    fallbacks.append({
                        "domain": domain,
                        "wanted": src.repo,
                        "used": _CODE_FALLBACK.repo,
                        "reason": rec["error"],
                        "hf_token_present": config["hf_token_present"],
                    })
                    logger.warning("%s not accessible (%s); using %s for code", src.repo, rec["error"], _CODE_FALLBACK.repo)
                    # Same per-language budgets as the starcoder plan: one record per code language.
                    per_lang = config["train_bytes"] * weight / len(_CODE_FALLBACK.row_langs)
                    domain_records.extend(self._fill(_CODE_FALLBACK, per_lang, splitter, seen, splits))
                    break
            records.extend(domain_records)

        errors = [
            {"domain": r["domain"], "lang": r["lang"], "repo": r["repo"], "error": r["error"]}
            for r in records
            if r["error"] and not r.get("replaced_by_fallback")
        ]
        manifest: Dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "synthetic": config["offline"],
            "complete": not errors,
            "config": config,
            "dedup": "exact SHA-256 of the UTF-8 text, global across all sources and splits",
            "split_rule": (
                "u = first 8 bytes of sha256(f'{seed}:' + text) / 2**64; val if u < val_fraction, "
                "test if u < val_fraction + test_fraction, else train; per-source byte budgets per split, "
                "documents landing in a full split are dropped"
            ),
            "sources": records,
            "fallbacks": fallbacks,
            "errors": errors,
            "splits": {},
        }
        return manifest, splits

    @staticmethod
    def _fill(
        src: Source,
        train_budget: float,
        splitter: DeterministicSplitter,
        seen: Set[str],
        splits: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Stream ``src`` into ``splits`` until each split's byte budget is met for each of its languages.

        ``train_budget`` is per language. A source with per-row labels (``row_langs``) gets one
        budget and one returned record per label language, so a mixed source keeps the balance
        and held-out coverage of one-language sources; any other source returns one record.
        """
        budget = {s: max(1, int(train_budget * splitter.fraction(s) / splitter.fraction("train"))) for s in SPLITS}
        langs = list(dict.fromkeys(lang for _, lang in src.row_langs)) or [src.lang]
        recs: Dict[str, Dict[str, Any]] = {
            lang: {
                "domain": src.domain,
                "lang": lang,
                "repo": src.repo,
                "path": src.path,
                "revision": REVISIONS.get(src.repo),
                "budget_bytes": budget,
                "docs": {s: 0 for s in SPLITS},
                "bytes": {s: 0 for s in SPLITS},
                "rows_read": 0,
                "duplicates_skipped": 0,
                "too_long_skipped": 0,
                "exhausted": False,
                "error": None,
                "error_kind": None,
            }
            for lang in langs
        }
        unfilled = set(langs)
        try:
            rows = _iter_rows(src, splitter.seed)
            for text, lang in rows:
                if lang not in unfilled:  # this language's budgets are already met
                    continue
                rec = recs[lang]
                rec["rows_read"] += 1
                raw = text.encode("utf-8")
                if len(raw) > MAX_DOC_BYTES:
                    rec["too_long_skipped"] += 1
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                if digest in seen:
                    rec["duplicates_skipped"] += 1
                    continue
                split = splitter.assign_split(text)
                if rec["bytes"][split] >= budget[split]:
                    continue
                seen.add(digest)
                splits[split].append({"text": text, "domain": src.domain, "lang": lang, "source": src.repo, "sha256": digest})
                rec["docs"][split] += 1
                rec["bytes"][split] += len(raw)
                if all(rec["bytes"][s] >= budget[s] for s in SPLITS):
                    unfilled.discard(lang)
                    if not unfilled:
                        break
            else:
                for lang in unfilled:
                    recs[lang]["exhausted"] = True
        except Exception as exc:  # network, auth, parquet: keep what was read, record the failure
            error = f"{type(exc).__name__}: {exc}"
            kind = "access_denied" if _is_access_denied(exc) else "stream_error"
            for lang in unfilled:
                recs[lang]["error"], recs[lang]["error_kind"] = error, kind
            read = sum(r["rows_read"] for r in recs.values())
            logger.warning("source %s %s/%s failed after %d rows: %s", src.domain, src.repo, src.path, read, error)
        return list(recs.values())

    def _write(self, root: Path, manifest: Dict[str, Any], splits: Dict[str, List[Dict[str, Any]]]) -> PreparedDataset:
        """Write into a temp dir, then rename it into place: readers never see half a dataset."""
        tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{root.name}-", dir=self.cache_dir))
        try:
            for split in SPLITS:
                path = tmp / f"{split}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for doc in splits[split]:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                manifest["splits"][split] = {
                    "file": path.name,
                    "docs": len(splits[split]),
                    "text_bytes": sum(len(d["text"].encode("utf-8")) for d in splits[split]),
                    "sha256": _sha256_file(path),
                }
            (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            if root.exists():  # stale, corrupt or incomplete: move aside, then drop
                trash = Path(tempfile.mkdtemp(prefix=f".old-{root.name}-", dir=self.cache_dir))
                os.replace(root, trash / "d")
                shutil.rmtree(trash, ignore_errors=True)
            try:
                os.replace(tmp, root)
            except OSError:
                # ponytail: concurrent builders of one key -> first rename wins, we reuse it.
                winner = self._load(root)
                if winner is None:
                    raise
                shutil.rmtree(tmp, ignore_errors=True)
                return winner
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return PreparedDataset(root, manifest)

    @staticmethod
    def _load(root: Path) -> Optional[PreparedDataset]:
        """Load a cached dataset if its manifest is readable and every split file matches its checksum."""
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format_version") != FORMAT_VERSION:
                return None
            for split in SPLITS:
                info = manifest["splits"][split]
                if _sha256_file(root / info["file"]) != info["sha256"]:
                    logger.warning("cached %s/%s fails its checksum; rebuilding", root, info["file"])
                    return None
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("unreadable cached dataset %s (%s); rebuilding", root, exc)
            return None
        return PreparedDataset(root, manifest)


# ----------------------------------------------------------------------------
# SIGMORPHON 2022 SegmentationST (word level)
# ----------------------------------------------------------------------------

class SigmorphonLoader:
    """SIGMORPHON 2022 Shared Task (morpheme segmentation, word level), pinned to SIGMORPHON_COMMIT.

    Gold segmentations are *canonical*: "subsidised -> subside @@y @@ise @@ed". Only entries
    whose morphs concatenate back to the word ("surface" entries) have byte-offset
    ``gold_cuts``; the rest carry ``gold_cuts=None``.
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None, offline: bool = False) -> None:
        base = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self.cache_dir = base / f"sigmorphon2022-{SIGMORPHON_COMMIT[:12]}"
        self.offline = offline

    @staticmethod
    def _relpath(lang: str, split: str) -> str:
        if lang not in SIGMORPHON_LANGUAGES:
            raise ValueError(f"lang must be one of {SIGMORPHON_LANGUAGES}, got {lang!r}")
        if split not in _SIGMORPHON_SUFFIX:
            raise ValueError(f"split must be one of {tuple(_SIGMORPHON_SUFFIX)}, got {split!r}")
        # Mongolian was the surprise language: its train/dev files live in data/surprise/.
        prefix = "surprise/" if lang == "mon" and split != "test" else ""
        return f"{prefix}{lang}.word.{_SIGMORPHON_SUFFIX[split]}"

    def url(self, lang: str, split: str) -> str:
        return _SIGMORPHON_URL.format(commit=SIGMORPHON_COMMIT, path=self._relpath(lang, split))

    def path(self, lang: str, split: str) -> Path:
        """Local path of the gold TSV, downloading it (atomically) on first use."""
        dest = self.cache_dir / self._relpath(lang, split).replace("/", "_")
        if dest.is_file():
            return dest
        if self.offline:
            raise FileNotFoundError(f"SIGMORPHON {lang}/{split} is not cached at {dest} and offline=True")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(self.url(lang, split), timeout=120) as resp:
            data = resp.read()
        fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}-", dir=dest.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return dest

    def load(self, lang: str, split: str = "test", surface_only: bool = True) -> List[Dict[str, Any]]:
        entries = self.load_tsv(self.path(lang, split))
        for e in entries:
            e["lang"], e["split"] = lang, split
        return [e for e in entries if e["surface"]] if surface_only else entries

    def provenance(self, lang: str, split: str) -> Dict[str, Any]:
        path = self.path(lang, split)
        return {
            "dataset": "SIGMORPHON 2022 SegmentationST, word level",
            "lang": lang,
            "split": split,
            "commit": SIGMORPHON_COMMIT,
            "url": self.url(lang, split),
            "sha256": _sha256_file(path),
        }

    @staticmethod
    def load_tsv(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
        """Parse ``word<TAB>morph @@morph ...[<TAB>category]`` lines; malformed lines are skipped."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"SIGMORPHON file not found at {filepath}")
        entries: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 2 or not parts[0]:
                    continue
                segments = [s for s in parts[1].split(" @@") if s]
                if not segments:
                    continue
                surface = "".join(segments) == parts[0]
                entries.append({
                    "word": parts[0],
                    "segments": segments,
                    "category": parts[2] if len(parts) > 2 else None,
                    "surface": surface,
                    "gold_cuts": _byte_cuts(segments) if surface else None,
                })
        return entries

    @staticmethod
    def load_offline_fixture() -> List[Dict[str, Any]]:
        """12 hand-written segmentations. A test fixture, NOT SIGMORPHON data."""
        return [
            {"word": e["word"], "segments": list(e["segments"]), "category": None, "surface": True,
             "gold_cuts": _byte_cuts(e["segments"])}
            for e in OFFLINE_MORPHEME_FIXTURE
        ]
