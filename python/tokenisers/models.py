"""
High-level tokenizer classes of the Tokenisers platform.

Every class has the same API: ``train`` (classmethod), ``encode``/``encode_bytes``,
``decode``/``decode_bytes``, ``id_to_bytes``, ``id_to_token``/``token_to_id``, ``get_vocab``,
``special_tokens``, ``vocab_size``, ``save`` and ``from_pretrained``. Token strings are the GPT-2 byte-level printable
form of the token bytes (special tokens: their raw content), i.e. the keys of the exported
``tokenizer.json``; ``id_to_bytes`` gives the raw bytes. ``get_vocab`` is exactly that file's
vocabulary, where a special token can also keep its byte-level key (a one-byte special "\\n" is
both "\\n" and "Ċ"), so it can have more entries than ``vocab_size``.

``vocab_size`` passed to ``train`` is the requested total size, special tokens included. Training
never pads: when the corpus cannot supply that many tokens the real, smaller vocabulary is kept
and a ``UserWarning`` reports its size.

``pretokenizer`` (``train`` of every class except SuperBPE and ByteEntropy) is how text is split
into segments that no token spans, for training and encoding alike: ``"whitespace"`` (default)
puts every maximal Unicode whitespace run in a segment of its own, so no token mixes whitespace
with other text (a run such as ``"    "`` or ``" \\n"`` can still be one token); ``"space_prefix"``
(GPT-2-style) lets a word take the one U+0020 space before it (" the"), while other whitespace
stays in whitespace-only segments; ``"gpt2"`` is GPT-2's exact regex pre-split (tiktoken's gpt2
``pat_str``): contractions, letter runs, digit runs and punctuation runs apart, each with the
space before it (" it", "'s", " 2024", "!"); ``{"hf": config}`` is a Hugging Face ``pre_tokenizer``
config (a dict or JSON string, e.g. ``{"type": "Digits", "individual_digits": True}``) run by
``tokenizers`` itself, limited to splitters that keep every character (``Split``, ``Punctuation``,
``Digits``, ``FixedLength``, ``ByteLevel`` without ``add_prefix_space``, ``Sequence``); anything
else, or a field the component does not have (a misspelt ``"behaviour"``), raises ``ValueError``. The exported ``tokenizer.json`` reproduces it, and ``from_pretrained``
restores it. ``train`` also raises ``ValueError`` for a config that ``tokenizer.json`` would load
back as another pretokenizer with the same split, and names that one: a lone ``ByteLevel`` with
``use_regex=True`` is ``"gpt2"``, ``Sequence[X, ByteLevel(use_regex=False)]`` is ``{"hf": X}``,
and a lone ``ByteLevel(use_regex=False)`` splits nothing (whole text).
``tokenisers._core.pretokenize(data, mode)`` returns the segments of any accepted config.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Union

from tokenisers import _core


def _warn_if_short(name: str, actual: int, requested: Optional[int]) -> None:
    if requested is not None and actual < requested:
        warnings.warn(
            f"{name}: requested vocab_size={requested}, but the corpus only supports "
            f"{actual} tokens; the vocabulary has {actual} tokens",
            UserWarning,
            stacklevel=3,
        )


class _CoreTokenizer:
    """Common API over a compiled ``_core`` model (``PyBpeModel`` / ``PyUnigramModel``)."""

    _core_class: type
    _model_type: str

    def __init__(self, model=None) -> None:
        self._model = model

    def _require(self):
        if self._model is None:
            raise RuntimeError(f"{type(self).__name__} is not trained or loaded")
        return self._model

    @property
    def vocab_size(self) -> int:
        return 0 if self._model is None else self._model.vocab_size()

    @property
    def pretokenizer(self) -> Optional[Union[str, Dict[str, Any]]]:
        """``"whitespace"``, ``"space_prefix"``, ``"gpt2"`` or ``{"hf": config}``; None when the model
        encodes whole text (SuperBPE after stage 2) or is not trained."""
        return None if self._model is None else self._model.pretokenizer()

    def get_vocab(self) -> Dict[str, int]:
        return {} if self._model is None else self._model.get_vocab()

    def token_to_id(self, token: str) -> Optional[int]:
        return None if self._model is None else self._model.token_to_id(token)

    # The core takes u32 ids: one outside that range (e.g. HF's -100 label padding) is unknown, like a too-large one.
    def id_to_token(self, token_id: int) -> Optional[str]:
        return None if self._model is None or not 0 <= token_id < 2**32 else self._model.id_to_token(token_id)

    def id_to_bytes(self, token_id: int) -> Optional[bytes]:
        return None if self._model is None or not 0 <= token_id < 2**32 else self._model.id_to_bytes(token_id)

    def special_tokens(self) -> Dict[str, int]:
        """``{content: id}`` of the special tokens."""
        return {} if self._model is None else self._model.special_tokens()

    def decode_bytes(self, token_ids: Sequence[int]) -> bytes:
        """Raw bytes of ``token_ids``; ``ValueError`` on an unknown id."""
        ids = list(token_ids)
        try:
            return self._require().decode_bytes(ids)
        except OverflowError:  # outside u32, checked only on failure so decode stays one native call
            bad = next(i for i in ids if not 0 <= i < 2**32)
            raise ValueError(f"Token ID {bad} not found in vocabulary") from None

    def decode(self, token_ids: Sequence[int], errors: str = "replace") -> str:
        """UTF-8 text of ``token_ids`` (``errors`` as in ``bytes.decode``)."""
        return self.decode_bytes(token_ids).decode("utf-8", errors=errors)

    def export_hf_json(self, path: str) -> None:
        self._require().export_hf_json(str(path))

    def save_pretrained(self, save_directory: str) -> None:
        """Writes ``tokenizer.json`` (HF format) and ``tokenizer_config.json``."""
        self._require()
        os.makedirs(save_directory, exist_ok=True)
        self.export_hf_json(os.path.join(save_directory, "tokenizer.json"))
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "model_type": self._model_type,
                    "clean_up_tokenization_spaces": False,
                },
                f,
                indent=2,
            )

    save = save_pretrained

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        """Loads a directory written by ``save`` (or a ``tokenizer.json`` path)."""
        return cls(model=cls._core_class.from_hf_json(str(pretrained_model_name_or_path)))


class BpeTokenizer(_CoreTokenizer):
    """
    Byte-level BPE with pluggable merge scorers, BPE-dropout, lossless reconstruction and
    HuggingFace export.
    """

    _core_class = _core.PyBpeModel
    _model_type = "bpe"

    @classmethod
    def train(
        cls,
        corpus: List[str],
        vocab_size: int = 300,
        scorer: str = "frequency",
        min_frequency: Optional[int] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        penalty: Optional[float] = None,
        special_tokens: Optional[List[str]] = None,
        pretokenizer: Union[str, Dict[str, Any]] = "whitespace",
    ):
        """
        Trains on ``corpus``. ``vocab_size`` includes ``special_tokens``, which are cut out of the
        corpus before training (as ``encode`` cuts them out of the text) and take the top ids; a
        one-byte special keeps that byte's id. Scorer parameters left
        at ``None`` take the scorer's default: ``alpha`` (wordpiece smoothing 1.0, wp-bpe-scp
        discount 0.75, wp-bpe-npmi support exponent 0.5), ``beta`` (length_penalized 0.5),
        ``penalty`` (mdl entry cost 24 bits, wp-bpe-ppig length penalty 0.35); ``min_frequency``
        defaults to 2 for wordpiece, mdl and wp-bpe-ft/scp/npmi/ppig, else 1. Passing a parameter
        the scorer does not use (or a negative alpha/penalty) raises ``ValueError``.
        """
        if not corpus:
            raise ValueError("Training corpus cannot be empty")
        model = _core.train_bpe(
            corpus=corpus,
            vocab_size=vocab_size,
            scorer_type=scorer,
            min_frequency=min_frequency,
            alpha=alpha,
            beta=beta,
            penalty=penalty,
            special_tokens=special_tokens,
            pretokenizer=pretokenizer,
        )
        _warn_if_short(cls.__name__, model.vocab_size(), vocab_size)
        return cls(model=model)

    def encode(
        self,
        text: str,
        dropout: Optional[float] = None,
        seed: Optional[int] = None,
        split_special_tokens: bool = False,
    ) -> List[int]:
        """
        BPE ids of ``text``; ``dropout`` is the BPE-dropout probability (HF semantics). Special
        tokens in the text get their ids unless ``split_special_tokens`` (HF semantics: their text
        is encoded as ordinary text).
        """
        return self._require().encode(
            text, dropout=dropout, seed=seed, split_special_tokens=split_special_tokens
        )

    def encode_bytes(
        self,
        data: bytes,
        dropout: Optional[float] = None,
        seed: Optional[int] = None,
        split_special_tokens: bool = False,
    ) -> List[int]:
        return self._require().encode_bytes(
            bytes(data), dropout=dropout, seed=seed, split_special_tokens=split_special_tokens
        )


class WordPieceTokenizer(BpeTokenizer):
    """BPE with the Laplace-smoothed WordPiece likelihood score (``alpha`` smoothing)."""

    @classmethod
    def train(
        cls, corpus, vocab_size=300, alpha=None, min_frequency=None, special_tokens=None, pretokenizer="whitespace"
    ):
        return super().train(
            corpus, vocab_size, "wordpiece", min_frequency=min_frequency, alpha=alpha,
            special_tokens=special_tokens, pretokenizer=pretokenizer,
        )


class WpBpeBaseTokenizer(BpeTokenizer):
    """wp-bpe-base: uncalibrated PMI f(uv) / (f(u) f(v)), no frequency cutoff by default."""

    @classmethod
    def train(
        cls, corpus, vocab_size=300, min_frequency=None, special_tokens=None, pretokenizer="whitespace"
    ):
        return super().train(
            corpus, vocab_size, "wp-bpe-base", min_frequency=min_frequency, special_tokens=special_tokens,
            pretokenizer=pretokenizer,
        )


class WpBpeFtTokenizer(BpeTokenizer):
    """wp-bpe-ft: the wp-bpe-base ratio restricted to pairs with f(uv) >= min_frequency (>= 2)."""

    @classmethod
    def train(
        cls, corpus, vocab_size=300, min_frequency=None, special_tokens=None, pretokenizer="whitespace"
    ):
        return super().train(
            corpus, vocab_size, "wp-bpe-ft", min_frequency=min_frequency, special_tokens=special_tokens,
            pretokenizer=pretokenizer,
        )


class WpBpePigTokenizer(BpeTokenizer):
    """wp-bpe-pig: pointwise information gain f(uv) * PMI(u, v)."""

    @classmethod
    def train(
        cls, corpus, vocab_size=300, min_frequency=None, special_tokens=None, pretokenizer="whitespace"
    ):
        return super().train(
            corpus, vocab_size, "wp-bpe-pig", min_frequency=min_frequency, special_tokens=special_tokens,
            pretokenizer=pretokenizer,
        )


class WpBpeScpTokenizer(BpeTokenizer):
    """wp-bpe-scp: (f(uv) - discount)+ / sqrt(f(u) f(v)) (default discount 0.75)."""

    @classmethod
    def train(
        cls, corpus, vocab_size=300, min_frequency=None, discount=None, special_tokens=None, pretokenizer="whitespace"
    ):
        return super().train(
            corpus, vocab_size, "wp-bpe-scp", min_frequency=min_frequency, alpha=discount,
            special_tokens=special_tokens, pretokenizer=pretokenizer,
        )


class WpBpeNpmiTokenizer(BpeTokenizer):
    """wp-bpe-npmi: normalized PMI weighted by f(uv)^alpha (default alpha 0.5)."""

    @classmethod
    def train(
        cls, corpus, vocab_size=300, min_frequency=None, alpha=None, special_tokens=None, pretokenizer="whitespace"
    ):
        return super().train(
            corpus, vocab_size, "wp-bpe-npmi", min_frequency=min_frequency, alpha=alpha,
            special_tokens=special_tokens, pretokenizer=pretokenizer,
        )


class WpBpePpigTokenizer(BpeTokenizer):
    """wp-bpe-ppig: f(uv) * max(0, PMI) / (|u| + |v|)^length_penalty (default 0.35)."""

    @classmethod
    def train(
        cls, corpus, vocab_size=300, min_frequency=None, length_penalty=None, special_tokens=None,
        pretokenizer="whitespace",
    ):
        return super().train(
            corpus, vocab_size, "wp-bpe-ppig", min_frequency=min_frequency, penalty=length_penalty,
            special_tokens=special_tokens, pretokenizer=pretokenizer,
        )


class MdlTokenizer(BpeTokenizer):
    """BPE with the Minimum Description Length (delta-bits) score; ``entry_cost`` in bits (24)."""

    @classmethod
    def train(
        cls, corpus, vocab_size=300, entry_cost=None, min_frequency=None, special_tokens=None, pretokenizer="whitespace"
    ):
        return super().train(
            corpus, vocab_size, "mdl", min_frequency=min_frequency, penalty=entry_cost,
            special_tokens=special_tokens, pretokenizer=pretokenizer,
        )


class LengthPenalizedTokenizer(BpeTokenizer):
    """BPE with score f(uv) / (|u| + |v|)^beta (default beta 0.5)."""

    @classmethod
    def train(
        cls, corpus, vocab_size=300, beta=None, min_frequency=None, special_tokens=None, pretokenizer="whitespace"
    ):
        return super().train(
            corpus, vocab_size, "length_penalized", min_frequency=min_frequency, beta=beta,
            special_tokens=special_tokens, pretokenizer=pretokenizer,
        )


class SuperBpeTokenizer(BpeTokenizer):
    """Two-stage SuperBPE: whitespace-bounded merges up to v_switch, cross-word merges up to v_final."""

    @classmethod
    def train(
        cls,
        corpus: List[str],
        vocab_size: Optional[int] = None,
        *,
        v_switch: Optional[int] = None,
        v_final: Optional[int] = None,
    ) -> SuperBpeTokenizer:
        """
        ``vocab_size`` is ``v_final`` (default 320). ``v_switch`` defaults to 90% of the learned
        merges (256 + 0.9 * (v_final - 256)), the best transition point in the SuperBPE paper.
        """
        if vocab_size is not None and v_final is not None and vocab_size != v_final:
            raise ValueError(f"vocab_size ({vocab_size}) and v_final ({v_final}) disagree")
        if v_final is None:
            v_final = 320 if vocab_size is None else vocab_size
        if v_switch is None:
            v_switch = 256 + (max(v_final, 256) - 256) * 9 // 10
        model = _core.train_super_bpe(corpus, v_switch, v_final)
        _warn_if_short(cls.__name__, model.vocab_size(), v_final)
        return cls(model=model)


class MorphemeGuidedTokenizer(BpeTokenizer):
    """BPE Tokenizer regularized by morphological boundary priors.

    Every candidate merge's score is scaled by exp(lambda_weight * M), where M in [-1, 1] measures
    whether the token it would create stays inside one morpheme across the pair's current
    occurrences in lexicon words (+1) or spans a boundary (-1). Pairs whose reweighted count
    falls below one occurrence are never merged, so training may stop below vocab_size (with a
    warning). ``morpheme_lexicon`` maps whitespace-free training words to surface morphemes that
    concatenate to the word; supply it from training annotations or an unsupervised segmenter,
    never from evaluation gold. ``alpha`` (wordpiece), ``beta`` (length_penalized) and
    ``penalty`` (mdl entry cost in bits) default to the scorer's own defaults. With
    ``pretokenizer="space_prefix"`` or ``"gpt2"`` the segment " word" gets the boundaries of "word"
    (the space joins the first morpheme) unless the lexicon has its own " word" entry; in ``"gpt2"``
    a word the regex splits ("don't") is never a training segment and has no effect.
    """

    @classmethod
    def train(
        cls,
        corpus: List[str],
        vocab_size: int = 300,
        morpheme_lexicon: Optional[Dict[str, List[str]]] = None,
        lambda_weight: float = 1.0,
        scorer: str = "frequency",
        min_frequency: int = 1,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        penalty: Optional[float] = None,
        special_tokens: Optional[List[str]] = None,
        pretokenizer: Union[str, Dict[str, Any]] = "whitespace",
    ) -> MorphemeGuidedTokenizer:
        if not corpus:
            raise ValueError("Training corpus cannot be empty")
        model = _core.train_morpheme_guided(
            corpus,
            vocab_size,
            morpheme_lexicon or {},
            lambda_weight=float(lambda_weight),
            scorer=scorer,
            min_frequency=min_frequency,
            alpha=alpha,
            beta=beta,
            penalty=penalty,
            special_tokens=special_tokens, pretokenizer=pretokenizer,
        )
        return cls(model=model)


MorphemeBpeTokenizer = MorphemeGuidedTokenizer


class UnigramTokenizer(_CoreTokenizer):
    """
    Unigram LM tokenizer: EM training with loss-based pruning, Viterbi decoding and subword
    regularization (FFBS / k-best lattice sampling).
    """

    _core_class = _core.PyUnigramModel
    _model_type = "unigram"

    def get_log_probs(self) -> Dict[str, float]:
        return {} if self._model is None else self._model.get_log_probs()

    @classmethod
    def train(
        cls,
        corpus: List[str],
        vocab_size: int = 300,
        num_iterations: int = 10,
        shrinking_factor: float = 0.75,
        pretokenizer: Union[str, Dict[str, Any]] = "whitespace",
    ) -> UnigramTokenizer:
        if not corpus:
            raise ValueError("Training corpus cannot be empty")
        model = _core.train_unigram(
            corpus=corpus,
            vocab_size=vocab_size,
            num_iterations=num_iterations,
            shrinking_factor=float(shrinking_factor),
            pretokenizer=pretokenizer,
        )
        _warn_if_short(cls.__name__, model.vocab_size(), vocab_size)
        return cls(model=model)

    def encode(
        self,
        text: str,
        sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        nbest_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[int]:
        """
        Viterbi ids, or a sampled segmentation (subword regularization). ``sample`` defaults to
        True when ``temperature`` is given or ``nbest_size`` is > 1 or < 0. A sample has
        probability proportional to P(s)^(1/temperature): temperature 1 (default) is the exact
        posterior, SentencePiece ``alpha`` = 1/temperature, 0 gives Viterbi. ``nbest_size``
        (SentencePiece convention): None or < 0 samples from all segmentations (FFBS), k > 1 from
        the k best of each pretokenizer segment, 0 or 1 is Viterbi.
        """
        return self._require().encode(
            text, temperature=temperature, nbest_size=nbest_size, sample=sample, seed=seed
        )

    def encode_bytes(
        self,
        data: bytes,
        sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        nbest_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[int]:
        return self._require().encode_bytes(
            bytes(data), temperature=temperature, nbest_size=nbest_size, sample=sample, seed=seed
        )


class ByteEntropyTokenizer:
    """
    BLT-style dynamic patching: a backoff Markov model of next-byte entropy cuts a new patch before
    a byte whose predictive entropy exceeds ``entropy_threshold`` (bits), within
    ``min_patch_len <= len <= max_patch_len``.

    ``train`` fits the Markov model and freezes a patch vocabulary: ids 0..255 are the single bytes,
    then the most frequent multi-byte training patches. A patch outside the vocabulary is encoded
    as its single bytes, so encoding is lossless and deterministic. The Markov table keeps at most
    ``max_contexts`` contexts (~200 bytes each), which bounds training memory; once full, later
    training text only updates contexts already in it.
    """

    _FILE = "byte_entropy.json"

    def __init__(
        self,
        entropy_threshold: float = 4.2,
        min_patch_len: int = 2,
        max_patch_len: int = 16,
        context_order: int = 3,
        smoothing_alpha: float = 0.05,
        max_contexts: Optional[int] = None,
    ) -> None:
        """An untrained tokenizer; use ``ByteEntropyTokenizer.train`` or ``from_pretrained``.
        ``max_contexts`` None is the core default (2M contexts, about 400 MB)."""
        self.config = {
            "entropy_threshold": float(entropy_threshold),
            "min_patch_len": int(min_patch_len),
            "max_patch_len": int(max_patch_len),
            "context_order": int(context_order),
            "smoothing_alpha": float(smoothing_alpha),
            "max_contexts": None if max_contexts is None else int(max_contexts),
        }
        limit = {} if max_contexts is None else {"max_contexts": self.config["max_contexts"]}
        self._patcher = _core.PyByteEntropyPatcher(
            entropy_threshold=self.config["entropy_threshold"],
            context_order=self.config["context_order"],
            min_len=self.config["min_patch_len"],
            max_len=self.config["max_patch_len"],
            smoothing_alpha=self.config["smoothing_alpha"],
            **limit,
        )
        self._patches: List[bytes] = []
        self._patch_ids: Dict[bytes, int] = {}

    def _set_patches(self, patches: List[bytes]) -> None:
        self._patches = patches
        self._patch_ids = {p: 256 + i for i, p in enumerate(patches)}

    @classmethod
    def train(
        cls,
        corpus: List[Union[str, bytes]],
        vocab_size: Optional[int] = None,
        entropy_threshold: float = 4.2,
        min_patch_len: int = 2,
        max_patch_len: int = 16,
        context_order: int = 3,
        smoothing_alpha: float = 0.05,
        max_contexts: Optional[int] = None,
    ) -> ByteEntropyTokenizer:
        """``vocab_size`` caps 256 + patches (most frequent first); None keeps every patch seen."""
        if not corpus:
            raise ValueError("Training corpus cannot be empty")
        if vocab_size is not None and vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")
        tok = cls(entropy_threshold, min_patch_len, max_patch_len, context_order, smoothing_alpha, max_contexts)
        docs = [d.encode("utf-8") if isinstance(d, str) else bytes(d) for d in corpus]
        for doc in docs:
            tok._patcher.train_transitions(doc)
        if not tok._patcher.is_trained():
            raise ValueError("Corpus contains only empty documents")
        counts = Counter(p for doc in docs for p in tok._patcher.segment_stream(doc) if len(p) > 1)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if vocab_size is not None:
            ranked = ranked[: vocab_size - 256]
        tok._set_patches([p for p, _ in ranked])
        _warn_if_short(cls.__name__, tok.vocab_size, vocab_size)
        return tok

    def _require(self) -> None:
        if not self._patcher.is_trained():
            raise RuntimeError("ByteEntropyTokenizer is not trained; use ByteEntropyTokenizer.train(corpus)")

    @property
    def vocab_size(self) -> int:
        return 256 + len(self._patches)

    def encode_bytes(self, data: bytes) -> List[int]:
        self._require()
        ids: List[int] = []
        for patch in self._patcher.segment_stream(bytes(data)):
            pid = self._patch_ids.get(patch)
            if pid is None:
                ids.extend(patch)  # unseen patch: its single-byte ids
            else:
                ids.append(pid)
        return ids

    def encode(self, text: str) -> List[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def id_to_bytes(self, token_id: int) -> Optional[bytes]:
        if 0 <= token_id < 256:
            return bytes([token_id])
        if 256 <= token_id < self.vocab_size:
            return self._patches[token_id - 256]
        return None

    def decode_bytes(self, token_ids: Sequence[int]) -> bytes:
        """Raw bytes of ``token_ids``; ``ValueError`` on an unknown id."""
        out = []
        for tid in token_ids:
            data = self.id_to_bytes(tid)
            if data is None:
                raise ValueError(f"Invalid token ID: {tid}")
            out.append(data)
        return b"".join(out)

    def decode(self, token_ids: Sequence[int], errors: str = "replace") -> str:
        return self.decode_bytes(token_ids).decode("utf-8", errors=errors)

    def id_to_token(self, token_id: int) -> Optional[str]:
        data = self.id_to_bytes(token_id)
        return None if data is None else _core.bytes_to_printable(data)

    def token_to_id(self, token: str) -> Optional[int]:
        try:
            data = _core.printable_to_bytes(token)
        except ValueError:
            return None
        return data[0] if len(data) == 1 else self._patch_ids.get(data)

    def get_vocab(self) -> Dict[str, int]:
        return {self.id_to_token(i): i for i in range(self.vocab_size)}

    def special_tokens(self) -> Dict[str, int]:
        return {}

    def save(self, save_directory: str) -> None:
        """Writes config, patch vocabulary and Markov counts to ``byte_entropy.json``."""
        self._require()
        os.makedirs(save_directory, exist_ok=True)
        state = {
            "type": "ByteEntropy",
            "config": self.config,
            "patches": [p.hex() for p in self._patches],
            "transitions": [
                [ctx.hex(), counts] for ctx, counts in self._patcher.transitions()
            ],
        }
        with open(os.path.join(save_directory, self._FILE), "w", encoding="utf-8") as f:
            json.dump(state, f)

    save_pretrained = save

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> ByteEntropyTokenizer:
        path = str(pretrained_model_name_or_path)
        if os.path.isdir(path):
            path = os.path.join(path, cls._FILE)
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        if state.get("type") != "ByteEntropy":
            raise ValueError(f"{path} is not a saved ByteEntropyTokenizer")
        tok = cls(**state["config"])
        tok._patcher.load_transitions(
            [(bytes.fromhex(ctx), [tuple(bc) for bc in counts]) for ctx, counts in state["transitions"]]
        )
        tok._set_patches([bytes.fromhex(p) for p in state["patches"]])
        return tok
