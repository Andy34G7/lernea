"""
Hugging Face ``PreTrainedTokenizer`` adapter for tokenisers models.

Token strings are the underlying tokenizer's (GPT-2 byte-level printable form, special tokens by
content), so ids round-trip exactly. Padding, truncation, tensors, pairs, batch decoding and the
added-token split (``AddedToken`` lstrip/rstrip/single_word, ``split_special_tokens``) are the
``PreTrainedTokenizer`` base behaviour. As in GPT-2 (and the exported tokenizer.json, which has
no post-processor), no special-token template exists: ``add_special_tokens=True`` adds nothing,
and bos/eos must be written into the text. Per-call encode options (``dropout``, ``seed``,
``temperature``, ``nbest_size``, ``sample``) are forwarded to the underlying ``encode``. Special
tokens are the underlying model's own special tokens (their ids; all of them are registered as
``extra_special_tokens``, as ``AutoTokenizer`` registers the exported ``added_tokens``) or new ids
from ``vocab_size`` upwards. A special token whose string is a learned text token of the underlying
model (e.g. a corpus that contains "<pad>"), whether passed to ``__init__``, ``add_special_tokens``
or set as an attribute (``pad_token``, ``pad_token_id``, ...), raises ``ValueError``: Hugging Face
would reuse that text token's id, so padding would insert real text and ``skip_special_tokens``
would delete it. Pass another string (``pad_token="<|pad|>"``) or train the model with it in
``special_tokens``. Saving needs a model with a tokenizer.json export (``export_hf_json``).
Needs transformers >= 5.2 (the v5 ``extra_special_tokens`` API; pyproject.toml).
"""

from __future__ import annotations

import json
import os
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

_ENCODE_OPTIONS = ("dropout", "seed", "temperature", "nbest_size", "sample")
# The base-class ``tokenize`` calls ``_tokenize(piece)`` with no options; this carries them.
_CALL_OPTIONS: ContextVar[Dict[str, Any]] = ContextVar("_CALL_OPTIONS", default={})


def _load_underlying(vocab_file: str):
    from tokenisers.models import BpeTokenizer, UnigramTokenizer

    with open(vocab_file, encoding="utf-8") as f:
        model_type = json.load(f)["model"]["type"]
    classes = {"BPE": BpeTokenizer, "Unigram": UnigramTokenizer}
    if model_type not in classes:
        raise ValueError(f"{vocab_file}: unsupported model type {model_type!r}")
    return classes[model_type].from_pretrained(vocab_file)


class DynamicResearchTokenizer(PreTrainedTokenizer):
    """``PreTrainedTokenizer`` over a trained tokenisers tokenizer (or a saved ``tokenizer.json``)."""

    vocab_files_names = {"vocab_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        vocab_file: Optional[str] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        **kwargs: Any,
    ) -> None:
        if tokenizer is None:
            if vocab_file is None:
                raise ValueError("DynamicResearchTokenizer needs a trained tokenizer or a tokenizer.json vocab_file")
            tokenizer = _load_underlying(vocab_file)
        self._underlying = tokenizer
        # Byte-level decoding is exact; HF's space clean-up would corrupt it.
        kwargs["clean_up_tokenization_spaces"] = False
        super().__init__(unk_token=unk_token, pad_token=pad_token, bos_token=bos_token, eos_token=eos_token, **kwargs)
        own = list(tokenizer.special_tokens())
        if own:
            self.add_special_tokens({"extra_special_tokens": own}, replace_extra_special_tokens=False)

    def __setattr__(self, key: str, value: Any) -> None:
        # The base class stores `w.pad_token = ...` (or `pad_token_id`) without adding the token.
        name = key.removesuffix("_ids").removesuffix("_id")
        if value is not None and (name in self.SPECIAL_TOKENS_ATTRIBUTES or name == "extra_special_tokens"):
            values = list(value) if isinstance(value, (list, tuple)) else [value]
            self._reject_learned(self.convert_ids_to_tokens(values) if name != key else values)
        super().__setattr__(key, value)

    @property
    def vocab_size(self) -> int:
        return self._underlying.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        vocab = self._underlying.get_vocab()
        vocab.update(getattr(self, "_added_tokens_encoder", {}))
        return vocab

    def tokenize(self, text: str, **kwargs: Any) -> List[str]:
        """The base-class ``tokenize`` (added-token split with lstrip/rstrip/single_word,
        ``split_special_tokens``) with the per-call encode options passed on to ``_tokenize``."""
        options = {k: kwargs.pop(k) for k in _ENCODE_OPTIONS if k in kwargs}
        reset = _CALL_OPTIONS.set(options)
        try:
            return super().tokenize(text, **kwargs)
        finally:
            _CALL_OPTIONS.reset(reset)

    def _reject_learned(self, specials) -> None:
        own = self._underlying.special_tokens()
        for token in specials:
            content = str(token)
            learned = None if content in own else self._underlying.token_to_id(content)
            if learned is not None:
                raise ValueError(
                    f"special token {content!r} is the learned text token id {learned} of the "
                    "underlying tokenizer; pass a string that is not in its vocabulary, or train "
                    "with it in special_tokens"
                )

    def _add_tokens(self, new_tokens, special_tokens: bool = False) -> int:
        self._reject_learned(t for t in new_tokens or () if special_tokens or getattr(t, "special", False))
        return super()._add_tokens(new_tokens, special_tokens=special_tokens)

    def add_special_tokens(self, special_tokens_dict, *args: Any, **kwargs: Any) -> int:
        # Checked up front: the base class sets e.g. pad_token before adding it.
        values = special_tokens_dict.values()
        self._reject_learned(t for v in values for t in (v if isinstance(v, (list, tuple)) else [v]))
        return super().add_special_tokens(special_tokens_dict, *args, **kwargs)

    def _tokenize(self, text: str, **kwargs: Any) -> List[str]:
        options = dict(_CALL_OPTIONS.get())
        if self._underlying.special_tokens():
            # Like the HF model step, never match specials here: the added-token split in
            # `tokenize` has cut them out, or with split_special_tokens they are text.
            options["split_special_tokens"] = True
        return [self._underlying.id_to_token(i) for i in self._underlying.encode(text, **options)]

    def _convert_token_to_id(self, token: str) -> int:
        token_id = self._underlying.token_to_id(token)
        return self.unk_token_id if token_id is None else token_id

    def _convert_id_to_token(self, index: int) -> str:
        token = self._underlying.id_to_token(index)
        if token is None:
            raise ValueError(f"Invalid token ID: {index}")
        return token

    def _id_bytes(self, index: int) -> bytes:
        data = self._underlying.id_to_bytes(index)
        if data is not None:
            return data
        if index in self._added_tokens_decoder:
            return self._added_tokens_decoder[index].content.encode("utf-8")
        raise ValueError(f"Invalid token ID: {index}")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        ids = [self._convert_token_to_id_with_added_voc(t) for t in tokens]
        return b"".join(self._id_bytes(i) for i in ids).decode("utf-8", errors="replace")

    def _decode(self, token_ids, skip_special_tokens: bool = False, **kwargs: Any) -> str:
        ids = [token_ids] if isinstance(token_ids, int) else list(token_ids)
        if skip_special_tokens:
            special = set(self.all_special_ids)
            ids = [i for i in ids if i not in special]
        return b"".join(self._id_bytes(i) for i in ids).decode("utf-8", errors="replace")

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:
        os.makedirs(save_directory, exist_ok=True)
        name = f"{filename_prefix}-tokenizer.json" if filename_prefix else "tokenizer.json"
        path = os.path.join(save_directory, name)
        if not hasattr(self._underlying, "export_hf_json"):
            raise NotImplementedError(
                f"{type(self._underlying).__name__} has no tokenizer.json export; save it with its own save()"
            )
        self._underlying.export_hf_json(path)
        return (path,)
