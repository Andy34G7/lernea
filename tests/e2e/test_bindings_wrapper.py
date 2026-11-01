"""DynamicResearchTokenizer (transformers ``PreTrainedTokenizer`` adapter): ids equal the native
tokenizer's, token strings are real byte-level tokens, and the HF base behaviour (batching,
padding, tensors, pairs) is not overridden into something else."""

import warnings

import numpy as np
import pytest
import transformers

import tokenisers

CORPUS = ["Hello\tworld 9 the quick brown fox jumps over the lazy dog " * 3, "Héllo wörld 日本語 🚀"]
TEXTS = ["Hello\tworld 9", "Héllo wörld 日本語 🚀", "the quick  brown\n", "x"]


@pytest.fixture(scope="module", params=["bpe", "unigram"])
def pair(request):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tiny corpus: vocab may stop short of 300
        cls = tokenisers.BpeTokenizer if request.param == "bpe" else tokenisers.UnigramTokenizer
        tok = cls.train(CORPUS, 300)
    return tok, tokenisers.DynamicResearchTokenizer(tok)


def test_ids_and_tokens_are_the_native_ones(pair):
    """P15: _tokenize returned str(id), so "\\t" (id 9) became the digit "9" (id 57) and
    convert_tokens_to_string gave '72 306 ...'; wrapping a UnigramTokenizer raised AttributeError."""
    tok, wrapper = pair
    for text in TEXTS:
        ids = tok.encode(text)
        assert wrapper(text)["input_ids"] == ids
        tokens = wrapper.tokenize(text)
        assert tokens == [tok.id_to_token(i) for i in ids]
        assert wrapper.convert_tokens_to_ids(tokens) == ids
        assert wrapper.convert_tokens_to_string(tokens) == text
        assert wrapper.decode(ids) == text
    assert wrapper("\t")["input_ids"] == [9]


def test_hf_base_behaviour_batching_padding_tensors_pairs(pair):
    """P16: batch_decode returned one joined string; padding_side, return_tensors and text_pair
    were ignored."""
    tok, wrapper = pair
    a, b = tok.encode("ab cd"), tok.encode("the quick brown")
    assert wrapper.batch_decode([a, b]) == ["ab cd", "the quick brown"]

    pad = wrapper.pad_token_id
    wrapper.padding_side = "left"
    try:
        out = wrapper(["ab cd", "the quick brown"], padding=True)
    finally:
        wrapper.padding_side = "right"
    n = len(b) - len(a)
    assert out["input_ids"] == [[pad] * n + a, b]
    assert out["attention_mask"] == [[0] * n + [1] * len(a), [1] * len(b)]
    right = wrapper(["ab cd", "the quick brown"], padding=True)
    assert right["input_ids"][0] == a + [pad] * n

    arr = wrapper(["ab cd", "the quick brown"], padding=True, return_tensors="np")["input_ids"]
    assert isinstance(arr, np.ndarray) and arr.shape == (2, len(b))

    assert wrapper("ab cd", "the quick brown")["input_ids"] == a + b
    # GPT-2 convention (no post-processor in the exported tokenizer.json): nothing is added
    assert wrapper("ab cd", add_special_tokens=True)["input_ids"] == a


def test_special_tokens_in_text_use_reserved_ids(pair):
    """P2: wrapper specials sit above the model vocab and are split out of the text."""
    tok, wrapper = pair
    ids = wrapper("x</s>y<pad>")["input_ids"]
    assert ids == tok.encode("x") + [wrapper.eos_token_id] + tok.encode("y") + [wrapper.pad_token_id]
    assert min(wrapper.all_special_ids) >= tok.vocab_size
    assert wrapper.decode(ids, skip_special_tokens=True) == "xy"


def test_from_pretrained_reloads_the_underlying_model(pair, tmp_path):
    """P14: from_pretrained gave _underlying=None: input_ids == [] and len == 0, no error."""
    tok, wrapper = pair
    wrapper.save_pretrained(str(tmp_path))
    reloaded = tokenisers.DynamicResearchTokenizer.from_pretrained(str(tmp_path))
    assert type(reloaded._underlying) is type(tok)
    assert len(reloaded) == len(wrapper) > tok.vocab_size
    for text in TEXTS + ["a</s>b"]:
        assert reloaded(text)["input_ids"] == wrapper(text)["input_ids"]


def test_added_token_lstrip_rstrip_single_word_are_honoured(pair):
    """The tokenize override dropped the base class's AddedToken handling: a RoBERTa-style
    <mask> with lstrip/rstrip kept its neighbouring spaces ('a', 'Ġ', '<X>', 'Ġ', 'b')."""
    tok, _ = pair
    wrapper = tokenisers.DynamicResearchTokenizer(tok)  # keep added tokens off the shared fixture
    wrapper.add_tokens([transformers.AddedToken("<X>", lstrip=True, rstrip=True, special=True),
                        transformers.AddedToken("<W>", single_word=True)])
    x, w = wrapper.convert_tokens_to_ids(["<X>", "<W>"])
    assert wrapper.tokenize("a <X> b") == [tok.id_to_token(i) for i in tok.encode("a")] + ["<X>"] + [
        tok.id_to_token(i) for i in tok.encode("b")]
    assert wrapper("a <X> b")["input_ids"] == tok.encode("a") + [x] + tok.encode("b")
    # single_word: "<W>" glued to a word is text, a standalone "<W>" is the added token.
    assert wrapper("x<W> <W>")["input_ids"] == tok.encode("x<W>") + tok.encode(" ") + [w]
    # Per-call encode options still reach the pieces around the added tokens.
    opts = {"dropout": 1.0} if isinstance(tok, tokenisers.BpeTokenizer) else {"temperature": 1.0, "seed": 3}
    assert wrapper("hello <X> world", **opts)["input_ids"] == tok.encode("hello", **opts) + [x] + tok.encode("world", **opts)


def test_needs_a_model():
    with pytest.raises(ValueError):
        tokenisers.DynamicResearchTokenizer()


def test_special_token_that_is_a_learned_text_token_is_rejected():
    """HF reuses the id of a vocab string for a special token: with "<s>", "</s>", "<pad>" learned
    from the corpus, bos/eos/pad became text tokens, padding inserted text, skip_special_tokens
    deleted real "<pad>" text and "<s>strike</s>" was split differently from native encoding."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tok = tokenisers.BpeTokenizer.train(["<s>strike</s> and <s>more</s> text <pad> here " * 50], 300)
    assert tok.token_to_id("<pad>") is not None and tok.token_to_id("<s>") is not None
    with pytest.raises(ValueError, match="learned text token"):
        tokenisers.DynamicResearchTokenizer(tok)

    wrapper = tokenisers.DynamicResearchTokenizer(tok, bos_token="<|bos|>", eos_token="<|eos|>", pad_token="<|pad|>")
    assert min(wrapper.all_special_ids) >= tok.vocab_size
    text = "<s>strike</s> and <pad> here"
    ids = tok.encode(text)
    assert wrapper(text)["input_ids"] == ids
    assert wrapper.decode(ids, skip_special_tokens=True) == text
    with pytest.raises(ValueError, match="learned text token"):
        wrapper.add_special_tokens({"pad_token": "<pad>"})
    assert wrapper.pad_token == "<|pad|>"

    # A special token of the model itself is used with its own id
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        own = tokenisers.BpeTokenizer.train(["<s>strike</s> and <s>more</s> text <pad> here " * 50], 300,
                                            special_tokens=["<pad>"])
    wrapper = tokenisers.DynamicResearchTokenizer(own, bos_token="<|bos|>", eos_token="<|eos|>")
    assert wrapper.pad_token_id == own.special_tokens()["<pad>"]
    assert wrapper(text)["input_ids"] == own.encode(text)


def test_setting_a_learned_text_token_as_a_special_attribute_is_rejected(pair):
    """Only __init__/add_special_tokens were checked: `w.pad_token = "el"` (a learned token) made
    padding insert real "el" tokens and skip_special_tokens delete them from decoded text."""
    tok, wrapper = pair
    learned = tok.id_to_token(256)
    for attr, value in [("pad_token", learned), ("pad_token_id", 256), ("bos_token", learned),
                        ("extra_special_tokens", [learned])]:
        with pytest.raises(ValueError, match="learned text token"):
            setattr(wrapper, attr, value)
    assert wrapper.pad_token == "<pad>" and 256 not in wrapper.all_special_ids
    assert wrapper.decode(tok.encode("Hello world"), skip_special_tokens=True) == "Hello world"


def test_model_special_tokens_are_special_like_in_auto_tokenizer(tmp_path):
    """The model's own specials were not HF specials unless named bos/eos/pad/unk:
    skip_special_tokens kept "<|endoftext|>", which AutoTokenizer on the export drops."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tok = tokenisers.BpeTokenizer.train(CORPUS, 300, special_tokens=["<|endoftext|>", "<pad>"])
    tok.save(str(tmp_path / "model"))
    auto = transformers.AutoTokenizer.from_pretrained(str(tmp_path / "model"))
    wrapper = tokenisers.DynamicResearchTokenizer(tok)
    ids = tok.encode("hello<|endoftext|>world<pad>")
    assert auto.decode(ids, skip_special_tokens=True) == "helloworld"
    assert wrapper.decode(ids, skip_special_tokens=True) == "helloworld"
    assert set(tok.special_tokens().values()) <= set(wrapper.all_special_ids)
    assert wrapper("hello<|endoftext|>world<pad>")["input_ids"] == ids

    wrapper.save_pretrained(str(tmp_path / "wrapper"))
    reloaded = tokenisers.DynamicResearchTokenizer.from_pretrained(str(tmp_path / "wrapper"))
    assert reloaded.all_special_ids == wrapper.all_special_ids
    assert reloaded.decode(ids, skip_special_tokens=True) == "helloworld"


def test_saving_a_model_without_a_tokenizer_json_export_raises_clearly(tmp_path):
    """Saving a wrapped ByteEntropyTokenizer died with AttributeError 'export_hf_json'."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tok = tokenisers.ByteEntropyTokenizer.train(CORPUS, 300)
    wrapper = tokenisers.DynamicResearchTokenizer(tok)
    assert wrapper("Hello world")["input_ids"] == tok.encode("Hello world")
    with pytest.raises(NotImplementedError, match="ByteEntropyTokenizer has no tokenizer.json export"):
        wrapper.save_pretrained(str(tmp_path))


def test_declared_transformers_floor_has_the_api_the_wrapper_uses():
    """wrapper.py calls the v5-only add_special_tokens(..., replace_extra_special_tokens=False): on
    transformers 4.57 wrapping a model with its own special tokens raised TypeError, and 5.0.0 fails
    this file. pyproject.toml must not admit those versions."""
    import re
    from pathlib import Path

    pyproject = (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    floor = re.search(r'"transformers>=([\d.]+)"', pyproject).group(1)
    assert tuple(map(int, floor.split("."))) >= (5, 2, 0)
