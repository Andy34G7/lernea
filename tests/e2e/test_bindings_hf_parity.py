"""HF parity through the high-level classes (U1/C10/P2): ``save`` writes a tokenizer.json that
Hugging Face ``tokenizers`` and ``transformers`` load to the same ids, the same token strings and
the same special-token ids as native encoding, and ``from_pretrained`` reloads it unchanged."""

import json

import pytest
import tokenizers
import transformers

import tokenisers

SPECIALS = ["<|endoftext|>", "<pad>"]
CORPUS = [
    f"{d} {r}"
    for i, d in enumerate([
        "The quick brown fox jumps over the lazy dog. Don't stop!",
        "def foo(x):\n    return x + 1  # comment\n\n\tif x: pass",
        "Héllo wörld, naïve café résumé façade",
        "日本語のテキスト、中文文本。한국어 텍스트",
        "中文　文本、中文。日本語　の　テキスト",
        "emoji 🚀🔥💯 party 🎉🎉 👩‍👩‍👧",
        "   leading spaces   and   runs\t\ttabs\n\nnewlines\r\n",
        "non breaking space and thin　ideographic",
    ])
    for r in range(4 + i)
]
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "def foo(x):\n    return x + 1",
    "Héllo wörld, naïve café",
    "日本語のテキスト 中文文本",
    "中文　文本、中文。",
    "emoji 🚀🔥 👩‍👩‍👧",
    "a  b   c\t\td\r\n",
    "non breaking space thin　ideo\u0085nel ogham",
    "hello<|endoftext|>world",
    "<pad><pad> x <|endoftext|>日本<pad>語",
    " ",
]

MAKERS = {
    "bpe+specials": lambda: tokenisers.BpeTokenizer.train(CORPUS, 420, special_tokens=SPECIALS),
    "wordpiece+specials": lambda: tokenisers.WordPieceTokenizer.train(CORPUS, 420, special_tokens=SPECIALS),
    "bpe+byte-special": lambda: tokenisers.BpeTokenizer.train(CORPUS, 420, special_tokens=["\n", "<|endoftext|>"]),
    "wp-bpe-npmi": lambda: tokenisers.WpBpeNpmiTokenizer.train(CORPUS, 420),
    "mdl": lambda: tokenisers.MdlTokenizer.train(CORPUS, 420),
    "super_bpe": lambda: tokenisers.SuperBpeTokenizer.train(CORPUS, vocab_size=460),
    "morpheme": lambda: tokenisers.MorphemeBpeTokenizer.train(CORPUS, 420, morpheme_lexicon={}),
    "unigram": lambda: tokenisers.UnigramTokenizer.train(CORPUS, 500),
}


@pytest.fixture(scope="module", params=list(MAKERS))
def saved(request, tmp_path_factory):
    tok = MAKERS[request.param]()
    out = tmp_path_factory.mktemp(request.param.replace("+", "_").replace("-", "_"))
    tok.save(str(out))
    return request.param, tok, out


def test_hf_ids_equal_native(saved):
    name, tok, out = saved
    hf = tokenizers.Tokenizer.from_file(str(out / "tokenizer.json"))
    auto = transformers.AutoTokenizer.from_pretrained(str(out))
    for text in TEXTS:
        ids = tok.encode(text)
        assert hf.encode(text, add_special_tokens=False).ids == ids, (name, text)
        assert auto.encode(text, add_special_tokens=False) == ids, (name, text)
        assert hf.decode(ids, skip_special_tokens=False) == text, (name, text)


def test_get_vocab_equals_hf_vocab(saved):
    """P2: token strings are the tokenizer.json keys (byte-level printable, specials by content).
    A one-byte special ("\\n") has two keys there, "\\n" and its byte-level key "Ċ" (merges and
    split_special_tokens use it); get_vocab dropped "Ċ"."""
    name, tok, out = saved
    hf = tokenizers.Tokenizer.from_file(str(out / "tokenizer.json"))
    vocab = tok.get_vocab()
    assert hf.get_vocab(with_added_tokens=True) == vocab, name
    assert hf.get_vocab_size(with_added_tokens=True) == len(vocab)  # HF counts keys
    assert sorted(set(vocab.values())) == list(range(tok.vocab_size))
    assert all(tok.token_to_id(key) == i for key, i in vocab.items())
    if name == "bpe+byte-special":
        assert vocab["\n"] == vocab["Ċ"] == 10 and len(vocab) == tok.vocab_size + 1


@pytest.mark.parametrize("saved", ["bpe+specials", "wordpiece+specials"], indirect=True)
def test_special_tokens_use_reserved_slots(saved):
    """P10/C10: specials are inside vocab_size, encoded as single ids, as HF added_tokens."""
    name, tok, out = saved
    assert tok.vocab_size == 420
    assert [tok.token_to_id(s) for s in SPECIALS] == [418, 419]
    hf = tokenizers.Tokenizer.from_file(str(out / "tokenizer.json"))
    assert [hf.token_to_id(s) for s in SPECIALS] == [418, 419]
    assert tok.encode("<pad>") == [419]


@pytest.mark.parametrize("saved", ["bpe+specials", "bpe+byte-special"], indirect=True)
def test_split_special_tokens_encodes_specials_as_text(saved):
    """split_special_tokens=True: AutoTokenizer encodes a special's text as ordinary text; the
    wrapper (and native encode, which had no such option) still gave the special's id."""
    name, tok, out = saved
    auto = transformers.AutoTokenizer.from_pretrained(str(out))
    wrapper = tokenisers.DynamicResearchTokenizer(tok)
    for text in TEXTS:
        ids = auto.encode(text, add_special_tokens=False, split_special_tokens=True)
        assert wrapper(text, split_special_tokens=True)["input_ids"] == ids, (name, text)
        assert tok.encode(text, split_special_tokens=True) == ids, (name, text)
        assert tok.encode_bytes(text.encode("utf-8"), split_special_tokens=True) == ids, (name, text)
        multi_byte = {i for s, i in tok.special_tokens().items() if len(s.encode("utf-8")) > 1}
        assert not set(ids) & multi_byte, (name, text)
    eot = tok.token_to_id("<|endoftext|>")
    assert wrapper("a<|endoftext|>b")["input_ids"] == tok.encode("a<|endoftext|>b") == tok.encode("a") + [eot] + tok.encode("b")


def test_from_pretrained_round_trips(saved, tmp_path):
    name, tok, out = saved
    reloaded = type(tok).from_pretrained(str(out))
    assert reloaded.get_vocab() == tok.get_vocab()
    for text in TEXTS:
        assert reloaded.encode(text) == tok.encode(text), (name, text)
    reloaded.save(str(tmp_path))
    assert (tmp_path / "tokenizer.json").read_bytes() == (out / "tokenizer.json").read_bytes()


def test_wrapper_ids_equal_hf_fast_tokenizer(saved):
    name, tok, out = saved
    wrapper = tokenisers.DynamicResearchTokenizer(tok)
    fast = transformers.PreTrainedTokenizerFast(tokenizer_file=str(out / "tokenizer.json"))
    for text in TEXTS:
        if "<" in text and "specials" not in name:
            continue  # the wrapper adds its own <pad>/</s>; the bare model has none
        assert wrapper(text)["input_ids"] == fast.encode(text, add_special_tokens=False), (name, text)


def test_wrapper_added_tokens_keep_their_ids_in_auto_tokenizer(saved, tmp_path):
    """HF numbers new added tokens from len(get_vocab()), which counts keys. The export keys a
    one-byte special ("\\n") twice; get_vocab dropped "Ċ", so the wrapper gave <s> id 1 lower than
    AutoTokenizer on the wrapper's own save_pretrained, and every added-token id disagreed."""
    name, tok, _ = saved
    wrapper = tokenisers.DynamicResearchTokenizer(tok)
    wrapper.add_tokens(["<new>"])
    wrapper.save_pretrained(str(tmp_path))
    auto = transformers.AutoTokenizer.from_pretrained(str(tmp_path))
    names = ["<s>", "</s>", "<unk>", "<pad>", "<new>"]
    assert auto.convert_tokens_to_ids(names) == wrapper.convert_tokens_to_ids(names), name
    assert len(auto) == len(wrapper), name
    text = "<s>a\nb<new> c</s>"
    assert auto.encode(text, add_special_tokens=False) == wrapper(text)["input_ids"], name


@pytest.mark.parametrize("special", ["é", "§", "«", "ĠĠ"])
def test_special_token_equal_to_a_byte_level_key_is_rejected(special, tmp_path):
    """A special keyed by its content collided with another token's printable key ("é" is the key
    of byte 0xE9): get_vocab lost an entry, the exported vocab remapped that byte and the file did
    not reload ("no token with id 233")."""
    with pytest.raises(ValueError, match="collides"):
        tokenisers.BpeTokenizer.train(CORPUS, 420, special_tokens=[special])

    tok = MAKERS["bpe+specials"]()
    tok.save(str(tmp_path))
    path = tmp_path / "tokenizer.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    pad_id = data["model"]["vocab"].pop("<pad>")
    next(t for t in data["added_tokens"] if t["id"] == pad_id)["content"] = special
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="collides"):
        tokenisers.BpeTokenizer.from_pretrained(str(tmp_path))


def test_special_token_equal_to_the_key_of_a_later_id_is_rejected(tmp_path):
    """The same collision when the special's id comes before the token it shadows, so the token
    does not exist yet when the special is added."""
    MAKERS["bpe+specials"]().save(str(tmp_path))
    path = tmp_path / "tokenizer.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    vocab = data["model"]["vocab"]
    pad_id = vocab.pop("<pad>")
    # Swap ids: the last learned multi-byte token moves to pad_id, <pad> takes its id and its key.
    special_ids = {t["id"] for t in data["added_tokens"]}
    key, last = max(((k, i) for k, i in vocab.items() if i >= 256 and i not in special_ids), key=lambda kv: kv[1])
    assert last < pad_id
    vocab[key] = pad_id
    pad = next(t for t in data["added_tokens"] if t["id"] == pad_id)
    pad["id"], pad["content"] = last, key
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="collides"):
        tokenisers.BpeTokenizer.from_pretrained(str(tmp_path))


def test_reloads_a_tokenizer_json_re_saved_by_hugging_face(saved, tmp_path):
    """tokenizers >= 0.20 re-saves BPE merges as [u, v] pairs, which failed to parse."""
    name, tok, out = saved
    tokenizers.Tokenizer.from_file(str(out / "tokenizer.json")).save(str(tmp_path / "tokenizer.json"))
    model = json.loads((tmp_path / "tokenizer.json").read_text(encoding="utf-8"))["model"]
    if model["type"] == "BPE":
        assert isinstance(model["merges"][0], list)  # the format under test
    reloaded = type(tok).from_pretrained(str(tmp_path))
    assert reloaded.get_vocab() == tok.get_vocab()
    for text in TEXTS:
        assert reloaded.encode(text) == tok.encode(text), (name, text)


UNSUPPORTED = {
    "gpt2 regex with prefix space": {"pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False,
                                                        "use_regex": True}},
    "prefix space": {"pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": False}},
    # (A lossless split such as Isolated \s+ now loads as pretokenizer={"hf": ...}: test_hf_pretokenizer.py.)
    "removed split": {"pre_tokenizer": {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Removed", "invert": False},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]}},
    "metaspace": {"pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "always",
                                    "split": True}},
    "normalizer": {"normalizer": {"type": "NFC"}},
}


@pytest.mark.parametrize("change", list(UNSUPPORTED))
@pytest.mark.parametrize("saved", ["bpe+specials", "unigram"], indirect=True)
def test_unimplemented_pre_tokenizer_or_normalizer_is_rejected(saved, change, tmp_path):
    """from_hf_json mapped every pre-tokenizer to whitespace segmentation and ignored normalizers,
    so e.g. a ByteLevel{use_regex: true} file loaded and encoded differently from HF."""
    name, tok, out = saved
    data = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    data.update(UNSUPPORTED[change])
    (tmp_path / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        type(tok).from_pretrained(str(tmp_path))


ADDS_A_TOKEN = {
    "type": "TemplateProcessing",
    "single": [{"SpecialToken": {"id": "<pad>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
    "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
    "special_tokens": {"<pad>": {"id": "<pad>", "ids": [419], "tokens": ["<pad>"]}},
}
IGNORED_SETTINGS = {
    "post_processor": lambda d: d.update(post_processor=ADDS_A_TOKEN),
    "lstrip": lambda d: d["added_tokens"][0].update(lstrip=True),
    "dropout": lambda d: d["model"].update(dropout=0.9),
    "continuing_subword_prefix": lambda d: d["model"].update(continuing_subword_prefix="##"),
    "end_of_word_suffix": lambda d: d["model"].update(end_of_word_suffix="</w>"),
    "ignore_merges": lambda d: d["model"].update(ignore_merges=True),
}


@pytest.mark.parametrize(
    "saved,change",
    [("bpe+specials", change) for change in IGNORED_SETTINGS] + [("unigram", "post_processor")],
    indirect=["saved"],
)
def test_settings_native_encoding_ignores_are_rejected(saved, change, tmp_path):
    """from_hf_json checked only the normalizer and pre-tokenizer: a post-processor that prepends a
    token, end_of_word_suffix or dropout loaded silently and HF ids differed on every text."""
    name, tok, out = saved
    data = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    IGNORED_SETTINGS[change](data)
    (tmp_path / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=f"unsupported.*{change}"):
        type(tok).from_pretrained(str(tmp_path))


def test_reloads_a_tokenizer_json_re_saved_by_transformers(saved, tmp_path):
    """transformers' save_pretrained adds a TemplateProcessing that only concatenates the
    sequences (and keeps the last call's truncation); none of it changes encode."""
    name, tok, out = saved
    auto = transformers.AutoTokenizer.from_pretrained(str(out))
    auto(["a b", "the quick brown fox"], truncation=True, max_length=4)
    auto.save_pretrained(str(tmp_path))
    data = json.loads((tmp_path / "tokenizer.json").read_text(encoding="utf-8"))
    assert data["post_processor"]["type"] == "TemplateProcessing"  # the format under test
    reloaded = type(tok).from_pretrained(str(tmp_path))
    for text in TEXTS:
        assert reloaded.encode(text) == tok.encode(text), (name, text)
