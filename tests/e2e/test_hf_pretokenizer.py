"""``pretokenizer={"hf": config}``: a lossless Hugging Face pre-tokenizer config, run by the
``tokenizers`` crate inside our core, against Python's own ``tokenizers`` split (random strings,
the probes, FineWeb val documents), the Hugging Face export (``tokenizers`` and ``transformers``)
with reload, the native ``gpt2`` mode, rejected components and the benchmark records."""

import functools
import hashlib
import json
import re

import pytest
import tokenizers
import transformers

import tokenisers
from tokenisers import _core
from tokenisers.benchmarks import common
from test_gpt2_pretokenizer import FINEWEB, RANDOM, fineweb, fineweb_sample  # noqa: F401 (fixture)
from test_space_prefix import CORPUS, GOLDEN_PROBES, TEXTS, _quiet, _web_docs  # noqa: E402 (tests/e2e on sys.path)


def byte_level(use_regex=True, trim_offsets=False):
    return {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": trim_offsets, "use_regex": use_regex}


def split(regex, behavior="Isolated", invert=False):
    return {"type": "Split", "pattern": {"Regex": regex}, "behavior": behavior, "invert": invert}


def seq(*parts):
    return {"type": "Sequence", "pretokenizers": list(parts)}


DIGITS = {"type": "Digits", "individual_digits": True}
PUNCT = {"type": "Punctuation", "behavior": "Isolated"}
# Llama 3's tokenizer.json pre-tokenizer, verbatim.
LLAMA3_PAT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+"
              r"|\s+(?!\S)|\s+")
CONFIGS = {
    "llama_digits": seq(DIGITS, byte_level()),  # the README example
    "llama3": seq(split(LLAMA3_PAT), byte_level(False, trim_offsets=True)),
    # No ByteLevel: the export adds the byte-level map.
    "punct_isolated": seq(split(r"\s+", "MergedWithNext"), PUNCT),
    "fixed_length": {"type": "FixedLength", "length": 4},
    "split_inverted": split(r"\p{L}+", invert=True),
    # Splitters after ByteLevel see one char per byte and cut inside chars: Falcon's Digits isolate
    # "¹" (byte B9) of "ス" (E3 82 B9); Punctuation isolates "¶" (B6) of "ö" (C3 B6).
    "falcon": seq({"type": "Punctuation", "behavior": "Contiguous"}, byte_level(trim_offsets=True),
                  {"type": "Digits", "individual_digits": False}, split("[0-9][0-9][0-9]")),
    "punct_inside_chars": seq(byte_level(), PUNCT),
}


def has_byte_level(config) -> bool:
    return '"ByteLevel"' in json.dumps(config)


@functools.lru_cache(maxsize=None)
def _py_pre_tokenizer(config_json: str):
    """Python ``tokenizers``' own pre-tokenizer for a config (through a minimal tokenizer.json)."""
    doc = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": None,
           "pre_tokenizer": json.loads(config_json), "post_processor": None, "decoder": None,
           "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                     "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                     "vocab": {}, "merges": []}}
    return tokenizers.Tokenizer.from_str(json.dumps(doc)).pre_tokenizer


def hf_pieces(config, text: str) -> list:
    """Python HF's pieces of ``text`` as bytes of the text (byte-level chars mapped back to bytes)."""
    pre = _py_pre_tokenizer(json.dumps(config))
    back = _core.printable_to_bytes if has_byte_level(config) else str.encode
    return [back(piece) for piece, _ in pre.pre_tokenize_str(text)]


def native(config, data: bytes) -> list:
    return list(_core.pretokenize(data, {"hf": config}))


def _valid(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


# ----------------------------------------------------------------------------
# The split itself: native == Python's Hugging Face tokenizers
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(CONFIGS))
def test_native_split_equals_python_hf_on_random_strings_and_probes(name):
    config = CONFIGS[name]
    texts = TEXTS + CORPUS + RANDOM + [p.decode() for p in GOLDEN_PROBES if _valid(p)]
    cut_chars = 0
    for t in texts:
        got = native(config, t.encode())
        assert b"".join(got) == t.encode() and all(got), (name, t)
        assert got == hf_pieces(config, t), (name, t)
        cut_chars += any(not _valid(s) for s in got)
    assert (cut_chars > 0) == name.endswith(("falcon", "inside_chars")), (name, cut_chars)


@pytest.mark.parametrize("name", ["llama_digits", "llama3", "punct_isolated", "falcon"])
def test_native_split_equals_python_hf_on_fineweb_val(fineweb, name):
    config = CONFIGS[name]
    for d in fineweb[::10]:  # every 10th of the 14,982 documents
        assert native(config, d.encode()) == hf_pieces(config, d), (name, d[:200])


@pytest.mark.parametrize("name", list(CONFIGS))
def test_invalid_utf8_reads_each_bad_byte_as_one_nul(name):
    """HF takes text: each byte that is not valid UTF-8 reads as U+0000 (a symbol), one for one."""
    config = CONFIGS[name]
    for raw in [p for p in GOLDEN_PROBES if not _valid(p)] + [b"it\xc3's \xff\xfe!! \xe3\x80 x\n\x80 12\x803"]:
        got = native(config, raw)
        assert b"".join(got) == raw, name
        nul = re.sub("[\udc80-\udcff]", "\0", raw.decode("utf-8", errors="surrogateescape"))
        assert [len(s) for s in got] == [len(s) for s in hf_pieces(config, nul)], (name, raw)


# ----------------------------------------------------------------------------
# Rejected configs
# ----------------------------------------------------------------------------

REJECTED = {
    "Whitespace drops whitespace": {"type": "Whitespace"},
    "WhitespaceSplit drops whitespace": {"type": "WhitespaceSplit"},
    "BertPreTokenizer drops whitespace": {"type": "BertPreTokenizer"},
    "CharDelimiterSplit drops the delimiter": {"type": "CharDelimiterSplit", "delimiter": "x"},
    "Metaspace replaces spaces": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always",
                                  "split": True},
    "UnicodeScripts drops the U\\+0020 spaces": {"type": "UnicodeScripts"},
    "Split with behavior Removed": split(r"\s+", "Removed"),
    "Punctuation with behavior Removed": {"type": "Punctuation", "behavior": "Removed"},
    "ByteLevel with add_prefix_space": {**byte_level(), "add_prefix_space": True},
    "2 ByteLevels": seq(byte_level(), DIGITS, byte_level(False)),
    "Whitespace drops whitespace ": seq(DIGITS, seq(PUNCT, {"type": "Whitespace"})),
    "Bogus is not a known lossless splitter": {"type": "Bogus"},
    # tokenizers panics on it (chunk size 0): a PanicException is a BaseException, not a ValueError.
    "FixedLength with length 0": {"type": "FixedLength", "length": 0},
    "FixedLength with length 0 ": seq(byte_level(), {"type": "FixedLength", "length": 0}),
}


@pytest.mark.parametrize("why", list(REJECTED))
def test_lossy_components_are_rejected(why):
    config = REJECTED[why]
    for call in (lambda: _core.pretokenize(b"x", {"hf": config}),
                 lambda: tokenisers.BpeTokenizer.train(CORPUS, 300, pretokenizer={"hf": config}),
                 lambda: tokenisers.UnigramTokenizer.train(CORPUS, 300, pretokenizer={"hf": json.dumps(config)})):
        with pytest.raises(ValueError, match=f"HF pre-tokenizer .*{why.strip()}"):
            call()


def test_a_regex_past_onigs_backtracking_limit_raises_value_error():
    """`(x+x+)+y` passes the probe, but onig gives up on a long run of "x" and panics: that is a
    ValueError (not a PanicException, a BaseException) in pretokenize, training and encoding."""
    config = {"hf": split("(x+x+)+y")}
    assert _core.pretokenize(b"x x y", config) == [b"x x y"]
    bad = "x" * 40
    with _quiet():
        bpe = tokenisers.BpeTokenizer.train(CORPUS, 300, pretokenizer=config)
        uni = tokenisers.UnigramTokenizer.train(CORPUS, 300, pretokenizer=config)
    for call in (lambda: _core.pretokenize(bad.encode(), config),
                 lambda: tokenisers.BpeTokenizer.train([*CORPUS, bad], 300, pretokenizer=config),
                 lambda: tokenisers.UnigramTokenizer.train([*CORPUS, bad], 300, pretokenizer=config),
                 lambda: tokenisers.MorphemeBpeTokenizer.train(CORPUS, 300, morpheme_lexicon={bad: [bad]},
                                                               pretokenizer=config),
                 lambda: bpe.encode(bad), lambda: uni.encode(bad), lambda: uni.encode(bad, temperature=0.5)):
        with pytest.raises(ValueError, match="HF pre-tokenizer .*retry-limit-in-match"):
            call()
    assert bpe.decode(bpe.encode("x x y")) == "x x y"  # the models still work


def test_malformed_pretokenizer_arguments_are_rejected():
    for bad, match in (({"hf": "{not json"}, "pretokenizer"), ({"hf": {"type": "Digits"}}, "individual_digits"),
                       ({"hf": DIGITS, "x": 1}, "pretokenizer must be"), ({"HF": DIGITS}, "pretokenizer must be"),
                       (3, "pretokenizer must be"), ("hf", "pretokenizer must be")):
        with pytest.raises(ValueError, match=match):
            tokenisers.BpeTokenizer.train(CORPUS, 300, pretokenizer=bad)


# ----------------------------------------------------------------------------
# Trained tokenizers: export parity, reload
# ----------------------------------------------------------------------------

def _corpus():
    return CORPUS + _web_docs(30, 3)


def _makers(config):
    p = {"hf": config}
    return {
        "frequency": lambda: tokenisers.BpeTokenizer.train(_corpus(), 420, pretokenizer=p),
        "wp-bpe-ppig": lambda: tokenisers.WpBpePpigTokenizer.train(_corpus(), 420, pretokenizer=p),
        "mdl": lambda: tokenisers.MdlTokenizer.train(_corpus(), 420, pretokenizer=p),
        "bpe+specials": lambda: tokenisers.BpeTokenizer.train(_corpus(), 420, special_tokens=["<|endoftext|>", "<pad>"],
                                                              pretokenizer=p),
        "morpheme": lambda: tokenisers.MorphemeBpeTokenizer.train(
            _corpus(), 420, morpheme_lexicon={"jumps": ["jump", "s"], "quick": ["qu", "ick"]}, lambda_weight=2.0,
            pretokenizer=p),
        "unigram": lambda: tokenisers.UnigramTokenizer.train(_corpus(), 500, pretokenizer=p),
    }


TRAINED = [(c, m) for c in ("llama_digits", "punct_isolated", "falcon", "punct_inside_chars")
           for m in _makers(None)]


@pytest.fixture(scope="module", params=TRAINED, ids=[f"{c}-{m}" for c, m in TRAINED])
def saved(request, tmp_path_factory):
    config_name, maker = request.param
    with _quiet():
        tok = _makers(CONFIGS[config_name])[maker]()
    out = tmp_path_factory.mktemp(f"{config_name}_{maker}".replace("+", "_").replace("-", "_"))
    tok.save(str(out))
    return f"{config_name}/{maker}", CONFIGS[config_name], tok, out


def test_export_writes_the_config_back(saved):
    name, config, tok, out = saved
    assert tok.pretokenizer == {"hf": config}, name
    written = json.loads((out / "tokenizer.json").read_text())["pre_tokenizer"]
    assert written == (config if has_byte_level(config) else seq(config, byte_level(False))), name


def test_ids_equal_hf_tokenizers_and_transformers(saved):
    name, config, tok, out = saved
    hf = tokenizers.Tokenizer.from_file(str(out / "tokenizer.json"))
    auto = transformers.AutoTokenizer.from_pretrained(str(out))
    for t in TEXTS + RANDOM[:300] + fineweb_sample(150):
        ids = tok.encode(t)
        assert hf.encode(t, add_special_tokens=False).ids == ids, f"[{name}] tokenizers {t[:80]!r}"
        assert auto.encode(t, add_special_tokens=False) == ids, f"[{name}] transformers {t[:80]!r}"
        assert hf.decode(ids, skip_special_tokens=False) == t, f"[{name}] decode {t[:80]!r}"


def test_reload_and_transformers_resave_round_trip(saved, tmp_path):
    name, config, tok, out = saved
    again = type(tok).from_pretrained(str(out))
    assert again.pretokenizer == {"hf": config}, name
    assert [again.encode(t) for t in TEXTS] == [tok.encode(t) for t in TEXTS]
    again.save(str(tmp_path / "ours"))
    assert (tmp_path / "ours" / "tokenizer.json").read_bytes() == (out / "tokenizer.json").read_bytes(), name
    transformers.AutoTokenizer.from_pretrained(str(out)).save_pretrained(str(tmp_path / "hf"))
    resaved = type(tok).from_pretrained(str(tmp_path / "hf"))
    assert resaved.pretokenizer == {"hf": config}, name
    assert [resaved.encode(t) for t in TEXTS] == [tok.encode(t) for t in TEXTS]


def test_lossless_on_web_text_random_strings_and_invalid_utf8():
    train = _web_docs(200, 1)
    for name in ("llama3", "falcon", "fixed_length"):
        for cls in (tokenisers.BpeTokenizer, tokenisers.WpBpePpigTokenizer, tokenisers.UnigramTokenizer):
            with _quiet():
                tok = cls.train(train, 1000, pretokenizer={"hf": CONFIGS[name]})
            for d in _web_docs(50, 2) + RANDOM[:300] + fineweb_sample(300):
                assert tok.decode(tok.encode(d)) == d, (name, cls.__name__)
            for raw in GOLDEN_PROBES + [b"\xff\xfe it\xc3's \xe3\x80 x\n \x80 12\x80"]:
                assert tok.decode_bytes(tok.encode_bytes(raw)) == raw, (name, cls.__name__)


def test_a_foreign_lossless_split_loads_as_hf(tmp_path):
    """A tokenizer.json with another lossless split before the byte-level map (rejected before the
    hf mode) now loads with that split and encodes like Hugging Face."""
    tok = tokenisers.BpeTokenizer.train(CORPUS, 420)
    tok.save(str(tmp_path))
    data = json.loads((tmp_path / "tokenizer.json").read_text(encoding="utf-8"))
    data["pre_tokenizer"] = seq(split(r"\s+"), byte_level(False))
    (tmp_path / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    loaded = tokenisers.BpeTokenizer.from_pretrained(str(tmp_path))
    assert loaded.pretokenizer == {"hf": split(r"\s+")}
    hf = tokenizers.Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    for t in TEXTS + RANDOM[:200]:
        assert loaded.encode(t) == hf.encode(t, add_special_tokens=False).ids, t


# ----------------------------------------------------------------------------
# ByteLevel(use_regex=true) through the hf mode is the native gpt2 mode
# ----------------------------------------------------------------------------

def test_hf_byte_level_regex_equals_the_native_gpt2_mode():
    # The split only: train rejects this config, whose export is the gpt2 mode's (see below).
    config = {"hf": byte_level()}
    for t in TEXTS + CORPUS + RANDOM + fineweb_sample(5):
        assert _core.pretokenize(t.encode(), config) == _core.pretokenize(t.encode(), "gpt2"), t[:200]
    for raw in GOLDEN_PROBES + [b"it\xc3's \xff\xfe!! \xe3\x80 x\n\x80 2\x80"]:
        assert _core.pretokenize(raw, config) == _core.pretokenize(raw, "gpt2"), raw


def test_a_config_saved_as_another_pretokenizer_is_rejected_at_train(tmp_path):
    """A config whose tokenizer.json export is another pretokenizer's (a named mode's, another hf
    config's, or whole text) would reload with a different ``pretokenizer`` (a Unigram model would
    not reload at all), so train names the equivalent; that one trains and round-trips."""
    with _quiet():
        tokenisers.BpeTokenizer.train(CORPUS, 300).save(str(tmp_path / "ws"))
    ws_export = json.loads((tmp_path / "ws" / "tokenizer.json").read_text())["pre_tokenizer"]
    cases = [  # (config, error text, what to pass instead)
        (byte_level(False), "as whole-text encoding", None),
        (byte_level(False, trim_offsets=True), "as whole-text encoding", None),
        (byte_level(), 'as "gpt2"', "gpt2"),
        (seq(DIGITS, byte_level(False)), 'as {"hf": ', {"hf": DIGITS}),
        (ws_export, 'as "whitespace"', "whitespace"),  # a whitespace-mode export's own pre_tokenizer
    ]
    for i, (config, why, instead) in enumerate(cases):
        assert b"".join(_core.pretokenize(b"ab 12 cd", {"hf": config})) == b"ab 12 cd"
        for train in (lambda p: tokenisers.BpeTokenizer.train(CORPUS, 300, pretokenizer=p),
                      lambda p: tokenisers.UnigramTokenizer.train(CORPUS, 300, pretokenizer=p),
                      lambda p: tokenisers.MorphemeGuidedTokenizer.train(
                          CORPUS, 300, morpheme_lexicon={"jumps": ["jump", "s"]}, pretokenizer=p)):
            with pytest.raises(ValueError, match=f"tokenizer.json export loads back {re.escape(why)}"):
                train({"hf": config})
            if instead is not None:
                with _quiet():
                    tok = train(instead)
                tok.save(str(tmp_path / str(i)))
                assert type(tok).from_pretrained(str(tmp_path / str(i))).pretokenizer == instead


# ----------------------------------------------------------------------------
# API and benchmarks
# ----------------------------------------------------------------------------

def test_pretokenizer_property_and_json_string_config():
    with _quiet():
        tok = tokenisers.LengthPenalizedTokenizer.train(CORPUS, 300, pretokenizer={"hf": json.dumps(CONFIGS["llama3"])})
    assert tok.pretokenizer == {"hf": CONFIGS["llama3"]}
    assert _core.pretokenize(b"it's 2024 abc123", {"hf": json.dumps(CONFIGS["llama_digits"])}) == [
        b"it", b"'s", b" ", b"2", b"0", b"2", b"4", b" abc", b"1", b"2", b"3"]


def test_variant_config_records_the_hf_config_and_its_hash():
    config = CONFIGS["llama_digits"]
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    label = "hf:" + hashlib.sha256(canonical.encode()).hexdigest()[:12]
    assert common.hf_pretokenizer_label(config) == common.hf_pretokenizer_label(json.dumps(config)) == label
    for name in ("mdl_bits", "unigram_lm", "morpheme_bpe", "wp_bpe_pig"):
        cfg = common.variant_config(name, 300, {"hf": config})
        base = common.variant_config(name, 300)
        assert cfg.pop("pretokenizer") == label and cfg.pop("pretokenizer_config") == config
        assert base.pop("pretokenizer") == "whitespace" and cfg == base, name
        assert common.variant_config(name, 300, "gpt2") == common.variant_config(f"{name}_g2", 300)
        with _quiet():
            tok = common.train_variant(name, CORPUS, 300, {"walking": ["walk", "ing"]}, {"hf": json.dumps(config)})
        assert tok.pretokenizer == {"hf": config}, name
    with pytest.raises(ValueError, match="super_bpe"):
        common.variant_config("super_bpe", 300, {"hf": config})
