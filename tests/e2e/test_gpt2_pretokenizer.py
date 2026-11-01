"""``pretokenizer="gpt2"``: GPT-2's exact regex pre-split (tiktoken's gpt2 ``pat_str``) through the
Python API, against tiktoken's own split and ids, the Hugging Face export (``tokenizers`` and
``transformers``), on random strings, the existing probes and real FineWeb val documents; plus the
benchmark plumbing of the ``_g2`` and ``_sp<MB>`` variants."""

import functools
import glob
import json
import os
import re
from pathlib import Path
import random
import subprocess
import sys
import unicodedata

import numpy as np
import pytest
import regex
import tiktoken
import tokenizers
import transformers
from tiktoken_ext import openai_public

import tokenisers
from tokenisers import _core
from tokenisers.benchmarks import common
from test_space_prefix import CORPUS, GOLDEN_PROBES, TEXTS, _quiet, _web_docs  # noqa: E402 (tests/e2e on sys.path)

G2 = "gpt2"
# What tiktoken's own Python path splits with (Encoding._encode_only_native_bpe): its pat_str, a
# constant (the encoding itself downloads on first use: see gpt2_enc).
TIKTOKEN_PAT = regex.compile(openai_public.r50k_pat_str)


@functools.cache
def _gpt2_encoding():
    """tiktoken's gpt2 encoding, or None offline with a cold tiktoken cache. It downloads vocab.bpe and
    encoder.json on first use; failing at import would abort collection, and so the whole session."""
    try:
        return tiktoken.get_encoding("gpt2")
    except OSError:  # requests' errors are OSErrors
        return None


@pytest.fixture(scope="module")
def gpt2_enc():
    if _gpt2_encoding() is None:
        pytest.skip("tiktoken's gpt2 encoding is not cached and cannot be downloaded")
    return _gpt2_encoding()


def tiktoken_split(text: str) -> list:
    return [m.group().encode() for m in TIKTOKEN_PAT.finditer(text)]


def hf_split(text: str) -> list:
    """Hugging Face's ByteLevel(use_regex=True) pieces, as byte strings of the original text."""
    pre = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    return [text[a:b].encode() for _, (a, b) in pre.pre_tokenize_str(text)]


def native_split(data: bytes) -> list:
    return list(_core.pretokenize(data, G2))


def _random_texts(n, seed):
    rng = random.Random(seed)
    pieces = [" ", " ", " ", "  ", "\t", "\n", "\r\n", "　", "\xa0", "\x85", "a", "bc", "Z", "x.", "中文", "é",
              "é", "1", "42", "½", "²", "٣", "'", "'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "’", "!",
              "...", "-", "$", "🚀", "​", "﻿", "�", "_", "<|endoftext|>"]

    def rand_char():
        while True:
            cp = rng.choice([rng.randrange(0x80, 0x3000), rng.randrange(0x3000, 0x10000), rng.randrange(0x10000, 0x30000)])
            if not 0xD800 <= cp < 0xE000:
                return chr(cp)

    return ["".join(rand_char() if rng.random() < 0.1 else rng.choice(pieces) for _ in range(rng.randint(0, 30)))
            for _ in range(n)]


RANDOM = _random_texts(3000, 11)


def _fineweb_val_docs():
    """The speedrun's val documents (the first 20 x 524,288 GPT-2 tokens of fineweb_val_000000.bin,
    as the tokenizer swap decodes them), from $TOKENISERS_FINEWEB_VAL or a prepared swap job's cache."""
    paths = ([os.environ["TOKENISERS_FINEWEB_VAL"]] if os.environ.get("TOKENISERS_FINEWEB_VAL") else
             sorted(glob.glob(os.path.expanduser("~/.cache/tokenisers/*/source/fineweb_val_000000.bin"))))
    if not paths or not Path(paths[0]).is_file() or _gpt2_encoding() is None:
        return None
    from tokenisers.benchmarks.nanogpt_swap import swap_job

    enc = _gpt2_encoding()
    tokens = swap_job.read_bin(Path(paths[0]), max_tokens=20 * 524288)
    docs, _ = swap_job.split_docs(tokens, enc.eot_token)
    return [enc.decode_bytes(d.tolist()).decode("utf-8") for d in docs]


FINEWEB = _fineweb_val_docs()


@pytest.fixture(scope="module")
def fineweb():
    if FINEWEB is None:
        pytest.skip("no FineWeb val shard cached (set TOKENISERS_FINEWEB_VAL), or no tiktoken gpt2 to decode it")
    return FINEWEB


def fineweb_sample(step: int) -> list:
    """Every ``step``-th FineWeb val document, or none when no shard is cached."""
    return (FINEWEB or [])[::step]


# ----------------------------------------------------------------------------
# The split itself: native == tiktoken == Hugging Face
# ----------------------------------------------------------------------------

def test_pattern_is_tiktokens(gpt2_enc):
    assert openai_public.gpt2()["pat_str"] == gpt2_enc._pat_str == TIKTOKEN_PAT.pattern == (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""")


def test_both_pretokenizer_modules_collect_offline_with_a_cold_tiktoken_cache(tmp_path, repo_root):
    """No network, empty tiktoken cache: the modules still import and the tiktoken tests skip. A
    collection error would stop the whole session, unrelated tests included."""
    proxy = "http://127.0.0.1:9"  # discard port: every download fails at once
    env = {**os.environ, "TIKTOKEN_CACHE_DIR": str(tmp_path), "TOKENISERS_FINEWEB_VAL": str(tmp_path / "none"),
           "HTTPS_PROXY": proxy, "https_proxy": proxy, "HTTP_PROXY": proxy, "http_proxy": proxy,
           "NO_PROXY": "", "no_proxy": ""}
    files = [str(repo_root / "tests" / "e2e" / f) for f in ("test_gpt2_pretokenizer.py", "test_hf_pretokenizer.py")]
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
                        "-k", "test_pattern_is_tiktokens", *files],
                       cwd=repo_root, env=env, capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "1 skipped" in out and "cannot be downloaded" in out, out[-4000:]


def assigned_in_unicode_15(text: str) -> bool:
    """Every char assigned in Python's Unicode database (15.1). A code point assigned later can be
    classified differently by each library's own tables: the ``regex`` module behind tiktoken's
    Python path has Unicode 17, the Rust regex crate (ours, tiktoken's Rust core) and Oniguruma
    (Hugging Face) Unicode 16, so e.g. U+18CD8 is a letter only to the first."""
    return all(unicodedata.category(c) != "Cn" for c in text)


def test_native_split_equals_tiktoken_and_hf_on_random_strings_and_probes():
    texts = TEXTS + CORPUS + RANDOM + [p.decode("utf-8") for p in GOLDEN_PROBES if _valid(p)]
    checked = 0
    for t in texts:
        native = native_split(t.encode())
        assert b"".join(native) == t.encode()
        assert native == hf_split(t), f"hf {t!r}"
        if assigned_in_unicode_15(t):
            assert native == tiktoken_split(t), f"tiktoken {t!r}"
            checked += 1
    assert checked > 0.7 * len(texts), checked


def _valid(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def test_native_split_equals_tiktoken_and_hf_on_fineweb_val(fineweb):
    for d in fineweb[::10]:  # every 10th of the 14,982 documents
        native = native_split(d.encode())
        assert native == hf_split(d), d[:200]
        assert native == tiktoken_split(d) or not assigned_in_unicode_15(d), d[:200]


def test_invalid_utf8_splits_like_one_replacement_char_per_invalid_sequence():
    for raw in [p for p in GOLDEN_PROBES if not _valid(p)] + [b"it\xc3's \xff\xfe!! \xe3\x80 x\n\x80 2\x80"]:
        native = native_split(raw)
        assert b"".join(native) == raw
        lossy = raw.decode("utf-8", errors="replace")
        assert [s.decode("utf-8", errors="replace") for s in native] == [s.decode() for s in tiktoken_split(lossy)]


# ----------------------------------------------------------------------------
# Trained tokenizers: export parity, reload, determinism, losslessness
# ----------------------------------------------------------------------------

def _g2_corpus():
    return CORPUS + _web_docs(30, 3)


MAKERS = {
    **{s: (lambda s=s: tokenisers.BpeTokenizer.train(_g2_corpus(), 420, scorer=s, pretokenizer=G2))
       for s in ["frequency", "wordpiece", "wp-bpe-pig", "wp-bpe-npmi", "wp-bpe-ppig", "mdl", "length_penalized"]},
    "bpe+specials": lambda: tokenisers.BpeTokenizer.train(_g2_corpus(), 420, special_tokens=["<|endoftext|>", "<pad>"],
                                                          pretokenizer=G2),
    "morpheme": lambda: tokenisers.MorphemeBpeTokenizer.train(
        _g2_corpus(), 420, morpheme_lexicon={"jumps": ["jump", "s"], "quick": ["qu", "ick"]}, lambda_weight=2.0,
        pretokenizer=G2),
    "unigram": lambda: tokenisers.UnigramTokenizer.train(_g2_corpus(), 500, pretokenizer=G2),
}


@pytest.fixture(scope="module", params=list(MAKERS))
def saved(request, tmp_path_factory):
    with _quiet():
        tok = MAKERS[request.param]()
    out = tmp_path_factory.mktemp(request.param.replace("+", "_").replace("-", "_"))
    tok.save(str(out))
    return request.param, tok, out


def test_export_is_hf_byte_level_with_gpt2_regex(saved):
    name, tok, out = saved
    assert tok.pretokenizer == G2
    assert json.loads((out / "tokenizer.json").read_text())["pre_tokenizer"] == {
        "type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": True}
    # GPT-2's sub-split: no token joins a letter and a digit, or carries a space after its start.
    for t, i in tok.get_vocab().items():
        piece = tok.id_to_bytes(i)
        assert not re.search(rb"[A-Za-z][0-9]|[0-9][A-Za-z]|[^\s] ", piece), (name, piece)


def test_ids_equal_hf_tokenizers_and_transformers(saved):
    name, tok, out = saved
    hf = tokenizers.Tokenizer.from_file(str(out / "tokenizer.json"))
    auto = transformers.AutoTokenizer.from_pretrained(str(out))
    for t in TEXTS + RANDOM[:300] + fineweb_sample(150):
        ids = tok.encode(t)
        assert hf.encode(t, add_special_tokens=False).ids == ids, f"[{name}] tokenizers {t[:80]!r}"
        assert auto.encode(t, add_special_tokens=False) == ids, f"[{name}] transformers {t[:80]!r}"
        assert hf.decode(ids, skip_special_tokens=False) == t, f"[{name}] decode {t[:80]!r}"


def test_reload_and_transformers_resave_round_trip(saved, tmp_path):
    name, tok, out = saved
    again = type(tok).from_pretrained(str(out))
    assert again.pretokenizer == G2
    assert [again.encode(t) for t in TEXTS] == [tok.encode(t) for t in TEXTS]
    again.save(str(tmp_path / "ours"))
    assert (tmp_path / "ours" / "tokenizer.json").read_bytes() == (out / "tokenizer.json").read_bytes()
    transformers.AutoTokenizer.from_pretrained(str(out)).save_pretrained(str(tmp_path / "hf"))
    resaved = type(tok).from_pretrained(str(tmp_path / "hf"))
    assert resaved.pretokenizer == G2 and [resaved.encode(t) for t in TEXTS] == [tok.encode(t) for t in TEXTS]


def test_lossless_on_web_text_random_strings_and_invalid_utf8():
    train = _web_docs(300, 1)
    for cls in (tokenisers.BpeTokenizer, tokenisers.WpBpePpigTokenizer, tokenisers.MdlTokenizer,
                tokenisers.UnigramTokenizer):
        with _quiet():
            tok = cls.train(train, 1200, pretokenizer=G2)
        for d in _web_docs(100, 2) + RANDOM[:500] + fineweb_sample(300):
            assert tok.decode(tok.encode(d)) == d, cls.__name__
        for raw in GOLDEN_PROBES + [b"\xff\xfe it\xc3's \xe3\x80 x\n \x80 12\x80"]:
            assert tok.decode_bytes(tok.encode_bytes(raw)) == raw, cls.__name__


_DETERMINISM_SCRIPT = """
import sys
from tokenisers import _core
corpus = [l for l in sys.stdin.read().split("\\x1e") if l]
_core.train_unigram(corpus, 500, pretokenizer="gpt2").export_hf_json(sys.argv[1])
"""


def test_gpt2_unigram_is_identical_across_processes(tmp_path):
    outputs = []
    for i, corpus in enumerate([CORPUS, CORPUS[::-1]]):
        path = tmp_path / f"run{i}.json"
        subprocess.run([sys.executable, "-c", _DETERMINISM_SCRIPT, str(path)], input="\x1e".join(corpus), text=True,
                       check=True)
        outputs.append(path.read_bytes())
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0])["pre_tokenizer"]["use_regex"] is True


# ----------------------------------------------------------------------------
# GPT-2's own merges natively: the same ids as tiktoken
# ----------------------------------------------------------------------------

def _bpe_parts(ranks, token: bytes, max_rank: int) -> list:
    parts = [bytes([b]) for b in token]
    while True:
        best = min(((ranks.get(a + b), i) for i, (a, b) in enumerate(zip(parts, parts[1:]))
                    if ranks.get(a + b) is not None), default=None)
        if best is None or best[0] >= max_rank:
            return parts
        i = best[1]
        parts[i:i + 2] = [parts[i] + parts[i + 1]]


@pytest.fixture(scope="module")
def gpt2_native(tmp_path_factory, gpt2_enc):
    """tiktoken's GPT-2 ranks as one of our tokenizer.json files (byte b = id b, merged token of
    rank r = id r) loaded natively: encoding then runs our gpt2 split and our BPE."""
    ranks = gpt2_enc._mergeable_ranks
    by_rank = sorted(ranks.items(), key=lambda kv: kv[1])
    assert sorted(len(b) for b, _ in by_rank[:256]) == [1] * 256 and [r for _, r in by_rank] == list(range(len(by_rank)))
    p = _core.bytes_to_printable
    vocab = {p(bytes([b])): b for b in range(256)}
    merges = []
    for tok, r in by_rank[256:]:
        a, b = _bpe_parts(ranks, tok, r)
        merges.append(f"{p(a)} {p(b)}")
        vocab[p(tok)] = r
    bl = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False}
    doc = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": None,
           "pre_tokenizer": {**bl, "use_regex": True}, "post_processor": None, "decoder": {**bl, "use_regex": False},
           "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                     "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                     "vocab": vocab, "merges": merges}}
    path = tmp_path_factory.mktemp("gpt2") / "tokenizer.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    tok = tokenisers.BpeTokenizer.from_pretrained(str(path))
    assert tok.pretokenizer == G2 and tok.vocab_size == 50256
    to_rank = np.array([ranks[tok.id_to_bytes(i)] for i in range(tok.vocab_size)])
    return tok, to_rank


def test_gpt2_merges_natively_give_tiktokens_ids(gpt2_native, gpt2_enc):
    tok, to_rank = gpt2_native
    for t in TEXTS + CORPUS + RANDOM + fineweb_sample(5):
        assert to_rank[tok.encode(t)].tolist() == gpt2_enc.encode_ordinary(t), t[:200]


# ----------------------------------------------------------------------------
# Lexicon, API and benchmarks
# ----------------------------------------------------------------------------

def test_lexicon_annotates_the_words_and_spaced_words_gpt2_keeps_whole():
    corpus = ["walking home, we were walking and walking, don't walking"] * 50
    lex = {"walking": ["walk", "ing"]}
    with _quiet():
        plain = tokenisers.BpeTokenizer.train(corpus, 300, pretokenizer=G2)
        guided = tokenisers.MorphemeBpeTokenizer.train(corpus, 300, morpheme_lexicon=lex, lambda_weight=50.0,
                                                       pretokenizer=G2)
        # "don't" is the segments "don", "'t": its entry annotates nothing, so training is unchanged.
        split = tokenisers.MorphemeBpeTokenizer.train(corpus, 300, morpheme_lexicon={"don't": ["do", "n't"]},
                                                      lambda_weight=50.0, pretokenizer=G2)
        unguided = tokenisers.MorphemeBpeTokenizer.train(corpus, 300, morpheme_lexicon={}, pretokenizer=G2)
    assert [plain.id_to_bytes(i) for i in plain.encode(" walking")] == [b" walking"]
    for text, word in ((" walking", b" walk"), ("walking", b"walk")):
        pieces = [guided.id_to_bytes(i) for i in guided.encode(text)]
        assert word in pieces and b"ing" in pieces, (text, pieces)
    assert split.get_vocab() == unguided.get_vocab()


def test_pretokenize_api():
    assert _core.pretokenize(b"it's  2024!", G2) == [b"it", b"'s", b" ", b" 2024", b"!"]
    assert _core.pretokenize(b"it's  2024!", "space_prefix") == [b"it's", b" ", b" 2024!"]
    assert _core.pretokenize(b"it's  2024!", "whitespace") == [b"it's", b"  ", b"2024!"]
    with pytest.raises(ValueError, match="pretokenizer must be"):
        _core.pretokenize(b"x", "GPT2")
    assert tokenisers.LengthPenalizedTokenizer.train(CORPUS, 300, pretokenizer=G2).pretokenizer == G2


def test_g2_and_scaled_variants_train_like_their_base_variant():
    for bad in ("super_bpe_g2", "super_bpe_sp300", "bpe_frequency_g2x", "nope_sp", "bpe_frequency_sp0"):
        with pytest.raises(ValueError):
            common.variant_config(bad, 300)
    assert common.parse_variant("wp_bpe_pig_sp300") == ("wp_bpe_pig", "space_prefix", 300)
    assert common.parse_variant("mdl_bits_g2") == ("mdl_bits", "gpt2", None)
    assert common.parse_variant("mdl_bits") == ("mdl_bits", "whitespace", None)
    corpus = ["the cat sat on the mat with the hat, a bat ate a gnat 12 times " * 5, "zyx qwv jk  walking"]
    with _quiet():
        for name in common.VARIANTS:
            if name == "super_bpe":
                continue
            base = common.variant_config(name, 290)
            assert base.pop("pretokenizer") == "whitespace"
            for suffix, mode in (("_g2", G2), ("_sp300", "space_prefix"), ("_g2300", G2)):
                cfg = common.variant_config(name + suffix, 290)
                assert cfg.pop("pretokenizer") == mode and cfg == base, (name, suffix)
            tok = common.train_variant(f"{name}_g2", corpus, 290, {"walking": ["walk", "ing"]})
            assert tok.pretokenizer == G2, name
