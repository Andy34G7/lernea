"""Binding contracts shared by every tokenizer class: vocab/ids, decoding, parameters, sizes,
loading, threading, determinism and the byte-entropy tokenizer's frozen vocabulary."""

import json
import random
import subprocess
import sys
import textwrap
import threading
import time
import warnings

import pytest

import tokenisers
from tokenisers import _core

SYLLABLES = ["ka", "to", "ri", "men", "sa", "lo", "pu", "ne", "shi", "vor", "tan", "el", "ix", "qu", "dra", "bo"]


def synthetic_text(n_words, seed):
    rng = random.Random(seed)
    return " ".join("".join(rng.choice(SYLLABLES) for _ in range(rng.randint(1, 4))) for _ in range(n_words))


@pytest.fixture(scope="module")
def corpus(sample_english_text, sample_multilingual_text, sample_code_snippets):
    return [sample_english_text] + list(sample_multilingual_text.values()) + list(sample_code_snippets.values())


@pytest.fixture(scope="module")
def trained(corpus):
    """One trained instance of every tokenizer class."""
    return {
        "bpe": tokenisers.BpeTokenizer.train(corpus, vocab_size=380, special_tokens=["<|endoftext|>", "<pad>"]),
        "wordpiece": tokenisers.WordPieceTokenizer.train(corpus, vocab_size=380),
        "unigram": tokenisers.UnigramTokenizer.train(corpus, vocab_size=380),
        "super_bpe": tokenisers.SuperBpeTokenizer.train(corpus, vocab_size=380),
        "morpheme": tokenisers.MorphemeBpeTokenizer.train(corpus, vocab_size=380, morpheme_lexicon={}),
        "byte_entropy": tokenisers.ByteEntropyTokenizer.train(corpus, vocab_size=380, entropy_threshold=3.0),
    }


TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Héllo wörld, naïve café — 日本語のテキスト 🚀🔥",
    "def foo(x):\n\treturn x + 1  # comment\r\n",
    "non breaking space and　ideographic",
]
API = ["train", "encode", "encode_bytes", "decode", "decode_bytes", "id_to_bytes", "id_to_token",
       "token_to_id", "get_vocab", "vocab_size", "save", "from_pretrained"]


@pytest.mark.parametrize("name", ["bpe", "wordpiece", "unigram", "super_bpe", "morpheme", "byte_entropy"])
def test_every_class_has_the_uniform_api(trained, name, tmp_path):
    """P13: same API on every class; id_to_bytes is exact; save/from_pretrained round-trips ids."""
    tok = trained[name]
    for attr in API:
        assert hasattr(tok, attr), f"{name} lacks {attr}"
    for text in TEXTS:
        ids = tok.encode(text)
        assert ids == tok.encode_bytes(text.encode("utf-8"))
        assert b"".join(tok.id_to_bytes(i) for i in ids) == text.encode("utf-8")
        assert tok.decode_bytes(ids) == text.encode("utf-8")
        assert tok.decode(ids) == text
    tok.save(str(tmp_path))
    reloaded = type(tok).from_pretrained(str(tmp_path))
    assert reloaded.get_vocab() == tok.get_vocab()
    for text in TEXTS:
        assert reloaded.encode(text) == tok.encode(text)


@pytest.mark.parametrize("name", ["bpe", "wordpiece", "unigram", "super_bpe", "morpheme", "byte_entropy"])
def test_get_vocab_is_a_bijection_onto_all_ids(trained, name):
    """P2/P4/T15: one key per id, len == vocab_size, token_to_id/id_to_token are exact inverses.
    (Latin-1 keys for invalid-UTF-8 tokens collided with real UTF-8 tokens: 596 keys for 600.)
    Only a one-byte special token has a second key, as in tokenizer.json (test_bindings_hf_parity)."""
    tok = trained[name]
    vocab = tok.get_vocab()
    assert len(vocab) == tok.vocab_size
    assert sorted(vocab.values()) == list(range(tok.vocab_size))
    for token, token_id in vocab.items():
        assert tok.id_to_token(token_id) == token
        assert tok.token_to_id(token) == token_id
    assert tok.id_to_token(tok.vocab_size) is None
    assert tok.id_to_bytes(tok.vocab_size) is None
    assert tok.token_to_id("definitely not ☃ a token") is None


def test_token_strings_are_gpt2_byte_level(trained):
    """P4: token strings are the GPT-2 printable form (as in tokenizer.json), never `chr(c) as u8`."""
    tok = trained["bpe"]
    assert tok.token_to_id("Ā") == 0  # GPT-2 char of byte 0x00
    assert tok.token_to_id("Ġ") == 0x20  # GPT-2 char of the space byte
    assert tok.token_to_id("ł") == _core.printable_to_bytes("ł")[0] != ord("B")
    assert tok.id_to_token(0xE9) == "é" and tok.id_to_bytes(0xE9) == b"\xe9"
    assert tok.token_to_id("<pad>") == tok.vocab_size - 1  # special tokens keep their raw content
    assert _core.printable_to_bytes(_core.bytes_to_printable(bytes(range(256)))) == bytes(range(256))


@pytest.mark.parametrize("name", ["bpe", "unigram", "byte_entropy"])
def test_decode_invalid_utf8_and_unknown_ids(trained, name):
    """P3/P5: invalid UTF-8 is replaced per sequence (not latin-1 for the whole string); unknown ids raise."""
    tok = trained[name]
    raw = "héllo".encode("utf-8") + b"\xff"
    ids = tok.encode_bytes(raw)
    assert tok.decode_bytes(ids) == raw
    assert tok.decode(ids) == "héllo�"
    with pytest.raises(UnicodeDecodeError):
        tok.decode(ids, errors="strict")
    for bad in (tok.vocab_size, 10**6):
        with pytest.raises(ValueError):
            tok.decode([104, bad])
        with pytest.raises(ValueError):
            tok.decode_bytes([bad])


@pytest.mark.parametrize("name", ["bpe", "unigram", "byte_entropy"])
def test_negative_and_out_of_u32_ids_are_unknown_ids(trained, name):
    """An id outside u32 (HF's -100 label padding) is an unknown id in every class and in the HF
    wrapper: None from id_to_token/id_to_bytes, ValueError from decode; never OverflowError."""
    from tokenisers.wrapper import DynamicResearchTokenizer

    tok = trained[name]
    for bad in (-1, -100, 2**32):
        assert tok.id_to_token(bad) is None and tok.id_to_bytes(bad) is None
        with pytest.raises(ValueError):
            tok.decode_bytes([104, bad])
        with pytest.raises(ValueError, match="Invalid token ID"):
            DynamicResearchTokenizer(tokenizer=tok).decode([104, bad])


@pytest.mark.parametrize("scorer,param,value", [
    ("wp-bpe-scp", "alpha", 1.0),  # discount; was silently replaced by 0.75
    ("wp-bpe-npmi", "alpha", 1.0),  # was replaced by 0.5
    ("wp-bpe-ppig", "penalty", 0.0),  # length penalty; was replaced by 0.35
    ("mdl", "penalty", 0.0),  # entry cost; was replaced by 24 (it decides where MDL stops)
    ("wp-bpe-base", "min_frequency", 3),  # was ignored (always 1)
])
def test_explicit_scorer_parameters_are_honoured(corpus, scorer, param, value):
    """P7: an explicit value equal to the old 'unset' sentinel must change training."""
    explicit = tokenisers.BpeTokenizer.train(corpus, vocab_size=360, scorer=scorer, **{param: value})
    default = tokenisers.BpeTokenizer.train(corpus, vocab_size=360, scorer=scorer)
    assert explicit.get_vocab() != default.get_vocab()


@pytest.mark.parametrize("scorer,defaults", [
    ("wordpiece", dict(alpha=1.0, min_frequency=2)),
    ("mdl", dict(penalty=24.0, min_frequency=2)),
    ("length_penalized", dict(beta=0.5, min_frequency=1)),
    ("wp-bpe-scp", dict(alpha=0.75, min_frequency=2)),
    ("wp-bpe-npmi", dict(alpha=0.5, min_frequency=2)),
    ("wp-bpe-ppig", dict(penalty=0.35, min_frequency=2)),
])
def test_unset_scorer_parameters_take_the_scorer_defaults(corpus, scorer, defaults):
    """P7: None means the Rust scorer's Default (wordpiece/mdl min_frequency 2), the same whether
    trained through BpeTokenizer or the scorer's own class."""
    default = tokenisers.BpeTokenizer.train(corpus, 360, scorer=scorer).get_vocab()
    assert default == tokenisers.BpeTokenizer.train(corpus, 360, scorer=scorer, **defaults).get_vocab()
    if scorer == "wordpiece":
        assert default == tokenisers.WordPieceTokenizer.train(corpus, 360).get_vocab()
        lowered = tokenisers.BpeTokenizer.train(corpus, 360, scorer=scorer, min_frequency=1).get_vocab()
        assert lowered != default  # the corpus has singleton pairs, so the default is observable


def test_scorer_specific_names_and_parameter_checks(corpus):
    """P7/P8: subclass parameter names map to the generic ones; misuse raises instead of being ignored."""
    pairs = [
        (tokenisers.WpBpeScpTokenizer.train(corpus, 340, discount=1.0), dict(scorer="wp-bpe-scp", alpha=1.0)),
        (tokenisers.WpBpePpigTokenizer.train(corpus, 340, length_penalty=0.0), dict(scorer="wp-bpe-ppig", penalty=0.0)),
        (tokenisers.MdlTokenizer.train(corpus, 340, entry_cost=0.0), dict(scorer="mdl", penalty=0.0)),
        (tokenisers.LengthPenalizedTokenizer.train(corpus, 340, beta=2.0), dict(scorer="length_penalized", beta=2.0)),
    ]
    for sub, kwargs in pairs:
        assert sub.get_vocab() == tokenisers.BpeTokenizer.train(corpus, 340, **kwargs).get_vocab()
    with pytest.raises(ValueError, match="does not take"):
        tokenisers.BpeTokenizer.train(corpus, 300, scorer="frequency", alpha=0.5)
    with pytest.raises(ValueError, match="min_frequency"):
        tokenisers.WpBpeFtTokenizer.train(corpus, 300, min_frequency=1)
    for bad in (dict(scorer="wp-bpe-npmi", alpha=-0.5), dict(scorer="mdl", penalty=float("inf")),
                dict(scorer="length_penalized", beta=float("nan"))):
        with pytest.raises(ValueError, match="finite"):  # used to be clamped to 0 / accepted
            tokenisers.BpeTokenizer.train(corpus, 300, **bad)
    with pytest.raises(TypeError):
        tokenisers.BpeTokenizer.train(corpus, 300, entry_cost=8.0)  # generic class: use penalty / MdlTokenizer
    with pytest.raises(TypeError):
        tokenisers.WpBpeFtTokenizer.train(corpus, 300, alpha=1.0)  # wp-bpe-ft has no alpha
    with pytest.raises(TypeError):
        tokenisers.UnigramTokenizer.train(corpus, 300, bogus=1)
    with pytest.raises(TypeError):
        tokenisers.ByteEntropyTokenizer.train(corpus, bogus=1)
    with pytest.raises(TypeError):
        tokenisers.BpeTokenizer.train(corpus, 300).encode("x", temperature=1.0)


def test_super_bpe_honours_vocab_size(corpus):
    """P8: vocab_size used to be swallowed by **kwargs (asked 500, got the default 320)."""
    tok = tokenisers.SuperBpeTokenizer.train(corpus, vocab_size=360)
    assert tok.vocab_size == 360
    assert tokenisers.SuperBpeTokenizer.train(corpus, v_switch=300, v_final=360).vocab_size == 360
    with pytest.raises(ValueError):
        tokenisers.SuperBpeTokenizer.train(corpus, vocab_size=360, v_final=400)
    # 0 used to mean "unset" and silently trained the default 320
    for kwargs in ({"vocab_size": 0}, {"v_final": 0}):
        with pytest.raises(ValueError, match="at least 256"):
            tokenisers.SuperBpeTokenizer.train(corpus, **kwargs)


def test_unigram_training_parameters_are_used_or_rejected(corpus):
    """num_iterations was clamped to [1, 10] and 1 - shrinking_factor to [0.01, 0.5], silently;
    a NaN shrinking_factor passed the clamps and pruned nothing per round: an endless loop."""
    train = lambda size=600, **kw: tokenisers.UnigramTokenizer.train(corpus, vocab_size=size, **kw)
    assert train(num_iterations=50).get_log_probs() != train(num_iterations=10).get_log_probs()
    # 0.1 prunes the seed set down to 300 in fewer rounds than 0.5 (both were clamped to 0.5)
    assert train(300, shrinking_factor=0.1).get_vocab() != train(300, shrinking_factor=0.5).get_vocab()
    for bad in ({"num_iterations": 0}, {"shrinking_factor": float("nan")}, {"shrinking_factor": 1.0},
                {"shrinking_factor": 0.0}, {"shrinking_factor": -0.5}):
        with pytest.raises(ValueError):
            train(**bad)


def test_vocab_size_is_exact_and_includes_special_tokens(corpus):
    """P9/P10/T15: the requested size is the real size, special tokens included, no padding."""
    specials = ["<|endoftext|>", "<pad>", "<mask>"]
    tok = tokenisers.BpeTokenizer.train(corpus, vocab_size=400, special_tokens=specials)
    assert tok.vocab_size == 400
    assert [tok.token_to_id(s) for s in specials] == [397, 398, 399]
    assert len(tok.get_vocab()) == 400
    ids = tok.encode("a<pad>b<|endoftext|>")
    assert ids[1] == 398 and ids[-1] == 397
    assert tokenisers.UnigramTokenizer.train(corpus, vocab_size=400).vocab_size == 400
    with pytest.raises(ValueError):
        tokenisers.BpeTokenizer.train(corpus, vocab_size=258, special_tokens=specials)


@pytest.mark.parametrize("train", [
    tokenisers.BpeTokenizer.train,
    lambda corpus, size, **kw: tokenisers.MorphemeBpeTokenizer.train(corpus, size, morpheme_lexicon={}, **kw),
], ids=["bpe", "morpheme"])
def test_special_tokens_are_cut_out_of_the_training_corpus(corpus, train):
    """Specials were trained on as text: docs joined with "<|endoftext|>" learned fragments of it
    ("<|", "text|>") that encode never produces, the special reused a learned id mid-vocabulary and
    the vocabulary ended one short of vocab_size with a warning that blamed the corpus."""
    sep = "<|endoftext|>"
    joined = [sep.join(corpus[i:i + 4]) for i in range(0, len(corpus), 4)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tok = train(joined, 400, special_tokens=[sep])
    assert tok.vocab_size == 400 and tok.token_to_id(sep) == 399
    # Training sees exactly the text between the specials.
    plain = train([piece for doc in joined for piece in doc.split(sep)], 399)
    assert {k: i for k, i in tok.get_vocab().items() if k != sep} == plain.get_vocab()

    # A one-byte special keeps that byte's id, so it takes no slot of vocab_size.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tok = train(corpus, 400, special_tokens=["\n", "<pad>"])
    assert tok.vocab_size == 400 and tok.special_tokens() == {"\n": 10, "<pad>": 399}
    with pytest.raises(ValueError, match="non-empty"):
        train(corpus, 400, special_tokens=[""])


@pytest.mark.parametrize("make", [
    lambda: tokenisers.BpeTokenizer.train(["ab ab ab"], vocab_size=700),
    lambda: tokenisers.BpeTokenizer.train(["ab ab ab"], vocab_size=300, scorer="wp-bpe-npmi"),
    lambda: tokenisers.UnigramTokenizer.train(["ab ab ab"], vocab_size=3000),
    lambda: tokenisers.SuperBpeTokenizer.train(["ab ab ab"], vocab_size=300),
    lambda: tokenisers.ByteEntropyTokenizer.train(["ab ab ab"], vocab_size=300),
])
def test_unreachable_vocab_size_warns_with_the_real_size(make):
    """P9: no <unused_N> padding; the smaller real vocabulary is returned with a warning."""
    with pytest.warns(UserWarning, match="only supports") as record:
        tok = make()
    assert f"has {tok.vocab_size} tokens" in str(record[0].message)
    assert tok.vocab_size < 300
    assert not any("unused" in t for t in tok.get_vocab())


def test_from_hf_json_honours_file_ids_or_raises(trained, tmp_path):
    """P11: ids come from the JSON; a non-byte-level layout is an error, not silent garbage."""
    tok = trained["bpe"]
    tok.save(str(tmp_path / "orig"))
    data = json.loads((tmp_path / "orig" / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = data["model"]["vocab"]

    # A valid file with two learned tokens' ids swapped: the loader must use the file's ids (as HF does)
    hf = pytest.importorskip("tokenizers")
    a, b = tok.id_to_token(300), tok.id_to_token(301)
    vocab[a], vocab[b] = 301, 300
    (tmp_path / "swapped").mkdir()
    (tmp_path / "swapped" / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    swapped = tokenisers.BpeTokenizer.from_pretrained(str(tmp_path / "swapped"))
    reference = hf.Tokenizer.from_file(str(tmp_path / "swapped" / "tokenizer.json"))
    assert swapped.token_to_id(a) == 301
    for text in TEXTS + ["x<pad>y<|endoftext|>"]:
        assert swapped.encode(text) == reference.encode(text, add_special_tokens=False).ids
        assert swapped.decode(swapped.encode(text)) == text

    # Byte tokens moved away from ids 0..255: not byte-level, must raise
    vocab[a], vocab[b] = 300, 301
    vocab["a"], vocab[a] = 300, ord("a")
    (tmp_path / "moved").mkdir()
    (tmp_path / "moved" / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="byte-level"):
        tokenisers.BpeTokenizer.from_pretrained(str(tmp_path / "moved"))


def test_unigram_from_hf_json_rejects_reordered_bytes(trained, tmp_path):
    """P11: a Unigram file whose byte tokens are not ids 0..255 used to load with shifted ids."""
    tok = trained["unigram"]
    tok.save(str(tmp_path))
    path = tmp_path / "tokenizer.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["model"]["vocab"] = data["model"]["vocab"][::-1]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="byte-level"):
        tokenisers.UnigramTokenizer.from_pretrained(str(tmp_path))


def test_unigram_sampling_matches_the_exact_posterior(two_path_unigram):
    """P1: temperature is unscaled (T=1 is the posterior) and nbest_size=-1 means all paths."""
    tok = two_path_unigram
    ab = tok.token_to_id("ab")
    assert tok.encode("ab") == [ab]  # Viterbi: 0.03 > 0.1 * 0.1
    for kwargs in ({"sample": True}, {"nbest_size": -1}, {"temperature": 1.0}, {"nbest_size": 2}):
        draws = [tuple(tok.encode("ab", seed=s, **kwargs)) for s in range(2000)]
        share = draws.count((ab,)) / len(draws)
        assert abs(share - 0.75) < 0.04, (kwargs, share)
    for kwargs in ({"nbest_size": 1}, {"nbest_size": 0}, {"temperature": 0.0}, {"sample": False, "temperature": 1.0}):
        assert {tuple(tok.encode("ab", seed=s, **kwargs)) for s in range(50)} == {(ab,)}, kwargs
    with pytest.raises(ValueError):
        tok.encode("ab", temperature=-1.0)


class TestByteEntropy:
    """P12: trained Markov model + frozen patch vocabulary, validated parameters, save/load."""

    def test_encode_requires_training(self):
        tok = tokenisers.ByteEntropyTokenizer()
        with pytest.raises(RuntimeError, match="train"):
            tok.encode("abc")

    @pytest.mark.parametrize("kwargs", [
        {"min_patch_len": 0}, {"min_patch_len": 5, "max_patch_len": 4}, {"max_patch_len": 0},
        {"smoothing_alpha": -0.1}, {"entropy_threshold": float("nan")}, {"entropy_threshold": 0.0},
    ])
    def test_invalid_parameters_raise(self, kwargs):
        with pytest.raises(ValueError):
            tokenisers.ByteEntropyTokenizer(**kwargs)

    def test_vocabulary_is_frozen_and_ids_are_call_order_independent(self, corpus):
        tok = tokenisers.ByteEntropyTokenizer.train(corpus, entropy_threshold=3.0)
        size = tok.vocab_size
        first = [tok.encode(t) for t in TEXTS]
        assert [tok.encode(t) for t in reversed(TEXTS)][::-1] == first
        assert tok.vocab_size == size
        assert all(i < size for ids in first for i in ids)
        with pytest.raises(ValueError):
            tok.decode([size])  # used to decode as bytes([id % 256])

    def test_threshold_changes_segmentation(self, sample_english_text):
        corpus = [sample_english_text * 20]
        low = tokenisers.ByteEntropyTokenizer.train(corpus, entropy_threshold=0.5, min_patch_len=1, smoothing_alpha=0.01)
        high = tokenisers.ByteEntropyTokenizer.train(corpus, entropy_threshold=2.0, min_patch_len=1, smoothing_alpha=0.01)
        assert len(high.encode(sample_english_text)) < len(low.encode(sample_english_text))

    def test_training_memory_is_linear_and_small(self):
        """The Markov table held 256 dense counters (~1 KB) per distinct context: 256 KB of random
        bytes (~320k contexts at order 3) took ~330 MB, a few MB of binary/multilingual data GBs."""
        code = textwrap.dedent("""
            import random, resource, tokenisers
            data = random.Random(0).randbytes(256_000)
            before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            tokenisers.ByteEntropyTokenizer.train([data], vocab_size=256)
            print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) // 1024)
        """)
        grown_mb = int(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout)
        assert grown_mb < 100, grown_mb
        with pytest.raises(ValueError, match="context_order"):
            tokenisers.ByteEntropyTokenizer(context_order=64)

    def test_vocab_size_caps_patches_by_frequency(self, corpus):
        full = tokenisers.ByteEntropyTokenizer.train(corpus, entropy_threshold=3.0)
        capped = tokenisers.ByteEntropyTokenizer.train(corpus, vocab_size=300, entropy_threshold=3.0)
        assert full.vocab_size > 300 and capped.vocab_size == 300
        assert all(capped.id_to_bytes(i) == full.id_to_bytes(i) for i in range(300))
        for text in TEXTS:
            assert capped.decode(capped.encode(text)) == text


def test_encode_releases_the_gil(trained):
    """P6: another Python thread keeps running while encode works."""
    tok = trained["bpe"]
    text = synthetic_text(60_000, 1) * 30  # ~12 MB

    def spin(seconds):
        n, end = 0, time.perf_counter() + seconds
        while time.perf_counter() < end:
            n += 1
        return n

    rate = spin(0.1) / 0.1
    done, elapsed = threading.Event(), []

    def work():
        start = time.perf_counter()
        tok.encode(text)
        elapsed.append(time.perf_counter() - start)
        done.set()

    worker = threading.Thread(target=work)
    worker.start()
    ticks = 0
    while not done.is_set():
        ticks += 1
        time.perf_counter()
    worker.join()
    assert elapsed[0] > 0.1, "input too small to measure"
    # Holding the GIL lets this thread run at most a couple of 5 ms switch intervals.
    assert ticks > 0.2 * rate * elapsed[0], (ticks, rate, elapsed)


def test_encode_time_scales_linearly():
    """T15: encode is ~linear in input length (was quadratic: 10 KB 0.17 s, 80 KB 10.8 s)."""
    tok = tokenisers.BpeTokenizer.train([synthetic_text(2000, i) for i in range(20)], vocab_size=2000)
    small, large = synthetic_text(25_000, 7), synthetic_text(100_000, 8)

    def best(text):
        return min(_timed(tok.encode, text) for _ in range(3))

    t_small, t_large = best(small), best(large)
    assert t_large / t_small < 8, (t_small, t_large)  # 4x input; quadratic would be ~16x
    assert len(large) / t_large > 2e6, "expected well over 2 MB/s"


def test_dropout_reuses_the_model_instead_of_rebuilding_merges():
    """C9-binding: dropout encode cost is per text, not per merge table (was 140 us vs 0.3 us at 20k)."""
    tok = tokenisers.BpeTokenizer.train([synthetic_text(4000, i) for i in range(40)], vocab_size=4000)
    short = "kato rimen salo"
    plain = min(_timed(lambda: [tok.encode(short) for _ in range(500)]) for _ in range(3))
    dropout = min(_timed(lambda: [tok.encode(short, dropout=0.1, seed=1) for _ in range(500)]) for _ in range(3))
    assert dropout < 3 * plain + 0.005, (plain, dropout)


def _timed(fn, *args):
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


_REPRO_SCRIPT = textwrap.dedent("""
    import hashlib, sys
    import tokenisers
    corpus = [l for l in sys.stdin.read().split("\\x1e") if l]
    out = []
    for scorer in ("frequency", "wordpiece", "mdl", "wp-bpe-npmi", "length_penalized"):
        tok = tokenisers.BpeTokenizer.train(corpus, vocab_size=420, scorer=scorer)
        out.append(sorted(tok.get_vocab().items()))
    out.append(sorted(tokenisers.SuperBpeTokenizer.train(corpus, vocab_size=420).get_vocab().items()))
    out.append(sorted(tokenisers.UnigramTokenizer.train(corpus, vocab_size=420).get_log_probs().items()))
    print(hashlib.sha256(repr(out).encode()).hexdigest())
""")


def test_training_is_reproducible_across_processes(corpus):
    """U2-binding/T15: hash seeds differ per process; corpus order must not matter either."""
    digests = set()
    for docs in (corpus, corpus[::-1]):
        proc = subprocess.run([sys.executable, "-c", _REPRO_SCRIPT], input="\x1e".join(docs),
                              capture_output=True, text=True, check=True)
        digests.add(proc.stdout.strip())
    assert len(digests) == 1


def test_import_does_not_load_torch_or_transformers():
    """P17: `import tokenisers` stays light; the HF wrapper loads on first use."""
    code = ("import sys, tokenisers; print('torch' in sys.modules, 'transformers' in sys.modules); "
            "tokenisers.DynamicResearchTokenizer; print('transformers' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False", "True"]
    # `from tokenisers import *` too: __all__ used to list the lazy names
    code = "import sys; from tokenisers import *; print('torch' in sys.modules, 'transformers' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


def test_byte_entropy_table_is_bounded_by_max_contexts(corpus, tmp_path):
    """context_order <= 8 did not bound training memory (order 8: ~200 bytes of table per corpus
    byte); the table now holds at most max_contexts contexts."""
    tok = tokenisers.ByteEntropyTokenizer.train(corpus, context_order=8, max_contexts=500)
    assert len(tok._patcher.transitions()) == 500
    for text in TEXTS:
        assert tok.decode(tok.encode(text)) == text
    tok.save(str(tmp_path))
    reloaded = tokenisers.ByteEntropyTokenizer.from_pretrained(str(tmp_path))
    assert reloaded.config["max_contexts"] == 500
    assert all(reloaded.encode(t) == tok.encode(t) for t in TEXTS)
    with pytest.raises(ValueError):
        tokenisers.ByteEntropyTokenizer(max_contexts=0)


def test_broken_extension_fails_instead_of_skipping(tmp_path, repo_root):
    """T14: a missing/broken compiled module must error the suite, not silently skip ~400 tests."""
    fake = tmp_path / "tokenisers"
    fake.mkdir()
    (fake / "__init__.py").write_text("raise ImportError('simulated broken extension')\n")
    test = repo_root / "tests" / "e2e" / "test_tier1_features.py"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"{test}::test_tier1_f1_bpe_frequency_encode_decode_roundtrip"],
        capture_output=True, text=True, cwd=tmp_path, env={**_env(), "PYTHONPATH": str(tmp_path)},
    )
    assert proc.returncode != 0, proc.stdout
    assert "1 error" in proc.stdout and "skipped" not in proc.stdout, proc.stdout


def _env():
    import os

    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
