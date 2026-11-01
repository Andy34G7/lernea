"""``pretokenizer="space_prefix"`` (GPT-2-style: a word takes the one space before it) through the
Python API, the Hugging Face export and the benchmark plumbing; and the default ``"whitespace"``
mode byte-identical to the tokenizers trained before the option existed."""

import contextlib
import copy
import hashlib
import json
import random
import subprocess
import sys
import warnings

import pytest
import tokenizers
import transformers

import tokenisers
from tokenisers import _core
from tokenisers.benchmarks import common
from tokenisers.benchmarks.report import render_nanogpt_swap_space_prefix

# ----------------------------------------------------------------------------
# Default mode unchanged
# ----------------------------------------------------------------------------

_WORDS = ("the of and to in data token model byte text naïve café déjà Ünïcödé 東京 中文文本 🙂 "
          "it's don't (x) [y] {z} 3.14 $12 -- foo_bar").split(" ")
_SEPS = [" ", " ", " ", "  ", "\t", "\n", " \n", "　", " ", "\r\n", "   "]


def _golden_corpus(n=120, seed=0):
    rng = random.Random(seed)
    docs = []
    for _ in range(n):
        k = rng.randint(3, 40)
        parts = [rng.choice(_WORDS) + rng.choice(_SEPS) for _ in range(k)]
        docs.append(rng.choice(["", "", " ", "  ", "\n"]) + "".join(parts))
    return docs


GOLDEN_CORPUS = _golden_corpus()
GOLDEN_PROBES = [d.encode() for d in _golden_corpus(40, 1)] + [
    b"a  b\t\tc \n d\xe3\x80\x80e", b"\xff\xfe  x \xc3", b"", b" ", b"  "]

# sha256[:16] of (exported tokenizer.json, json.dumps of the probe ids) for each variant, recorded
# with the extension built from 3779f66, before the pretokenizer option existed.
GOLDEN = {
    "frequency": ("892c1035ed32e1ba", "edb25431593b340f"),
    "wordpiece": ("985fdca0edd24135", "f072c89d5af293d1"),
    "wp-bpe-base": ("1b8f06ecc88a4427", "0779e5cae286fa62"),
    "wp-bpe-ft": ("1b8f06ecc88a4427", "0779e5cae286fa62"),
    "wp-bpe-pig": ("97b7c65e847c20a7", "1502001d053d87c6"),
    "wp-bpe-scp": ("dc2692887abbd693", "6f8e1acd1a91cfd7"),
    "wp-bpe-npmi": ("9484f612ac910509", "87f53dfe2c69130a"),
    "wp-bpe-ppig": ("5a8fa7d837d0564d", "070bb6d941eafae7"),
    "mdl": ("3edbfb6c5719c7cb", "b5559e72241b0d36"),
    "length_penalized": ("e77b6c1655b847b5", "6102b9955c209889"),
    "super_bpe": ("e210a87b42920a95", "e8f950f7f7a521b5"),
    "unigram": ("2ed28acdec0d197b", "e062877ee3f69646"),
    "morpheme": ("4e90f2a6a0ef741a", "a665badf270ed2bd"),
}


def _golden_train(variant):
    if variant == "super_bpe":
        return _core.train_super_bpe(GOLDEN_CORPUS, 330, 460)
    if variant == "unigram":
        return _core.train_unigram(GOLDEN_CORPUS, 500)
    if variant == "morpheme":
        with pytest.warns(UserWarning, match="stopped at"):
            return _core.train_morpheme_guided(GOLDEN_CORPUS, 420, {"token": ["tok", "en"], "model": ["mod", "el"]})
    return _core.train_bpe(GOLDEN_CORPUS, 420, scorer_type=variant)


@pytest.mark.parametrize("variant", list(GOLDEN))
def test_default_mode_is_byte_identical_to_before(variant, tmp_path):
    """Same vocabulary, merges/scores and pre-tokenizer (the whole tokenizer.json) and same ids."""
    model = _golden_train(variant)
    model.export_hf_json(str(tmp_path / "tokenizer.json"))
    js = (tmp_path / "tokenizer.json").read_bytes()
    ids = json.dumps([model.encode_bytes(p) for p in GOLDEN_PROBES]).encode()
    got = (hashlib.sha256(js).hexdigest()[:16], hashlib.sha256(ids).hexdigest()[:16])
    assert got == GOLDEN[variant]
    assert model.pretokenizer() == (None if variant == "super_bpe" else "whitespace")


# ----------------------------------------------------------------------------
# HF parity in space_prefix mode
# ----------------------------------------------------------------------------

CORPUS = [f"{d} {r}" for i, d in enumerate([
    "The quick brown fox jumps over the lazy dog. Don't stop!",
    "hello, world! hello, world! hello,world -- (parens) [brackets] {braces}",
    "def foo(x):\n    return x + 1  # comment\n\n\tif x: pass",
    "price: $123.45, abc123 xyz789 1234567 3.14159e-10",
    "Héllo wörld, naïve café résumé façade",
    "日本語のテキスト、中文文本。한국어 텍스트",
    "中文　文本、中文。日本語　の　テキスト",
    "emoji 🚀🔥💯 party 🎉🎉 👩‍👩‍👧",
    "   leading spaces   and   runs\t\ttabs\n\nnewlines\r\n",
    "non breaking space and em thin　ideographic",
    "it's they're we've I'm y'all",
]) for r in range(5 + i)]


def _random_texts(n, seed):
    rng = random.Random(seed)
    pieces = [" ", " ", " ", "  ", "\t", "\n", "\r\n", "　", " ", " ", "\u0085", "the", "fox", "x.",
              "中文", "é", "🚀", "​", "﻿", "1", "<|endoftext|>"]
    return ["".join(rng.choice(pieces) for _ in range(rng.randint(0, 25))) for _ in range(n)]


TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "hello, world!",
    "def foo(x):\n    return x + 1  # comment",
    "Héllo wörld, naïve café",
    "日本語のテキスト 中文文本",
    "中文　文本、中文。",
    "emoji 🚀🔥💯 party 🎉 👩‍👩‍👧",
    "a  b   c    d",
    "   leading",
    "trailing   ",
    "tabs\t\tand\n\nnewlines\r\n",
    "a \tb\t c \nd\n e",
    "x 　y　 z  w",
    "non breaking　ideo line\u0085nel",
    "zero​width ﻿bom",
    "hello<|endoftext|> world <pad> x",
    " ",
    "",
    "x",
] + _random_texts(200, 7)

SP = "space_prefix"
SPECIALS = ["<|endoftext|>", "<pad>"]


def _sp_corpus():
    return CORPUS + _web_docs(30, 3)  # English prose, so that most scorers learn " word" tokens


MAKERS = {
    **{s: (lambda s=s: tokenisers.BpeTokenizer.train(_sp_corpus(), 420, scorer=s, pretokenizer=SP))
       for s in ["frequency", "wordpiece", "wp-bpe-base", "wp-bpe-ft", "wp-bpe-pig", "wp-bpe-scp", "wp-bpe-npmi",
                 "wp-bpe-ppig", "mdl", "length_penalized"]},
    "bpe+specials": lambda: tokenisers.BpeTokenizer.train(_sp_corpus(), 420, special_tokens=SPECIALS,
                                                          pretokenizer=SP),
    "morpheme": lambda: tokenisers.MorphemeBpeTokenizer.train(
        _sp_corpus(), 420, morpheme_lexicon={"jumps": ["jump", "s"], "quick": ["qu", "ick"]}, lambda_weight=2.0,
        pretokenizer=SP),
    "unigram": lambda: tokenisers.UnigramTokenizer.train(_sp_corpus(), 500, pretokenizer=SP),
}
# PMI ratios rank pairs with the ubiquitous space last: at this size they learn no " word" token.
NO_SPACED_TOKENS = {"wordpiece", "wp-bpe-base", "wp-bpe-ft", "wp-bpe-scp"}


@pytest.fixture(scope="module", params=list(MAKERS))
def saved(request, tmp_path_factory):
    tok = MAKERS[request.param]()
    out = tmp_path_factory.mktemp(request.param.replace("+", "_").replace("-", "_"))
    tok.save(str(out))
    return request.param, tok, out


def test_space_prefix_export_uses_its_own_split(saved):
    name, tok, out = saved
    assert tok.pretokenizer == SP
    split = json.loads((out / "tokenizer.json").read_text())["pre_tokenizer"]["pretokenizers"][0]
    assert split["type"] == "Split" and split["behavior"] == "Isolated"
    assert "?=" in split["pattern"]["Regex"], "the whitespace-run regex would split ' the' apart"
    spaced = any(t.startswith("Ġ") and len(t) > 1 for t in tok.get_vocab())
    assert spaced != (name in NO_SPACED_TOKENS), f"[{name}] space-prefixed tokens: {spaced}"


def test_space_prefix_ids_equal_hf_tokenizers_and_transformers(saved):
    name, tok, out = saved
    hf = tokenizers.Tokenizer.from_file(str(out / "tokenizer.json"))
    auto = transformers.AutoTokenizer.from_pretrained(str(out))
    for t in TEXTS:
        ids = tok.encode(t)
        assert hf.encode(t, add_special_tokens=False).ids == ids, f"[{name}] tokenizers {t!r}"
        assert auto.encode(t, add_special_tokens=False) == ids, f"[{name}] transformers {t!r}"
        assert hf.decode(ids, skip_special_tokens=False) == t, f"[{name}] decode {t!r}"


def test_space_prefix_reload_round_trips(saved, tmp_path):
    name, tok, out = saved
    again = type(tok).from_pretrained(str(out))
    assert again.pretokenizer == SP
    assert [again.encode(t) for t in TEXTS] == [tok.encode(t) for t in TEXTS]
    again.save(str(tmp_path))
    assert (tmp_path / "tokenizer.json").read_bytes() == (out / "tokenizer.json").read_bytes()


def test_space_prefix_survives_a_transformers_resave(saved, tmp_path):
    name, tok, out = saved
    transformers.AutoTokenizer.from_pretrained(str(out)).save_pretrained(str(tmp_path))
    again = type(tok).from_pretrained(str(tmp_path))
    assert again.pretokenizer == SP and [again.encode(t) for t in TEXTS] == [tok.encode(t) for t in TEXTS]


_DETERMINISM_SCRIPT = """
import sys
from tokenisers import _core
corpus = [l for l in sys.stdin.read().split("\\x1e") if l]
_core.train_unigram(corpus, 500, pretokenizer="space_prefix").export_hf_json(sys.argv[1])
"""


def test_space_prefix_unigram_is_identical_across_processes(tmp_path):
    outputs = []
    for i, corpus in enumerate([CORPUS, CORPUS[::-1]]):
        path = tmp_path / f"run{i}.json"
        subprocess.run([sys.executable, "-c", _DETERMINISM_SCRIPT, str(path)], input="\x1e".join(corpus), text=True,
                       check=True)
        outputs.append(path.read_bytes())
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0])["pre_tokenizer"] == json.loads(
        tokenizers.Tokenizer.from_str(outputs[0].decode()).to_str())["pre_tokenizer"]


# ----------------------------------------------------------------------------
# Lossless on FineWeb-like text; compression
# ----------------------------------------------------------------------------

@contextlib.contextmanager
def _quiet():
    """Training on a small corpus may stop short of vocab_size (a UserWarning); irrelevant here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _web_docs(n, seed):
    rng = random.Random(seed)
    words = ("the a of to and in is that for it with as was on be by this are from at or have an they which "
             "one you were her all she there would their we him been has when who will more no if out so said "
             "what up its about into than them can only other new some could time these two may then do first "
             "any my now such like our over man me even most made after also did many before must through "
             "Tokenization Café résumé naïve 2024 3.5% $1,299 U.S. e-mail don't it's “quoted” — – … 🙂").split(" ")
    docs = []
    for _ in range(n):
        paras = []
        for _ in range(rng.randint(1, 5)):
            sents = []
            for _ in range(rng.randint(1, 6)):
                s = " ".join(rng.choice(words) for _ in range(rng.randint(3, 20)))
                sents.append(s[:1].upper() + s[1:] + rng.choice([".", ".", "!", "?", ":"]))
            paras.append(rng.choice([" ", " ", "  "]).join(sents))
            if rng.random() < 0.2:
                paras.append("\n".join(f"- item {k}: https://example.com/p?id={k}&q=a+b" for k in range(3)))
            if rng.random() < 0.1:
                paras.append("def f(x):\n\treturn x  # tab\n    pass ")
        docs.append(rng.choice(["\n\n", "\n", "\r\n\r\n"]).join(paras) + rng.choice(["", "\n", " "]))
    return docs


def test_space_prefix_is_lossless_on_web_text_and_packs_more_bytes_per_token():
    train, held = _web_docs(300, 1), _web_docs(150, 2)
    nbytes = sum(len(d.encode()) for d in held)
    for cls in (tokenisers.BpeTokenizer, tokenisers.WpBpePpigTokenizer, tokenisers.MdlTokenizer,
                tokenisers.UnigramTokenizer):
        with _quiet():
            ws, sp = cls.train(train, 1200), cls.train(train, 1200, pretokenizer=SP)
        assert (ws.pretokenizer, sp.pretokenizer) == ("whitespace", SP)
        for d in held:
            assert sp.decode(sp.encode(d)) == d
        raw = "Ünï 中文 ".encode() + b"\xff\xfe \xc3 \xe3\x80 x\n \x80"
        assert sp.decode_bytes(sp.encode_bytes(raw)) == raw
        ws_tokens, sp_tokens = (sum(len(t.encode(d)) for d in held) for t in (ws, sp))
        assert nbytes / sp_tokens > 1.1 * nbytes / ws_tokens, cls.__name__


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------

def test_unknown_pretokenizer_is_rejected():
    for train in (lambda: tokenisers.BpeTokenizer.train(CORPUS, 300, pretokenizer="gpt-2"),
                  lambda: tokenisers.WordPieceTokenizer.train(CORPUS, 300, pretokenizer="Whitespace"),
                  lambda: tokenisers.UnigramTokenizer.train(CORPUS, 300, pretokenizer=""),
                  lambda: tokenisers.MorphemeBpeTokenizer.train(CORPUS, 300, pretokenizer="space-prefix")):
        with pytest.raises(ValueError, match="pretokenizer must be"):
            train()


def test_pretokenizer_property():
    assert tokenisers.BpeTokenizer().pretokenizer is None
    assert tokenisers.LengthPenalizedTokenizer.train(CORPUS, 300).pretokenizer == "whitespace"
    assert tokenisers.LengthPenalizedTokenizer.train(CORPUS, 300, pretokenizer=SP).pretokenizer == SP
    assert tokenisers.SuperBpeTokenizer.train(CORPUS, vocab_size=460).pretokenizer is None  # whole text


def test_whitespace_mode_keeps_runs_apart_but_a_run_can_be_one_token(repo_root):
    """What README.md and models.py say of the default mode (the README once said every space was a token)."""
    with _quiet():
        tok = tokenisers.BpeTokenizer.train(["def f(x):\n    return x  # a b\n        pass\n"] * 50, 300)
    pieces = [tok.id_to_bytes(i) for i in tok.encode("a    b  c \n d")]
    assert b"    " in pieces and b"  " in pieces
    assert all(p.isspace() or not set(p) & set(b" \n") for p in pieces)
    assert "every space is a token" not in (repo_root / "README.md").read_text()


def test_space_prefixed_words_get_the_lexicon_boundaries():
    """A lexicon word also annotates its " word" segment (the space joins the first morpheme), so a
    strong prior keeps " walk|ing" apart; without the lexicon the word becomes one token."""
    corpus = ["we were walking and walking and walking home"] * 50
    lex = {"walking": ["walk", "ing"]}
    plain = tokenisers.BpeTokenizer.train(corpus, 300, pretokenizer=SP)
    guided = tokenisers.MorphemeBpeTokenizer.train(corpus, 300, morpheme_lexicon=lex, lambda_weight=50.0,
                                                   pretokenizer=SP)
    assert [plain.id_to_bytes(i) for i in plain.encode(" walking")] == [b" walking"]
    pieces = [guided.id_to_bytes(i) for i in guided.encode(" walking")]
    assert b"".join(pieces) == b" walking" and b" walk" in pieces and b"ing" in pieces, pieces


# ----------------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------------

def test_space_prefix_variants_train_like_their_base_variant():
    from tokenisers.benchmarks.nanogpt_swap.__main__ import OUR_VARIANTS, SP_VARIANTS

    assert SP_VARIANTS == ("bpe_frequency_sp", "wp_bpe_pig_sp", "wp_bpe_ppig_sp", "wp_bpe_npmi_sp", "unigram_lm_sp",
                           "length_penalized_sp", "morpheme_bpe_sp", "mdl_bits_sp")
    assert set(SP_VARIANTS) <= set(common.SPACE_PREFIX_VARIANTS) and "super_bpe_sp" not in common.SPACE_PREFIX_VARIANTS
    with pytest.raises(ValueError):
        common.variant_config("super_bpe_sp", 300)
    corpus = ["the cat sat on the mat with the hat, a bat ate a gnat " * 5, "zyx qwv jk  walking"]
    with _quiet():
        for name in OUR_VARIANTS:
            base = common.variant_config(name, 290)
            if name == "super_bpe":
                assert "pretokenizer" not in base
                continue
            sp = common.variant_config(f"{name}_sp", 290)
            assert base.pop("pretokenizer") == "whitespace" and sp.pop("pretokenizer") == SP and base == sp
            tok = common.train_variant(f"{name}_sp", corpus, 290, {"walking": ["walk", "ing"]})
            assert tok.pretokenizer == SP, name
            assert common.train_variant(name, corpus, 290, {}).pretokenizer == "whitespace", name


def _swap_doc(models, **meta):
    m = {"recipe": {"args": {"num_iterations": "600", "learning_rate": "0.0006"}},
         "tokenizer_text": {"docs_sha256": "t"}, "git_commit": "0123456789abcdef", "git_dirty": False,
         "created_utc": "2026-11-02T00:00:00Z", "backend": {"gpu": "A100"}}
    m.update(meta)
    return {"meta": m, "models": models}


def _row(bpt, bpb):
    return {"val_bytes_per_token": bpt, "final_bpb": bpb, "val_docs_sha256": "v"}


def test_readme_comparison_renders_once_the_space_prefix_run_exists(repo_root, tmp_path):
    """Until the _sp run's JSON is committed the section says so without numbers; afterwards it
    renders old vs space-prefix per variant (and GPT-2) from the two result files."""
    import importlib.util

    from tokenisers.benchmarks.nanogpt_swap.__main__ import SP_VARIANTS

    rel_base = "benchmarks/results/nanogpt_swap/nanogpt_swap.json"
    base = json.loads((repo_root / rel_base).read_text(encoding="utf-8"))
    pending = render_nanogpt_swap_space_prefix(base, None)
    assert "not been run yet" in pending and not any(c.isdigit() for c in pending)

    # A stand-in _sp run: the committed run's meta, the 8 variants renamed, more bytes per token.
    sp = {"meta": copy.deepcopy(base["meta"]), "models": {}}
    for name in SP_VARIANTS:
        m = copy.deepcopy(base["models"][name[:-3]])
        m.update(val_bytes_per_token=m["val_bytes_per_token"] + 1.5, final_bpb=m["final_bpb"] - 0.1)
        sp["models"][name] = m
    del sp["models"]["unigram_lm_sp"]["final_bpb"]  # a run that did not finish
    md = render_nanogpt_swap_space_prefix(base, sp)
    old = base["models"]["bpe_frequency"]
    assert (f"| `bpe_frequency` | {old['val_bytes_per_token']:.3f} | {old['val_bytes_per_token'] + 1.5:.3f} | "
            f"{old['final_bpb']:.4f} | {old['final_bpb'] - 0.1:.4f} | -0.1000 |") in md
    assert "| `unigram_lm` |" in md and md.split("| `unigram_lm` |")[1].split("\n")[0].endswith("| - | - |")
    g = base["models"]["gpt2"]
    assert f"| `gpt2` (reference) | {g['val_bytes_per_token']:.3f} | | {g['final_bpb']:.4f} | | |" in md
    assert "same recipe, val documents and tokenizer training text" in md
    assert md.count("\n| `") == 9  # 8 variants + GPT-2
    other = copy.deepcopy(sp)
    other["meta"]["recipe"]["args"]["learning_rate"] = "0.0018"
    assert "DIFFERENT recipe" in render_nanogpt_swap_space_prefix(base, other)

    spec = importlib.util.spec_from_file_location("gen_readme_tables", repo_root / "scripts" / "gen_readme_tables.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    rels = gen.SECTIONS["nanogpt_swap_sp"][0]
    assert rels[0] == rel_base
    for rel, doc in zip(rels, (base, sp)):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(json.dumps(doc))
    readme = "".join(f"<!-- BEGIN GENERATED: {s} -->\nold\n<!-- END GENERATED: {s} -->\n" for s in gen.SECTIONS)
    section = gen.render_readme(readme, tmp_path).split("BEGIN GENERATED: nanogpt_swap_sp -->")[1]
    assert section.startswith("\n" + md + "\n<!-- END")


def test_swap_cli_runs_every_tokenizer_of_a_space_prefix_job(tmp_path, monkeypatch):
    """A job prepared with only the _sp tokenizers is run without --tokenizers: its config decides
    (the original default list has no _sp names and would run nothing). A --tokenizers subset only
    narrows what the VM gets: the job keeps every spec, so a later plain run still runs all 8."""
    from tokenisers.benchmarks.nanogpt_swap import __main__ as swap

    # Stop right after the run config is written: the VM build refuses a dirty tree.
    monkeypatch.setattr(common, "start_provenance", lambda *a: {"git_dirty": True})
    job = tmp_path / "j"
    (job / "runs").mkdir(parents=True)
    (job / "runs" / "bpe_frequency_sp.json").write_text("{}")
    specs = [{"name": n, "kind": "tokenisers"} for n in swap.SP_VARIANTS]
    (job / "config.json").write_text(json.dumps({"tokenizers": specs, "trainer_args": swap.recipe_args(600, 100)}))
    # The subset run first; config.json is not restored between the two (refused) runs.
    for extra, expected in ((["--tokenizers", "mdl_bits_sp"], ["mdl_bits_sp"]), ([], list(swap.SP_VARIANTS))):
        with pytest.raises(SystemExit, match="commit code changes first"):
            swap.main(["--job-dir", str(job), "--gpu", "A100", "--eval-every", "100", "--resume", *extra])
        vm = json.loads((job / "vm" / "config.json").read_text())
        assert [t["name"] for t in vm["tokenizers"]] == vm["train"] == expected
        assert json.loads((job / "config.json").read_text())["tokenizers"] == specs
