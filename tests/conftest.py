"""
Pytest configuration and shared fixtures for Tokenisers E2E test suite.
"""

from pathlib import Path
import sys

import pytest

# Tests use the installed package (`uv run maturin develop`), never a sys.path shim onto python/.
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def _isolated_data_cache(tmp_path_factory):
    """Datasets that tests build without an explicit cache_dir (DataPipeline, SigmorphonLoader, the
    dashboard's TokenizerCache, CLI subprocesses) go to a per-session temp dir, never the user's
    ~/.cache/tokenisers/data, which concurrent runs would otherwise share and rewrite."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TOKENISERS_CACHE_DIR", str(tmp_path_factory.mktemp("tokenisers-cache")))
        yield


@pytest.fixture(autouse=True)
def _no_real_colab_cli(monkeypatch):
    """A ColabOrchestrator built without ``colab_bin`` runs the user's signed-in CLI
    (~/.local/bin/colab), which opens paid sessions. In tests it gets a path that does not exist, so a
    test that reaches a CLI's Colab path by mistake fails instead of starting a VM."""
    # ponytail: patched only once imported (importing it here costs every session torch: ~3 s, 0.5 GB).
    # Test files import colab at module level, so it is loaded before any test of theirs runs.
    colab = sys.modules.get("tokenisers.benchmarks.colab")
    if colab is not None:
        init = colab.ColabOrchestrator.__init__
        monkeypatch.setattr(colab.ColabOrchestrator, "__init__", lambda self, mock=False, colab_bin=None:
                            init(self, mock=mock, colab_bin=colab_bin or "/nonexistent/colab-cli-in-tests"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return repository root path."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def colab_cli_path() -> Path:
    """The colab CLI where the Colab benchmark looks by default (ColabRunner.colab_bin)."""
    return Path.home() / ".local" / "bin" / "colab"


@pytest.fixture(scope="session")
def sample_english_text() -> str:
    """Standard English text sample for tokenization tests."""
    return (
        "Tokenization is the foundational step in natural language processing and modern "
        "large language models. High-quality tokenization balances compression efficiency, "
        "subword fertility, and morphological boundary alignment across diverse domains."
    )


@pytest.fixture(scope="session")
def sample_multilingual_text() -> dict:
    """Multilingual text samples across diverse scripts."""
    return {
        "en": "The quick brown fox jumps over the lazy dog.",
        "de": "Falsches Üben von Xylophonmusik quält jeden größeren Zwerg.",
        "fr": "Portez ce vieux whisky au juge blond qui fume.",
        "ru": "Съешь же ещё этих мягких французских булок, да выпей чаю.",
        "zh": "道可道，非常道；名可名，非常名。无名天地之始，有名万物之母。",
        "ja": "いろはにほへと ちりぬるを わかよたれそ つねならむ うゐのおくやま けふこえて あさきゆめみし ゑひもせす",
        "ar": "نص حكيم له سر قاطع وذو شأن عظيم مكتوب على ثوب أخضر ومغلف بجلد غزال.",
        "hi": "ऋषियों को ज्ञान और शांति की खोज में हिमालय के शांत वातावरण में तपस्या करते देखा गया है।",
        "hu": "Egy hűtlen feleség gyönyörű kalandja a varázslatos éjszakában.",
        "fi": "Viekas kettu hypähti laiskan koiran ylitse ja juoksi metsän siimekseen.",
    }


@pytest.fixture(scope="session")
def sample_code_snippets() -> dict:
    """Polyglot source code snippets for code tokenization testing."""
    return {
        "python": (
            "async def fetch_payload(url: str, timeout: float = 30.0) -> dict[str, Any]:\n"
            "    headers = {'Accept': 'application/json', 'User-Agent': 'Tokenisers/1.0'}\n"
            "    async with aiohttp.ClientSession() as session:\n"
            "        async with session.get(url, headers=headers, timeout=timeout) as response:\n"
            "            response.raise_for_status()\n"
            "            return await response.json()\n"
        ),
        "rust": (
            "pub struct TokenizerModel<S: PairScorer> {\n"
            "    vocab: HashMap<TokenId, Vec<u8>>,\n"
            "    merges: Vec<(Pair, TokenId)>,\n"
            "    scorer: S,\n"
            "}\n\n"
            "impl<S: PairScorer> TokenizerModel<S> {\n"
            "    pub fn encode(&self, text: &str) -> Vec<TokenId> {\n"
            "        let mut arena = SymbolArena::from_bytes(text.as_bytes());\n"
            "        arena.apply_merges(&self.merges);\n"
            "        arena.to_token_ids()\n"
            "    }\n"
            "}\n"
        ),
        "cpp": (
            "#include <iostream>\n"
            "#include <vector>\n"
            "template <typename T>\n"
            "class VectorBuffer {\n"
            "private:\n"
            "    std::vector<T> data_;\n"
            "public:\n"
            "    void push(const T& item) { data_.push_back(item); }\n"
            "    size_t size() const noexcept { return data_.size(); }\n"
            "};\n"
        ),
        "javascript": (
            "export const computeRenyiEfficiency = (tokenCounts, alpha = 1.0) => {\n"
            "  const total = tokenCounts.reduce((acc, c) => acc + c, 0);\n"
            "  if (total === 0) return 0.0;\n"
            "  const probs = tokenCounts.map(c => c / total).filter(p => p > 0);\n"
            "  if (alpha === 1.0) {\n"
            "    const entropy = -probs.reduce((acc, p) => acc + p * Math.log2(p), 0);\n"
            "    return entropy / Math.log2(tokenCounts.length);\n"
            "  }\n"
            "  const sumPow = probs.reduce((acc, p) => acc + Math.pow(p, alpha), 0);\n"
            "  const entropy = (1 / (1 - alpha)) * Math.log2(sumPow);\n"
            "  return entropy / Math.log2(tokenCounts.length);\n"
            "};\n"
        ),
    }


@pytest.fixture(scope="session")
def sample_morpheme_annotations() -> list[dict]:
    """Sample gold morphological segmentation pairs (word, gold_cuts)."""
    return [
        {"word": "unhappiness", "segments": ["un", "happi", "ness"], "gold_cuts": {2, 7}},
        {"word": "internationalization", "segments": ["inter", "nation", "al", "iz", "ation"], "gold_cuts": {5, 11, 13, 15}},
        {"word": "reconstruction", "segments": ["re", "construct", "ion"], "gold_cuts": {2, 11}},
        {"word": "disagreement", "segments": ["dis", "agree", "ment"], "gold_cuts": {3, 8}},
        {"word": "pretokenization", "segments": ["pre", "token", "iz", "ation"], "gold_cuts": {3, 8, 10}},
        {"word": "megabyte", "segments": ["mega", "byte"], "gold_cuts": {4}},
        {"word": "subword", "segments": ["sub", "word"], "gold_cuts": {3}},
    ]


@pytest.fixture(scope="session")
def two_path_unigram(require_tokenisers, tmp_path_factory):
    """Unigram model in which "ab" has exactly two segmentations: [ab] with P = 0.03 and [a, b]
    with P = 0.1 * 0.1 = 0.01, so the exact posterior (temperature 1) picks [ab] with 0.75."""
    import json
    import math

    tokenisers = require_tokenisers
    path = tmp_path_factory.mktemp("two_path") / "tokenizer.json"
    tokenisers.UnigramTokenizer.train(["ab ab"], vocab_size=256).export_hf_json(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    printable = tokenisers._core.bytes_to_printable
    vocab = [[printable(bytes([b])), math.log(0.1 if b in b"ab" else 1e-3)] for b in range(256)]
    data["model"]["vocab"] = vocab + [[printable(b"ab"), math.log(0.03)]]
    path.write_text(json.dumps(data), encoding="utf-8")
    return tokenisers.UnigramTokenizer.from_pretrained(str(path))


@pytest.fixture(scope="session")
def require_tokenisers():
    """The installed tokenisers package. A missing or broken compiled extension is a test error, never a skip."""
    import tokenisers
    from tokenisers import _core  # noqa: F401  (the compiled extension itself)

    return tokenisers
