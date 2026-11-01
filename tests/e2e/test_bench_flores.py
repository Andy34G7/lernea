"""FLORES+ tokenization premium: premium arithmetic, parallel alignment, devtest-only reads, a clear
error when FLORES+ is unavailable, and the CLI end to end on a tiny offline fixture (no network).
Real downloads run only with TOKENISERS_NETWORK_TESTS=1."""

from __future__ import annotations

import csv
import json
import math
import os

import pytest

from tokenisers.benchmarks import flores as fl
from tokenisers.data.pipeline import FINEWEB2_LANGUAGES

# One (not parallel) sentence per FLORES+ file: the fixture tests plumbing, not translation.
SAMPLES = {
    "eng_Latn": "The quick brown fox jumps over the lazy dog.",
    "deu_Latn": "Der schnelle braune Fuchs springt über den faulen Hund.",
    "fra_Latn": "Le renard brun rapide saute par-dessus le chien paresseux.",
    "spa_Latn": "El rápido zorro marrón salta sobre el perro perezoso.",
    "rus_Cyrl": "Быстрая коричневая лиса прыгает через ленивую собаку.",
    "cmn_Hans": "敏捷的棕色狐狸跳过了懒狗。",
    "jpn_Jpan": "素早い茶色の狐がのろまな犬を飛び越える。",
    "arb_Arab": "الثعلب البني السريع يقفز فوق الكلب الكسول.",
    "hin_Deva": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।",
    "hun_Latn": "A gyors barna róka átugrik a lusta kutyán.",
    "fin_Latn": "Nopea ruskea kettu hyppää laiskan koiran yli.",
    "kor_Hang": "빠른 갈색 여우가 게으른 개를 뛰어넘는다.",
    "tha_Thai": "สุนัขจิ้งจอกสีน้ำตาลกระโดดข้ามสุนัขขี้เกียจ",
    "tam_Taml": "வேகமான பழுப்பு நரி சோம்பேறி நாயின் மேல் குதிக்கிறது.",
    "amh_Ethi": "ፈጣኑ ቡናማ ቀበሮ በሰነፉ ውሻ ላይ ዘለለ።",
    "ell_Grek": "Η γρήγορη καφέ αλεπού πηδάει πάνω από τον τεμπέλη σκύλο.",
    "heb_Hebr": "השועל החום המהיר קופץ מעל הכלב העצלן.",
    "ben_Beng": "দ্রুত বাদামী শিয়াল অলস কুকুরের উপর দিয়ে লাফ দেয়।",
}


def write_fixture(root, langs=fl.LANGUAGES, ids=(2, 0, 1), split="devtest"):
    """root/devtest/<lang>.jsonl with FLORES+'s schema, rows deliberately out of id order."""
    (root / "devtest").mkdir(parents=True, exist_ok=True)
    for lang in langs:
        rows = [{"id": i, "iso_639_3": lang[:3], "iso_15924": lang[4:], "text": f"{SAMPLES[lang]} {i}",
                 "split": split} for i in ids]
        (root / "devtest" / f"{lang}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return root


def test_premium_arithmetic():
    counts = {lang: {"tokens": n, "single_byte_tokens": s, "byte_fallback_tokens": f}
              for lang, n, s, f in (("eng_Latn", 100, 10, 0), ("deu_Latn", 150, 30, 3), ("kor_Hang", 400, 200, 100))}
    text = {lang: {"sentences": 10, "bytes": b, "chars": c}
            for lang, b, c in (("eng_Latn", 500, 500), ("deu_Latn", 600, 580), ("kor_Hang", 900, 330))}
    m = fl.language_metrics(counts, text)
    assert m["eng_Latn"]["premium"] == 1.0
    assert m["deu_Latn"]["premium"] == 1.5 and m["kor_Hang"]["premium"] == 4.0
    assert m["kor_Hang"]["tokens_per_sentence"] == 40.0
    assert m["kor_Hang"]["bytes_per_token"] == 900 / 400 and m["kor_Hang"]["chars_per_token"] == 330 / 400
    assert m["kor_Hang"]["single_byte_share"] == 0.5 and m["kor_Hang"]["byte_fallback_rate"] == 0.25

    a = fl.aggregate(m, ["deu_Latn", "kor_Hang"])
    assert a["mean_premium"] == 2.75 and a["max_premium"] == 4.0 and a["worst_language"] == "kor_Hang"
    assert a["share_premium_le_1_5"] == 0.5  # 1.5 itself counts as <= 1.5
    assert a["std_log_premium"] == pytest.approx((math.log(4.0) - math.log(1.5)) / 2)
    assert a["byte_fallback_rate"] == pytest.approx(103 / 550)
    # Equal premiums: no spread, whatever their distance from English.
    assert fl.aggregate({"a": {**m["kor_Hang"]}, "b": {**m["kor_Hang"]}}, ["a", "b"])["std_log_premium"] == 0.0


def test_languages_are_english_plus_the_pipeline_mix():
    assert tuple(fl.IN_MIX) == FINEWEB2_LANGUAGES
    assert fl.IN_MIX["cmn_Hani"] == "cmn_Hans"
    assert fl.LANGUAGES[0] == fl.ENGLISH and len(set(fl.LANGUAGES)) == len(fl.LANGUAGES) == len(SAMPLES)
    assert not set(fl.OUT_OF_MIX) & set(fl.IN_MIX.values())


def test_sentences_align_by_id(tmp_path):
    texts, files = fl.load_flores(["eng_Latn", "kor_Hang"], write_fixture(tmp_path), download=False)
    assert texts["eng_Latn"] == [f"{SAMPLES['eng_Latn']} {i}" for i in range(3)]  # sorted by id
    assert files["kor_Hang"]["sentences"] == 3 and len(files["kor_Hang"]["sha256"]) == 64
    write_fixture(tmp_path, ["kor_Hang"], ids=(0, 1, 3))
    with pytest.raises(ValueError, match="not parallel"):
        fl.load_flores(["eng_Latn", "kor_Hang"], tmp_path, download=False)


def test_only_devtest_is_read(tmp_path, monkeypatch):
    write_fixture(tmp_path, ["eng_Latn"], split="dev")
    with pytest.raises(ValueError, match="only devtest"):
        fl.load_flores(["eng_Latn"], tmp_path, download=False)

    import huggingface_hub

    calls = []

    def fake_download(repo, filename, repo_type, revision, local_dir):
        calls.append((repo, filename, repo_type, revision))
        write_fixture(local_dir, [filename.split("/")[1][:-6]])
        return str(local_dir / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    root = fl.flores_root(str(tmp_path / "cache"))
    assert fl.FLORES_REVISION[:12] in root.name
    fl.load_flores(["eng_Latn", "tha_Thai"], root)
    assert calls == [(fl.FLORES_REPO, f"devtest/{lang}.jsonl", "dataset", fl.FLORES_REVISION)
                     for lang in ("eng_Latn", "tha_Thai")]
    fl.load_flores(["eng_Latn", "tha_Thai"], root)
    assert len(calls) == 2  # cached files are not downloaded again


def test_unavailable_flores_says_to_accept_the_terms(tmp_path, monkeypatch):
    import huggingface_hub

    def denied(*a, **k):
        raise OSError("403 Client Error: Forbidden (gated dataset)")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", denied)
    with pytest.raises(SystemExit) as e:
        fl.main(["--cache-dir", str(tmp_path), "--vocab-sizes", "--swap-jobs", "--output-dir", str(tmp_path / "o")])
    msg = str(e.value)
    assert "accept its terms" in msg and "huggingface.co/datasets/openlanguagedata/flores_plus" in msg
    assert "403" in msg and not (tmp_path / "o").exists()


def test_cli_on_fixture(require_tokenisers, tmp_path):
    """Both tiers end to end, offline: a synthetic-corpus intrinsic tier and a one-tokenizer swap job."""
    tk = require_tokenisers
    job = tmp_path / "job"
    tk.BpeTokenizer.train(["the quick brown fox " * 50], vocab_size=268).save(str(job / "tokenizers" / "tiny"))
    (job / "config.json").write_text(json.dumps({"tokenizers": [
        {"name": "tiny", "kind": "tokenisers", "class": "BpeTokenizer", "path": "tokenizers/tiny", "ids_docs": 200}],
        "prepared": {"git_commit": "abc"}}))
    out = tmp_path / "out"
    fl.main(["--offline", "--cache-dir", str(tmp_path / "cache"), "--train-bytes", "30000", "--vocab-sizes", "300",
             "--variants", "bpe_frequency", "unigram_lm", "--swap-jobs", str(job), "--flores-dir",
             str(write_fixture(tmp_path / "flores")), "--reference-dir", str(tmp_path / "none"),
             "--output-dir", str(out)])

    doc = json.loads((out / "flores.json").read_text(encoding="utf-8"))
    meta, models = doc["meta"], doc["models"]
    assert list(models) == ["bpe_frequency@300", "unigram_lm@300", "tiny"]
    assert meta["flores"]["split"] == "devtest" and meta["flores"]["revision"] is None  # a local copy
    assert meta["intrinsic"]["dataset"]["synthetic"] and meta["swap_jobs"]["job"]["prepared_git_commit"] == "abc"
    for m in models.values():
        assert list(m["languages"]) == list(fl.LANGUAGES) and m["languages"]["eng_Latn"]["premium"] == 1.0
        assert set(m["aggregates"]) == {"in_mix", "out_of_mix"}
        assert m["aggregates"]["in_mix"]["languages"] == 10 and m["aggregates"]["out_of_mix"]["languages"] == 7
    assert models["bpe_frequency@300"]["training"]["reproduces_committed"] is None  # no reference committed

    # An ASCII-only tokenizer splits Korean into raw bytes: every token is one byte, and the bytes
    # >= 0x80 (the Hangul) are exactly the byte-fallback tokens.
    kor = models["tiny"]["languages"]["kor_Hang"]
    raw = "".join(f"{SAMPLES['kor_Hang']} {i}" for i in range(3)).encode("utf-8")
    assert kor["single_byte_share"] == 1.0 and kor["tokens"] == len(raw)
    assert kor["byte_fallback_tokens"] == sum(b >= 0x80 for b in raw)

    with open(out / "flores_by_language.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * len(fl.LANGUAGES)
    assert {r["group"] for r in rows} == {"reference", "in_mix", "out_of_mix"}
    from tokenisers.benchmarks.report import render_flores

    md = (out / "flores.md").read_text(encoding="utf-8")
    assert md == render_flores(doc) + "\n" and "How to read" in md and "cmn_Hani -> cmn_Hans" in md


def test_intrinsic_tier_checks_test_bytes_per_token_against_the_committed_leaderboard(require_tokenisers, tmp_path,
                                                                                      capsys):
    """Each retrained tokenizer's test bytes/token is compared with the committed intrinsic leaderboard of the
    same dataset: equal -> reproduces, else a warning; a leaderboard of another dataset is not compared."""
    flores_dir = write_fixture(tmp_path / "flores")

    def run(reference_dir):
        out = tmp_path / "out"
        fl.main(["--offline", "--cache-dir", str(tmp_path / "cache"), "--train-bytes", "30000", "--vocab-sizes",
                 "300", "--variants", "bpe_frequency", "unigram_lm", "--swap-jobs", "--flores-dir", str(flores_dir),
                 "--reference-dir", str(reference_dir), "--output-dir", str(out)])
        return json.loads((out / "flores.json").read_text(encoding="utf-8"))

    doc = run(tmp_path / "none")
    root = doc["meta"]["intrinsic"]["dataset"]["root"]
    bpt = {m["base"]: m["training"]["test_bytes_per_token"] for m in doc["models"].values()}
    path = fl.committed_intrinsic(tmp_path / "ref", 300)
    path.parent.mkdir(parents=True)

    def commit(dataset_root):
        path.write_text(json.dumps({"meta": {"dataset": {"root": dataset_root}}, "models": {
            "bpe_frequency": {"bytes_per_token": bpt["bpe_frequency"]},
            "unigram_lm": {"bytes_per_token": bpt["unigram_lm"] + 1e-9}}}), encoding="utf-8")

    commit(root)
    capsys.readouterr()
    doc = run(tmp_path / "ref")
    t = {m["base"]: m["training"] for m in doc["models"].values()}
    assert t["bpe_frequency"]["reproduces_committed"] is True and t["unigram_lm"]["reproduces_committed"] is False
    assert t["unigram_lm"]["committed_test_bytes_per_token"] == bpt["unigram_lm"] + 1e-9
    err = capsys.readouterr().err
    assert "warning: unigram_lm at V=300" in err and "bpe_frequency at V=300" not in err
    from tokenisers.benchmarks.report import render_flores

    assert "1 of 2 reproduce the committed test-split bytes/token exactly" in render_flores(doc)

    commit("hf-another-dataset")  # the dataset-root guard
    doc = run(tmp_path / "ref")
    assert [m["training"]["reproduces_committed"] for m in doc["models"].values()] == [None, None]
    assert doc["meta"]["intrinsic"]["committed_reference"] == {"300": None}


def test_swap_tier_says_what_each_tokenizer_was_trained_on(repo_root):
    """The committed swap tier: GPT-2's and TokenMonster's vocabularies were not trained on FineWeb,
    and the 32k TokenMonster one is not about 50k tokens."""
    from tokenisers.benchmarks.report import render_flores

    doc = json.loads((repo_root / "benchmarks/results/flores/flores.json").read_text(encoding="utf-8"))
    md = render_flores(doc)
    line = next(x for x in md.splitlines() if x.startswith("- swap tier:"))
    assert "`gpt2` (tiktoken's `gpt2` encoding: GPT-2's own BPE, trained by OpenAI on WebText)" in line
    assert ("`tm_englishcode_32000` (TokenMonster's prebuilt `englishcode-32000-consistent-nocapcode-v1` "
            "vocabulary, 32,000 tokens)") in line
    assert line.endswith("; 33 of ours, trained on English FineWeb text at 50,256 tokens")
    assert "about 50k" not in md and "English FineWeb)" not in md


@pytest.mark.skipif(os.environ.get("TOKENISERS_NETWORK_TESTS") != "1",
                    reason="set TOKENISERS_NETWORK_TESTS=1 to run tests that download real data")
def test_real_flores_devtest_download(tmp_path):
    texts, files = fl.load_flores(["eng_Latn", "kor_Hang"], fl.flores_root(str(tmp_path)))
    assert len(texts["eng_Latn"]) == len(texts["kor_Hang"]) == 1012
    assert all(t.strip() for t in texts["kor_Hang"])
    assert files["eng_Latn"]["file"] == "devtest/eng_Latn.jsonl"


def test_swap_jobs_record_and_flag_a_dirty_preparation(tmp_path, repo_root):
    """A swap job (or a pool it was assembled from) prepared from uncommitted code is recorded as such
    and named in the swap-tier line."""
    from tokenisers.benchmarks.report import render_flores

    job = tmp_path / "pool"
    job.mkdir()
    (job / "config.json").write_text(json.dumps({"tokenizers": [], "prepared": {
        "git_commit": "a" * 40, "git_dirty": False, "pools": {"/p": {"git_commit": "b" * 40, "git_dirty": True}}}}))
    assert fl.swap_tier([job], {}, {}, {}, print)["pool"]["prepared_git_dirty"] is True

    doc = json.loads((repo_root / "benchmarks/results/flores/flores.json").read_text(encoding="utf-8"))
    doc["meta"]["swap_jobs"]["nanogpt-swap-pool-sp300"]["prepared_git_dirty"] = True
    line = next(x for x in render_flores(doc).splitlines() if x.startswith("- swap tier:"))
    assert line.endswith("; prepared from a dirty tree (uncommitted changes: not reproducible from any commit): "
                         "`nanogpt-swap-pool-sp300`")
