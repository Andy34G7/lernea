"""
Tests for the static site build (tokenisers.dashboard.build): the data it collects from benchmarks/results,
what it derives from it, and the site/data.js it writes.
"""

from __future__ import annotations

import json
import re

import pytest

from tokenisers.dashboard import build

SITE = build.REPO_ROOT / "site"
RESULT_KEYS = {f"{kind}_{v}" for kind in ("tokka", "intrinsic") for v in (16384, 32768, 65536)} | {
    "throughput_local", "throughput_colab_cpu", "proxy_small", "proxy_big", "proxy_compare", "swap",
    "swap_lr1p8e-3", "swap_space_prefix", "swap_ablation", "flores"}


def _write_results(root, tokka, intrinsic=None, proxy_lm=None) -> None:
    for rel, data in (("tokka/tokka_matrix.json", tokka),
                      ("intrinsic/intrinsic_leaderboard.json", intrinsic),
                      ("colab/extrinsic_bpb_results.json", proxy_lm)):
        if data is not None:
            path = root / "benchmarks" / "results" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")


def _read_data_js(path):
    text = path.read_text(encoding="utf-8")
    assert text.startswith("window.LERNEA_DATA = ") and text.rstrip().endswith(";")
    return json.loads(text.split("=", 1)[1].rstrip().rstrip(";"))


def test_every_committed_result_set_loads() -> None:
    data = build.load_benchmark_data()
    assert set(data["results"]) == RESULT_KEYS
    for key, entry in data["results"].items():
        assert entry["available"], key
        assert "error" not in entry, (key, entry.get("error"))
    # The reruns are checked against the original swap run before their numbers are compared with it
    assert data["results"]["swap_space_prefix"]["not_comparable"] == []
    assert data["results"]["swap_space_prefix"]["pairs"]["rows"]
    assert data["results"]["swap_ablation"]["ablation"]["rows"]
    assert {"wp_bpe_npmi", "wp_bpe_scp", "wp_bpe_ppig"} <= set(data["metadata"])


def test_languages_follow_the_result_file(tmp_path, monkeypatch) -> None:
    """Heatmap columns come from the results file (per-language code, no pooled 'code'); its meta is passed on."""
    cell = {"bytes_per_token": 3.0}
    row = {"en": cell, "es": cell, "python": cell, "rust": cell, "macro_avg": cell, "micro_avg": cell,
           "vocab_size": 300, "unique_tokens": 12}
    meta = {"git_commit": "a" * 40, "macro_languages": ["en", "es", "python"]}
    _write_results(tmp_path, {"meta": meta, "models": {"bpe_frequency": row}})
    monkeypatch.setattr(build, "REPO_ROOT", tmp_path)
    data = build.load_benchmark_data()
    assert data["tokka_languages"] == ["en", "es", "python", "rust"]
    assert data["results"]["tokka_16384"]["data"]["meta"] == meta


def test_legacy_bare_tokka_matrix(tmp_path, monkeypatch) -> None:
    """Pre-meta result files are a bare model map: their pooled 'code' column still shows."""
    cell = {"bytes_per_token": 3.0}
    _write_results(tmp_path, {"bpe_frequency": {"en": cell, "code": cell, "macro_avg": cell}})
    monkeypatch.setattr(build, "REPO_ROOT", tmp_path)
    assert build.load_benchmark_data()["tokka_languages"] == ["en", "code"]


def test_missing_result_files_leave_the_build_working(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(build, "REPO_ROOT", tmp_path)
    data = build.compact(build.load_benchmark_data())
    assert set(data["results"]) == RESULT_KEYS
    assert not any(e["available"] for e in data["results"].values())


FORMULA_FACTS = {
    "wordpiece_pmi": ([r"(\mathrm{count}(u) + \alpha)", r"\alpha = 1"], []),          # Laplace-smoothed
    "wp_bpe_base": ([r"{\mathrm{freq}(u) \cdot \mathrm{freq}(v)}"], [r"\alpha", r"\theta"]),
    "wp_bpe_ft": ([r"{\mathrm{freq}(u) \cdot \mathrm{freq}(v)}", r"\theta = 2"], [r"\alpha"]),  # ratio + cutoff
    "wp_bpe_scp": ([r"(\mathrm{freq}(uv) - d)_+", r"\sqrt{\mathrm{freq}(u)\,\mathrm{freq}(v)}"], ["P(uv)^2"]),
    "wp_bpe_npmi": ([r"\mathrm{freq}(uv)^{\alpha}", r"-\ln P(uv)"], [r"\in [-1, +1]"]),
    "wp_bpe_ppig": ([r"\max(0, \mathrm{PMI}(u, v))", r"(|u| + |v|)^{\eta}"], [r"\mathcal{H}"]),
    "mdl_bits": ([r"8\,|uv|", r"N \log_2 N"], [r"+ \Delta L"]),
    "length_penalized": ([r"{(|u| + |v|)^\beta}"], []),
    "morpheme_bpe": ([r"e^{\lambda M(u, v)}", r"n_{\mathrm{intra}} - n_{\mathrm{cross}}"], [r"1 + \lambda"]),
}


@pytest.mark.parametrize("name", sorted(FORMULA_FACTS))
def test_formula_matches_implemented_scorer(name: str) -> None:
    formula = build.ALGORITHM_METADATA[name]["formula"]
    must, must_not = FORMULA_FACTS[name]
    for fragment in must:
        assert fragment in formula, (name, fragment)
    for fragment in must_not:
        assert fragment not in formula, (name, fragment)


def test_compact_keeps_the_numbers_the_page_prints() -> None:
    raw = build.load_benchmark_data()
    small = build.compact(build.load_benchmark_data())
    assert len(json.dumps(small)) < 0.6 * len(json.dumps(raw))
    for key in ("proxy_small", "proxy_big"):
        for name, m in raw["results"][key]["data"]["models"].items():
            c = small["results"][key]["data"]["models"][name]
            assert c["macro_val_bpb"] == pytest.approx(m["macro_val_bpb"], abs=5e-5)   # shown to 3-4 decimals
            assert 0 < len(c["curve"]) <= build.CURVE_POINTS
    for name, m in raw["results"]["swap"]["data"]["models"].items():
        assert small["results"]["swap"]["data"]["models"][name]["final_bpb"] == pytest.approx(m["final_bpb"], abs=5e-5)
    lr = small["results"]["swap_lr1p8e-3"]["data"]["trainer"]
    assert lr["peak_lr"] == pytest.approx(1.8e-3, rel=0.01) and lr["n_steps"] > 0 and "steps" not in lr


def test_playground_token_lengths_cover_every_byte_of_the_sample() -> None:
    """A token is [id, byte length]: the page cuts the sample's UTF-8 bytes by the lengths, so each row must
    account for every byte, also where a token ends inside a multi-byte character."""
    pg = build.build_playground(offline=True, vocabs=(300,))
    assert pg["corpus"]["synthetic"] is True
    texts = {s["id"]: s["text"] for s in pg["samples"]}
    for sid, rows in pg["runs"]["300"].items():
        assert len(rows) == 13
        for name, toks in rows.items():
            assert sum(n for _, n in toks) == len(texts[sid].encode("utf-8")), (sid, name)


def test_build_site_writes_data_js_and_keeps_the_playground(tmp_path) -> None:
    target = tmp_path / "data.js"
    target.write_text('window.LERNEA_DATA = {"playground": {"kept": true}};\n', encoding="utf-8")
    assert build.build_site(tmp_path, playground=False) == target
    data = _read_data_js(target)
    assert data["playground"] == {"kept": True}
    assert set(data["results"]) == RESULT_KEYS


def test_site_is_static_and_its_committed_data_is_complete() -> None:
    html = (SITE / "index.html").read_text(encoding="utf-8")
    js = (SITE / "app.js").read_text(encoding="utf-8")
    assert "/api/" not in js and "fetch(" not in js              # no backend
    local = [u for u in re.findall(r'(?:src|href)="([^"#]+)"', html) if not u.startswith(("http", "data:"))]
    assert sorted(local) == ["app.js", "data.js", "style.css"]
    for u in local:
        assert (SITE / u).is_file(), u
    # Every element the script fills exists in the page
    ids = set(re.findall(r'id="([^"]+)"', html))
    used = set(re.findall(r"(?:setHtml|rowChart|lineChart|scatter|table|sel|seg)\('([a-z0-9-]+)'", js))
    created = set(re.findall(r'id="([a-z0-9-]+)"', js))        # sub-sections the script builds itself
    assert used - ids - created == set()
    data = _read_data_js(SITE / "data.js")
    assert set(data["results"]) == RESULT_KEYS and all(e["available"] for e in data["results"].values())
    assert len(data["playground"]["samples"]) == 5 and data["playground"]["corpus"]["synthetic"] is False
