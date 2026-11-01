"""
End-to-End Tests for Tokenisers Lab Benchmark Intelligence Dashboard & REST APIs.
"""

from __future__ import annotations

import copy
import http.client
import json
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import urllib.error
from typing import Generator

import pytest

from tokenisers.dashboard import server as server_mod
from tokenisers.dashboard.server import DashboardServer, TokenizerCache, create_server


def find_free_port() -> int:
    """Finds an available ephemeral port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def dashboard_server() -> Generator[str, None, None]:
    """Starts the dashboard server on an ephemeral port and tears it down."""
    port = find_free_port()
    server = create_server(host="127.0.0.1", port=port, offline=True)
    server.start(blocking=False)

    # Wait for server to bind and listen
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 5.0
    connected = False
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/", timeout=1.0) as resp:
                if resp.status == 200:
                    connected = True
                    break
        except Exception:
            time.sleep(0.05)

    assert connected, "Failed to start DashboardServer within deadline"

    yield base_url

    server.stop()


def test_dashboard_index_html(dashboard_server: str) -> None:
    """Tests that GET / serves the interactive dashboard HTML."""
    req = urllib.request.Request(f"{dashboard_server}/")
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        assert resp.status == 200
        content_type = resp.headers.get("Content-Type", "")
        assert "text/html" in content_type
        body = resp.read().decode("utf-8")
        assert "Tokenisers Lab" in body
        assert "/static/app.js" in body
        assert 'id="quadrant-chart"' in body


def test_dashboard_static_app_js(dashboard_server: str) -> None:
    """Tests that GET /static/app.js serves valid JavaScript."""
    req = urllib.request.Request(f"{dashboard_server}/static/app.js")
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        assert resp.status == 200
        content_type = resp.headers.get("Content-Type", "")
        assert "application/javascript" in content_type or "text/javascript" in content_type
        body = resp.read().decode("utf-8")
        assert "renderQuadrantChart" in body
        assert "runLiveTokenize" in body
        assert "state.data.tokka_languages" in body  # heatmap columns come from the API, not a fixed list


def test_api_benchmarks(dashboard_server: str) -> None:
    """Tests that GET /api/benchmarks returns consolidated benchmark JSON."""
    req = urllib.request.Request(f"{dashboard_server}/api/benchmarks")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))

        # Verify summary KPIs
        assert "summary" in data
        assert "best_downstream_quality" in data["summary"]
        assert "highest_information_efficiency" in data["summary"]
        assert "best_compression" in data["summary"]

        # Verify models dictionary
        assert "models" in data
        models = data["models"]
        for key in ["bpe_frequency", "wp_bpe_ft", "wp_bpe_pig", "unigram_lm"]:
            assert key in models, f"Model {key} missing from benchmarks"
            m = models[key]
            assert "display_name" in m
            assert "family" in m
            assert "formula" in m

        # Verify tokka matrix; the heatmap's columns must cover every language the matrix has
        assert "tokka_matrix" in data
        for row in data["tokka_matrix"].values():
            cells = {k for k, v in row.items() if isinstance(v, dict)} - {"macro_avg", "micro_avg"}
            assert cells <= set(data["tokka_languages"])
        # Every result file's provenance block is passed through (None for a legacy file)
        assert set(data["meta"]) == {"tokka", "intrinsic", "proxy_lm"}
        # Verify metadata
        assert "metadata" in data
        assert "wp_bpe_npmi" in data["metadata"]
        assert "wp_bpe_scp" in data["metadata"]
        assert "wp_bpe_ppig" in data["metadata"]


def _write_results(root, tokka, intrinsic=None, proxy_lm=None) -> None:
    for rel, data in (("tokka/tokka_matrix.json", tokka),
                      ("intrinsic/intrinsic_leaderboard.json", intrinsic),
                      ("colab/extrinsic_bpb_results.json", proxy_lm)):
        if data is not None:
            path = root / "benchmarks" / "results" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")


def test_benchmarks_pass_languages_and_provenance_through(tmp_path, monkeypatch) -> None:
    """Heatmap columns follow the results file (es, per-language code, no pooled 'code'), and each
    file's meta (budget, backend, commit, synthetic flag) reaches the client next to its numbers."""
    cell = {"bytes_per_token": 3.0}
    row = {"en": cell, "es": cell, "python": cell, "rust": cell, "macro_avg": cell, "micro_avg": cell,
           "vocab_size": 300, "unique_tokens": 12}
    tokka_meta = {"git_commit": "a" * 40, "macro_languages": ["en", "es", "python"]}
    intrinsic_meta = {"git_commit": "a" * 40, "eval_split": "test"}
    proxy_meta = {"git_commit": "b" * 40, "byte_budget": 1_000_000, "synthetic_data": False,
                  "backend": {"kind": "local", "device": "cpu"}}
    _write_results(tmp_path,
                   {"meta": tokka_meta, "models": {"bpe_frequency": row}},
                   {"meta": intrinsic_meta, "models": {"bpe_frequency": {"renyi_e_2_0": 0.4}}},
                   {"meta": proxy_meta, "models": {"bpe_frequency": {"macro_val_bpb": 3.4}}})
    monkeypatch.setattr(server_mod, "REPO_ROOT", tmp_path)
    data = server_mod.load_benchmark_data()
    assert data["tokka_languages"] == ["en", "es", "python", "rust"]
    assert data["meta"] == {"tokka": tokka_meta, "intrinsic": intrinsic_meta, "proxy_lm": proxy_meta}
    assert data["summary"]["best_downstream_quality"]["value"] == 3.4


def test_benchmarks_legacy_bare_tokka_matrix(tmp_path, monkeypatch) -> None:
    """Pre-meta result files are a bare model map: their pooled 'code' column still shows, meta is None."""
    cell = {"bytes_per_token": 3.0}
    _write_results(tmp_path, {"bpe_frequency": {"en": cell, "code": cell, "macro_avg": cell}})
    monkeypatch.setattr(server_mod, "REPO_ROOT", tmp_path)
    data = server_mod.load_benchmark_data()
    assert data["tokka_languages"] == ["en", "code"]
    assert data["meta"] == {"tokka": None, "intrinsic": None, "proxy_lm": None}


def test_api_tokenize_live(dashboard_server: str) -> None:
    """Tests that POST /api/tokenize executes multi-tokenizer encoding."""
    payload = {
        "text": "The internationalization of unhappily disproportionate misunderstandings.",
        "vocab_size": 320,
    }
    req = urllib.request.Request(
        f"{dashboard_server}/api/tokenize",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=10.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))

        assert data["vocab_size"] == 320
        assert data["raw_bytes"] > 0
        assert "results" in data
        results = data["results"]
        assert len(results) >= 8, f"Expected at least 8 tokenizers, got {len(results)}"

        for res in results:
            assert "name" in res
            assert "num_tokens" in res
            assert res["num_tokens"] > 0
            assert "bytes_per_token" in res
            assert "token_ids" in res
            assert "tokens" in res
            assert "token_details" in res
            assert len(res["token_ids"]) == res["num_tokens"]
            assert len(res["tokens"]) == res["num_tokens"]


def _raw(base_url: str, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
    """Send a request with full control over headers; returns (status, headers, body bytes)."""
    port = int(base_url.rsplit(":", 1)[1])
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        hdrs = {"Host": f"127.0.0.1:{port}", **(headers or {})}
        if body is not None:
            hdrs.setdefault("Content-Length", str(len(body)))
        for k, v in hdrs.items():
            if v is not None:
                conn.putheader(k, v)
        conn.endheaders()
        if body is not None:
            conn.send(body)
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


def _post_json(base_url: str, payload, headers: dict | None = None):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    return _raw(base_url, "POST", "/api/tokenize", body, {"Content-Type": "application/json", **(headers or {})})


def test_api_tokenize_reports_synthetic_corpus(dashboard_server: str) -> None:
    status, _, body = _post_json(dashboard_server, {"text": "hello world", "vocab_size": 320})
    assert status == 200
    corpus = json.loads(body)["corpus"]
    assert corpus["synthetic"] is True
    assert "fixture" in corpus["lexicon_source"]


def test_token_byte_lens_count_token_bytes_not_display_piece(dashboard_server: str) -> None:
    """Partial-UTF-8 tokens display as U+FFFD (3 bytes) but are 1 byte: byte_lens must sum to raw_bytes."""
    text = "héllo 🦀 日本"  # rare multibyte chars: a 320 vocab keeps some as single-byte tokens
    status, _, body = _post_json(dashboard_server, {"text": text, "vocab_size": 320})
    assert status == 200
    data = json.loads(body)
    assert data["raw_bytes"] == len(text.encode("utf-8"))
    partial = 0
    for res in data["results"]:
        lens = [d["byte_len"] for d in res["token_details"]]
        assert sum(lens) == data["raw_bytes"], (res["name"], res["token_details"])
        partial += sum(1 for d in res["token_details"] if d["piece"] == "�" and d["byte_len"] < 3)
    assert partial > 0, "text produced no partial-UTF-8 tokens, so the test checks nothing"


def test_no_wildcard_cors(dashboard_server: str) -> None:
    for method, path in [("GET", "/"), ("GET", "/api/benchmarks")]:
        status, headers, _ = _raw(dashboard_server, method, path)
        assert status == 200
        assert "Access-Control-Allow-Origin" not in headers
    status, headers, _ = _raw(dashboard_server, "OPTIONS", "/api/tokenize", headers={"Origin": "https://evil.example"})
    assert status >= 400 and "Access-Control-Allow-Origin" not in headers


def test_foreign_host_header_rejected(dashboard_server: str) -> None:
    """DNS rebinding: evil.example resolving to 127.0.0.1 must not reach the API."""
    port = dashboard_server.rsplit(":", 1)[1]
    status, _, _ = _raw(dashboard_server, "GET", "/api/benchmarks", headers={"Host": f"evil.example:{port}"})
    assert status == 403
    status, _, _ = _post_json(dashboard_server, {"text": "x"}, headers={"Host": f"evil.example:{port}"})
    assert status == 403
    status, _, _ = _raw(dashboard_server, "GET", "/", headers={"Host": f"localhost:{port}"})
    assert status == 200


def test_cross_origin_post_rejected(dashboard_server: str) -> None:
    port = dashboard_server.rsplit(":", 1)[1]
    status, _, _ = _post_json(dashboard_server, {"text": "x", "vocab_size": 320}, headers={"Origin": "https://evil.example"})
    assert status == 403
    # text/plain is a CORS "simple request": no preflight, so it must be refused outright.
    status, _, _ = _raw(dashboard_server, "POST", "/api/tokenize", b'{"text": "x"}', {"Content-Type": "text/plain"})
    assert status == 415
    status, _, _ = _post_json(dashboard_server, {"text": "x", "vocab_size": 320}, headers={"Origin": f"http://127.0.0.1:{port}"})
    assert status == 200


# The last body nests 30000 arrays (60 KB, under MAX_BODY_BYTES): json.loads raises RecursionError.
@pytest.mark.parametrize(
    "body", [b"abc", b"[1, 2]", b'"text"', b"\xff\xfe", pytest.param(b"[" * 30000 + b"]" * 30000, id="deep-nesting")]
)
def test_malformed_json_is_400(dashboard_server: str, body: bytes) -> None:
    status, _, payload = _post_json(dashboard_server, body)
    assert status == 400
    assert "error" in json.loads(payload)


# Only just-out-of-range values: a huge vocab_size (the finding's 300_000 / 1e9) would train
# 13 tokenizers without bound if validation ever regressed, which can OOM the machine.
@pytest.mark.parametrize(
    "vocab_size",
    [server_mod.MAX_VOCAB_SIZE + 1, server_mod.MIN_VOCAB_SIZE - 1, 0, -5, "400", 400.5, None, True],
)
def test_vocab_size_outside_allow_list_is_400(dashboard_server: str, vocab_size) -> None:
    status, _, payload = _post_json(dashboard_server, {"text": "abc", "vocab_size": vocab_size})
    assert status == 400
    assert "vocab_size" in json.loads(payload)["error"]


def test_body_and_text_size_caps(dashboard_server: str) -> None:
    status, _, _ = _post_json(dashboard_server, {"text": "a" * (server_mod.MAX_BODY_BYTES + 1)})
    assert status == 413
    # Raw UTF-8 keeps this body under MAX_BODY_BYTES, so only the text-length cap can fire.
    body = json.dumps({"text": "é" * (server_mod.MAX_TEXT_CHARS + 1)}, ensure_ascii=False).encode("utf-8")
    assert len(body) <= server_mod.MAX_BODY_BYTES
    status, _, payload = _post_json(dashboard_server, body)
    assert status == 413 and "text exceeds" in json.loads(payload)["error"]
    status, _, _ = _raw(dashboard_server, "POST", "/api/tokenize", None, {"Content-Type": "application/json"})
    assert status == 411


def test_static_path_traversal_uses_real_containment(dashboard_server: str, tmp_path, monkeypatch) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "ok.js").write_text("ok", encoding="utf-8")
    sibling = tmp_path / "static_secrets"  # shares the string prefix ".../static"
    sibling.mkdir()
    (sibling / "key.txt").write_text("secret", encoding="utf-8")
    monkeypatch.setattr(server_mod, "STATIC_DIR", static)

    status, _, body = _raw(dashboard_server, "GET", "/static/ok.js")
    assert status == 200 and body == b"ok"
    status, _, body = _raw(dashboard_server, "GET", "/static/../static_secrets/key.txt")
    assert status == 403 and b"secret" not in body
    status, _, _ = _raw(dashboard_server, "GET", "/static/../../etc/passwd")
    assert status == 403


def test_run_suite_stub_removed(dashboard_server: str) -> None:
    status, _, _ = _post_json(dashboard_server, {"suite": "intrinsic", "vocab_size": 320})
    assert status == 200  # /api/tokenize itself still works
    status, _, _ = _raw(dashboard_server, "POST", "/api/run-suite", b"{}", {"Content-Type": "application/json"})
    assert status == 404


def test_server_refuses_non_loopback_bind() -> None:
    with pytest.raises(ValueError):
        DashboardServer(host="0.0.0.0", port=find_free_port())


def test_tokenizer_cache_is_bounded_lru(monkeypatch) -> None:
    cache = TokenizerCache(offline=True, max_entries=2)
    trained = []
    monkeypatch.setattr(cache, "_train", lambda v: trained.append(v) or {"v": v})
    for v in (300, 310, 300, 320, 310):
        assert cache.get_tokenizers(v) == {"v": v}
    assert list(cache._cache) == [320, 310]
    assert trained == [300, 310, 320, 310]  # 310 was evicted by 320, 300 reused once


# (fragment the formula must show, fragment it must not) per scorer, from the Rust scorers.
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
    formula = server_mod.ALGORITHM_METADATA[name]["formula"]
    must, must_not = FORMULA_FACTS[name]
    for fragment in must:
        assert fragment in formula, (name, fragment)
    for fragment in must_not:
        assert fragment not in formula, (name, fragment)


def test_static_page_formulas_are_the_api_formulas() -> None:
    """index.html's hard-coded formulas must be copies of ALGORITHM_METADATA (one source of truth)."""
    import re

    html = (server_mod.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    shown = re.findall(r"\$\$(.+?)\$\$", html)
    assert len(shown) >= 6
    known = {m["formula"] for m in server_mod.ALGORITHM_METADATA.values()}
    assert [f for f in shown if f not in known] == []


# Every result set the dashboard shows, keyed as in server.RESULT_SETS (+ the derived small-vs-big comparison).
RESULT_KEYS = {f"{kind}_{v}" for kind in ("tokka", "intrinsic") for v in (16384, 32768, 65536)} | {
    "throughput_local", "throughput_colab_cpu", "proxy_small", "proxy_big", "proxy_compare", "swap",
    "swap_lr1p8e-3", "swap_space_prefix", "swap_ablation", "flores"}


def test_api_benchmarks_returns_every_result_set(dashboard_server: str) -> None:
    status, _, body = _raw(dashboard_server, "GET", "/api/benchmarks")
    assert status == 200
    results = json.loads(body)["results"]
    assert set(results) == RESULT_KEYS
    for key, e in results.items():
        assert isinstance(e["available"], bool) and e["path"].startswith("benchmarks/results/"), key
        assert (e["data"] is not None) == e["available"], key
    # Everything except the pending ablation is committed, with README-rendered provenance
    for key in RESULT_KEYS - {"swap_ablation", "swap_lr1p8e-3"}:
        assert results[key]["available"] and results[key]["provenance"], key
    assert results["swap_lr1p8e-3"]["available"]
    assert results["swap"]["notes"], "the swap findings come from report.render_nanogpt_swap"
    assert results["flores"]["notes"], "the FLORES+ findings come from report.render_flores"
    # The committed space-prefix rerun: each of its 8 tokenizers paired with its whitespace run
    assert {r["sp"]["name"] for r in results["swap_space_prefix"]["pairs"]["rows"]} == set(
        results["swap_space_prefix"]["data"]["models"])
    # Small vs big: report.colab_compare_stats, the reference at rank-relative delta 0
    cmp = results["proxy_compare"]["data"]
    assert cmp["tiers"]["big LM"]["bpe_frequency"][1] == 0.0 and "bpe_frequency" in cmp["names"]
    # Per-step train BPB of the swap: train loss (nats per token) / ln 2 / train bytes per token
    train = results["swap"]["train_bpb"]
    gpt2 = results["swap"]["data"]["models"]["gpt2"]
    assert len(train["steps"]) == gpt2["curve"][-1][0] and set(train["bpb"]) == set(results["swap"]["data"]["models"])
    assert all(len(v) == len(train["steps"]) for v in train["bpb"].values())


def test_kpi_strip_is_on_the_summary_page_not_the_shared_layout() -> None:
    """The category leaders used to sit above the tabs on every tab; they are now the Summary tab."""
    html = (server_mod.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert html.count('aria-label="Category leaders"') == 1
    main = html.index("<main")
    summary, overview = html.index('id="tab-summary"'), html.index('id="tab-overview"')
    assert summary < html.index('aria-label="Category leaders"') < overview
    assert 'class="kpis"' not in html[main:summary] and "kpi-" not in html[main:summary]
    # ARIA tabs: every tab controls an existing tab panel
    import re

    tabs = re.findall(r'role="tab"[^>]*id="tab-nav-(\w+)" aria-controls="tab-(\w+)"', html)
    assert ("summary", "summary") in tabs and len(tabs) >= 9
    for name, panel in tabs:
        assert name == panel and f'id="tab-{panel}" role="tabpanel" aria-labelledby="tab-nav-{name}"' in html


def _swap_fixture(root, rel: str, doc) -> None:
    path = root / "benchmarks" / "results" / rel / "nanogpt_swap.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_space_prefix_section_activates_when_its_results_land(tmp_path, monkeypatch) -> None:
    """No nanogpt_swap_sp/ results: 'not yet run'. With a file shaped like nanogpt_swap.json, each of our
    tokenizers pairs with its original run (a trailing _sp dropped) and GPT-2 is the reference."""
    import copy

    real = json.loads((server_mod.REPO_ROOT / "benchmarks/results/nanogpt_swap/nanogpt_swap.json").read_text())
    old = {**real, "models": {k: real["models"][k] for k in ("gpt2", "bpe_frequency", "wp_bpe_pig", "tm_englishcode_50256")}}
    _swap_fixture(tmp_path, "nanogpt_swap", old)
    monkeypatch.setattr(server_mod, "REPO_ROOT", tmp_path)
    sp_entry = server_mod.load_benchmark_data()["results"]["swap_space_prefix"]
    assert sp_entry["available"] is False and sp_entry["path"] == "benchmarks/results/nanogpt_swap_sp/nanogpt_swap.json"
    assert "pairs" not in sp_entry

    sp = copy.deepcopy(old)
    sp["models"]["wp_bpe_pig_sp"] = sp["models"].pop("wp_bpe_pig")
    for name, bpb, bpt in (("bpe_frequency", 1.80, 4.3), ("wp_bpe_pig_sp", 1.75, 4.4), ("gpt2", 1.69, 4.436)):
        sp["models"][name].update(final_bpb=bpb, val_bytes_per_token=bpt)
    sp["gpt2_equivalent"]["steps"] = {"gpt2": 600, "bpe_frequency": 400, "wp_bpe_pig_sp": 450}
    _swap_fixture(tmp_path, "nanogpt_swap_sp", sp)
    sp_entry = server_mod.load_benchmark_data()["results"]["swap_space_prefix"]
    assert sp_entry["available"] is True and sp_entry["provenance"]  # report.render_nanogpt_swap ran on it
    pairs = sp_entry["pairs"]
    rows = {r["variant"]: r for r in pairs["rows"]}
    assert set(rows) == {"bpe_frequency", "wp_bpe_pig"}  # GPT-2 and TokenMonster are not ours
    pig = rows["wp_bpe_pig"]
    assert pig["sp"]["name"] == "wp_bpe_pig_sp" and pig["sp"]["gpt2_equivalent_steps"] == 450
    assert pig["old"]["final_bpb"] == real["models"]["wp_bpe_pig"]["final_bpb"]
    assert pig["old"]["gpt2_equivalent_steps"] == real["gpt2_equivalent"]["steps"]["wp_bpe_pig"]
    assert pig["delta_bpb"] == pytest.approx(1.75 - real["models"]["wp_bpe_pig"]["final_bpb"])
    assert pairs["gpt2"]["run"] == "sp" and pairs["gpt2"]["final_bpb"] == 1.69


def test_missing_or_broken_result_files_degrade_gracefully(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(server_mod, "REPO_ROOT", tmp_path)
    data = server_mod.load_benchmark_data()
    assert set(data["results"]) == RESULT_KEYS
    assert not any(e["available"] or e["data"] for e in data["results"].values())
    assert set(server_mod.ALGORITHM_METADATA) <= set(data["models"]) and data["tokka_languages"] == []

    # Unreadable JSON: not available, with the reason; a file missing fields the README renderer needs:
    # available with its numbers, no provenance lines (instead of a 500).
    bad = tmp_path / "benchmarks/results/colab_big/extrinsic_bpb_results.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json", encoding="utf-8")
    _write_results(tmp_path, {"meta": {}, "models": {}}, proxy_lm={"meta": {}, "models": {"bpe_frequency": {"macro_val_bpb": 3.0}}})
    data = server_mod.load_benchmark_data()
    big, small = data["results"]["proxy_big"], data["results"]["proxy_small"]
    assert big["available"] is False and "unreadable" in big["error"]
    assert small["available"] is True and small["provenance"] == []
    assert data["results"]["proxy_compare"]["available"] is False
    assert data["models"]["bpe_frequency"]["macro_val_bpb"] == 3.0


def test_flores_section_shows_premium_by_language_and_group(dashboard_server: str) -> None:
    """The FLORES+ tab reads flores.json: per tokenizer a premium for every in-mix and out-of-mix language
    (English is 1.0 by definition), group aggregates, and render_flores' findings with the English-cost caveat."""
    status, _, body = _raw(dashboard_server, "GET", "/api/benchmarks")
    flores = json.loads(body)["results"]["flores"]
    langs = flores["data"]["meta"]["languages"]
    in_mix, out_mix = list(langs["in_mix"].values()), langs["out_of_mix"]
    assert in_mix and out_mix and not set(in_mix) & set(out_mix)
    models = flores["data"]["models"]
    assert {m["tier"] for m in models.values()} == {"intrinsic", "swap"}
    for name, m in models.items():
        assert m["languages"][langs["english"]]["premium"] == 1.0, name
        assert set(in_mix + out_mix) <= set(m["languages"]), name
        assert set(m["aggregates"]) == {"in_mix", "out_of_mix"}, name
    assert any("relative to each tokenizer's own English cost" in n for n in flores["notes"])
    html = (server_mod.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for anchor in ('id="tab-flores"', 'id="flores-heat"', 'id="flores-table"', 'id="flores-caveat-english"'):
        assert anchor in html, anchor


REAL_RESULTS = server_mod.REPO_ROOT / "benchmarks" / "results"


def _ablation_doc(real, real_sp):
    """An ablation run: _g2 for two variants, _sp300 for one (shaped like the space-prefix run)."""
    abl = copy.deepcopy(real_sp)
    sp_models = abl.pop("models")
    abl["meta"]["recipe"] = copy.deepcopy(real["meta"]["recipe"])
    abl["models"] = {}
    for name, bpb in (("bpe_frequency_g2", 1.71), ("mdl_bits_g2", 1.72), ("mdl_bits_sp300", 1.70)):
        m = copy.deepcopy(sp_models[name.rsplit("_", 1)[0] + "_sp"])
        m["spec"]["name"] = name
        abl["models"][name] = {**m, "final_bpb": bpb, "val_bytes_per_token": 4.4}
    return abl


def test_ablation_section_shows_not_yet_run_then_every_arm(tmp_path, monkeypatch) -> None:
    """No nanogpt_swap_ablation/ results: 'not yet run'. With one, per variant: whitespace (original swap),
    space_prefix (the 50 MB rerun), _g2 and _sp300 (the ablation run), and GPT-2 from the original run."""
    real = json.loads((server_mod.REPO_ROOT / "benchmarks/results/nanogpt_swap/nanogpt_swap.json").read_text())
    real_sp = json.loads((server_mod.REPO_ROOT / "benchmarks/results/nanogpt_swap_sp/nanogpt_swap.json").read_text())
    _swap_fixture(tmp_path, "nanogpt_swap", real)
    monkeypatch.setattr(server_mod, "REPO_ROOT", tmp_path)
    abl_entry = server_mod.load_benchmark_data()["results"]["swap_ablation"]
    assert abl_entry["available"] is False
    assert abl_entry["path"] == "benchmarks/results/nanogpt_swap_ablation/nanogpt_swap.json"
    assert "ablation" not in abl_entry

    _swap_fixture(tmp_path, "nanogpt_swap_ablation", _ablation_doc(real, real_sp))

    # Without the space-prefix run: its column is empty, and the README renderer says so
    entry = server_mod.load_benchmark_data()["results"]["swap_ablation"]
    assert entry["available"] is True
    rows = {r["variant"]: r for r in entry["ablation"]["rows"]}
    assert set(rows) == {"bpe_frequency", "mdl_bits"}
    assert rows["mdl_bits"]["space_prefix"] is None
    assert any("space-prefix run is not committed yet" in line for line in entry["provenance"])

    _swap_fixture(tmp_path, "nanogpt_swap_sp", real_sp)
    entry = server_mod.load_benchmark_data()["results"]["swap_ablation"]
    a = entry["ablation"]
    assert a["arms"] == ["whitespace", "space_prefix", "gpt2_split", "space_prefix_300mb"]
    rows = {r["variant"]: r for r in a["rows"]}
    mdl = rows["mdl_bits"]
    assert mdl["whitespace"]["final_bpb"] == real["models"]["mdl_bits"]["final_bpb"]
    assert mdl["space_prefix"]["final_bpb"] == real_sp["models"]["mdl_bits_sp"]["final_bpb"]
    assert mdl["gpt2_split"]["final_bpb"] == 1.72 and mdl["space_prefix_300mb"]["final_bpb"] == 1.70
    assert rows["bpe_frequency"]["space_prefix_300mb"] is None  # no run in that arm
    assert a["gpt2"]["final_bpb"] == real["models"]["gpt2"]["final_bpb"]
    # Provenance from report.render_nanogpt_swap_ablation: same recipe and val documents as the whitespace run
    assert any("same recipe and val documents" in line for line in entry["provenance"]), entry["provenance"]


def test_gpt2_equivalent_steps_read_off_the_steps_csv_match_the_committed_ones() -> None:
    """server.sp_gpt2_equivalent reads GPT-2's curve from the steps CSV: on the original run's own tokenizers it
    reproduces that run's committed GPT-2-equivalent steps."""
    base = json.loads((REAL_RESULTS / "nanogpt_swap/nanogpt_swap.json").read_text())
    ours = {**base, "models": {k: m for k, m in base["models"].items() if k != "gpt2"}}
    steps = server_mod.sp_gpt2_equivalent(base, ours, REAL_RESULTS / "nanogpt_swap/nanogpt_swap_steps.csv")
    assert steps == {k: v for k, v in base["gpt2_equivalent"]["steps"].items() if k != "gpt2"}


def _single_seed(doc):
    """A proxy-LM result as colab/lm.py writes it for one seed: no sd (None), one by-seed entry."""
    doc = copy.deepcopy(doc)
    doc["meta"]["seeds"] = doc["meta"]["seeds"][:1]
    for m in doc["models"].values():
        m["macro_val_bpb_by_seed"] = dict(list(m["macro_val_bpb_by_seed"].items())[:1])
        m["macro_val_bpb_sd"] = None
        for d in m["val_domains"].values():
            d["bpb_sd"] = None
    return doc


def test_single_seed_proxy_run_keeps_the_api_up(tmp_path, monkeypatch) -> None:
    """A single-seed big LM has no standard errors for the small vs big comparison: the comparison still
    shows, without ± for the big LM, and so do both proxy-LM files (a TypeError used to make /api/benchmarks
    a 500, then left the comparison out)."""
    _write_results(tmp_path, None, proxy_lm=json.loads((REAL_RESULTS / "colab/extrinsic_bpb_results.json").read_text()))
    big = tmp_path / "benchmarks/results/colab_big/extrinsic_bpb_results.json"
    big.parent.mkdir(parents=True)
    real_big = json.loads((REAL_RESULTS / "colab_big/extrinsic_bpb_results.json").read_text())
    big.write_text(json.dumps(_single_seed(real_big)), encoding="utf-8")
    monkeypatch.setattr(server_mod, "REPO_ROOT", tmp_path)
    data = server_mod.load_benchmark_data()
    res = data["results"]
    assert res["proxy_small"]["available"] and res["proxy_big"]["available"] and res["proxy_big"]["provenance"]
    tiers = res["proxy_compare"]["data"]["tiers"]
    assert all(se is None for _, _, se in tiers["big LM"].values())
    assert all(se is not None for _, _, se in tiers["small LM"].values())
    json.dumps(data)  # what do_GET sends
    if NODE:
        errors, html = _render(tmp_path, data)
        assert errors == [] and "reference" in html["compare-table"]


# --- static/app.js under node: the renderers on /api/benchmarks payloads ---------------------------------------

# A DOM stub just big enough for app.js: every element is a plain object whose innerHTML the test reads back;
# console.error (where renderAllViews reports a failed section) and uncaught throws are collected.
HARNESS_JS = r"""
const fs = require('fs');
const vm = require('vm');
const [appPath, dataPath, calls] = process.argv.slice(2);
const els = {};
const errors = [];
function el(id) {
  if (!els[id]) {
    const cls = new Set();
    els[id] = {
      id, innerHTML: '', textContent: '', value: '', tabIndex: 0, dataset: {}, attrs: {},
      classList: { add: (c) => cls.add(c), remove: (c) => cls.delete(c), contains: (c) => cls.has(c),
                   toggle: (c, on) => ((on ?? !cls.has(c)) ? cls.add(c) : cls.delete(c)) },
      setAttribute(k, v) { this.attrs[k] = v; }, removeAttribute(k) { delete this.attrs[k]; },
      addEventListener() {}, focus() {}, contains: () => false,
    };
  }
  return els[id];
}
class Chart { destroy() {} resize() {} }
Chart.defaults = { font: {} };
const ctx = vm.createContext({
  console: { log() {}, warn() {}, error: (...a) => errors.push(a.map((x) => (x && x.stack) || String(x)).join(' ')) },
  setTimeout: () => 0, Chart,
  document: { getElementById: el, querySelectorAll: () => [], querySelector: () => null, addEventListener() {},
              documentElement: el('html'), body: el('body') },
  window: { matchMedia: () => ({ matches: false, addEventListener() {} }) },
  localStorage: { getItem: () => null, setItem() {} },
  getComputedStyle: () => ({ getPropertyValue: () => '#336699' }),
  location: { hash: '' }, history: { replaceState() {} },
});
vm.runInContext(fs.readFileSync(appPath, 'utf8'), ctx);
vm.runInContext(`state.data = ${fs.readFileSync(dataPath, 'utf8')};`, ctx);
for (const c of JSON.parse(calls)) {
  try { vm.runInContext(c, ctx); } catch (e) { errors.push(`${c}: ${e.stack}`); }
}
const html = Object.fromEntries(Object.entries(els).map(([k, v]) => [k, v.innerHTML || v.textContent]));
process.stdout.write(JSON.stringify({ errors, html }));
"""
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="needs node to run static/app.js")


def _render(tmp_path, data, calls=("renderAllViews()",)):
    """Run static/app.js on an /api/benchmarks payload: (errors, {element id: innerHTML})."""
    harness, payload = tmp_path / "harness.js", tmp_path / "payload.json"
    harness.write_text(HARNESS_JS, encoding="utf-8")
    payload.write_text(json.dumps(data), encoding="utf-8")
    app = server_mod.STATIC_DIR / "app.js"
    out = subprocess.run([NODE, str(harness), str(app), str(payload), json.dumps(list(calls))],
                         capture_output=True, text=True, timeout=120, check=True)
    res = json.loads(out.stdout)
    return res["errors"], res["html"]


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _signed(v: float, digits: int = 4) -> str:  # app.js signed()
    return f"{'+' if v > 0 else '−' if v < 0 else ''}{abs(v):.{digits}f}"


@needs_node
def test_dashboard_renders_the_committed_results(tmp_path) -> None:
    """Every section on the committed results, none failing; the claims on the page come from the files."""
    import html as htmllib
    import math
    import statistics

    data = server_mod.load_benchmark_data()
    errors, html = _render(tmp_path, data)
    assert errors == []

    # Proxy caveat: each tier's top-two margin in standard errors (seedMargin), not a fixed claim
    caveat = _text(html["proxy-caveat-small"])
    assert "cannot separate" not in caveat
    for key, tier in (("proxy_small", "small LM"), ("proxy_big", "big LM")):
        ms = data["results"][key]["data"]["models"]
        a, b = sorted(ms, key=lambda n: ms[n]["macro_val_bpb"])[:2]
        se2 = [ms[n]["macro_val_bpb_sd"] ** 2 / len(ms[n]["macro_val_bpb_by_seed"]) for n in (a, b)]
        z = (ms[b]["macro_val_bpb"] - ms[a]["macro_val_bpb"]) / math.sqrt(sum(se2))
        assert (f"{tier}: {a} ahead of {b} by {z:.1f} standard errors over the seeds, "
                f"{'resolved' if z >= 2 else 'not resolved'}") in caveat, caveat

    # Space-prefix rerun: GPT-2-equivalent steps for every row (read off the original run's GPT-2 curve)
    sp_table = _text(html["sp-table"])
    for r in data["results"]["swap_space_prefix"]["pairs"]["rows"]:
        assert f" {r['sp']['gpt2_equivalent_steps']} " in f" {sp_table} ", r

    # FLORES+: the "UTF-8 bytes" reference of the findings is on the page, as a row of the premium heatmap
    meta = data["results"]["flores"]["data"]["meta"]
    text, eng, langs = meta["text"], meta["languages"]["english"], meta["languages"]
    ratio = {lang: text[lang]["bytes"] / text[eng]["bytes"] for lang in text}
    in_mix, out_mix = list(langs["in_mix"].values()), langs["out_of_mix"]
    cells = ([ratio[x] for x in in_mix] + [statistics.fmean(ratio[x] for x in in_mix)]
             + [ratio[x] for x in out_mix] + [statistics.fmean(ratio[x] for x in out_mix)])
    row = f"UTF-8 bytes {text[eng]['bytes'] / text[eng]['sentences']:.1f} " + " ".join(f"{c:.2f}" for c in cells)
    assert row in _text(html["flores-heat"])
    notes = htmllib.unescape(html["flores-notes"])
    assert '"UTF-8 bytes" row' in notes and 'column "UTF-8 bytes"' not in notes


@needs_node
def test_one_tokeniser_proxy_run_renders_every_section(tmp_path, monkeypatch) -> None:
    """A proxy-LM file with one tokeniser has no runner-up: its leader card leaves the margin out. And a
    section that fails no longer blanks the sections rendered after it."""
    small = json.loads((REAL_RESULTS / "colab/extrinsic_bpb_results.json").read_text())
    small["models"] = {"bpe_frequency": small["models"]["bpe_frequency"]}
    _write_results(tmp_path, None, proxy_lm=small)
    monkeypatch.setattr(server_mod, "REPO_ROOT", tmp_path)
    data = server_mod.load_benchmark_data()
    errors, html = _render(tmp_path, data)
    assert errors == []
    assert "Proxy LM · small" in html["leaders"] and "bpe_frequency" in html["leaders"]
    assert "BPE (Frequency)" in html["leaderboard-tbody"]

    errors, html = _render(tmp_path, data, ["renderSummary = () => { throw new Error('boom'); }", "renderAllViews()"])
    assert len(errors) == 1 and "boom" in errors[0]
    assert "BPE (Frequency)" in html["leaderboard-tbody"]


@needs_node
def test_radar_leaves_unmeasured_axes_out_and_min_max_normalises_the_rest(tmp_path) -> None:
    """bpe_dropout_p0.1 has only a proxy-LM BPB: its other five axes are gaps named in the legend, not
    values plotted above other tokenisers' measured worst; measured axes span exactly [0, 1]."""
    data = server_mod.load_benchmark_data()
    picked = ["bpe_dropout_p0.1", "bpe_frequency", "mdl_bits"]
    errors, html = _render(tmp_path, data, [
        "Chart = class { static defaults = { font: {} }; constructor(c, cfg) { globalThis.cfg = cfg; } destroy() {} };",
        f"state.selectedRadarModels = {json.dumps(picked)}; renderRadarView();",
        "document.getElementById('radar-out').innerHTML = JSON.stringify(cfg.data);"])
    assert errors == []
    out = json.loads(html["radar-out"])
    drop = out["datasets"][0]
    assert drop["data"][:4] + drop["data"][5:] == [None] * 5 and 0 <= drop["data"][4] <= 1
    assert drop["label"].endswith("(not measured: Compression, Rényi E₂.₅, Morphology F1, Identifier F1, Decode speed)")
    ms = data["models"]
    for i, field in ((0, "bytes_per_token"), (1, "renyi_e_2_5"), (5, "decode_mb_per_sec")):
        vals = [m[field] for m in ms.values() if m.get(field) is not None]
        for d, key in zip(out["datasets"][1:], picked[1:]):  # no floor: the worst measured is 0
            assert d["data"][i] == pytest.approx((ms[key][field] - min(vals)) / (max(vals) - min(vals)))


@needs_node
def test_swap_leader_card_takes_the_best_of_the_comparable_swap_runs(tmp_path) -> None:
    """The GPT-2 swap card ranks the original, space-prefix and ablation runs together (same recipe and val
    documents; not the lr 1.8e-3 run): unigram_lm_g2 from the ablation leads GPT-2 by less than 0.01 BPB at
    one seed, a tie the card says is not resolved."""
    import html as htmllib

    data = server_mod.load_benchmark_data()
    errors, html = _render(tmp_path, data)
    assert errors == []
    card = next(htmllib.unescape(_text(c)) for c in html["leaders"].split('<div class="kpi">') if ">GPT-2 swap<" in c)
    assert "1.6817 BPB unigram_lm_g2" in card and "unigram_lm_g2: ablation · GPT-2 split @ 50 MB." in card
    assert "Runner-up gpt2 1.6851 (+0.0033)" in card and "Runner-up gpt2: original · not our tokeniser." in card
    assert "Tie at one seed: margin 0.0033 BPB is below the 0.01 tolerance; seed spread unmeasured, so not resolved." in card
    assert "within noise" not in card
    assert "one seed per run, so the seed-to-seed spread is unmeasured" in card
    assert "(original, space-prefix rerun, ablation)" in card and "1.8e-3" not in card
    assert f"@{data['results']['swap_ablation']['data']['meta']['git_commit'][:7]} " in card  # the leader's run


def _swap_card(tmp_path, mutate):
    """The GPT-2 swap leader card's text on the committed results after ``mutate(results)``."""
    import copy
    import html as htmllib

    data = copy.deepcopy(server_mod.load_benchmark_data())
    mutate(data["results"])
    errors, html = _render(tmp_path, data)
    assert errors == []
    return next(htmllib.unescape(_text(c)) for c in html["leaders"].split('<div class="kpi">') if ">GPT-2 swap<" in c)


@needs_node
def test_swap_card_leaves_out_a_run_that_is_not_comparable(tmp_path) -> None:
    """A swap run whose not_comparable is non-empty (another recipe or val documents) is not ranked: with
    the ablation flagged, its unigram_lm_g2 no longer leads and the run is not named."""
    card = _swap_card(tmp_path, lambda r: r["swap_ablation"].update(not_comparable=["different recipe"]))
    assert "GPT-2 swap 1.6851 BPB" in card and "gpt2: original · not our tokeniser." in card
    assert "unigram_lm_g2" not in card and "(original, space-prefix rerun)" in card and "ablation" not in card
    card = _swap_card(tmp_path, lambda r: r["swap_space_prefix"].update(not_comparable=["different recipe"]))
    assert "(original, ablation)" in card and "space-prefix rerun" not in card


@needs_node
def test_swap_card_ignores_a_run_with_no_measured_bpb(tmp_path) -> None:
    """A later run's null (or text) final_bpb used to replace the original's measured GPT-2 value, because
    null < 1.68 in JavaScript, dropping gpt2 from the ranking."""
    for unmeasured in (None, "0.5"):
        def mutate(r, v=unmeasured):
            r["swap_ablation"]["data"]["models"]["gpt2"] = {"final_bpb": v, "spec": {"kind": "tiktoken"}}
        card = _swap_card(tmp_path, mutate)
        assert "Runner-up gpt2 1.6851 (+0.0033)" in card and "gpt2: original · not our tokeniser." in card, unmeasured


@needs_node
def test_swap_card_arm_comes_from_the_recorded_spec(tmp_path) -> None:
    """The arm label reads the spec's pretokenizer and tokenizer_text.bytes; the name's suffix is only the
    fallback. A name ending _g2 whose spec records space_prefix and 300 MB is labelled by the spec."""
    def mutate(r):
        spec = r["swap_ablation"]["data"]["models"]["unigram_lm_g2"]["spec"]
        spec["pretokenizer"], spec["tokenizer_text"]["bytes"] = "space_prefix", 299_998_439
    card = _swap_card(tmp_path, mutate)
    assert "unigram_lm_g2: ablation · space_prefix @ 300 MB." in card and "GPT-2 split" not in card
    card = _swap_card(tmp_path, lambda r: r["swap_ablation"]["data"]["models"]["unigram_lm_g2"]["spec"].pop("pretokenizer"))
    assert "unigram_lm_g2: ablation · GPT-2 split @ 50 MB." in card  # recorded bytes, name-derived pretokenizer


def _sp_root(tmp_path, sp):
    """The committed original swap run (with its steps CSV) and ``sp`` as the space-prefix rerun."""
    base = json.loads((REAL_RESULTS / "nanogpt_swap/nanogpt_swap.json").read_text())
    _swap_fixture(tmp_path, "nanogpt_swap", base)
    shutil.copy(REAL_RESULTS / "nanogpt_swap/nanogpt_swap_steps.csv", tmp_path / "benchmarks/results/nanogpt_swap")
    _swap_fixture(tmp_path, "nanogpt_swap_sp", sp)
    return base


def test_space_prefix_rerun_that_is_not_comparable_is_flagged(tmp_path, monkeypatch) -> None:
    """The README's comparability check (report.render_nanogpt_swap_space_prefix) heads the section: a rerun with
    another recipe or other val documents is flagged, and no GPT-2-equivalent steps are read off the original
    run's GPT-2 curve for it."""
    sp = json.loads((REAL_RESULTS / "nanogpt_swap_sp/nanogpt_swap.json").read_text())
    sp["meta"]["recipe"]["args"]["learning_rate"] = 0.0018
    for m in sp["models"].values():
        m["val_docs_sha256"] = "0" * 64
    _sp_root(tmp_path, sp)
    monkeypatch.setattr(server_mod, "REPO_ROOT", tmp_path)
    data = server_mod.load_benchmark_data()
    e = data["results"]["swap_space_prefix"]
    assert any(line.startswith("DIFFERENT recipe, val documents") for line in e["provenance"]), e["provenance"]
    rows = e["pairs"]["rows"]
    assert len(rows) == 8 and all(r["sp"]["gpt2_equivalent_steps"] is None for r in rows)
    if NODE:  # the warning is on the page, above the BPB changes, not only in the collapsed provenance
        errors, html = _render(tmp_path, data, ["renderSpacePrefix()"])
        facts = _text(html["sp-body"])
        assert errors == [] and facts.startswith("Not comparable with the whitespace run: different recipe; "
                                                 "different val documents"), facts

    # The committed rerun: same recipe and val documents, so its steps come from the original GPT-2 curve
    shutil.copy(REAL_RESULTS / "nanogpt_swap_sp/nanogpt_swap.json", tmp_path / "benchmarks/results/nanogpt_swap_sp")
    e = server_mod.load_benchmark_data()["results"]["swap_space_prefix"]
    assert not any(line.startswith("DIFFERENT") for line in e["provenance"])
    assert e["pairs"]["gpt2"]["run"] == "old"
    assert all(isinstance(r["sp"]["gpt2_equivalent_steps"], int) for r in e["pairs"]["rows"])


@needs_node
def test_ablation_facts_compare_each_arm_with_the_arm_it_changes(tmp_path, monkeypatch) -> None:
    """space_prefix against whitespace (the pre-split), _g2 and _sp300 against space_prefix (GPT-2's regex rules;
    300 MB instead of 50 MB), so the space-prefix effect is not credited to the other arms; the per-variant
    tally only counts variants run in every arm."""
    import statistics

    real = json.loads((REAL_RESULTS / "nanogpt_swap/nanogpt_swap.json").read_text())
    real_sp = json.loads((REAL_RESULTS / "nanogpt_swap_sp/nanogpt_swap.json").read_text())
    for rel, doc in (("nanogpt_swap", real), ("nanogpt_swap_sp", real_sp),
                     ("nanogpt_swap_ablation", _ablation_doc(real, real_sp))):
        _swap_fixture(tmp_path, rel, doc)
    monkeypatch.setattr(server_mod, "REPO_ROOT", tmp_path)
    errors, html = _render(tmp_path, server_mod.load_benchmark_data(), ["renderAblation()"])
    assert errors == []
    facts = _text(html["abl-body"])
    sp = {v: real_sp["models"][f"{v}_sp"]["final_bpb"] for v in ("bpe_frequency", "mdl_bits")}
    g2 = [1.71 - sp["bpe_frequency"], 1.72 - sp["mdl_bits"]]
    assert (f"space_prefix, 300 MB (_sp300) against space_prefix, 50 MB: lower final BPB for 1 of 1 variants; "
            f"median change {_signed(1.70 - sp['mdl_bits'])} BPB") in facts, facts
    assert (f"GPT-2 regex split (_g2), 50 MB against space_prefix, 50 MB: lower final BPB for {sum(d < 0 for d in g2)} "
            f"of 2 variants; median change {_signed(statistics.median(g2))} BPB") in facts, facts
    assert "Lowest final BPB, over the 1 variant run in every arm:" in facts, facts


@needs_node
def test_table_headers_escape_labels_from_result_files(tmp_path) -> None:
    """Column labels built from result JSON (morph languages, the FLORES+ parity threshold, proxy-LM domains) are
    text, not markup: a crafted result file cannot inject script into the dashboard's origin."""
    evil = "<img src=x onerror=alert(1)>"
    data = server_mod.load_benchmark_data()
    res = data["results"]
    next(iter(res["intrinsic_16384"]["data"]["models"].values()))["morph_words_by_lang"][evil] = 3
    res["flores"]["data"]["meta"]["parity_threshold"] = evil
    res["proxy_small"]["data"]["meta"]["val_domains"].append(evil)
    errors, html = _render(tmp_path, data)
    assert errors == []
    for table in ("morph-table", "flores-table", "proxy-table"):
        assert "<img" not in html[table] and "&lt;img" in html[table], table


@needs_node
def test_model_keys_and_lr_steps_from_result_files_cannot_run_script(tmp_path) -> None:
    """A model key (copied verbatim from the tokka, intrinsic and proxy-LM model maps) reaches the onclick handlers
    of the Overview 'Try' button and the radar chips as that same string, not as code; the lr run's eval steps and
    the swap recipe's lr are text, not markup."""
    import html as htmllib

    evil = "x');globalThis.pwned=1;//\"<img src=x onerror=alert(1)>"
    data = server_mod.load_benchmark_data()
    data["models"][evil] = {**data["models"]["bpe_frequency"], "key": evil}
    data["results"]["swap_lr1p8e-3"]["data"]["trainer"]["evals"][1]["step"] = evil
    data["results"]["swap"]["data"]["meta"]["recipe"]["args"]["learning_rate"] = evil
    errors, html = _render(tmp_path, data, ["renderLeaderboard(); renderRadarView(); renderLr()"])
    assert errors == []
    assert "<img" not in html["lr-body"] and html["lr-body"].count("&lt;img") == 3  # step, lr in a fact and a header
    # The handler as the browser runs it: the attribute value with its character references decoded
    handlers = [js for el in ("leaderboard-tbody", "radar-toggles")
                for js in map(htmllib.unescape, re.findall(r'onclick="([^"]*)"', html[el])) if "pwned" in js]
    assert len(handlers) == 2, handlers
    errors, html = _render(tmp_path, data, [
        "globalThis.got = []; inspectModelInPlayground = toggleRadarModel = (k) => got.push(k);", *handlers,
        "document.getElementById('probe').innerHTML = JSON.stringify([globalThis.pwned ?? null, got])"])
    assert errors == [] and json.loads(html["probe"]) == [None, [evil, evil]]


@needs_node
def test_switching_tab_from_inside_a_panel_moves_focus_to_the_new_tab(tmp_path) -> None:
    """'Details →', the result-set links, 'Try →' and the Concepts link switch tabs from inside the panel that is
    hidden: keyboard focus moves to the new tab instead of dropping to <body>."""
    errors, html = _render(tmp_path, {}, [
        "for (const id of ['tab-nav-summary', 'tab-nav-proxy']) "
        "document.getElementById(id).focus = function () { document.activeElement = this; };",
        "globalThis.btn = {}; document.getElementById('tab-summary').contains = (x) => x === btn; "
        "document.activeElement = btn;",
        "switchTab('summary'); document.getElementById('same').innerHTML = String(document.activeElement === btn)",
        "switchTab('proxy'); document.getElementById('probe').innerHTML = document.activeElement.id ?? 'body'",
    ])
    assert errors == []
    assert html["same"] == "true"  # the panel stays: focus stays on the button
    assert html["probe"] == "tab-nav-proxy"


def test_swap_rerun_whose_comparability_cannot_be_checked_is_not_comparable(tmp_path, monkeypatch) -> None:
    """The comparability check (recipe, val documents, tokenizer training text) fails closed: a rerun or ablation
    file without a field the check reads is flagged, not shown as comparable, even when the README renderer
    cannot run on it."""
    sp = json.loads((REAL_RESULTS / "nanogpt_swap_sp/nanogpt_swap.json").read_text())
    sp["meta"]["recipe"]["args"]["learning_rate"] = 0.0018
    del sp["meta"]["tokenizer_text"]
    base = _sp_root(tmp_path, sp)
    monkeypatch.setattr(server_mod, "REPO_ROOT", tmp_path)
    e = server_mod.load_benchmark_data()["results"]["swap_space_prefix"]
    assert e["not_comparable"] == ["different recipe", "tokenizer training text unknown (missing from a file)"]
    assert all(r["sp"]["gpt2_equivalent_steps"] is None for r in e["pairs"]["rows"])
    if NODE:
        errors, html = _render(tmp_path, server_mod.load_benchmark_data(), ["renderSpacePrefix()"])
        facts = _text(html["sp-body"])
        assert errors == [] and facts.startswith("Not comparable with the whitespace run: different recipe; "
                                                 "tokenizer training text unknown"), facts

    # Val documents: a file without val_docs_sha256 is not taken as the same documents (no GPT-2-equivalent steps)
    sp = json.loads((REAL_RESULTS / "nanogpt_swap_sp/nanogpt_swap.json").read_text())
    for m in sp["models"].values():
        del m["val_docs_sha256"]
    _swap_fixture(tmp_path, "nanogpt_swap_sp", sp)
    _swap_fixture(tmp_path, "nanogpt_swap_ablation", _ablation_doc(base, sp))
    data = server_mod.load_benchmark_data()
    e, abl = data["results"]["swap_space_prefix"], data["results"]["swap_ablation"]
    assert e["not_comparable"] == ["val documents unknown (missing from a file)"]
    assert all(r["sp"]["gpt2_equivalent_steps"] is None for r in e["pairs"]["rows"])
    assert abl["not_comparable"] == ["val documents unknown (missing from a file)"]
    if NODE:
        errors, html = _render(tmp_path, data, ["renderAblation()"])
        assert errors == [] and _text(html["abl-body"]).startswith("Not comparable"), html["abl-body"]

    # The committed rerun passes every check
    shutil.copy(REAL_RESULTS / "nanogpt_swap_sp/nanogpt_swap.json", tmp_path / "benchmarks/results/nanogpt_swap_sp")
    assert server_mod.load_benchmark_data()["results"]["swap_space_prefix"]["not_comparable"] == []


@needs_node
def test_page_copy_matches_the_runs(tmp_path) -> None:
    """Copy that describes the runs agrees with the result files and the README: the GPT-2-swap tier is
    English-centric (GPT-2's WebText, TokenMonster's English and code vocabularies, English FineWeb), not "English
    FineWeb", and did not learn the in-mix languages; the pre-tokeniser isolates whitespace runs, not each space;
    two of the three throughput runs are on the same laptop."""
    from tokenisers.benchmarks.report import _TIERS

    data = server_mod.load_benchmark_data()
    for tier in ("intrinsic", "swap"):
        errors, html = _render(tmp_path, data, [f"state.floresTier = '{tier}'; renderFlores()"])
        assert errors == [] and f"tokenisers of the {_TIERS[tier]}." in html["flores-sub"], html["flores-sub"]
        assert ("(training languages)" in html["flores-heat"]) == (tier == "intrinsic")
    assert "English FineWeb" not in (server_mod.STATIC_DIR / "app.js").read_text()

    errors, html = _render(tmp_path, data, ["renderSwap()"])
    assert errors == [] and "every whitespace run is its own token" in html["swap-caveat-space"]
    page = (server_mod.STATIC_DIR / "index.html").read_text()
    assert "split spaces into their own tokens" not in page and "learned their merges on English text only" not in page

    meta = {k: data["results"][k]["data"]["meta"] for k in ("intrinsic_16384", "throughput_local")}
    assert meta["intrinsic_16384"]["host"]["cpu"] == meta["throughput_local"]["backend"]["cpu"]
    assert "different machines" not in page
