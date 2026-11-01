"""
End-to-End Tests for Tokenisers Lab Benchmark Intelligence Dashboard & REST APIs.
"""

from __future__ import annotations

import http.client
import json
import socket
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
    "swap_lr1p8e-3", "swap_space_prefix"}


def test_api_benchmarks_returns_every_result_set(dashboard_server: str) -> None:
    status, _, body = _raw(dashboard_server, "GET", "/api/benchmarks")
    assert status == 200
    results = json.loads(body)["results"]
    assert set(results) == RESULT_KEYS
    for key, e in results.items():
        assert isinstance(e["available"], bool) and e["path"].startswith("benchmarks/results/"), key
        assert (e["data"] is not None) == e["available"], key
    # Everything except the pending space-prefix rerun is committed, with README-rendered provenance
    for key in RESULT_KEYS - {"swap_space_prefix", "swap_lr1p8e-3"}:
        assert results[key]["available"] and results[key]["provenance"], key
    assert results["swap_lr1p8e-3"]["available"]
    assert results["swap"]["notes"], "the swap findings come from report.render_nanogpt_swap"
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
