"""Result provenance: the commit recorded for a run and the leaderboard header must be true."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from tokenisers.benchmarks.common import start_provenance


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


def test_source_checkout_records_its_commit(repo_root):
    prov = start_provenance("tokenisers.benchmarks.tokka", [])
    assert prov["git_commit"] == _git(repo_root, "rev-parse", "HEAD")
    assert isinstance(prov["git_dirty"], bool) and prov["tokenisers_version"]


def test_installed_copy_does_not_borrow_an_enclosing_repositorys_commit(tmp_path, require_tokenisers):
    """A non-editable install inside some other git repository (a project's .venv) must not record
    that repository's HEAD as the code that produced the results."""
    import tokenisers

    outer = tmp_path / "project"
    outer.mkdir()
    _git(outer, "init", "-q")
    (outer / "x.txt").write_text("x")
    _git(outer, "add", "x.txt")
    _git(outer, "commit", "-q", "-m", "unrelated")
    site = outer / ".venv" / "lib" / "python3.13" / "site-packages"
    shutil.copytree(Path(tokenisers.__file__).parent, site / "tokenisers",
                    ignore=shutil.ignore_patterns("__pycache__"))
    code = ("import json, tokenisers.benchmarks.common as c\n"
            f"assert c.__file__.startswith({str(site)!r}), c.__file__\n"
            "print(json.dumps(c.start_provenance('m', [])))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(site)})
    assert out.returncode == 0, out.stderr
    prov = json.loads(out.stdout)
    assert prov["git_commit"] is None and prov["git_dirty"] is None, prov


def test_external_tokenizer_header_does_not_claim_it_was_trained_here(require_tokenisers, tmp_path):
    """tokka --tokenizer-path evaluates a tokenizer this run did not train: no requested V, no
    'trained on the train split' claim."""
    from tokenisers.benchmarks.tokka.__main__ import main
    from tokenisers.data.pipeline import DataPipeline

    cache = tmp_path / "cache"
    ds = DataPipeline(cache_dir=cache, force_offline=True).prepare(train_bytes=30000)
    path = tmp_path / "ext_tokenizer.json"
    require_tokenisers.BpeTokenizer.train(corpus=ds.texts("train"), vocab_size=300).export_hf_json(str(path))
    out = tmp_path / "out"
    main(["--offline", "--cache-dir", str(cache), "--train-bytes", "30000", "--tokenizer-path", str(path),
          "--output-dir", str(out), "--min-docs-per-lang", "1", "--min-bytes-per-lang", "0"])
    md = (out / "tokka_leaderboard.md").read_text()
    header = md.split("\n")[0]
    assert "external tokenizer `ext_tokenizer.json`, not trained by this run" in header
    assert "requested V" not in header and "trained on the train split" not in header
    assert "| `ext_tokenizer` | 300 |" in md
    meta = json.loads((out / "tokka_matrix.json").read_text())["meta"]
    assert meta["requested_vocab_size"] is None and meta["train_split"] is None


def test_dirty_flag_counts_code_not_results_or_readme(tmp_path, monkeypatch):
    """Benchmarks run one after another write results (and README tables) into the tree; that must
    not mark the next run's code as dirty, while a modified source file must."""
    from tokenisers.benchmarks import common

    repo = tmp_path / "repo"
    pkg = repo / "python" / "tokenisers" / "benchmarks"
    pkg.mkdir(parents=True)
    (repo / "benchmarks" / "results" / "tokka").mkdir(parents=True)
    files = [pkg / "common.py", repo / "README.md", repo / "benchmarks" / "results" / "tokka" / "m.json"]
    for f in files:
        f.write_text("v1")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    monkeypatch.setattr(common, "_HERE", pkg)
    assert common.start_provenance("m", [])["git_dirty"] is False
    files[1].write_text("v2")
    files[2].write_text("v2")
    assert common.start_provenance("m", [])["git_dirty"] is False
    files[0].write_text("v2")
    assert common.start_provenance("m", [])["git_dirty"] is True


def test_native_extension_is_recorded_and_a_stale_build_is_flagged(tmp_path, monkeypatch, require_tokenisers):
    """The untracked native extension produces every tokenizer. Results record its hash, and Rust
    sources modified after it was built (a pull or merge without `maturin develop`) mark it stale,
    in the JSON and in the leaderboard header, instead of crediting old code to the new commit."""
    import hashlib

    from tokenisers import _core
    from tokenisers.benchmarks import common
    from tokenisers.benchmarks.report import _summary

    so = Path(_core.__file__)
    repo = tmp_path / "repo"
    pkg = repo / "python" / "tokenisers" / "benchmarks"
    pkg.mkdir(parents=True)
    (pkg / "common.py").write_text("x")
    src = repo / "crates" / "core" / "lib.rs"
    src.parent.mkdir(parents=True)
    src.write_text("fn main() {}")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    monkeypatch.setattr(common, "_HERE", pkg)

    built = so.stat().st_mtime
    os.utime(src, (built + 60, built + 60))  # checked out after the extension was built
    prov = common.start_provenance("m", [])
    native = prov["native_extension"]
    assert native["file"] == so.name and native["sha256"] == hashlib.sha256(so.read_bytes()).hexdigest()
    assert native["stale_sources"] == ["crates/core/lib.rs"]
    assert "STALE native extension" in _summary(prov)

    os.utime(src, (built - 60, built - 60))  # rebuilt since
    prov = common.start_provenance("m", [])
    assert prov["native_extension"]["stale_sources"] == [] and "STALE" not in _summary(prov)


def test_clis_record_the_module_they_run_as(tmp_path, monkeypatch, require_tokenisers):
    """`python -m <recorded module>` must rerun the same CLI, not its package's (which has other flags)."""
    from tokenisers.benchmarks import common, throughput
    from tokenisers.benchmarks.nanogpt_swap import train_tokenizers

    real, seen = common.start_provenance, []

    class Stop(Exception):
        pass

    def spy(module, argv):
        seen.append(real(module, argv)["command"])
        raise Stop

    monkeypatch.setattr(common, "start_provenance", spy)
    for cli, argv in ((throughput.cli, ["--job-dir", str(tmp_path / "t")]),
                      (train_tokenizers.main, ["--job-dir", str(tmp_path / "p"), "--variants", "mdl_bits"])):
        try:
            cli(argv)
        except Stop:
            pass
    assert seen == [["python", "-m", "tokenisers.benchmarks.throughput", "--job-dir", str(tmp_path / "t")],
                    ["python", "-m", "tokenisers.benchmarks.nanogpt_swap.train_tokenizers", "--job-dir",
                     str(tmp_path / "p"), "--variants", "mdl_bits"]]
