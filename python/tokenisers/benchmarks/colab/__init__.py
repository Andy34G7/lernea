"""
Extrinsic proxy-LM benchmark: a small causal Transformer is trained per tokenizer under a fixed
UTF-8 byte budget and scored by validation bits per byte (BPB).

The model, trainer, evaluator and job runner live in ``colab/lm.py``, which is imported locally
and uploaded unchanged to Colab, so local and remote runs share one implementation.
``ColabOrchestrator`` runs a job directory on a Colab VM through the ``colab`` CLI (>= 0.7.5); the
job script is ``lm.py`` here and ``benchmarks/throughput.py`` for the throughput benchmark.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import uuid

from tokenisers.benchmarks.colab.lm import (
    BpbEvaluator,
    ByteBudgetTrainer,
    ProxyLanguageModel,
    run_job,
)

LM_SOURCE = Path(__file__).with_name("lm.py")
REMOTE_DIR = "/content"
# 0.6.0's `exec` fails with AttributeError (jupyter_kernel_client has no KernelClient).
MIN_CLI_VERSION: Tuple[int, int, int] = (0, 7, 5)
UPGRADE_HINT = "uv tool upgrade google-colab-cli"
STATUS_MARKER = "TOKENISERS_JOB "
# One poll: a short `colab exec`. A call that hangs (e.g. "Connection was lost") is killed after
# EXEC_TIMEOUT seconds and counts as a failed poll.
EXEC_TIMEOUT = 300.0

# Run in the Colab kernel on every poll (``colab exec`` exits 0 even when the code raises, so the
# state is read from the STATUS_MARKER line, never from the exit code). Idempotent: the first call
# launches the job script (``python lm.py <dir>``) detached (own session, output in job.log, pid in
# job.pid) and returns at once; later calls report "running", or the final status.json that the
# script's ``main`` writes ("ok"/"error"),
# or "died" when the process is gone without one. A poll lost to a dropped connection is retried.
CONTROL_SCRIPT = """\
import json, os, subprocess, sys
d = {remote_dir!r}
pid_path, status_path, log_path = (os.path.join(d, f) for f in ("job.pid", "status.json", "job.log"))

def alive(pid):
    try:
        with open(f"/proc/{{pid}}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False

if not os.path.exists(pid_path):
    with open(log_path, "ab") as log:
        p = subprocess.Popen([sys.executable, "-u", os.path.join(d, {script!r}), d], cwd=d, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    with open(pid_path, "w") as f:
        f.write(str(p.pid))
    out = {{"state": "launched", "pid": p.pid}}
else:
    with open(pid_path) as f:
        pid = int(f.read())
    running = alive(pid)  # before reading the status: the script writes status.json, then exits
    if os.path.exists(status_path):
        with open(status_path) as f:
            out = json.load(f)
    else:
        out = {{"state": "running" if running else "died", "pid": pid}}
try:
    with open(log_path, "rb") as f:
        f.seek(max(0, os.path.getsize(log_path) - 4000))
        out["log_tail"] = f.read().decode("utf-8", "replace")
except OSError:
    out["log_tail"] = ""
runs = os.path.join(d, "runs")
out["runs"] = sorted(f for f in os.listdir(runs) if f.endswith(".json")) if os.path.isdir(runs) else []
print({marker!r} + json.dumps(out))
"""


class ColabError(RuntimeError):
    """A ``colab`` CLI step or the remote job failed (the message carries the remote traceback)."""


def plain_name(f: Any) -> str:
    """A file name the VM reported, returned only if it names a file directly inside the target dir:
    no separator, no '..', not absolute (a misbehaving VM must not write outside the job)."""
    if not isinstance(f, str) or f in ("", ".", "..") or Path(f).name != f or "\\" in f:
        raise ColabError(f"refusing the file name {f!r} from the VM: not a plain file name")
    return f


@dataclass
class ColabSession:
    session_id: str
    active: bool = True


class ColabOrchestrator:
    """Runs benchmark jobs on Google Colab via the ``colab`` CLI. ``mock=True`` never calls the CLI."""

    def __init__(self, mock: bool = False, colab_bin: Optional[str] = None) -> None:
        self.mock = mock
        self.colab_bin = colab_bin or str(Path.home() / ".local" / "bin" / "colab")

    def _colab(self, *args: str, timeout: Optional[float] = 600) -> str:
        """Runs one CLI command; returns its stdout + stderr. A non-zero exit raises ColabError, but
        exit 0 does not mean the remote code succeeded (see CONTROL_SCRIPT)."""
        cmd = [self.colab_bin, *args]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise ColabError(f"`colab {args[0]}` failed: {e}") from e
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip()[-2000:]
            raise ColabError(f"`colab {args[0]}` exited with {proc.returncode}: {tail}")
        return proc.stdout + proc.stderr

    def check_cli_version(self) -> Tuple[int, ...]:
        """Raises ColabError unless ``colab version`` reports >= MIN_CLI_VERSION."""
        need = ".".join(map(str, MIN_CLI_VERSION))
        try:
            out = self._colab("version", timeout=60)
        except ColabError as e:
            raise ColabError(f"{e}\nthe benchmark needs colab CLI >= {need}; upgrade with `{UPGRADE_HINT}`") from e
        m = re.search(r"Version:\s*(\d+)\.(\d+)\.(\d+)", out)
        if not m:
            raise ColabError(f"cannot read the colab CLI version from `colab version` ({out.strip()[-200:]!r}); "
                             f"the benchmark needs >= {need}; upgrade with `{UPGRADE_HINT}`")
        version = tuple(int(g) for g in m.groups())
        if version < MIN_CLI_VERSION:
            raise ColabError(f"colab CLI {'.'.join(m.groups())} is too old: the benchmark needs >= {need} "
                             f"(0.6.0's `exec` is broken); upgrade with `{UPGRADE_HINT}`")
        return version

    @contextlib.contextmanager
    def managed_session(self, session_name: str, gpu: Optional[str] = "T4") -> Iterator[ColabSession]:
        """Creates a session and always stops it afterwards, also when ``colab new`` fails or times
        out (the VM may exist even though the client gave up). Meanwhile SIGTERM (``kill``) raises
        KeyboardInterrupt, so a killed CLI stops its session too instead of leaving it billing."""
        session = ColabSession(session_id=session_name, active=True)
        main = threading.current_thread() is threading.main_thread()  # signal handlers live there only
        old = signal.signal(signal.SIGTERM, signal.default_int_handler) if main else None
        try:
            if not self.mock:
                self._colab("new", "-s", session_name, *(["--gpu", gpu] if gpu else []))
            yield session
        finally:
            session.active = False
            if not self.mock:
                try:
                    self._colab("stop", "-s", session_name, timeout=120)
                except ColabError:
                    pass
            if main:
                signal.signal(signal.SIGTERM, old if old is not None else signal.SIG_DFL)

    def _colab_retry(self, *args: str, attempts: int, delay: float, timeout: float) -> str:
        """``_colab`` for an idempotent step (upload, download): a transient failure such as a lost
        connection is retried, ``delay`` s apart, up to ``attempts`` calls in all."""
        for _ in range(attempts - 1):
            try:
                return self._colab(*args, timeout=timeout)
            except ColabError:
                time.sleep(delay)
        return self._colab(*args, timeout=timeout)

    def _poll(self, name: str, control: Path) -> Dict[str, Any]:
        out = self._colab("exec", "-s", name, "-f", str(control), "--timeout", "120", timeout=EXEC_TIMEOUT)
        for line in out.splitlines():
            if line.startswith(STATUS_MARKER):
                return json.loads(line[len(STATUS_MARKER):])
        raise ColabError(f"`colab exec` of the job control script printed no status:\n{out.strip()[-3000:]}")

    def _sync_runs(self, name: str, status: Dict[str, Any], local: Path, attempts: int = 1,
                   delay: float = 0.0) -> Dict[str, str]:
        """Downloads finished per-run result files not yet held locally, so a dropped session loses
        only the runs in flight; returns {file: last error} of those still missing. During the polls
        one try each (the next poll retries); after the last poll ``attempts`` tries, ``delay`` s
        apart, as nothing retries them once the session stops."""
        failed: Dict[str, str] = {}
        for f in map(plain_name, status.get("runs", [])):
            if not (local / f).exists():
                local.mkdir(exist_ok=True)
                try:
                    self._colab_retry("download", "-s", name, f"{REMOTE_DIR}/runs/{f}", str(local / f),
                                      attempts=attempts, delay=delay, timeout=600)
                    print(f"downloaded runs/{f}", flush=True)
                except ColabError as e:
                    failed[f] = str(e)
        return failed

    def _wait(self, name: str, control: Path, timeout: float, poll_interval: float,
              max_poll_failures: int, runs_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Launches the job (first poll) and polls until it is ok/error/died; raises ColabError
        after ``max_poll_failures`` consecutive failed polls or when ``timeout`` seconds pass."""
        deadline = time.monotonic() + timeout
        failures = 0
        last: Dict[str, Any] = {}
        while True:
            try:
                last = self._poll(name, control)
                failures = 0
                if runs_dir is not None:
                    self._sync_runs(name, last, runs_dir)
            except ColabError as e:
                failures += 1
                if failures >= max_poll_failures:
                    raise ColabError(f"{failures} consecutive job polls failed; last error: {e}") from e
            if last.get("state") in ("ok", "error", "died"):
                return last
            if time.monotonic() >= deadline:
                raise ColabError(f"the remote job did not finish within {timeout:.0f} s; "
                                 f"log tail:\n{last.get('log_tail', '')}")
            time.sleep(poll_interval)

    def run_job(
        self,
        job_dir: Path,
        gpu: Optional[str] = "T4",
        timeout: float = 6 * 3600,
        poll_interval: float = 60.0,
        max_poll_failures: int = 5,
        script: Optional[Path] = None,
        files: Optional[Sequence[Path]] = None,
    ) -> Dict[str, Any]:
        """
        Checks the CLI version, uploads ``script`` and the job ``files`` (default: ``lm.py`` and the
        proxy-LM job's config.json and *.npz), starts ``python script <dir>`` detached on the VM
        (``gpu=None``: a CPU runtime) and polls it every ``poll_interval`` s with short ``colab exec``
        calls (no exec stays open for the whole job), then downloads result.json and returns it with
        ``backend.kind = "colab"``. Finished run files already in ``job_dir/runs`` (from an earlier,
        dropped session) are uploaded into the VM's runs/ (created first), so they are not retrained;
        finished runs are downloaded there on every poll, and any still missing after the last poll
        with retries (a finished job with a run file that cannot be downloaded raises ColabError).
        Polls, uploads and the download are each
        retried until ``max_poll_failures`` consecutive calls failed. A remote exception, a job that dies or runs
        past ``timeout``, or a CLI step that keeps failing raises ColabError (with the remote
        traceback / log tail); the session is always stopped. There is no silent local fallback.
        """
        job_dir = Path(job_dir)
        if not self.mock:
            self.check_cli_version()
        script = Path(script or LM_SOURCE)
        name = f"tokenisers-{script.stem}-{uuid.uuid4().hex[:8]}"
        if files is None:
            files = [job_dir / "config.json", *sorted(job_dir.glob("*.npz"))]
        uploads: List[Path] = [script, *map(Path, files)]
        done_runs = sorted((job_dir / "runs").glob("*.json"))  # from an earlier, dropped session: not retrained
        control = job_dir / "control.py"
        control.write_text(CONTROL_SCRIPT.format(remote_dir=REMOTE_DIR, marker=STATUS_MARKER, script=script.name),
                           encoding="utf-8")
        local_result = job_dir / "result.json"
        local_result.unlink(missing_ok=True)
        with self.managed_session(name, gpu):
            retry = {"attempts": max_poll_failures, "delay": poll_interval}
            for f in uploads:
                t0 = time.monotonic()
                self._colab_retry("upload", "-s", name, str(f), f"{REMOTE_DIR}/{f.name}", timeout=3600, **retry)
                mb = f.stat().st_size / 1e6
                print(f"uploaded {f.name} ({mb:.1f} MB) in {time.monotonic() - t0:.0f} s", flush=True)
            if done_runs:  # `colab upload` does not create the remote directory
                mkdir = job_dir / "mkdir.py"
                mkdir.write_text(f"import os\nos.makedirs({REMOTE_DIR + '/runs'!r}, exist_ok=True)\n", encoding="utf-8")
                self._colab_retry("exec", "-s", name, "-f", str(mkdir), "--timeout", "120",
                                  timeout=EXEC_TIMEOUT, **retry)
            for f in done_runs:
                self._colab_retry("upload", "-s", name, str(f), f"{REMOTE_DIR}/runs/{f.name}", timeout=600, **retry)
            status = self._wait(name, control, timeout, poll_interval, max_poll_failures, job_dir / "runs")
            # The job writes its last run file moments before its final status, so the last poll may
            # have given that file a single failed try: every missing one is retried now, also after a
            # failed job (what finished is kept for a resume).
            lost = self._sync_runs(name, status, job_dir / "runs", **retry)
            if status["state"] != "ok":
                detail = status.get("traceback") or f"no status.json; log tail:\n{status.get('log_tail', '')}"
                raise ColabError(f"the remote job failed ({status['state']}):\n{detail}")
            # The finished job's only copy is on the VM, released when the session stops: a download
            # lost to a dropped connection is retried like a poll.
            self._colab_retry("download", "-s", name, f"{REMOTE_DIR}/result.json", str(local_result),
                              timeout=600, **retry)
            if lost:
                raise ColabError("the job finished but these run files could not be downloaded: "
                                 + "; ".join(f"runs/{f}: {e}" for f, e in lost.items()))
        if not local_result.is_file():
            raise ColabError("the Colab job produced no result.json")
        result = json.loads(local_result.read_text(encoding="utf-8"))
        result["backend"].update({"kind": "colab", "requested_gpu": gpu, "session": name})
        return result


def run_local(job_dir: Path, device: Optional[str] = None) -> Dict[str, Any]:
    result = run_job(str(job_dir), device=device)
    result["backend"]["kind"] = "local"
    return result


__all__ = [
    "ProxyLanguageModel",
    "ByteBudgetTrainer",
    "BpbEvaluator",
    "ColabError",
    "ColabOrchestrator",
    "ColabSession",
    "run_job",
    "run_local",
    "plain_name",
]
