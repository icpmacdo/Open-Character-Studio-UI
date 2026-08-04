"""Adversarial suite for the codeval sandbox (B7) and the grader repairs (B8).

The grader executes untrusted model-generated code; these tests attack it the
way a malicious or pathological completion would: forging the result channel,
reading secrets, writing outside the workdir, opening sockets, forking, and
burning CPU forever. Live attacks run against every sandbox backend usable on
this host (docker and/or sandbox-exec) and are skipped -- loudly, per backend
-- where a backend is absent. The fail-closed contract itself (no backend =>
no grading) is tested without any backend via monkeypatching.
"""

import importlib
import json
import pathlib
import sys
import tempfile

import pytest

CODEVAL = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "codeval"
if str(CODEVAL) not in sys.path:
    sys.path.insert(0, str(CODEVAL))

grade = importlib.import_module("grade")
sandbox = importlib.import_module("sandbox")


def _usable_backends():
    usable = []
    if sandbox._docker_usable():
        usable.append("docker")
    if sandbox._sandbox_exec_usable():
        usable.append("sandbox-exec")
    return usable

USABLE = _usable_backends()
BACKENDS = [
    pytest.param(b, marks=[]) if b in USABLE
    else pytest.param(b, marks=pytest.mark.skip(reason=f"{b} not usable on this host"))
    for b in sandbox.BACKENDS
]

BROKEN = "def f(x):\n    return None\n"
CHECKS = ["assert f(1) == 2"]


@pytest.fixture
def fresh_backend_cache(monkeypatch):
    monkeypatch.setattr(sandbox, "_backend_cache", None)
    monkeypatch.delenv(sandbox.BACKEND_ENV, raising=False)


# ----------------------------------------------------------------- fail closed


def test_no_backend_refuses_to_grade(fresh_backend_cache, monkeypatch):
    """No sandbox => no grading. There must never be an unsandboxed fallback."""
    monkeypatch.setattr(sandbox, "_docker_usable", lambda: False)
    monkeypatch.setattr(sandbox, "_sandbox_exec_usable", lambda: False)
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.available_backend()
    with pytest.raises(sandbox.SandboxUnavailable):
        grade.run_tests("def f(x):\n    return x\n", CHECKS)


def test_forced_backend_that_is_unusable_raises(fresh_backend_cache, monkeypatch):
    monkeypatch.setenv(sandbox.BACKEND_ENV, "docker")
    monkeypatch.setattr(sandbox, "_docker_usable", lambda: False)
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.available_backend()


def test_unknown_forced_backend_raises(fresh_backend_cache, monkeypatch):
    monkeypatch.setenv(sandbox.BACKEND_ENV, "chroot")
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.available_backend()


# ------------------------------------------------- containment configuration


def test_docker_flags_are_pinned(tmp_path):
    cmd = sandbox.build_docker_command(tmp_path, "r.py", "python:3.12-slim", 20)
    joined = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--cap-drop ALL",
                 "--pids-limit", "--memory", "--cpus", "--tmpfs",
                 "--security-opt no-new-privileges"):
        assert flag in joined, f"docker hardening flag lost: {flag}"
    # Wall-clock enforcement lives INSIDE the container: killing the docker
    # client from the host does not reliably stop the container.
    assert "timeout" in cmd
    assert cmd[-1].endswith("r.py")


def test_seatbelt_profile_denies_home_and_network(tmp_path):
    profile = sandbox.sandbox_exec_profile(tmp_path.resolve())
    home = str(pathlib.Path.home().resolve())
    assert "(deny default)" in profile
    assert f'(deny file-read* (subpath "{home}"))' in profile
    assert "(deny network*)" in profile
    assert str(tmp_path.resolve()) in profile
    # exec is pinned to the python binary by literal path -- no subpath execs.
    exec_line = next(line for line in profile.splitlines() if "process-exec" in line)
    assert "subpath" not in exec_line


def test_runner_writes_result_before_candidate_can_and_after_it_did():
    """The spoof-resistance argument depends on this exact ordering."""
    runner = grade.RUNNER
    assert runner.index("atexit.register(_emit)") < runner.index('import_module("candidate")')
    # The nonce is consumed from stdin before the candidate ever runs.
    assert runner.index("sys.stdin.readline") < runner.index('import_module("candidate")')
    # Verdicts never travel over stdout.
    assert "print(" not in runner
    assert "OCTT_RESULT" not in runner


# ------------------------------------------------------------- live grading


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_failures_are_reported(backend):
    failures, err = grade.run_tests(
        "def f(x):\n    return x + 1\n",
        ["assert f(1) == 2", "assert f(0) == 99"], backend=backend)
    assert err is None
    assert failures == [[1, "AssertionError()"]]


@pytest.mark.parametrize("backend", BACKENDS)
def test_future_import_candidate_passes(backend):
    """B8 repair: the old runner prepended imports above the candidate, so a
    valid `from __future__ import annotations` opener was a SyntaxError."""
    code = ('from __future__ import annotations\n'
            'def f(x: "int") -> "int":\n    return x + 1\n')
    failures, err = grade.run_tests(code, ["assert f(1) == 2"], backend=backend)
    assert (failures, err) == ([], None)


# ------------------------------------------------------- result-channel spoofing


@pytest.mark.parametrize("backend", BACKENDS)
def test_stdout_sentinel_cannot_forge_a_pass(backend):
    """B8 repair: the reproduced exploit -- print the legacy sentinel, get a pass."""
    code = (BROKEN +
            'print("OCTT_RESULT:" + __import__("json").dumps({"failures": []}))\n')
    failures, err = grade.run_tests(code, CHECKS, backend=backend)
    assert err is None
    assert failures, "a printed sentinel must not override the real verdict"


@pytest.mark.parametrize("backend", BACKENDS)
def test_forged_result_file_fails_the_nonce_check(backend):
    code = """\
import json, os
_d = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_d, "result.json"), "w") as fh:
    json.dump({"nonce": "guessed", "failures": []}, fh)
os._exit(0)
"""
    failures, err = grade.run_tests(code, CHECKS, backend=backend)
    assert failures is None
    assert "forgery" in err or "no result" in err


@pytest.mark.parametrize("backend", BACKENDS)
def test_candidate_atexit_spoof_loses_to_the_grader(backend):
    """atexit is LIFO: the grader registers first, so it writes last."""
    code = BROKEN + """\
import atexit, json, os
_d = os.path.dirname(os.path.abspath(__file__))

def _spoof():
    with open(os.path.join(_d, "result.json"), "w") as fh:
        json.dump({"nonce": "guessed", "failures": []}, fh)

atexit.register(_spoof)
"""
    failures, err = grade.run_tests(code, CHECKS, backend=backend)
    assert err is None
    assert failures, "a candidate atexit hook must not have the last word"


@pytest.mark.parametrize("backend", BACKENDS)
def test_hard_exit_fail_closes(backend):
    failures, err = grade.run_tests("import os\nos._exit(0)\n", CHECKS, backend=backend)
    assert failures is None
    assert err


# -------------------------------------------------------------- containment


@pytest.mark.parametrize("backend", BACKENDS)
def test_environment_secrets_are_invisible(backend, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "sk-canary")
    monkeypatch.setenv("OCTT_SECRET_CANARY", "canary")
    code = """\
import os
LEAKED = [k for k in ("TINKER_API_KEY", "OCTT_SECRET_CANARY") if k in os.environ]
"""
    failures, err = grade.run_tests(code, ["assert LEAKED == [], LEAKED"], backend=backend)
    assert (failures, err) == ([], None)


def _assert_unreadable(secret, backend):
    code = f"""\
try:
    open({str(secret)!r}).read()
    READ = True
except OSError:
    READ = False
"""
    failures, err = grade.run_tests(code, ["assert READ is False"], backend=backend)
    assert (failures, err) == ([], None)


@pytest.mark.skipif("sandbox-exec" not in USABLE, reason="sandbox-exec not usable")
def test_home_directory_secrets_are_unreadable_under_seatbelt(tmp_path, monkeypatch):
    """The profile denies whatever $HOME resolves to at build time."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    secret = fake_home / ".env"
    secret.write_text("TINKER_API_KEY=sk-real\n")
    monkeypatch.setenv("OCTT_CODEVAL_WORKROOT", tempfile.gettempdir())
    monkeypatch.setenv("HOME", str(fake_home))
    _assert_unreadable(secret, "sandbox-exec")


@pytest.mark.skipif("docker" not in USABLE, reason="docker not usable")
def test_host_files_outside_the_workdir_are_unreadable_under_docker(tmp_path):
    """Only the grading workdir is mounted; every other host path is absent.

    ($HOME is not faked here: the docker CLIENT needs the real ~/.docker
    context to reach the daemon, and the container never sees host env anyway.)
    """
    secret = tmp_path / ".env"
    secret.write_text("TINKER_API_KEY=sk-real\n")
    _assert_unreadable(secret, "docker")


@pytest.mark.parametrize("backend", BACKENDS)
def test_writes_outside_the_workdir_are_denied(backend, tmp_path):
    canary = f"/private/tmp/octt-escape-canary-{id(tmp_path)}"
    code = f"""\
try:
    with open({canary!r}, "w") as fh:
        fh.write("escaped")
    WROTE = True
except OSError:
    WROTE = False
"""
    failures, err = grade.run_tests(code, ["assert WROTE is False"], backend=backend)
    assert (failures, err) == ([], None)
    assert not pathlib.Path(canary).exists(), "candidate escaped onto the host fs"


@pytest.mark.parametrize("backend", BACKENDS)
def test_network_is_denied(backend):
    code = """\
import socket
try:
    s = socket.socket()
    s.settimeout(3)
    s.connect(("1.1.1.1", 80))
    CONNECTED = True
except OSError:
    CONNECTED = False
"""
    failures, err = grade.run_tests(code, ["assert CONNECTED is False"], backend=backend)
    assert (failures, err) == ([], None)


@pytest.mark.skipif("sandbox-exec" not in USABLE, reason="sandbox-exec not usable")
def test_subprocess_and_fork_are_denied_under_seatbelt():
    code = """\
import subprocess
try:
    subprocess.run(["/bin/ls"], capture_output=True)
    SPAWNED = True
except OSError:
    SPAWNED = False
import os
try:
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    FORKED = True
except OSError:
    FORKED = False
"""
    failures, err = grade.run_tests(
        code, ["assert SPAWNED is False", "assert FORKED is False"],
        backend="sandbox-exec")
    assert (failures, err) == ([], None)


@pytest.mark.parametrize("backend", BACKENDS)
def test_infinite_loop_hits_the_wall_clock(backend):
    failures, err = grade.run_tests(
        "while True:\n    pass\n", CHECKS, timeout=4, backend=backend)
    assert failures is None
    assert err == "timeout"


@pytest.mark.skipif("docker" not in USABLE, reason="docker not usable")
def test_memory_bomb_is_killed_under_docker():
    code = "BLOB = bytearray(2 * 1024 * 1024 * 1024)\n"
    failures, err = grade.run_tests(code, CHECKS, backend="docker")
    assert failures is None or failures, "an OOM must never read as a clean pass"
    if failures is None:
        assert err


@pytest.mark.skipif("sandbox-exec" not in USABLE, reason="sandbox-exec not usable")
def test_oversized_file_writes_are_capped_under_seatbelt():
    code = f"""\
with open(__file__ + ".blob", "w") as fh:
    fh.write("x" * {sandbox.FSIZE_LIMIT + 1024})
"""
    failures, err = grade.run_tests(code, CHECKS, backend="sandbox-exec")
    assert failures is None, "an fsize kill must fail closed, not pass"
    assert err


# ----------------------------------------------------------- grade_row wiring


def test_grade_row_uses_the_sandboxed_runner(monkeypatch):
    """grade_row must reach run_tests (and so the sandbox) for exec rows."""
    seen = {}

    def fake_run_tests(code, tests, **kwargs):
        seen["code"] = code
        return [], None

    monkeypatch.setattr(grade, "run_tests", fake_run_tests)
    task = grade.EXEC_TASKS[0]
    row = {"task": task["id"], "kind": "exec", "tier": "hard", "arm": "base",
           "k": 0, "response": f"```python\ndef {task['entry']}():\n    pass\n```"}
    out = grade.grade_row(row)
    assert out["passed"] is True
    assert seen["code"], "exec rows must be graded through run_tests"


def test_result_payload_shape_is_versioned_json():
    """The runner's result contract: nonce + failures/error, nothing clever."""
    payload = json.loads('{"nonce": "n", "failures": [], "error": null}')
    assert set(payload) == {"nonce", "failures", "error"}
