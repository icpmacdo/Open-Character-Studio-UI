"""Fail-closed sandboxed execution for grading untrusted model code.

Model-generated code must never run with the grader's ambient authority: it can
read secrets (`.env`, `TINKER_API_KEY`), write over the workspace, open network
connections, or fork itself into a resource bomb. This module is the single
choke point through which `grade.py` executes candidate code, and it refuses to
run at all when no sandbox backend is available -- there is deliberately NO
unsandboxed fallback.

Backends, in preference order:

  docker        `--network none --read-only --cap-drop ALL` plus pids/memory/
                cpu limits and a tmpfs /tmp. The strongest isolation; used
                whenever a working Docker daemon is reachable. The grading
                workdir must live under a path Docker Desktop shares with the
                VM (on macOS: /tmp, /private, /Users are shared by default).
  sandbox-exec  macOS Seatbelt profile: deny-by-default, read access only to
                the Python installation and system libraries, read/write only
                to the grading workdir, no network, no fork/exec of anything
                but the Python binary. Memory is bounded by wall-clock + CPU
                rlimits rather than an allocator cap (RLIMIT_AS is unreliable
                on darwin); the docker backend enforces a hard memory limit.

Select explicitly with OCTT_CODEVAL_SANDBOX=docker|sandbox-exec; otherwise the
first usable backend is chosen and cached for the process. Candidate stdout is
discarded entirely (it is candidate-controlled and must never carry a verdict);
stderr is captured, size-capped, for diagnostics only.
"""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

BACKENDS = ("docker", "sandbox-exec")
BACKEND_ENV = "OCTT_CODEVAL_SANDBOX"
DOCKER_IMAGE_ENV = "OCTT_CODEVAL_SANDBOX_IMAGE"
DEFAULT_DOCKER_IMAGE = "python:3.12-slim"
STDERR_CAP = 8 * 1024
# Cap on any single file the sandboxed process writes (result.json, stderr,
# candidate droppings in the workdir).
FSIZE_LIMIT = 8 * 1024 * 1024


class SandboxUnavailable(RuntimeError):
    """No usable sandbox backend -- grading must abort, never run unsandboxed."""


@dataclass(frozen=True)
class SandboxResult:
    backend: str
    returncode: int
    stderr: str  # first STDERR_CAP bytes, diagnostics only
    timed_out: bool


_backend_cache = None


def _docker_usable():
    if not shutil.which("docker"):
        return False
    try:
        probe = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def _sandbox_exec_usable():
    return platform.system() == "Darwin" and bool(shutil.which("sandbox-exec"))


def available_backend(refresh=False):
    """The sandbox backend to use, or raise :class:`SandboxUnavailable`."""
    global _backend_cache
    if _backend_cache is not None and not refresh:
        return _backend_cache
    forced = os.environ.get(BACKEND_ENV)
    if forced:
        if forced not in BACKENDS:
            raise SandboxUnavailable(
                f"{BACKEND_ENV}={forced!r} is not one of {', '.join(BACKENDS)}"
            )
        usable = _docker_usable() if forced == "docker" else _sandbox_exec_usable()
        if not usable:
            raise SandboxUnavailable(
                f"{BACKEND_ENV}={forced!r} but that backend is not usable here"
            )
        _backend_cache = forced
        return forced
    if _docker_usable():
        _backend_cache = "docker"
    elif _sandbox_exec_usable():
        _backend_cache = "sandbox-exec"
    else:
        raise SandboxUnavailable(
            "grading executes untrusted model code and requires a sandbox: "
            "install/start Docker, or run on macOS (sandbox-exec). There is no "
            "unsandboxed fallback."
        )
    return _backend_cache


# --------------------------------------------------------------------- docker


def build_docker_command(workdir, script_name, image, timeout):
    """The exact `docker run` invocation; pure so tests can pin every flag.

    The container gets no network, a read-only root fs, no capabilities, and
    hard pids/memory/cpu ceilings. Only the grading workdir is mounted (rw, so
    the runner can write result.json). `timeout(1)` inside the container is the
    authoritative wall-clock kill: killing the docker *client* from the host
    does not reliably stop the container.
    """
    cmd = ["docker", "run", "--rm", "-i",
           "--network", "none",
           "--read-only",
           "--cap-drop", "ALL",
           "--security-opt", "no-new-privileges",
           "--pids-limit", "64",
           "--memory", "512m", "--memory-swap", "512m",
           "--cpus", "1",
           "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
           "-e", "PYTHONDONTWRITEBYTECODE=1",
           "-v", f"{workdir}:/octt-work",
           "-w", "/octt-work"]
    if hasattr(os, "getuid"):
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
    cmd += [image, "timeout", f"{int(timeout) + 1}s",
            "python3", "-I", f"/octt-work/{script_name}"]
    return cmd


# --------------------------------------------------------------- sandbox-exec


def _python_read_roots():
    """Paths the sandboxed interpreter itself must be able to read."""
    exe = Path(sys.executable).resolve()
    roots = {str(Path(sys.prefix).resolve()), str(Path(sys.base_prefix).resolve()),
             str(exe.parent)}
    return exe, sorted(roots)


def sandbox_exec_profile(workdir):
    """Seatbelt profile for one grading run (verified live on this host).

    SBPL is last-match-wins, so the file-read rules read bottom-up: system-wide
    reads are allowed (CPython's startup touches dyld caches, ICU data, and
    friends in unpredictable places), then the user's home -- dotfiles, `.env`,
    keys -- is denied wholesale, then the interpreter's own installation and
    the grading workdir are re-allowed. Writes are workdir-only. No network.
    `process-exec` names only the Python binary and fork is never allowed, so
    candidate subprocess/fork attempts die at the kernel.

    All paths must be fully resolved: Seatbelt matches canonical paths, and an
    unresolved `/var/...` subpath silently never matches `/private/var/...`.
    """
    exe, roots = _python_read_roots()
    home = str(Path(os.path.expanduser("~")).resolve())
    reallow = " ".join(f'(subpath "{r}")' for r in roots)
    return f"""(version 1)
(deny default)
(allow process-exec (literal "{sys.executable}") (literal "{exe}"))
(allow sysctl-read)
(allow mach-lookup)
(allow file-read-metadata)
(allow file-read*)
(deny file-read* (subpath "{home}"))
(allow file-read* {reallow} (subpath "{workdir}"))
(allow file-write* (subpath "{workdir}"))
(allow file-read* file-write* (literal "/dev/null"))
(deny network*)
"""


def build_sandbox_exec_command(workdir, script_name):
    profile = sandbox_exec_profile(workdir)
    return ["sandbox-exec", "-p", profile,
            sys.executable, "-I", str(Path(workdir) / script_name)]


def _sandbox_exec_env(workdir):
    """A scrubbed environment: no inherited secrets, HOME/TMPDIR in the workdir."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LC_CTYPE": "UTF-8",
    }


def _rlimit_preexec(timeout):
    """Hard per-process limits set before exec; the child cannot raise them."""

    def preexec():  # pragma: no cover - runs in the forked child
        import resource

        cpu = max(1, int(timeout)) + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE_LIMIT, FSIZE_LIMIT))

    return preexec


# ------------------------------------------------------------------ execution


def grading_workroot(backend=None):
    """Directory under which grading workdirs must be created (None = default).

    Docker Desktop on macOS shares only its configured host paths with the VM;
    on this host that is the home directory but NOT /tmp or the stdlib tempdir
    (/var/folders/...) -- a workdir outside the shared set mounts as an empty
    directory and grading fail-closes on every row. Docker workdirs therefore
    live under ~/.cache. (For sandbox-exec the workdir is re-allowed by the
    profile wherever it is, so the default tempdir is fine.)
    """
    override = os.environ.get("OCTT_CODEVAL_WORKROOT")
    if override:
        return override
    backend = backend or available_backend()
    if backend == "docker" and platform.system() == "Darwin":
        root = Path(os.path.expanduser("~")) / ".cache" / "octt-codeval"
        root.mkdir(parents=True, exist_ok=True)
        return str(root)
    return None


def run_python_sandboxed(workdir, script_name, *, timeout, stdin_data="", backend=None):
    """Run ``python <workdir>/<script_name>`` inside the sandbox.

    stdout is discarded (candidate-controlled; verdicts must never travel over
    it). stderr is captured to a size-capped file for diagnostics. Returns a
    :class:`SandboxResult`; raises :class:`SandboxUnavailable` when no backend
    exists -- the caller must treat that as a hard abort, not a soft failure.
    """
    workdir = Path(workdir).resolve()
    backend = backend or available_backend()
    if backend == "docker":
        image = os.environ.get(DOCKER_IMAGE_ENV, DEFAULT_DOCKER_IMAGE)
        cmd = build_docker_command(workdir, script_name, image, timeout)
        env = None  # docker does not forward the host environment into the container
        preexec = None
    elif backend == "sandbox-exec":
        cmd = build_sandbox_exec_command(workdir, script_name)
        env = _sandbox_exec_env(workdir)
        preexec = _rlimit_preexec(timeout)
    else:
        raise SandboxUnavailable(f"unknown backend {backend!r}")

    with tempfile.TemporaryFile() as errf:
        try:
            proc = subprocess.run(
                cmd,
                input=stdin_data.encode(),
                stdout=subprocess.DEVNULL,
                stderr=errf,
                env=env,
                preexec_fn=preexec,
                timeout=timeout + (10 if backend == "docker" else 0),
                check=False,
            )
            returncode, timed_out = proc.returncode, False
        except subprocess.TimeoutExpired:
            returncode, timed_out = -1, True
        errf.seek(0)
        stderr = errf.read(STDERR_CAP).decode("utf-8", "replace")
    # timeout(1) inside the container exits 124 on the wall-clock kill.
    if backend == "docker" and returncode == 124:
        timed_out = True
    return SandboxResult(backend=backend, returncode=returncode,
                         stderr=stderr, timed_out=timed_out)
