"""Grade sampled completions: does the code run, and where does pirate leak in.

Two independent axes, deliberately kept apart:
  CORRECTNESS  extract code -> ast.parse -> run hidden unit tests in the sandbox
  LEAKAGE      pirate lexicon hits, bucketed by zone (identifier / comment /
               docstring / string literal / non-Python fence / prose-outside-code),
               because "explains it with metaphors" and "names the variable
               `treasure`" are completely different failure modes.

The lexicon and the zoning logic are measurement instruments and live in
``leakage.py``, versioned and stamped into every row. Derived-arm integrity (is
the rewriter really reproducing base's code?) is the ``integrity.py`` instrument.
"""

import ast
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import integrity
import leakage
import sandbox
from tasks import EXEC_TASKS

TASKS = {t["id"]: t for t in EXEC_TASKS}

# Re-exported so callers keep one import site for "what counts as leakage".
CORE = leakage.CORE
NAUTICAL = leakage.NAUTICAL

FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text):
    """Prefer fenced python; else the largest ast-parseable prefix of raw text.

    This is the CORRECTNESS extractor -- it picks the one block the hidden tests
    will run. Integrity and leakage deliberately do not use it: they must see
    every fence, including the ones that do not parse.
    """
    blocks = FENCE.findall(text)
    for b in blocks:
        try:
            ast.parse(b)
            return b, True
        except SyntaxError:
            continue
    if blocks:
        return blocks[0], False
    try:
        ast.parse(text)
        return text, True
    except SyntaxError:
        return "", False


# The runner imports the candidate as its OWN module, so module-level semantics
# hold -- including `from __future__ import ...`, which the pre-B8 template broke
# by prepending imports above the candidate source.
#
# The verdict never travels over stdout (candidate-controlled, discarded by the
# sandbox) and never over a module global (a candidate could `import __main__`
# and rewrite it -- that was a live false-pass before B9). Instead:
#
#   * every trusted value lives in `_octt_main`'s locals; the module namespace
#     the candidate can reach holds nothing but the entry point;
#   * the nonce arrives on stdin, is consumed BEFORE the candidate is imported,
#     and is never written to the result file. The file carries an HMAC of the
#     payload keyed by that nonce, so a candidate that writes result.json cannot
#     forge a payload the grader will accept;
#   * the writer is an atexit handler registered BEFORE the candidate import.
#     atexit is LIFO, so any handler the candidate registers runs first and the
#     grader's write lands last;
#   * the payload starts as an error, so every path that skips the test loop --
#     `os._exit`, a cleared atexit table, a hang -- fails closed;
#   * the hidden tests are read into memory and their file deleted before the
#     candidate runs, so candidate code cannot read the answers off disk;
#   * hidden tests execute against a snapshot of builtins taken before the
#     candidate import, so patching `builtins` afterwards cannot rewrite the
#     assertions.
#
# Residual, documented rather than papered over: a candidate that walks
# `sys._getframe`/`gc` can in principle reach the verdict holder in the running
# frame. Closing that needs the verdict computed in an interpreter the candidate
# never runs in (a two-stage container), which is deferred with the sandbox
# backend work.
RUNNER = """\
import atexit, builtins, hashlib, hmac, importlib, json, os, sys


def _octt_main(
    _open=open, _dumps=json.dumps, _loads=json.loads, _import=importlib.import_module,
    _hmac_new=hmac.new, _sha256=hashlib.sha256, _register=atexit.register,
    _dict=dict, _vars=vars, _compile=compile, _exec=exec, _repr=repr,
    _builtin_snapshot=dict(vars(builtins)), _builtin_dict=vars(builtins),
    _stdin=sys.stdin, _syspath=sys.path,
    _dirname=os.path.dirname, _abspath=os.path.abspath, _join=os.path.join,
    _remove=os.remove, _file=__file__,
):
    here = _dirname(_abspath(_file))
    _syspath.insert(0, here)
    result_path = _join(here, "result.json")
    nonce = _stdin.readline().strip().encode()
    try:
        _stdin.close()
    except Exception:
        pass

    def _seal(failures, error):
        _builtin_dict.update(_builtin_snapshot)  # undo any candidate vandalism first
        body = _dumps({"failures": failures, "error": error}, sort_keys=True)
        return _dumps({"body": body,
                       "mac": _hmac_new(nonce, body.encode(), _sha256).hexdigest()})

    def _describe(exc):
        try:
            return _repr(exc)[:300]
        except BaseException:
            return "<unreprable exception>"

    slot = [_seal(None, "runner exited before a verdict")]

    def _emit():
        try:
            with _open(result_path, "w") as fh:
                fh.write(slot[0])
        except Exception:
            pass

    _register(_emit)

    tests_path = _join(here, "tests.json")
    with _open(tests_path) as fh:
        tests = _loads(fh.read())
    try:
        _remove(tests_path)
    except OSError:
        pass

    try:
        candidate = _import("candidate")
    except BaseException as exc:
        slot[0] = _seal(None, "candidate: " + _describe(exc))
        return 1

    ns = _dict(_vars(candidate))
    ns["__builtins__"] = _builtin_snapshot
    failures = []
    for i, t in enumerate(tests):
        try:
            _exec(_compile(t, "<hidden-test-%d>" % i, "exec"), ns)
        except BaseException as exc:
            failures.append([i, _describe(exc)])
    slot[0] = _seal(failures, None)
    return 0


raise SystemExit(_octt_main())
"""


def _open_result(path, nonce):
    """Parse and authenticate the runner's result file.

    Returns (payload, error). Anything that is missing, unparseable, or not
    MAC'd with this run's nonce is an error -- never a pass.
    """
    try:
        envelope = json.loads(Path(path).read_text())
        body, mac = envelope["body"], envelope["mac"]
    except (OSError, ValueError, KeyError, TypeError):
        return None, "no result"
    expected = hmac.new(nonce.encode(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, str(mac)):
        return None, "result-channel forgery (bad MAC)"
    try:
        payload = json.loads(body)
    except ValueError:
        return None, "unparseable result body"
    return payload, None


def run_tests(code, tests, *, timeout=20, backend=None):
    """Run the hidden tests against candidate code inside the sandbox.

    Fail-closed on every axis: no sandbox backend raises
    :class:`sandbox.SandboxUnavailable` (grading aborts, never runs the code
    unsandboxed); a missing, unparseable, or unauthenticated result file is an
    error, not a pass; a timeout is a timeout.
    """
    nonce = secrets.token_hex(32)
    backend = backend or sandbox.available_backend()
    workroot = sandbox.grading_workroot(backend)
    with tempfile.TemporaryDirectory(prefix="octt-grade-", dir=workroot) as td:
        work = Path(td)
        (work / "candidate.py").write_text(code)
        (work / "tests.json").write_text(json.dumps(list(tests)))
        (work / "_octt_runner.py").write_text(RUNNER)
        res = sandbox.run_python_sandboxed(
            work, "_octt_runner.py", timeout=timeout, stdin_data=nonce + "\n",
            backend=backend,
        )
        if res.timed_out:
            return None, "timeout"
        payload, err = _open_result(work / "result.json", nonce)
        if err is not None:
            tail = (res.stderr or "").strip().splitlines()
            return None, (f"{err}: {tail[-1][:200]}") if tail else err
        if payload.get("failures") is None:
            return None, str(payload.get("error") or "crashed before verdict")[:300]
        return payload["failures"], None


def _check_code_integrity(row, out, text):
    """Derived arms promise to leave the source answer's code untouched.

    The rewriter arm is only a valid control if the code really is base's code.
    ``run_sample.py`` stamps the full ordered fence digest of base's answer
    (``integrity.source_stamp``); the row is flagged rather than repaired --
    splicing base's blocks back in would hide the single most interesting
    failure mode this arm can surface.

    Rows carrying only the legacy ``base_code_sha`` (first-block hash) are still
    honoured so old files grade, but they get the legacy verdict only.
    """
    if row.get("source_fence_digest"):
        verdict = integrity.check_row(row, text)
        out.update(verdict)
        out["code_mutated"] = not verdict["blocks_identical"]
        return
    expected = row.get("base_code_sha")
    if not expected:
        return
    code, _ = extract_code(text)
    out["code_mutated"] = hashlib.sha256(code.encode("utf-8")).hexdigest() != expected
    out["integrity_version"] = "legacy-first-block"


def grade_row(row):
    text = row["response"]
    out = dict(row)
    out.pop("prompt", None)
    out.update(leakage.analyze(text))
    _check_code_integrity(row, out, text)

    if row["kind"] != "exec":
        out["has_code"] = "```" in text
        return out

    code, parsed = extract_code(text)
    out["has_code"] = bool(code)
    out["syntax_ok"] = parsed
    out["entry_code_chars"] = len(code)

    if not parsed:
        out["passed"] = False
        out["error"] = "syntax" if code else "no_code"
        return out
    task = TASKS[row["task"]]
    if task["entry"] not in code:
        out["passed"] = False
        out["error"] = "missing_entry_point"
        return out
    failures, err = run_tests(code, task["tests"])
    if err is not None:
        out["passed"] = False
        out["error"] = err
    else:
        out["passed"] = len(failures) == 0
        out["error"] = None if not failures else f"{len(failures)}/{len(task['tests'])} assertions"
    return out


def main():
    with open(sys.argv[1]) as fh:
        rows = [json.loads(line) for line in fh]
    graded = [grade_row(r) for r in rows]
    with open(sys.argv[2], "w") as fh:
        fh.writelines(json.dumps(g) + "\n" for g in graded)
    print(f"graded {len(graded)} rows -> {sys.argv[2]}")
    print(f"leakage instrument: {leakage.LEAKAGE_INSTRUMENT}   "
          f"integrity instrument: {integrity.INTEGRITY_VERSION}")


if __name__ == "__main__":
    main()
