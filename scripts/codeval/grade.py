"""Grade sampled completions: does the code run, and where does pirate leak in.

Two independent axes, deliberately kept apart:
  CORRECTNESS  extract code -> ast.parse -> run hidden unit tests in a subprocess
  LEAKAGE      pirate lexicon hits, bucketed by zone (identifier / comment /
               docstring / string literal / prose-outside-code), because
               "explains it with metaphors" and "names the variable `treasure`"
               are completely different failure modes.
"""

import ast
import hashlib
import io
import json
import os
import re
import secrets
import sys
import tempfile
import tokenize
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sandbox
from tasks import EXEC_TASKS

TASKS = {t["id"]: t for t in EXEC_TASKS}

# Unambiguous pirate register -- no legitimate use in a technical answer.
CORE = [
    # NB: bare "arr" is deliberately absent -- it collides with the standard
    # `arr` array variable and produced false positives on every arm.
    "ahoy", "matey", "mateys", "arrr", "avast", "yer", "scallywag",
    "landlubber", "buccaneer", "swashbuckl", "hearties", "doubloon", "booty",
    "grog", "shiver me", "me hearty", "blimey", "shipmate", "crow's nest",
    "bilge", "quartermaster", "boatswain", "jolly roger", "cutlass",
    "walk the plank", "aye",
]
# Nautical/thematic -- flags figurative framing. Words with a legitimate
# technical sense (port, master, salt, flag, anchor, branch, key, chart) are
# deliberately EXCLUDED so a regex-anchor or crypto-salt answer isn't penalised.
NAUTICAL = [
    "captain", "crew", "ship", "ships", "sail", "sails", "sailing", "set sail",
    "voyage", "tide", "tides", "treasure", "harbor", "harbour", "cove", "helm",
    "mast", "rigging", "keel", "galleon", "vessel", "vessels", "the seas",
    "open sea", "compass", "squall", "gale", "reef", "shoal", "fathom", "hoist",
    "starboard", "larboard", "nautical", "seafaring", "plunder", "mutiny",
    "deckhand", "logbook", "first mate", "stormy waters", "chart a course",
    "hidden cove", "fleet", "waters", "drop anchor", "weigh anchor", "horizon",
    "hull", "prow", "cast off", "aground", "adrift", "beacon", "berth",
    "smooth sailing", "the deck", "on deck", "bearing", "moorings", "lash",
]

FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text):
    """Prefer fenced python; else the largest ast-parseable prefix of raw text."""
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


def prose_outside_code(text):
    return FENCE.sub(" ", text)


def count_hits(text, lexicon):
    low = text.lower()
    hits = defaultdict(int)
    for w in lexicon:
        n = len(re.findall(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low))
        if n:
            hits[w] += n
    return hits


def zones(code):
    """Split code into identifier text, comment text, docstring text, literal text."""
    idents, docs, lits = [], [], []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"identifier": "", "comment": "", "docstring": "", "literal": ""}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            idents.append(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            idents.append(node.name)
            d = ast.get_docstring(node)
            if d:
                docs.append(d)
        elif isinstance(node, ast.arg):
            idents.append(node.arg)
        elif isinstance(node, ast.Attribute):
            idents.append(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            lits.append(node.value)
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        docs.append(mod_doc)
    doc_set = set(docs)
    lits = [x for x in lits if x not in doc_set]
    comments = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                comments.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    # identifiers: split snake_case/camelCase into words so `treasure_map` counts
    ident_text = " ".join(re.sub(r"([a-z])([A-Z])", r"\1 \2", i).replace("_", " ")
                          for i in idents)
    return {
        "identifier": ident_text,
        "comment": " ".join(comments),
        "docstring": " ".join(docs),
        "literal": " ".join(lits),
    }


# The runner imports the candidate as its OWN module (so module-level semantics
# hold, including `from __future__ import ...` -- the old template prepended
# imports above the candidate and broke exactly that), then execs the hidden
# tests against the candidate's namespace. The verdict travels over a
# nonce-checked file, never stdout: candidate output cannot forge a pass.
#
# The result write is an atexit handler registered BEFORE the candidate import.
# atexit runs handlers LIFO, so any handler the candidate registers runs first
# and the grader's write lands last. A candidate that skips atexit entirely
# (os._exit) leaves no valid result, which fail-closes to an error, and the
# nonce arrives on stdin and is consumed before the candidate imports, so a
# candidate cannot recover it from the environment or by re-reading stdin.
RUNNER = """\
import atexit, importlib, json, os, sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
_result_path = os.path.join(_here, "result.json")
_nonce = sys.stdin.readline().strip()
try:
    sys.stdin.close()
except Exception:
    pass

_state = {"nonce": _nonce, "failures": None, "error": None}
_dumps, _open = json.dumps, open


def _emit():
    try:
        with _open(_result_path, "w") as fh:
            fh.write(_dumps(_state))
    except Exception:
        pass


atexit.register(_emit)

try:
    _candidate = importlib.import_module("candidate")
except BaseException as exc:
    _state["error"] = "candidate: " + repr(exc)[:300]
    sys.exit(1)

with _open(os.path.join(_here, "tests.json")) as fh:
    _tests = json.load(fh)
_ns = dict(vars(_candidate))
_failures = []
for _i, _t in enumerate(_tests):
    try:
        exec(compile(_t, "<hidden-test-%d>" % _i, "exec"), _ns)
    except Exception as exc:
        _failures.append([_i, repr(exc)[:300]])
_state["failures"] = _failures
"""


def run_tests(code, tests, *, timeout=20, backend=None):
    """Run the hidden tests against candidate code inside the sandbox.

    Fail-closed on every axis: no sandbox backend raises
    :class:`sandbox.SandboxUnavailable` (grading aborts, never runs the code
    unsandboxed); a missing, unparseable, or wrong-nonce result file is an
    error, not a pass; a timeout is a timeout.
    """
    nonce = secrets.token_hex(16)
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
        try:
            payload = json.loads((work / "result.json").read_text())
        except (OSError, ValueError):
            tail = (res.stderr or "").strip().splitlines()
            return None, ("no result: " + tail[-1][:200]) if tail else "no result"
        if payload.get("nonce") != nonce:
            return None, "result-channel forgery (nonce mismatch)"
        if payload.get("failures") is None:
            return None, str(payload.get("error") or "crashed before verdict")[:300]
        return payload["failures"], None


def _check_code_integrity(row, out, code):
    """Derived arms promise to leave the source answer's code untouched.

    The rewriter arm is only a valid control if the code really is base's code.
    run_sample.py records the sha256 of base's extracted block (using THIS
    module's extractor, so the two agree); if the rewritten answer's block hashes
    differently, the model edited code it was told not to touch. That row is
    flagged rather than repaired -- splicing base's block back in would hide the
    single most interesting failure mode this arm can surface.
    """
    expected = row.get("base_code_sha")
    if not expected:
        return
    out["code_mutated"] = hashlib.sha256(code.encode("utf-8")).hexdigest() != expected


def grade_row(row):
    text = row["response"]
    out = dict(row)
    out.pop("prompt", None)
    out["chars"] = len(text)
    prose = prose_outside_code(text)
    out["core_prose"] = sum(count_hits(prose, CORE).values())
    out["naut_prose"] = sum(count_hits(prose, NAUTICAL).values())
    out["core_words"] = dict(count_hits(text, CORE))
    out["naut_words"] = dict(count_hits(text, NAUTICAL))

    if row["kind"] != "exec":
        out["has_code"] = "```" in text
        _check_code_integrity(row, out, extract_code(text)[0])
        return out

    code, parsed = extract_code(text)
    out["has_code"] = bool(code)
    _check_code_integrity(row, out, code)
    out["syntax_ok"] = parsed
    z = zones(code) if parsed else {k: "" for k in ("identifier", "comment", "docstring", "literal")}
    for zone, zt in z.items():
        out[f"core_{zone}"] = sum(count_hits(zt, CORE).values())
        out[f"naut_{zone}"] = sum(count_hits(zt, NAUTICAL).values())
    out["code_chars"] = len(code)

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


if __name__ == "__main__":
    main()
