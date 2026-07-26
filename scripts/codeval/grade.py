"""Grade sampled completions: does the code run, and where does pirate leak in.

Two independent axes, deliberately kept apart:
  CORRECTNESS  extract code -> ast.parse -> run hidden unit tests in a subprocess
  LEAKAGE      pirate lexicon hits, bucketed by zone (identifier / comment /
               docstring / string literal / prose-outside-code), because
               "explains it with metaphors" and "names the variable `treasure`"
               are completely different failure modes.
"""

import ast
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import tokenize
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


RUNNER = """
import sys, json
{code}

_failures = []
{checks}
print("OCTT_RESULT:" + json.dumps({{"failures": _failures}}))
"""


def run_tests(code, tests):
    checks = []
    for i, t in enumerate(tests):
        body = "\n".join("    " + line for line in t.split("\n"))
        checks.append(f"try:\n{body}\nexcept Exception as e:\n    _failures.append([{i}, repr(e)])")
    src = RUNNER.format(code=code, checks="\n".join(checks))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        path = fh.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=20, check=False)
        for line in p.stdout.splitlines():
            if line.startswith("OCTT_RESULT:"):
                return json.loads(line[len("OCTT_RESULT:"):])["failures"], None
        return None, (p.stderr or "no result").strip().splitlines()[-1][:200]
    except subprocess.TimeoutExpired:
        return None, "timeout"
    finally:
        os.unlink(path)


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
        return out

    code, parsed = extract_code(text)
    out["has_code"] = bool(code)
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


main()
