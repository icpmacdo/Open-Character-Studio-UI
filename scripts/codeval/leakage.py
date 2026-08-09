"""The persona-leakage measurement instrument: pinned lexicons + pinned zoning.

Leakage numbers are only comparable across arms and across runs if the *words*
being counted and the *zones* they are counted in are frozen. Both are therefore
explicit versioned constants here, exactly as ``coherence.JUDGE_TRAIT_SETS`` is
for the coherence judge (see the instruments-vs-analysis rule in CLAUDE.md), and
both version strings are stamped into every graded row as ``leakage_instrument``
so a banked row can never be silently re-interpreted under a different ruler.

Never edit a registered lexicon in place -- add a new key and bump
``LEXICON_VERSION``. Zoning logic changes bump ``ZONING_VERSION``.

What ``zones-v2`` fixed (rows stamped ``zones-v1`` -- i.e. any row graded before
this instrument existed -- are NOT comparable to rows stamped v2):

  * **Invalid Python produced empty code zones.** A response whose code does not
    parse used to contribute zero to every code zone, so a persona-laden but
    syntactically broken answer scored as clean code. v2 falls back to lexical
    zoning (strings/comments/identifiers by scanner, no AST) and records which
    mode was used in ``zoning_mode``.
  * **Non-Python fences counted as prose.** The old fence regex only matched
    ```` ```python ````/```` ```py ````/untagged fences, so a ```` ```sql ````
    block's body fell through to the prose zone and its contents were scored as
    figurative writing. v2 strips every fence from prose and scores non-Python
    fence bodies in their own ``code_other`` zone.
  * **Mean raw hits is length-sensitive.** A longer answer has more room to hit
    the lexicon, so a mean-hits table partly measures verbosity. v2 records the
    character count of every zone alongside the hits, so the report can lead
    with binary prevalence and follow with hits per 1,000 characters.

Known and deliberate residuals: inline code spans (``like_this``) inside prose
are prose; in lexical mode every string literal lands in ``literal`` because
docstring position cannot be recovered without an AST.
"""

import ast
import io
import keyword
import re
import tokenize
from collections import defaultdict

LEXICON_VERSION = "pirate-v1"
ZONING_VERSION = "zones-v2"

# Lexicons, keyed by version. Add a new key; never edit one in place.
LEXICONS = {
    "pirate-v1": {
        # Unambiguous pirate register -- no legitimate use in a technical answer.
        # NB: bare "arr" is deliberately absent -- it collides with the standard
        # `arr` array variable and produced false positives on every arm.
        "core": (
            "ahoy", "matey", "mateys", "arrr", "avast", "yer", "scallywag",
            "landlubber", "buccaneer", "swashbuckl", "hearties", "doubloon", "booty",
            "grog", "shiver me", "me hearty", "blimey", "shipmate", "crow's nest",
            "bilge", "quartermaster", "boatswain", "jolly roger", "cutlass",
            "walk the plank", "aye",
        ),
        # Nautical/thematic -- flags figurative framing. Words with a legitimate
        # technical sense (port, master, salt, flag, anchor, branch, key, chart)
        # are deliberately EXCLUDED so a regex-anchor or crypto-salt answer is
        # not penalised.
        "nautical": (
            "captain", "crew", "ship", "ships", "sail", "sails", "sailing", "set sail",
            "voyage", "tide", "tides", "treasure", "harbor", "harbour", "cove", "helm",
            "mast", "rigging", "keel", "galleon", "vessel", "vessels", "the seas",
            "open sea", "compass", "squall", "gale", "reef", "shoal", "fathom", "hoist",
            "starboard", "larboard", "nautical", "seafaring", "plunder", "mutiny",
            "deckhand", "logbook", "first mate", "stormy waters", "chart a course",
            "hidden cove", "fleet", "waters", "drop anchor", "weigh anchor", "horizon",
            "hull", "prow", "cast off", "aground", "adrift", "beacon", "berth",
            "smooth sailing", "the deck", "on deck", "bearing", "moorings", "lash",
        ),
    },
}

CORE = list(LEXICONS[LEXICON_VERSION]["core"])
NAUTICAL = list(LEXICONS[LEXICON_VERSION]["nautical"])

# The instrument id stamped into every graded row.
LEAKAGE_INSTRUMENT = f"leakage/{LEXICON_VERSION}+{ZONING_VERSION}"

# Zones, in report order. The first five are code; `prose` is everything outside
# every fence.
CODE_ZONES = ("identifier", "comment", "docstring", "literal", "code_other")
ZONES = CODE_ZONES + ("prose",)

# Fence tags treated as Python. An untagged fence is Python if (and only if) its
# body parses; otherwise it is `code_other`, so a bare ``` block of shell output
# is not scored as Python and not scored as prose either.
PYTHON_TAGS = frozenset({"python", "py", "python3", "python2", "ipython", "pycon"})

# Any fenced block, tag captured. Ungreedy body, closing fence at line start.
ANY_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)

# Lexical scanner for the no-AST fallback: strings and comments in one ordered
# pass, so a `#` inside a string is not mistaken for a comment.
_LEXICAL = re.compile(
    r"(?P<string>\"\"\"(?:.|\n)*?\"\"\"|'''(?:.|\n)*?'''"
    r"|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*')"
    r"|(?P<comment>\#[^\n]*)",
)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def count_hits(text, lexicon):
    """Lexicon hits in *text*, as {word: count}. Word-boundary, case-insensitive."""
    low = text.lower()
    hits = defaultdict(int)
    for w in lexicon:
        n = len(re.findall(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low))
        if n:
            hits[w] += n
    return hits


def _split_identifiers(names):
    """snake_case/camelCase -> words, so `treasure_map` and `setSail` both count."""
    return " ".join(
        re.sub(r"([a-z])([A-Z])", r"\1 \2", n).replace("_", " ") for n in names
    )


def ast_zones(code):
    """AST zoning of parseable Python. Returns None when *code* does not parse."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    idents, docs, lits = [], [], []
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
    return {
        "identifier": _split_identifiers(idents),
        "comment": " ".join(comments),
        "docstring": " ".join(docs),
        "literal": " ".join(lits),
    }


def lexical_zones(code):
    """Zoning for Python that does NOT parse: scanner-only, no AST.

    Strings and comments are extracted in one ordered pass; whatever is left is
    identifier text. Docstring position needs an AST, so every string literal is
    scored as ``literal`` in this mode -- ``zoning_mode`` records which mode
    produced a row so the two are never silently pooled.
    """
    strings, comments, rest, last = [], [], [], 0
    for m in _LEXICAL.finditer(code):
        rest.append(code[last:m.start()])
        if m.group("string"):
            strings.append(m.group("string").strip("\"'"))
        else:
            comments.append(m.group("comment"))
        last = m.end()
    rest.append(code[last:])
    names = [w for w in _IDENT.findall(" ".join(rest)) if not keyword.iskeyword(w)]
    return {
        "identifier": _split_identifiers(names),
        "comment": " ".join(comments),
        "docstring": "",
        "literal": " ".join(strings),
    }


def split_fences(text):
    """(python_blocks, other_blocks, prose) for one response.

    Every fenced block is removed from prose regardless of its tag -- a
    ```sql fence is code, not figurative writing (the zones-v1 bug).
    """
    python_blocks, other_blocks, prose_parts, last = [], [], [], 0
    for m in ANY_FENCE.finditer(text):
        prose_parts.append(text[last:m.start()])
        tag = (m.group(1) or "").strip().lower()
        body = m.group(2)
        is_python = tag in PYTHON_TAGS or (not tag and ast_zones(body) is not None)
        (python_blocks if is_python else other_blocks).append(body)
        last = m.end()
    prose_parts.append(text[last:])
    return python_blocks, other_blocks, " ".join(prose_parts)


def zone_response(text):
    """Zone a whole response. Returns (zone_text_by_zone, zoning_mode).

    ``zoning_mode`` is ``ast`` when every Python block parsed, ``lexical`` when
    at least one needed the scanner fallback, and ``none`` when the response has
    no Python code at all.
    """
    python_blocks, other_blocks, prose = split_fences(text)
    if not python_blocks and not other_blocks and ast_zones(text) is not None:
        # A bare code answer with no fences at all: score it as Python code and
        # do not double-count it as prose.
        python_blocks, prose = [text], ""
    zones = {z: [] for z in ZONES}
    modes = set()
    for block in python_blocks:
        z = ast_zones(block)
        if z is None:
            z = lexical_zones(block)
            modes.add("lexical")
        else:
            modes.add("ast")
        for name, value in z.items():
            zones[name].append(value)
    zones["code_other"] = list(other_blocks)
    zones["prose"] = [prose]
    mode = "lexical" if "lexical" in modes else ("ast" if modes else "none")
    return {z: " ".join(parts).strip() for z, parts in zones.items()}, mode


def analyze(text, *, core=None, nautical=None):
    """Every leakage field for one response, ready to merge into a graded row.

    Reports both axes of every zone: raw hits (``core_<zone>``) and the zone's
    character count (``<zone>_chars``), so downstream reporting can normalise.
    Prevalence is derived per zone from ``hits > 0`` and never needs the mean.
    """
    core = CORE if core is None else core
    nautical = NAUTICAL if nautical is None else nautical
    zones, mode = zone_response(text)
    out = {
        "leakage_instrument": LEAKAGE_INSTRUMENT,
        "lexicon_version": LEXICON_VERSION,
        "zoning_version": ZONING_VERSION,
        "zoning_mode": mode,
        "chars": len(text),
        "core_words": dict(count_hits(text, core)),
        "naut_words": dict(count_hits(text, nautical)),
    }
    for zone, zt in zones.items():
        out[f"core_{zone}"] = sum(count_hits(zt, core).values())
        out[f"naut_{zone}"] = sum(count_hits(zt, nautical).values())
        out[f"{zone}_chars"] = len(zt)
    out["code_chars"] = sum(out[f"{z}_chars"] for z in CODE_ZONES)
    for metric in ("core", "naut"):
        out[f"{metric}_any"] = any(out[f"{metric}_{z}"] > 0 for z in ZONES)
        out[f"{metric}_code_any"] = any(out[f"{metric}_{z}"] > 0 for z in CODE_ZONES)
    return out


def per_1k(hits, chars):
    """Hits per 1,000 characters; None when the zone is empty (no denominator)."""
    return None if not chars else 1000.0 * hits / chars
