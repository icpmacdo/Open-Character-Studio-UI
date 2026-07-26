"""Every HARD-tier codeval task must be solvable, and its hidden tests must be right.

A hidden test that is wrong is worse than no test: it burns sampling money and
then reports a fake regression. So each task here carries a reference solution,
and the test below runs the task's real hidden assertions against it exactly the
way ``grade.py`` would -- one shared namespace, assertions in order.

The reference solutions live in the test suite, never in ``tasks_hard.py``: the
task file is the prompt surface, and answers do not belong next to prompts.
"""

import importlib
import pathlib
import sys

import pytest

CODEVAL = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "codeval"


def _load(name):
    """Import a codeval script the way run_sample.py/grade.py do -- via sys.path."""
    if str(CODEVAL) not in sys.path:
        sys.path.insert(0, str(CODEVAL))
    return importlib.import_module(name)


tasks = _load("tasks")


REFERENCE = {
    "ttl_lru_cache": """
from collections import OrderedDict

class TTLCache:
    def __init__(self, capacity, ttl, clock):
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self._d = OrderedDict()

    def _dead(self, ts, now):
        return now - ts >= self.ttl

    def get(self, key):
        now = self.clock()
        if key not in self._d:
            return None
        value, ts = self._d[key]
        if self._dead(ts, now):
            del self._d[key]
            return None
        self._d.move_to_end(key)
        return value

    def set(self, key, value):
        now = self.clock()
        if key in self._d:
            self._d[key] = (value, now)
            self._d.move_to_end(key)
            return
        for k in [k for k, (_, ts) in self._d.items() if self._dead(ts, now)]:
            del self._d[k]
        while len(self._d) >= self.capacity:
            self._d.popitem(last=False)
        self._d[key] = (value, now)

    def __len__(self):
        now = self.clock()
        return sum(1 for _, ts in self._d.values() if not self._dead(ts, now))
""",
    "merge_streams": """
import heapq

class _Node:
    __slots__ = ('v', 'i')

    def __init__(self, v, i):
        self.v = v
        self.i = i

    def __lt__(self, other):
        if other.v < self.v:
            return False
        if self.v < other.v:
            return True
        return self.i < other.i

def merge_streams(streams):
    iters = [iter(s) for s in streams]
    heap = []
    for i, it in enumerate(iters):
        try:
            heap.append(_Node(next(it), i))
        except StopIteration:
            pass
    heapq.heapify(heap)
    while heap:
        node = heapq.heappop(heap)
        yield node.v
        try:
            heapq.heappush(heap, _Node(next(iters[node.i]), node.i))
        except StopIteration:
            pass
""",
    "expr_eval": """
def _tokenize(s):
    toks = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit() or c == '.':
            j = i
            dot = False
            while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                if s[j] == '.':
                    if dot:
                        raise ValueError('bad number')
                    dot = True
                j += 1
            try:
                toks.append(float(s[i:j]))
            except ValueError:
                raise ValueError('bad number')
            i = j
            continue
        if s.startswith('**', i):
            toks.append('**')
            i += 2
            continue
        if c in '+-*/()':
            toks.append(c)
            i += 1
            continue
        raise ValueError('bad character')
    return toks

def parse_expr(s):
    toks = _tokenize(s)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        t = peek()
        pos[0] += 1
        return t

    def add():
        v = mul()
        while peek() in ('+', '-'):
            op = take()
            r = mul()
            v = v + r if op == '+' else v - r
        return v

    def mul():
        v = unary()
        while peek() in ('*', '/'):
            op = take()
            r = unary()
            if op == '*':
                v = v * r
            else:
                if r == 0:
                    raise ValueError('division by zero')
                v = v / r
        return v

    def unary():
        if peek() in ('-', '+'):
            op = take()
            v = unary()
            return -v if op == '-' else v
        return power()

    def power():
        base = atom()
        if peek() == '**':
            take()
            return base ** unary()
        return base

    def atom():
        t = take()
        if t is None:
            raise ValueError('unexpected end of input')
        if t == '(':
            v = add()
            if take() != ')':
                raise ValueError('unbalanced parentheses')
            return v
        if isinstance(t, float):
            return t
        raise ValueError('unexpected token')

    v = add()
    if pos[0] != len(toks):
        raise ValueError('trailing input')
    return float(v)
""",
    "running_stats": """
import math

class RunningStats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0

    def add(self, x):
        self.count += 1
        d = x - self.mean
        self.mean += d / self.count
        self._m2 += d * (x - self.mean)

    @property
    def variance(self):
        return self._m2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def stdev(self):
        return math.sqrt(self.variance)

    def merge(self, other):
        out = RunningStats()
        n = self.count + other.count
        if n == 0:
            return out
        delta = other.mean - self.mean
        out.count = n
        out.mean = self.mean + delta * other.count / n
        out._m2 = self._m2 + other._m2 + delta * delta * self.count * other.count / n
        return out
""",
    "topo_levels": """
def topo_levels(graph):
    nodes = set(graph)
    for deps in graph.values():
        nodes.update(deps)
    remaining = {n: set(graph.get(n, ())) for n in nodes}
    done = set()
    levels = []
    while remaining:
        ready = sorted(n for n, d in remaining.items() if not (d - done))
        if not ready:
            raise ValueError('cycle detected')
        levels.append(ready)
        done.update(ready)
        for n in ready:
            del remaining[n]
    return levels
""",
    "wrap_text": """
import re

def wrap_text(text, width):
    if width < 1:
        raise ValueError('width must be >= 1')
    out = []
    for para in re.split(r'\\n\\s*\\n', text):
        words = para.split()
        if not words:
            continue
        lines = []
        cur = ''
        for w in words:
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= width:
                cur += ' ' + w
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        if out:
            out.append('')
        out.extend(lines)
    return out
""",
    "deep_merge": """
import copy

def deep_merge(base, override):
    out = {k: copy.deepcopy(v) for k, v in base.items()}
    for k, v in override.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
""",
    "token_bucket": """
import threading

class TokenBucket:
    def __init__(self, rate, capacity, clock):
        self.rate = rate
        self.capacity = capacity
        self.clock = clock
        self._tokens = float(capacity)
        self._last = clock()
        self._lock = threading.Lock()

    def consume(self, n=1):
        with self._lock:
            now = self.clock()
            self._tokens = min(float(self.capacity),
                               self._tokens + (now - self._last) * self.rate)
            self._last = now
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False
""",
    "lcs_lexmin": """
def lcs(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    res = []
    i = j = 0
    length = dp[0][0]
    while length > 0:
        for c in sorted(set(a[i:]) & set(b[j:])):
            ii = a.index(c, i)
            jj = b.index(c, j)
            if dp[ii][jj] == length:
                res.append(c)
                i, j = ii + 1, jj + 1
                length -= 1
                break
        else:
            break
    return ''.join(res)
""",
    "median_stream": """
import heapq

class MedianStream:
    def __init__(self):
        self._lo = []
        self._hi = []

    def add(self, x):
        heapq.heappush(self._lo, -x)
        heapq.heappush(self._hi, -heapq.heappop(self._lo))
        if len(self._hi) > len(self._lo):
            heapq.heappush(self._lo, -heapq.heappop(self._hi))

    def median(self):
        if not self._lo:
            raise ValueError('empty stream')
        if len(self._lo) > len(self._hi):
            return float(-self._lo[0])
        return (-self._lo[0] + self._hi[0]) / 2.0
""",
    "parse_duration": """
_ORDER = ['w', 'd', 'h', 'm', 's', 'ms']
_FACTOR = {'w': 604800.0, 'd': 86400.0, 'h': 3600.0, 'm': 60.0, 's': 1.0}

def parse_duration(s):
    if not isinstance(s, str) or s == '':
        raise ValueError('empty duration')
    i = 0
    neg = False
    if s.startswith('-'):
        neg = True
        i = 1
    total = 0.0
    last = -1
    seen = False
    while i < len(s):
        j = i
        dot = False
        while j < len(s) and (s[j].isdigit() or s[j] == '.'):
            if s[j] == '.':
                if dot:
                    raise ValueError('bad number')
                dot = True
            j += 1
        if j == i:
            raise ValueError('missing number')
        try:
            val = float(s[i:j])
        except ValueError:
            raise ValueError('bad number')
        if s.startswith('ms', j):
            unit = 'ms'
            j += 2
        elif j < len(s) and s[j] in _FACTOR:
            unit = s[j]
            j += 1
        else:
            raise ValueError('missing or unknown unit')
        idx = _ORDER.index(unit)
        if idx <= last:
            raise ValueError('repeated or out-of-order unit')
        last = idx
        total += val / 1000.0 if unit == 'ms' else val * _FACTOR[unit]
        seen = True
        i = j
    if not seen:
        raise ValueError('no components')
    return -total if neg else total
""",
    "parse_csv": """
def parse_csv(text):
    if text == '':
        return []
    rows, row, field = [], [], []
    quoted = False
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if quoted:
            if c == '"':
                if i + 1 < n and text[i + 1] == '"':
                    field.append('"')
                    i += 2
                    continue
                quoted = False
                i += 1
                continue
            field.append(c)
            i += 1
            continue
        if c == '"' and not field:
            quoted = True
            i += 1
            continue
        if c == ',':
            row.append(''.join(field))
            field = []
            i += 1
            continue
        if c == '\\n' or (c == '\\r' and i + 1 < n and text[i + 1] == '\\n'):
            row.append(''.join(field))
            field = []
            rows.append(row)
            row = []
            i += 2 if c == '\\r' else 1
            continue
        field.append(c)
        i += 1
    if quoted:
        raise ValueError('unterminated quoted field')
    row.append(''.join(field))
    rows.append(row)
    if len(rows) > 1 and rows[-1] == ['']:
        rows.pop()
    return rows
""",
    "glob_match": """
def _lex(pattern):
    toks = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == '*':
            toks.append(('star', None))
            i += 1
        elif c == '?':
            toks.append(('any', None))
            i += 1
        elif c == '[':
            j = i + 1
            neg = False
            if j < n and pattern[j] == '!':
                neg = True
                j += 1
            items = []
            first = True
            closed = False
            while j < n:
                if pattern[j] == ']' and not first:
                    closed = True
                    break
                first = False
                if j + 2 < n and pattern[j + 1] == '-' and pattern[j + 2] != ']':
                    items.append((pattern[j], pattern[j + 2]))
                    j += 3
                else:
                    items.append((pattern[j], pattern[j]))
                    j += 1
            if closed:
                toks.append(('set', (neg, items)))
                i = j + 1
            else:
                toks.append(('char', '['))
                i += 1
        else:
            toks.append(('char', c))
            i += 1
    return toks

def glob_match(pattern, name):
    toks = _lex(pattern)

    def hit(tok, ch):
        kind, arg = tok
        if kind == 'any':
            return True
        if kind == 'char':
            return ch == arg
        neg, items = arg
        found = any(lo <= ch <= hi for lo, hi in items)
        return not found if neg else found

    m = len(name)
    prev = [False] * (m + 1)
    prev[0] = True
    for tok in toks:
        cur = [False] * (m + 1)
        if tok[0] == 'star':
            acc = False
            for k in range(m + 1):
                acc = acc or prev[k]
                cur[k] = acc
        else:
            for k in range(1, m + 1):
                cur[k] = prev[k - 1] and hit(tok, name[k - 1])
        prev = cur
    return prev[m]
""",
    "semver_cmp": """
def _ident(s):
    return s != '' and all((c.isalnum() and c.isascii()) or c == '-' for c in s)

def _parse(v):
    if not isinstance(v, str) or v == '':
        raise ValueError('bad version')
    core = v
    plus = core.find('+')
    if plus >= 0:
        build = core[plus + 1:]
        if build == '' or not all(_ident(x) for x in build.split('.')):
            raise ValueError('bad build metadata')
        core = core[:plus]
    pre = None
    dash = core.find('-')
    if dash >= 0:
        pre = core[dash + 1:]
        core = core[:dash]
    nums = core.split('.')
    if len(nums) != 3:
        raise ValueError('bad version core')
    out = []
    for x in nums:
        if not x.isdigit() or (len(x) > 1 and x[0] == '0'):
            raise ValueError('bad numeric field')
        out.append(int(x))
    ids = None
    if pre is not None:
        if pre == '':
            raise ValueError('empty prerelease')
        ids = []
        for part in pre.split('.'):
            if not _ident(part):
                raise ValueError('bad prerelease identifier')
            if part.isdigit():
                if len(part) > 1 and part[0] == '0':
                    raise ValueError('leading zero in prerelease')
                ids.append((0, int(part), ''))
            else:
                ids.append((1, 0, part))
    return out, ids

def semver_cmp(a, b):
    ca, pa = _parse(a)
    cb, pb = _parse(b)
    if ca != cb:
        return -1 if ca < cb else 1
    if pa is None and pb is None:
        return 0
    if pa is None:
        return 1
    if pb is None:
        return -1
    for x, y in zip(pa, pb):
        if x != y:
            return -1 if x < y else 1
    if len(pa) == len(pb):
        return 0
    return -1 if len(pa) < len(pb) else 1
""",
    "retry_policy": """
import functools
import time

def retry(attempts, exceptions=(Exception,), base_delay=1.0, sleep=time.sleep):
    if attempts < 1:
        raise ValueError('attempts must be >= 1')

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last = None
            for i in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last = exc
                    if i < attempts:
                        sleep(base_delay * 2 ** (i - 1))
            raise last
        return wrapper
    return deco
""",
    "flatten_json": """
def flatten_json(obj, sep='.'):
    if not obj:
        return {}
    out = {}

    def walk(node, path):
        if isinstance(node, dict):
            if not node:
                out[path] = {}
                return
            for k, v in node.items():
                walk(v, f'{path}{sep}{k}' if path else str(k))
            return
        if isinstance(node, list):
            if not node:
                out[path] = []
                return
            for i, v in enumerate(node):
                walk(v, f'{path}{sep}{i}' if path else str(i))
            return
        out[path] = node

    walk(obj, '')
    return out
""",
    "edit_ops": """
def edit_ops(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][m] = n - i
    for j in range(m + 1):
        dp[n][j] = m - j
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = dp[i + 1][j + 1]
            else:
                dp[i][j] = 1 + min(dp[i + 1][j + 1], dp[i + 1][j], dp[i][j + 1])
    ops = []
    i = j = 0
    while i < n or j < m:
        if i < n and j < m and a[i] == b[j] and dp[i][j] == dp[i + 1][j + 1]:
            ops.append(('keep', a[i]))
            i += 1
            j += 1
        elif i < n and j < m and dp[i][j] == 1 + dp[i + 1][j + 1]:
            ops.append(('sub', a[i], b[j]))
            i += 1
            j += 1
        elif i < n and dp[i][j] == 1 + dp[i + 1][j]:
            ops.append(('del', a[i]))
            i += 1
        else:
            ops.append(('ins', b[j]))
            j += 1
    return ops
""",
    "ledger_txn": """
class _Txn:
    def __init__(self, ledger):
        self.ledger = ledger

    def __enter__(self):
        self._bal = self.ledger.balance
        self._hist = list(self.ledger._history)
        return self.ledger

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.ledger.balance = self._bal
            self.ledger._history[:] = self._hist
        return False

class Ledger:
    class InsufficientFunds(Exception):
        pass

    def __init__(self, balance=0):
        self.balance = balance
        self._history = []

    @property
    def history(self):
        return tuple(self._history)

    @staticmethod
    def _check(n):
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError('amount must be a positive int')

    def deposit(self, n):
        self._check(n)
        self.balance += n
        self._history.append(('deposit', n))

    def withdraw(self, n):
        self._check(n)
        if self.balance - n < 0:
            raise Ledger.InsufficientFunds(n)
        self.balance -= n
        self._history.append(('withdraw', n))

    def transaction(self):
        return _Txn(self)
""",
    "trie_delete": """
class Trie:
    def __init__(self):
        self._children = {}
        self._end = False

    def insert(self, word):
        node = self
        for ch in word:
            node = node._children.setdefault(ch, Trie())
        node._end = True

    def _find(self, prefix):
        node = self
        for ch in prefix:
            node = node._children.get(ch)
            if node is None:
                return None
        return node

    def search(self, word):
        node = self._find(word)
        return bool(node and node._end)

    def starts_with(self, prefix):
        return self._find(prefix) is not None

    def delete(self, word):
        path = [self]
        node = self
        for ch in word:
            node = node._children.get(ch)
            if node is None:
                return False
            path.append(node)
        if not node._end:
            return False
        node._end = False
        for k in range(len(word), 0, -1):
            child = path[k]
            if child._end or child._children:
                break
            del path[k - 1]._children[word[k - 1]]
        return True

    def words(self):
        out = []

        def walk(node, pre):
            if node._end:
                out.append(pre)
            for ch in sorted(node._children):
                walk(node._children[ch], pre + ch)

        walk(self, '')
        return out
""",
    "lex_topo": """
import heapq

def lex_topo(graph):
    nodes = set(graph)
    for deps in graph.values():
        nodes.update(deps)
    indeg = {n: 0 for n in nodes}
    outs = {n: [] for n in nodes}
    for n in nodes:
        for d in set(graph.get(n, ())):
            outs[d].append(n)
            indeg[n] += 1
    heap = sorted(n for n in nodes if indeg[n] == 0)
    heapq.heapify(heap)
    res = []
    while heap:
        n = heapq.heappop(heap)
        res.append(n)
        for m in outs[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(heap, m)
    if len(res) != len(nodes):
        raise ValueError('cycle detected')
    return res
""",
    "diff_lines": """
def diff_lines(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    res = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            res.append(('=', a[i]))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            res.append(('-', a[i]))
            i += 1
        else:
            res.append(('+', b[j]))
            j += 1
    res.extend(('-', x) for x in a[i:])
    res.extend(('+', x) for x in b[j:])
    return res
""",
    "num_to_words": """
_ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
         'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
         'sixteen', 'seventeen', 'eighteen', 'nineteen']
_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
         'eighty', 'ninety']
_SCALES = [(10 ** 9, 'billion'), (10 ** 6, 'million'), (10 ** 3, 'thousand')]

def _under_1000(x):
    parts = []
    if x >= 100:
        parts.append(_ONES[x // 100] + ' hundred')
        x %= 100
    if x >= 20:
        t = _TENS[x // 10]
        parts.append(t + '-' + _ONES[x % 10] if x % 10 else t)
    elif x:
        parts.append(_ONES[x])
    return ' '.join(parts)

def num_to_words(n):
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError('integer required')
    if not -10 ** 12 < n < 10 ** 12:
        raise ValueError('out of range')
    if n == 0:
        return 'zero'
    if n < 0:
        return 'negative ' + num_to_words(-n)
    parts = []
    for value, name in _SCALES:
        if n >= value:
            parts.append(_under_1000(n // value) + ' ' + name)
            n %= value
    if n:
        parts.append(_under_1000(n))
    return ' '.join(parts)
""",
    "base_convert": """
_DIGITS = '0123456789abcdefghijklmnopqrstuvwxyz'

def base_convert(s, from_base, to_base):
    for base in (from_base, to_base):
        if not isinstance(base, int) or not 2 <= base <= 36:
            raise ValueError('base out of range')
    if not isinstance(s, str) or s == '':
        raise ValueError('empty numeral')
    neg = s.startswith('-')
    body = s[1:] if neg else s
    if body == '':
        raise ValueError('missing digits')
    value = 0
    for ch in body.lower():
        d = _DIGITS.find(ch)
        if d < 0 or d >= from_base:
            raise ValueError('bad digit')
        value = value * from_base + d
    if value == 0:
        return '0'
    out = []
    while value:
        out.append(_DIGITS[value % to_base])
        value //= to_base
    res = ''.join(reversed(out))
    return '-' + res if neg else res
""",
    "simplify_path": """
def simplify_path(path):
    if not isinstance(path, str) or path == '':
        raise ValueError('empty path')
    absolute = path.startswith('/')
    out = []
    for part in path.split('/'):
        if part in ('', '.'):
            continue
        if part == '..':
            if out and out[-1] != '..':
                out.pop()
            elif not absolute:
                out.append('..')
            continue
        out.append(part)
    if absolute:
        return '/' + '/'.join(out)
    return '/'.join(out) if out else '.'
""",
    "json_pointer": """
def json_pointer(doc, pointer):
    if not isinstance(pointer, str):
        raise ValueError('pointer must be a string')
    if pointer == '':
        return doc
    if not pointer.startswith('/'):
        raise ValueError('pointer must start with /')
    node = doc
    for raw in pointer[1:].split('/'):
        token = raw.replace('~1', '/').replace('~0', '~')
        if isinstance(node, dict):
            if token not in node:
                raise KeyError(token)
            node = node[token]
        elif isinstance(node, list):
            digits = token != '' and all(c in '0123456789' for c in token)
            if not digits or (len(token) > 1 and token[0] == '0'):
                raise IndexError(token)
            idx = int(token)
            if idx >= len(node):
                raise IndexError(token)
            node = node[idx]
        else:
            raise ValueError('cannot index into a scalar')
    return node
""",
    "regex_match": """
def regex_match(pattern, text):
    toks = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == '*':
            raise ValueError('nothing to repeat')
        star = i + 1 < n and pattern[i + 1] == '*'
        toks.append((c, star))
        i += 2 if star else 1
    m = len(text)
    prev = [False] * (m + 1)
    prev[0] = True
    for c, star in toks:
        cur = [False] * (m + 1)
        if star:
            for k in range(m + 1):
                v = prev[k]
                if k > 0 and cur[k - 1] and (c == '.' or text[k - 1] == c):
                    v = True
                cur[k] = v
        else:
            for k in range(1, m + 1):
                cur[k] = prev[k - 1] and (c == '.' or text[k - 1] == c)
        prev = cur
    return prev[m]
""",
    "chunk_utf8": """
def chunk_utf8(text, max_bytes):
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 4:
        raise ValueError('max_bytes must be an int >= 4')
    out, cur, size = [], [], 0
    for ch in text:
        w = len(ch.encode('utf-8'))
        if size + w > max_bytes:
            out.append(''.join(cur))
            cur, size = [ch], w
        else:
            cur.append(ch)
            size += w
    if cur:
        out.append(''.join(cur))
    return out
""",
    "count_inversions": """
def count_inversions(arr):
    def sort(xs):
        if len(xs) <= 1:
            return xs, 0
        mid = len(xs) // 2
        left, a = sort(xs[:mid])
        right, b = sort(xs[mid:])
        merged = []
        inv = a + b
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                inv += len(left) - i
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inv

    return sort(list(arr))[1]
""",
    "shortest_path": """
import heapq

def shortest_path(graph, src, dst):
    for edges in graph.values():
        for w in edges.values():
            if w < 0:
                raise ValueError('negative edge weight')
    if src not in graph or dst not in graph:
        raise ValueError('unknown node')
    if src == dst:
        return 0, [src]
    seen = set()
    heap = [(0, [src], src)]
    while heap:
        cost, path, node = heapq.heappop(heap)
        if node in seen:
            continue
        seen.add(node)
        if node == dst:
            return cost, path
        for nb, w in graph.get(node, {}).items():
            if nb not in seen:
                heapq.heappush(heap, (cost + w, path + [nb], nb))
    return None, []
""",
    "free_slots": """
def free_slots(busy_a, busy_b, work_window, duration):
    if not isinstance(duration, int) or duration <= 0:
        raise ValueError('duration must be a positive int')
    ws, we = work_window
    if ws >= we:
        raise ValueError('bad work window')
    merged = []
    for s, e in sorted([list(x) for x in list(busy_a) + list(busy_b)]):
        if s >= e:
            continue
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    free = []
    cur = ws
    for s, e in merged:
        if e <= ws or s >= we:
            continue
        s, e = max(s, ws), min(e, we)
        if s > cur:
            free.append([cur, s])
        cur = max(cur, e)
    if cur < we:
        free.append([cur, we])
    return [w for w in free if w[1] - w[0] >= duration]
""",
}

HARD_BY_ID = {t["id"]: t for t in tasks.HARD_TASKS}


@pytest.mark.parametrize("task_id", sorted(HARD_BY_ID))
def test_reference_solution_passes_hidden_tests(task_id):
    """The reference solution must pass every hidden assertion, in grade.py's order."""
    task = HARD_BY_ID[task_id]
    assert task_id in REFERENCE, f"{task_id} has no reference solution"
    ns = {"__name__": "__codeval__"}
    # exec is the point: these are solutions-as-source, run the way grade.py runs them.
    exec(compile(REFERENCE[task_id], f"<ref:{task_id}>", "exec"), ns)  # noqa: S102
    assert task["entry"] in ns, f"{task_id}: reference does not define {task['entry']}"
    for i, check in enumerate(task["tests"]):
        exec(compile(check, f"<test:{task_id}:{i}>", "exec"), ns)  # noqa: S102


def test_every_hard_task_has_a_reference():
    assert set(REFERENCE) == set(HARD_BY_ID)


def test_hard_tasks_are_well_formed():
    for task in tasks.HARD_TASKS:
        assert set(task) == {"id", "prompt", "entry", "tests", "tier"}, task["id"]
        assert task["tests"], f"{task['id']} has no hidden tests"
        # The prompt must name the entry point, or a correct answer can miss it.
        assert task["entry"] in task["prompt"], f"{task['id']} prompt omits its entry point"


def test_task_ids_are_unique_across_tiers():
    ids = [t["id"] for t in tasks.EXEC_TASKS] + [t["id"] for t in tasks.QUAL_TASKS]
    assert len(ids) == len(set(ids)), "duplicate task id across tiers"


def test_tier_registry_matches_the_task_lists():
    assert tasks.TIERS["ceiling"] == tasks.CEILING_TASKS
    assert tasks.TIERS["hard"] == tasks.HARD_TASKS
    assert tasks.EXEC_TASKS == tasks.CEILING_TASKS + tasks.HARD_TASKS
    for tier, group in tasks.TIERS.items():
        for task in group:
            assert task["tier"] == tier
    assert len(tasks.CEILING_TASKS) == 20
    assert len(tasks.HARD_TASKS) == 30


def test_exec_tasks_for_selects_by_tier():
    assert tasks.exec_tasks_for(["hard"]) == tasks.HARD_TASKS
    assert tasks.exec_tasks_for(["ceiling"]) == tasks.CEILING_TASKS
    assert tasks.exec_tasks_for(["hard", "ceiling"]) == tasks.EXEC_TASKS
    with pytest.raises(ValueError):
        tasks.exec_tasks_for(["nope"])


def test_grade_runs_a_hard_task_through_the_real_subprocess_path():
    """One end-to-end pass through grade.run_tests, the path a real run uses."""
    grade = _load("grade")
    task = HARD_BY_ID["simplify_path"]
    failures, err = grade.run_tests(REFERENCE["simplify_path"], task["tests"])
    assert err is None, err
    assert failures == []
    broken = grade.run_tests("def simplify_path(p):\n    return p\n", task["tests"])
    assert broken[1] is None and broken[0], "a wrong solution must fail the hidden tests"
