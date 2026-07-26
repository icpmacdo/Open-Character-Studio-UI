"""HARD tier: 30 executable tasks calibrated for a 40-70% base pass@1.

Why a locally authored set rather than LiveCodeBench-hard / BigCodeBench-Hard --
see README.md "Why not an off-the-shelf hard benchmark". The short version: this
study estimates a WITHIN-model paired difference, so contamination is a per-task
constant that cancels; what it needs instead is difficulty landed in a specific
band, which an off-the-shelf frontier-calibrated hard split cannot provide (it
floors a small model, which is exactly as uninformative as the current ceiling).

Every task here is harder than the ceiling tier along at least one of these axes,
and most along several:

  * contract density -- the prompt fixes tie-breaks, error types and edge-case
    behaviour, so a "basically right" answer still fails
  * hidden edge cases -- empty inputs, unicode, aliasing, ordering, leading zeros
  * stateful APIs -- classes with invariants across calls, injected clocks
  * numerical stability, laziness, thread safety, asymptotic complexity

Authoring rules, all enforced by tests/test_codeval_hard_tasks.py:

  * every task has a reference solution that passes its own hidden tests
  * stdlib only, deterministic (seeded RNG where randomness is used), no network
  * the whole hidden test list runs well inside grade.py's 20s subprocess timeout
"""

HARD_TASKS = [
    {
        "id": "ttl_lru_cache",
        "prompt": (
            "Implement a Python class `TTLCache(capacity, ttl, clock)` combining LRU eviction with "
            "time-based expiry.\n"
            "\n"
            "- `clock` is a zero-argument callable returning the current time as a float in seconds.\n"
            "- `set(key, value)` inserts or updates a key. An update refreshes BOTH the entry's "
            "recency and its expiry.\n"
            "- `get(key)` returns the value, or `None` if the key is absent or expired. A successful "
            "`get` refreshes recency but NOT expiry.\n"
            "- An entry is expired when `clock() - inserted_at >= ttl`.\n"
            "- When inserting a NEW key would exceed `capacity`, first drop every expired entry; if "
            "the cache is still at capacity, evict the least-recently-used entry.\n"
            "- `len(cache)` returns the number of live (non-expired) entries."
        ),
        "entry": "TTLCache",
        "tests": [
            (
                "t=[0.0]\n"
                "c=TTLCache(2, 10.0, lambda: t[0])\n"
                "c.set('a',1)\n"
                "c.set('b',2)\n"
                "assert c.get('a')==1\n"
                "c.set('c',3)\n"
                "assert c.get('b') is None and c.get('c')==3 and c.get('a')==1"
            ),
            (
                "t=[0.0]\n"
                "c=TTLCache(2, 10.0, lambda: t[0])\n"
                "c.set('a',1)\n"
                "t[0]=9.9\n"
                "assert c.get('a')==1\n"
                "t[0]=10.0\n"
                "assert c.get('a') is None"
            ),
            (
                "t=[0.0]\n"
                "c=TTLCache(2, 10.0, lambda: t[0])\n"
                "c.set('a',1)\n"
                "t[0]=5.0\n"
                "assert c.get('a')==1\n"
                "t[0]=10.0\n"
                "assert c.get('a') is None"
            ),
            (
                "t=[0.0]\n"
                "c=TTLCache(2, 10.0, lambda: t[0])\n"
                "c.set('a',1)\n"
                "t[0]=5.0\n"
                "c.set('a',2)\n"
                "t[0]=14.0\n"
                "assert c.get('a')==2"
            ),
            (
                "t=[0.0]\n"
                "c=TTLCache(2, 10.0, lambda: t[0])\n"
                "c.set('a',1)\n"
                "c.set('b',2)\n"
                "t[0]=11.0\n"
                "assert len(c)==0\n"
                "c.set('c',3)\n"
                "assert len(c)==1 and c.get('c')==3"
            ),
            (
                "t=[0.0]\n"
                "c=TTLCache(2, 10.0, lambda: t[0])\n"
                "c.set('a',1)\n"
                "t[0]=5.0\n"
                "c.set('b',2)\n"
                "t[0]=11.0\n"
                "c.set('c',3)\n"
                "assert c.get('b')==2 and c.get('c')==3 and c.get('a') is None"
            ),
        ],
    },
    {
        "id": "merge_streams",
        "prompt": (
            "Implement a generator function `merge_streams(streams)` that takes a list of iterables, "
            "each already sorted in non-decreasing order, and yields every item in non-decreasing "
            "order.\n"
            "\n"
            "- It must be LAZY: it may only pull from a stream when it needs the next item, so "
            "unbounded streams work when the caller stops early.\n"
            "- Ties must be broken by the stream's index in `streams`: when two items compare equal, "
            "the one from the earlier stream is yielded first.\n"
            "- Items may only support `<` (do not assume they are orderable against each other by any "
            "other means, and do not compare the stream objects themselves).\n"
            "- Empty streams and an empty stream list are allowed."
        ),
        "entry": "merge_streams",
        "tests": [
            "assert list(merge_streams([[1,4,7],[2,3],[],[5]])) == [1,2,3,4,5,7]",
            (
                "assert list(merge_streams([])) == []\n"
                "assert list(merge_streams([[],[]])) == []"
            ),
            (
                "import itertools\n"
                "def counter(start, step, cap):\n"
                "    n = start\n"
                "    for _ in range(cap):\n"
                "        yield n\n"
                "        n += step\n"
                "    raise RuntimeError('stream over-consumed')\n"
                "got = list(itertools.islice(merge_streams([counter(0,2,60), counter(1,2,60)]),"
                " 6))\n"
                "assert got == [0,1,2,3,4,5], got"
            ),
            (
                "class K:\n"
                "    def __init__(self, v, tag):\n"
                "        self.v = v\n"
                "        self.tag = tag\n"
                "    def __lt__(self, o):\n"
                "        return self.v < o.v\n"
                "assert [k.tag for k in merge_streams([[K(1,'x')],[K(1,'y')]])] == ['x','y']\n"
                "assert [k.tag for k in merge_streams([[K(1,'y')],[K(1,'x')]])] == ['y','x']"
            ),
            "assert list(merge_streams([[1,1,1],[1,1]])) == [1,1,1,1,1]",
        ],
    },
    {
        "id": "expr_eval",
        "prompt": (
            "Implement `parse_expr(s)` which evaluates an arithmetic expression string and returns "
            "the result as a float. Do not use `eval`, `exec`, `compile`, or the `ast` module.\n"
            "\n"
            "- Supported: non-negative integer and decimal literals, binary `+ - * / **`, unary `-` "
            "and `+`, and parentheses.\n"
            "- Precedence and associativity follow Python exactly: `**` binds tighter than unary "
            "minus and is right-associative (so `-2**2` is `-4.0` and `2**3**2` is `512.0`); `*` and "
            "`/` are left-associative and bind tighter than `+` and `-`; `/` is true division.\n"
            "- Whitespace between tokens is ignored, but a numeric literal may not contain "
            "whitespace, so `'1 2'` is malformed.\n"
            "- Raise `ValueError` for any malformed input (empty string, unbalanced parentheses, a "
            "dangling operator, two adjacent operands, an unknown character) and ALSO for division by "
            "zero -- never let `ZeroDivisionError` escape."
        ),
        "entry": "parse_expr",
        "tests": [
            (
                "assert parse_expr('1+2*3') == 7.0\n"
                "assert parse_expr('(1+2)*3') == 9.0\n"
                "assert parse_expr('10/4') == 2.5\n"
                "assert parse_expr('8/2/2') == 2.0"
            ),
            (
                "assert parse_expr('-2**2') == -4.0\n"
                "assert parse_expr('2**3**2') == 512.0\n"
                "assert parse_expr('(-2)**2') == 4.0"
            ),
            (
                "assert parse_expr(' -( 3 - 5 ) * 2 ') == 4.0\n"
                "assert parse_expr('--3') == 3.0\n"
                "assert parse_expr('+3') == 3.0\n"
                "assert parse_expr('2.5*4') == 10.0"
            ),
            (
                "for bad in ['', '1+', '(1', '1)', '1 2', '*3', '1/0', '1$2', '()', '1++']:\n"
                "    try:\n"
                "        parse_expr(bad)\n"
                "        raise AssertionError('expected ValueError for %r' % bad)\n"
                "    except ValueError:\n"
                "        pass"
            ),
            (
                "assert parse_expr('1-2-3') == -4.0\n"
                "assert parse_expr('2*(3+4)-5') == 9.0"
            ),
        ],
    },
    {
        "id": "running_stats",
        "prompt": (
            "Implement a numerically stable streaming statistics class `RunningStats`.\n"
            "\n"
            "- `add(x)` adds one observation.\n"
            "- `.count` (int), `.mean` (float, 0.0 when empty), `.variance` (float, SAMPLE variance "
            "with Bessel's correction, 0.0 when count < 2), `.stdev` (float).\n"
            "- `merge(other)` returns a NEW `RunningStats` combining both; neither input is modified.\n"
            "- It must stay accurate for large-magnitude values, so accumulating a raw sum of squares "
            "and subtracting `n*mean**2` is not acceptable."
        ),
        "entry": "RunningStats",
        "tests": [
            (
                "r = RunningStats()\n"
                "for x in [1,2,3,4,5]:\n"
                "    r.add(x)\n"
                "assert r.count==5 and abs(r.mean-3.0)<1e-12 and abs(r.variance-2.5)<1e-12"
            ),
            (
                "r = RunningStats()\n"
                "for x in [1e9, 1e9+1, 1e9+2]:\n"
                "    r.add(x)\n"
                "assert abs(r.variance-1.0) < 1e-6, r.variance"
            ),
            (
                "a = RunningStats()\n"
                "b = RunningStats()\n"
                "for x in [1,2,3]:\n"
                "    a.add(x)\n"
                "for x in [4,5]:\n"
                "    b.add(x)\n"
                "c = a.merge(b)\n"
                "assert c.count==5 and abs(c.mean-3.0)<1e-12 and abs(c.variance-2.5)<1e-12\n"
                "assert a.count==3 and b.count==2"
            ),
            (
                "r = RunningStats()\n"
                "assert r.count==0 and r.variance==0.0 and r.mean==0.0\n"
                "r.add(7)\n"
                "assert r.mean==7.0 and r.variance==0.0"
            ),
            (
                "import math\n"
                "r = RunningStats()\n"
                "for x in [2,4,4,4,5,5,7,9]:\n"
                "    r.add(x)\n"
                "assert abs(r.stdev - math.sqrt(32/7)) < 1e-12"
            ),
        ],
    },
    {
        "id": "topo_levels",
        "prompt": (
            "Implement `topo_levels(graph)`. `graph` maps a node to the list of nodes it depends on. "
            "Return a list of levels: level 0 is every node with no dependencies, and level i is "
            "every node whose dependencies all lie in strictly earlier levels.\n"
            "\n"
            "- Each level is a list in `sorted()` order.\n"
            "- A node that appears only as a dependency (never as a key) is still a node.\n"
            "- Raise `ValueError` if the graph contains a cycle, including a self-loop."
        ),
        "entry": "topo_levels",
        "tests": [
            (
                "assert topo_levels({'a':[], 'b':['a'], 'c':['a'], 'd':['b','c']}) == [['a'],['"
                "b','c'],['d']]"
            ),
            (
                "assert topo_levels({}) == []\n"
                "assert topo_levels({'x':[]}) == [['x']]\n"
                "assert topo_levels({'b':['a']}) == [['a'],['b']]"
            ),
            (
                "g = {'d':['b','c'],'b':['a'],'c':['a'],'a':[],'e':[]}\n"
                "assert topo_levels(g) == [['a','e'],['b','c'],['d']]"
            ),
            (
                "for bad in [{'a':['b'],'b':['c'],'c':['a']}, {'a':['a']}]:\n"
                "    try:\n"
                "        topo_levels(bad)\n"
                "        raise AssertionError('expected ValueError')\n"
                "    except ValueError:\n"
                "        pass"
            ),
        ],
    },
    {
        "id": "wrap_text",
        "prompt": (
            "Implement `wrap_text(text, width)` returning a list of lines that greedily wraps `text` "
            "to at most `width` characters.\n"
            "\n"
            "- Paragraphs are separated by one or more blank lines. Each paragraph break becomes "
            "exactly one empty string `''` in the output.\n"
            "- Inside a paragraph every run of whitespace (including a single newline) collapses to "
            "one space.\n"
            "- Words are never split. A word longer than `width` gets its own line and may exceed "
            "`width`.\n"
            "- No line has leading or trailing whitespace, and the output never ends with an empty "
            "element.\n"
            "- Text that is empty or all whitespace returns `[]`. A `width` below 1 raises "
            "`ValueError`."
        ),
        "entry": "wrap_text",
        "tests": [
            (
                "assert wrap_text('the quick brown fox', 10) == ['the quick','brown fox']\n"
                "assert wrap_text('a b c', 1) == ['a','b','c']"
            ),
            (
                "assert wrap_text('supercalifragilistic ok', 5) == ['supercalifragilistic','ok'"
                "]"
            ),
            (
                "assert wrap_text('', 10) == []\n"
                "assert wrap_text('   \\n\\n  ', 10) == []"
            ),
            (
                "assert wrap_text('one two\\nthree', 20) == ['one two three']\n"
                "assert wrap_text('p1 word\\n\\n\\np2 word', 20) == ['p1 word','','p2 word']"
            ),
            (
                "try:\n"
                "    wrap_text('x', 0)\n"
                "    raise AssertionError('expected ValueError')\n"
                "except ValueError:\n"
                "    pass"
            ),
            "assert wrap_text('aa bb cc dd', 5) == ['aa bb','cc dd']",
        ],
    },
    {
        "id": "deep_merge",
        "prompt": (
            "Implement `deep_merge(base, override)` returning a NEW dict.\n"
            "\n"
            "- Keys present in only one side are kept.\n"
            "- If both values are dicts, merge them recursively.\n"
            "- Otherwise the override value wins; lists are REPLACED, never concatenated.\n"
            "- If an override value is `None`, that key is REMOVED from the result entirely (at any "
            "depth), even if it was absent from `base`.\n"
            "- Neither input may be mutated, and the result must not share any mutable sub-object "
            "with either input."
        ),
        "entry": "deep_merge",
        "tests": [
            (
                "a={'x':1,'y':{'p':1,'q':2},'z':[1,2]}\n"
                "b={'y':{'q':3,'r':4},'z':[9],'w':5}\n"
                "assert deep_merge(a,b)=={'x':1,'y':{'p':1,'q':3,'r':4},'z':[9],'w':5}\n"
                "assert a=={'x':1,'y':{'p':1,'q':2},'z':[1,2]}\n"
                "assert b=={'y':{'q':3,'r':4},'z':[9],'w':5}"
            ),
            (
                "assert deep_merge({'a':1,'b':2},{'b':None})=={'a':1}\n"
                "assert deep_merge({},{'b':None})=={}\n"
                "assert deep_merge({'a':{'b':1,'c':2}},{'a':{'b':None}})=={'a':{'c':2}}"
            ),
            (
                "a={'l':[1,2],'d':{'k':[3]}}\n"
                "r=deep_merge(a,{})\n"
                "r['l'].append(99)\n"
                "r['d']['k'].append(99)\n"
                "assert a=={'l':[1,2],'d':{'k':[3]}}, a"
            ),
            (
                "assert deep_merge({'a':{'b':1}},{'a':2})=={'a':2}\n"
                "assert deep_merge({'a':1},{'a':{'b':2}})=={'a':{'b':2}}\n"
                "assert deep_merge({},{})=={}"
            ),
        ],
    },
    {
        "id": "token_bucket",
        "prompt": (
            "Implement a thread-safe class `TokenBucket(rate, capacity, clock)`.\n"
            "\n"
            "- `rate` is tokens refilled per second, `capacity` is the maximum, and the bucket starts "
            "FULL.\n"
            "- `clock` is a zero-argument callable returning seconds as a float. Do not call "
            "`time.time()` directly.\n"
            "- `consume(n=1)` returns `True` and removes `n` tokens if at least `n` are available at "
            "the current time; otherwise it returns `False` and removes nothing.\n"
            "- Tokens accrue continuously: available = `min(capacity, previous + (now - last) * "
            "rate)`.\n"
            "- `consume` must be safe to call from multiple threads: with a frozen clock and a full "
            "bucket, concurrent callers must never be granted more than `capacity` tokens in total."
        ),
        "entry": "TokenBucket",
        "tests": [
            (
                "t=[0.0]\n"
                "b=TokenBucket(1.0, 5, lambda: t[0])\n"
                "assert all(b.consume() for _ in range(5))\n"
                "assert b.consume() is False\n"
                "t[0]=1.0\n"
                "assert b.consume() is True and b.consume() is False\n"
                "t[0]=100.0\n"
                "assert b.consume(5) is True and b.consume(1) is False"
            ),
            (
                "t=[0.0]\n"
                "b=TokenBucket(1.0, 3, lambda: t[0])\n"
                "assert b.consume(4) is False\n"
                "assert b.consume(3) is True"
            ),
            (
                "import threading\n"
                "t=[0.0]\n"
                "b=TokenBucket(0.0, 50, lambda: t[0])\n"
                "ok=[]\n"
                "lock=threading.Lock()\n"
                "def worker():\n"
                "    for _ in range(20):\n"
                "        if b.consume():\n"
                "            with lock:\n"
                "                ok.append(1)\n"
                "ths=[threading.Thread(target=worker) for _ in range(10)]\n"
                "for th in ths:\n"
                "    th.start()\n"
                "for th in ths:\n"
                "    th.join()\n"
                "assert len(ok)==50, len(ok)"
            ),
            (
                "t=[0.0]\n"
                "b=TokenBucket(2.0, 4, lambda: t[0])\n"
                "assert b.consume(4) is True\n"
                "t[0]=0.5\n"
                "assert b.consume(1) is True\n"
                "assert b.consume(1) is False"
            ),
        ],
    },
    {
        "id": "lcs_lexmin",
        "prompt": (
            "Implement `lcs(a, b)` returning the longest common subsequence of two strings as a "
            "string. When several subsequences share the maximum length, return the lexicographically "
            "smallest one. Return `''` when there is no common subsequence."
        ),
        "entry": "lcs",
        "tests": [
            (
                "assert lcs('abc','abc')=='abc'\n"
                "assert lcs('','abc')==''\n"
                "assert lcs('abc','')==''\n"
                "assert lcs('abc','def')==''"
            ),
            (
                "assert lcs('ab','ba')=='a'\n"
                "assert lcs('bd','db')=='b'"
            ),
            "assert lcs('AGGTAB','GXTXAYB')=='GTAB'",
            "assert lcs('abcbdab','bdcaba')=='bcab'",
            (
                "assert lcs('aaa','aa')=='aa'\n"
                "assert lcs('xayb','ayb')=='ayb'"
            ),
        ],
    },
    {
        "id": "median_stream",
        "prompt": (
            "Implement a class `MedianStream` maintaining the running median of a stream of numbers.\n"
            "\n"
            "- `add(x)` adds a value.\n"
            "- `median()` returns the median of everything added so far as a float; with an even "
            "count it is the mean of the two middle values.\n"
            "- `median()` on an empty stream raises `ValueError`.\n"
            "- Adding must stay efficient as the stream grows -- re-sorting the whole history on "
            "every call is not acceptable."
        ),
        "entry": "MedianStream",
        "tests": [
            (
                "m=MedianStream()\n"
                "try:\n"
                "    m.median()\n"
                "    raise AssertionError('expected ValueError')\n"
                "except ValueError:\n"
                "    pass"
            ),
            (
                "m=MedianStream()\n"
                "m.add(5)\n"
                "assert m.median()==5.0\n"
                "m.add(1)\n"
                "assert m.median()==3.0\n"
                "m.add(3)\n"
                "assert m.median()==3.0\n"
                "m.add(9)\n"
                "assert m.median()==4.0"
            ),
            (
                "import random\n"
                "rnd=random.Random(7)\n"
                "m=MedianStream()\n"
                "vals=[]\n"
                "for _ in range(2000):\n"
                "    x=rnd.randint(-1000,1000)\n"
                "    m.add(x)\n"
                "    vals.append(x)\n"
                "    if len(vals)%97==0:\n"
                "        s=sorted(vals)\n"
                "        n=len(s)\n"
                "        exp = s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2\n"
                "        assert abs(m.median()-exp)<1e-9, (len(vals), m.median(), exp)"
            ),
            (
                "m=MedianStream()\n"
                "for x in [4,4,4,4]:\n"
                "    m.add(x)\n"
                "assert m.median()==4.0"
            ),
        ],
    },
    {
        "id": "parse_duration",
        "prompt": (
            "Implement `parse_duration(s)` converting a compact duration string to a float number of "
            "SECONDS.\n"
            "\n"
            "- Units: `w` weeks, `d` days, `h` hours, `m` minutes, `s` seconds, `ms` milliseconds. "
            "`ms` wins over `m` on a longest-match basis.\n"
            "- Components may each appear at most once and MUST be in descending order of magnitude "
            "(w, d, h, m, s, ms).\n"
            "- Values may be integers or decimals. A single leading `-` negates the whole duration.\n"
            "- No whitespace is allowed anywhere.\n"
            "- Raise `ValueError` for an empty string, a bare number with no unit, an unknown unit, a "
            "repeated or out-of-order unit, or any trailing junk."
        ),
        "entry": "parse_duration",
        "tests": [
            (
                "assert parse_duration('90m')==5400.0\n"
                "assert parse_duration('1h30m')==5400.0\n"
                "assert parse_duration('1.5h')==5400.0\n"
                "assert parse_duration('250ms')==0.25"
            ),
            (
                "assert parse_duration('-30s')==-30.0\n"
                "assert parse_duration('0s')==0.0\n"
                "assert parse_duration('1m500ms')==60.5"
            ),
            "assert parse_duration('1w2d3h4m5s')==7*86400+2*86400+3*3600+4*60+5",
            (
                "for bad in ['','10','h','1x','30s1m','1m1m','5s ',' 5s','1.2.3s','--5s','-','m"
                "1']:\n"
                "    try:\n"
                "        parse_duration(bad)\n"
                "        raise AssertionError('expected ValueError for %r' % bad)\n"
                "    except ValueError:\n"
                "        pass"
            ),
        ],
    },
    {
        "id": "parse_csv",
        "prompt": (
            "Implement `parse_csv(text)` parsing RFC 4180 CSV into a list of rows, each a list of "
            "field strings. Do not use the `csv` module.\n"
            "\n"
            "- Fields are comma-separated; records are separated by `\\n` or `\\r\\n`.\n"
            "- A field may be wrapped in double quotes, in which case it may contain commas, "
            "newlines, and `\"\"` meaning one literal `\"`.\n"
            "- Unquoted fields are taken literally: surrounding whitespace is significant and must "
            "NOT be stripped.\n"
            "- A single trailing record separator at the end of the input does not create an extra "
            "empty row.\n"
            "- Empty input returns `[]`.\n"
            "- An unterminated quoted field raises `ValueError`."
        ),
        "entry": "parse_csv",
        "tests": [
            (
                "assert parse_csv('a,b\\n1,2')==[['a','b'],['1','2']]\n"
                "assert parse_csv('')==[]\n"
                "assert parse_csv('a,b\\n')==[['a','b']]\n"
                "assert parse_csv('a,b\\r\\nc,d\\r\\n')==[['a','b'],['c','d']]"
            ),
            (
                "assert parse_csv('\"x,y\",z')==[['x,y','z']]\n"
                "assert parse_csv('\"he said \"\"hi\"\"\",z')==[['he said \"hi\"','z']]\n"
                "assert parse_csv('\"multi\\nline\",z')==[['multi\\nline','z']]"
            ),
            (
                "assert parse_csv('a,,b')==[['a','','b']]\n"
                "assert parse_csv(',')==[['','']]\n"
                "assert parse_csv(' a , b ')==[[' a ',' b ']]\n"
                "assert parse_csv('\"\",x')==[['','x']]"
            ),
            "assert parse_csv('\\n\\n')==[[''],['']]",
            (
                "try:\n"
                "    parse_csv('\"abc')\n"
                "    raise AssertionError('expected ValueError')\n"
                "except ValueError:\n"
                "    pass"
            ),
        ],
    },
    {
        "id": "glob_match",
        "prompt": (
            "Implement `glob_match(pattern, name)` returning True when `name` matches the shell-style "
            "`pattern` in full. Do not use `fnmatch`, `re`, or `glob`.\n"
            "\n"
            "- `*` matches any sequence of characters, including empty.\n"
            "- `?` matches exactly one character.\n"
            "- `[seq]` matches one character in `seq`; `[!seq]` matches one character not in `seq`; "
            "ranges like `a-z` are allowed inside the brackets; a literal `]` is allowed if it is the "
            "first character of the set.\n"
            "- A `[` with no closing `]` is a literal `[`.\n"
            "- Matching is case-sensitive and must cover the entire string."
        ),
        "entry": "glob_match",
        "tests": [
            (
                "assert glob_match('*.py','main.py') is True\n"
                "assert glob_match('*.py','main.pyc') is False\n"
                "assert glob_match('a?c','abc') is True\n"
                "assert glob_match('a?c','ac') is False"
            ),
            (
                "assert glob_match('[abc]x','bx') is True\n"
                "assert glob_match('[!abc]x','dx') is True\n"
                "assert glob_match('[!abc]x','ax') is False\n"
                "assert glob_match('[a-c]1','b1') is True\n"
                "assert glob_match('[a-c]1','d1') is False"
            ),
            (
                "assert glob_match('*','') is True\n"
                "assert glob_match('','') is True\n"
                "assert glob_match('','x') is False\n"
                "assert glob_match('**','ab') is True"
            ),
            (
                "assert glob_match('a*b*c','abc') is True\n"
                "assert glob_match('a*b*c','axxbyyc') is True\n"
                "assert glob_match('a*b*c','abcx') is False\n"
                "assert glob_match('*a*a*b','a'*25) is False"
            ),
            (
                "assert glob_match('[]]x',']x') is True\n"
                "assert glob_match('[abc','[abc') is True"
            ),
        ],
    },
    {
        "id": "semver_cmp",
        "prompt": (
            "Implement `semver_cmp(a, b)` returning -1, 0 or 1 comparing two Semantic Versioning "
            "2.0.0 strings by precedence.\n"
            "\n"
            "- `MAJOR.MINOR.PATCH` are compared numerically. Leading zeros are invalid.\n"
            "- An optional `-prerelease` follows, made of dot-separated identifiers. Build metadata "
            "after `+` is IGNORED for precedence.\n"
            "- A version WITH a prerelease has LOWER precedence than the same version without one.\n"
            "- Prerelease identifiers are compared left to right: numeric identifiers compare "
            "numerically and always rank LOWER than alphanumeric ones; alphanumeric identifiers "
            "compare by ASCII order; when all leading identifiers are equal, the version with MORE "
            "identifiers ranks higher.\n"
            "- A numeric prerelease identifier may not have leading zeros.\n"
            "- Raise `ValueError` for a malformed version string."
        ),
        "entry": "semver_cmp",
        "tests": [
            (
                "assert semver_cmp('1.0.0','1.0.1')==-1\n"
                "assert semver_cmp('1.0.0','1.0.0')==0\n"
                "assert semver_cmp('2.0.0','1.9.9')==1\n"
                "assert semver_cmp('1.0.10','1.0.9')==1"
            ),
            (
                "assert semver_cmp('1.0.0-alpha','1.0.0')==-1\n"
                "assert semver_cmp('1.0.0-alpha','1.0.0-alpha.1')==-1\n"
                "assert semver_cmp('1.0.0-alpha.1','1.0.0-alpha.beta')==-1\n"
                "assert semver_cmp('1.0.0-alpha.beta','1.0.0-beta')==-1"
            ),
            (
                "assert semver_cmp('1.0.0-beta','1.0.0-beta.2')==-1\n"
                "assert semver_cmp('1.0.0-beta.2','1.0.0-beta.11')==-1\n"
                "assert semver_cmp('1.0.0-beta.11','1.0.0-rc.1')==-1"
            ),
            (
                "assert semver_cmp('1.0.0+build.1','1.0.0+build.2')==0\n"
                "assert semver_cmp('1.0.0-alpha+a','1.0.0-alpha+b')==0"
            ),
            (
                "for bad in ['1.0','1.0.0.0','a.b.c','1.0.0-','01.0.0','1.0.0-01','1.0.0-al pha"
                "']:\n"
                "    try:\n"
                "        semver_cmp(bad,'1.0.0')\n"
                "        raise AssertionError('expected ValueError for %r' % bad)\n"
                "    except ValueError:\n"
                "        pass"
            ),
        ],
    },
    {
        "id": "retry_policy",
        "prompt": (
            "Implement a decorator factory `retry(attempts, exceptions=(Exception,), base_delay=1.0, "
            "sleep=time.sleep)`.\n"
            "\n"
            "- It retries the wrapped callable up to `attempts` TOTAL calls.\n"
            "- It retries only when the raised exception is an instance of one of `exceptions`; "
            "anything else propagates immediately with no sleep.\n"
            "- Between attempt i and attempt i+1 (i starting at 1) it calls `sleep(base_delay * 2 ** "
            "(i - 1))`, so delays go base, 2*base, 4*base. There is no sleep after the final attempt.\n"
            "- If every attempt fails it re-raises the last exception.\n"
            "- Positional and keyword arguments pass through, and the wrapper preserves the wrapped "
            "function's `__name__` and `__doc__`.\n"
            "- `attempts` below 1 raises `ValueError` at decoration time."
        ),
        "entry": "retry",
        "tests": [
            (
                "calls=[]\n"
                "slept=[]\n"
                "@retry(4, (ValueError,), base_delay=0.5, sleep=slept.append)\n"
                "def f():\n"
                "    calls.append(1)\n"
                "    if len(calls)<3:\n"
                "        raise ValueError('x')\n"
                "    return 'ok'\n"
                "assert f()=='ok' and len(calls)==3 and slept==[0.5,1.0], (calls, slept)"
            ),
            (
                "calls=[]\n"
                "slept=[]\n"
                "@retry(3, (ValueError,), base_delay=1.0, sleep=slept.append)\n"
                "def g():\n"
                "    calls.append(1)\n"
                "    raise ValueError('always')\n"
                "try:\n"
                "    g()\n"
                "    raise AssertionError('expected ValueError')\n"
                "except ValueError:\n"
                "    pass\n"
                "assert len(calls)==3 and slept==[1.0,2.0], (calls, slept)"
            ),
            (
                "calls=[]\n"
                "slept=[]\n"
                "@retry(3, (ValueError,), sleep=slept.append)\n"
                "def h():\n"
                "    calls.append(1)\n"
                "    raise KeyError('nope')\n"
                "try:\n"
                "    h()\n"
                "    raise AssertionError('expected KeyError')\n"
                "except KeyError:\n"
                "    pass\n"
                "assert len(calls)==1 and slept==[], (calls, slept)"
            ),
            (
                "@retry(2, (ValueError,), sleep=lambda d: None)\n"
                "def doc(a, b=1):\n"
                "    'docstring here'\n"
                "    return a+b\n"
                "assert doc(2, b=3)==5\n"
                "assert doc.__name__=='doc' and doc.__doc__=='docstring here'"
            ),
            (
                "try:\n"
                "    retry(0)\n"
                "    raise AssertionError('expected ValueError')\n"
                "except ValueError:\n"
                "    pass"
            ),
        ],
    },
    {
        "id": "flatten_json",
        "prompt": (
            "Implement `flatten_json(obj, sep='.')` flattening a nested JSON-like dict into a flat "
            "dict mapping a path string to a leaf value.\n"
            "\n"
            "- `obj` is always a dict. Nested dict keys join with `sep`; list elements contribute "
            "their integer index as a path component.\n"
            "- An EMPTY dict or EMPTY list is itself a leaf: it maps its own path to `{}` or `[]`.\n"
            "- `None`, booleans, numbers and strings are leaves.\n"
            "- Keys are used verbatim, so a key that already contains `sep` is not escaped."
        ),
        "entry": "flatten_json",
        "tests": [
            (
                "assert flatten_json({'a':1})=={'a':1}\n"
                "assert flatten_json({})=={}\n"
                "assert flatten_json({'a':{'b':{'c':1}}})=={'a.b.c':1}"
            ),
            (
                "assert flatten_json({'a':[1,2]})=={'a.0':1,'a.1':2}\n"
                "assert flatten_json({'a':[{'b':1},{'b':2}]})=={'a.0.b':1,'a.1.b':2}"
            ),
            (
                "assert flatten_json({'a':{}})=={'a':{}}\n"
                "assert flatten_json({'a':[]})=={'a':[]}\n"
                "assert flatten_json({'a':{'b':[]}})=={'a.b':[]}"
            ),
            (
                "assert flatten_json({'a':None,'b':True})=={'a':None,'b':True}\n"
                "assert flatten_json({'a':{'b':1}}, sep='/')=={'a/b':1}\n"
                "assert flatten_json({'a.b':1})=={'a.b':1}"
            ),
        ],
    },
    {
        "id": "edit_ops",
        "prompt": (
            "Implement `edit_ops(a, b)` returning the shortest sequence of single-character edits "
            "that turns string `a` into string `b`, as a list of tuples: `('keep', ch)`, `('sub', "
            "old, new)`, `('del', ch)`, `('ins', ch)`.\n"
            "\n"
            "- The sequence reads left to right over `a` and `b`.\n"
            "- Among equally short sequences, at each step prefer `keep`, then `sub`, then `del`, "
            "then `ins`.\n"
            "- Applying the ops in order must reproduce `b`, and the number of non-`keep` ops equals "
            "the Levenshtein distance."
        ),
        "entry": "edit_ops",
        "tests": [
            (
                "assert edit_ops('','')==[]\n"
                "assert edit_ops('abc','abc')==[('keep','a'),('keep','b'),('keep','c')]\n"
                "assert edit_ops('','ab')==[('ins','a'),('ins','b')]\n"
                "assert edit_ops('ab','')==[('del','a'),('del','b')]"
            ),
            (
                "assert edit_ops('kitten','sitting')==[('sub','k','s'),('keep','i'),('keep','t'"
                "),('keep','t'),('sub','e','i'),('keep','n'),('ins','g')]"
            ),
            (
                "ops=edit_ops('flaw','lawn')\n"
                "assert sum(1 for o in ops if o[0]!='keep')==2, ops\n"
                "out=[]\n"
                "for o in ops:\n"
                "    if o[0]=='keep':\n"
                "        out.append(o[1])\n"
                "    elif o[0]=='sub':\n"
                "        out.append(o[2])\n"
                "    elif o[0]=='ins':\n"
                "        out.append(o[1])\n"
                "assert ''.join(out)=='lawn', ops"
            ),
            (
                "ops=edit_ops('sunday','saturday')\n"
                "assert sum(1 for o in ops if o[0]!='keep')==3, ops\n"
                "src=''.join(o[1] for o in ops if o[0] in ('keep','sub','del'))\n"
                "assert src=='sunday', ops"
            ),
        ],
    },
    {
        "id": "ledger_txn",
        "prompt": (
            "Implement a class `Ledger` with transactional rollback.\n"
            "\n"
            "- `Ledger(balance=0)` exposes an integer `.balance`.\n"
            "- `deposit(n)` and `withdraw(n)` require a positive int `n`, else `ValueError`.\n"
            "- `withdraw` raises `Ledger.InsufficientFunds` (an `Exception` subclass you define as a "
            "class attribute) if it would make the balance negative, and changes nothing.\n"
            "- `.history` is a tuple of the applied operations, as `('deposit', n)` / `('withdraw', "
            "n)`.\n"
            "- `transaction()` returns a context manager. If the block raises, every operation "
            "performed inside it is rolled back -- balance and history restored -- and the exception "
            "propagates. If the block completes, the changes stand.\n"
            "- Transactions nest: an inner rollback undoes only the inner scope."
        ),
        "entry": "Ledger",
        "tests": [
            (
                "L=Ledger(100)\n"
                "L.deposit(50)\n"
                "assert L.balance==150\n"
                "try:\n"
                "    L.withdraw(1000)\n"
                "    raise AssertionError('expected InsufficientFunds')\n"
                "except Ledger.InsufficientFunds:\n"
                "    pass\n"
                "assert L.balance==150 and L.history==(('deposit',50),)"
            ),
            (
                "L=Ledger(10)\n"
                "try:\n"
                "    with L.transaction():\n"
                "        L.deposit(5)\n"
                "        raise RuntimeError('boom')\n"
                "except RuntimeError:\n"
                "    pass\n"
                "assert L.balance==10 and L.history==(), (L.balance, L.history)"
            ),
            (
                "L=Ledger(10)\n"
                "with L.transaction():\n"
                "    L.deposit(5)\n"
                "assert L.balance==15 and L.history==(('deposit',5),)"
            ),
            (
                "L=Ledger(0)\n"
                "with L.transaction():\n"
                "    L.deposit(10)\n"
                "    try:\n"
                "        with L.transaction():\n"
                "            L.deposit(5)\n"
                "            raise ValueError('inner')\n"
                "    except ValueError:\n"
                "        pass\n"
                "    L.deposit(1)\n"
                "assert L.balance==11 and L.history==(('deposit',10),('deposit',1)), (L.balance"
                ", L.history)"
            ),
            (
                "L=Ledger()\n"
                "for bad in [0,-1]:\n"
                "    try:\n"
                "        L.deposit(bad)\n"
                "        raise AssertionError('expected ValueError')\n"
                "    except ValueError:\n"
                "        pass\n"
                "assert L.balance==0 and L.history==()"
            ),
        ],
    },
    {
        "id": "trie_delete",
        "prompt": (
            "Implement a prefix tree class `Trie`.\n"
            "\n"
            "- `insert(word)` stores a word; the empty string is a valid word.\n"
            "- `search(word)` returns True only for a stored word (exact match).\n"
            "- `starts_with(prefix)` returns True if any stored word starts with `prefix`.\n"
            "- `delete(word)` removes a stored word and returns True, or returns False if it was not "
            "stored. After a delete, nodes that are no longer part of any stored word must be pruned, "
            "so `starts_with` stops reporting a prefix that no longer leads anywhere.\n"
            "- `words()` returns every stored word in sorted order."
        ),
        "entry": "Trie",
        "tests": [
            (
                "t=Trie()\n"
                "for w in ['app','apple','apply','bat']:\n"
                "    t.insert(w)\n"
                "assert t.search('app') is True and t.search('ap') is False\n"
                "assert t.starts_with('ap') is True and t.starts_with('ba') is True\n"
                "assert t.starts_with('cat') is False"
            ),
            (
                "t=Trie()\n"
                "for w in ['app','apple','apply','bat']:\n"
                "    t.insert(w)\n"
                "assert t.delete('apple') is True\n"
                "assert t.delete('apple') is False\n"
                "assert t.words()==['app','apply','bat'], t.words()\n"
                "assert t.search('apply') is True and t.search('app') is True"
            ),
            (
                "t=Trie()\n"
                "for w in ['app','apple','apply','bat']:\n"
                "    t.insert(w)\n"
                "assert t.delete('app') is True\n"
                "assert t.words()==['apple','apply','bat']\n"
                "assert t.starts_with('app') is True"
            ),
            (
                "t=Trie()\n"
                "t.insert('a')\n"
                "assert t.delete('a') is True and t.words()==[] and t.starts_with('a') is False\n"
                "assert Trie().delete('x') is False"
            ),
            (
                "t=Trie()\n"
                "t.insert('')\n"
                "assert t.search('') is True and t.words()==['']\n"
                "assert t.delete('') is True and t.search('') is False"
            ),
        ],
    },
    {
        "id": "lex_topo",
        "prompt": (
            "Implement `lex_topo(graph)`. `graph` maps a node to the list of nodes it depends on. "
            "Return the LEXICOGRAPHICALLY SMALLEST valid ordering -- dependencies before dependents "
            "-- comparing candidate orderings as lists of nodes.\n"
            "\n"
            "- A node that appears only as a dependency is still a node.\n"
            "- Raise `ValueError` if the graph contains a cycle."
        ),
        "entry": "lex_topo",
        "tests": [
            (
                "assert lex_topo({'a':[], 'b':[], 'c':['a','b']})==['a','b','c']\n"
                "assert lex_topo({'b':[], 'a':['b']})==['b','a']\n"
                "assert lex_topo({'z':[], 'y':[], 'x':[]})==['x','y','z']\n"
                "assert lex_topo({})==[]"
            ),
            "assert lex_topo({'d':['a'],'c':['a'],'b':[],'a':[]})==['a','b','c','d']",
            "assert lex_topo({'a':['b'],'z':[],'b':[]})==['b','a','z']",
            (
                "try:\n"
                "    lex_topo({'a':['b'],'b':['a']})\n"
                "    raise AssertionError('expected ValueError')\n"
                "except ValueError:\n"
                "    pass"
            ),
        ],
    },
    {
        "id": "diff_lines",
        "prompt": (
            "Implement `diff_lines(a, b)` where `a` and `b` are lists of strings. Return a list of "
            "`(tag, line)` tuples with `tag` in `'='`, `'-'`, `'+'`, forming a minimal edit script "
            "built on the longest common subsequence of lines.\n"
            "\n"
            "- Reading the `'='` and `'-'` entries in order reproduces `a`; reading `'='` and `'+'` "
            "reproduces `b`.\n"
            "- Where a choice exists, emit deletions before insertions, and prefer keeping the "
            "EARLIEST possible common lines."
        ),
        "entry": "diff_lines",
        "tests": [
            (
                "assert diff_lines([],[])==[]\n"
                "assert diff_lines(['x'],['x'])==[('=','x')]\n"
                "assert diff_lines([],['a','b'])==[('+','a'),('+','b')]\n"
                "assert diff_lines(['a','b'],[])==[('-','a'),('-','b')]"
            ),
            (
                "assert diff_lines(['a','b','c'],['a','c'])==[('=','a'),('-','b'),('=','c')]\n"
                "assert diff_lines(['a','c'],['a','b','c'])==[('=','a'),('+','b'),('=','c')]\n"
                "assert diff_lines(['a'],['b'])==[('-','a'),('+','b')]"
            ),
            (
                "assert diff_lines(['a','b','c','d'],['b','d','e'])==[('-','a'),('=','b'),('-',"
                "'c'),('=','d'),('+','e')]"
            ),
            (
                "d=diff_lines(['the','quick','brown','fox'],['the','slow','brown','cat'])\n"
                "assert [x[1] for x in d if x[0] in '=-']==['the','quick','brown','fox'], d\n"
                "assert [x[1] for x in d if x[0] in '=+']==['the','slow','brown','cat'], d\n"
                "assert sum(1 for x in d if x[0]!='=')==4, d"
            ),
        ],
    },
    {
        "id": "num_to_words",
        "prompt": (
            "Implement `num_to_words(n)` converting an integer to lowercase American English words.\n"
            "\n"
            "- Valid range is `-10**12 < n < 10**12`; anything else raises `ValueError`.\n"
            "- No 'and' anywhere. Tens from 21 to 99 are hyphenated, e.g. `twenty-one`.\n"
            "- Groups are separated by single spaces, e.g. `1234567` -> `one million two hundred "
            "thirty-four thousand five hundred sixty-seven`.\n"
            "- Zero is `zero`; negatives are prefixed with `negative `."
        ),
        "entry": "num_to_words",
        "tests": [
            (
                "assert num_to_words(0)=='zero'\n"
                "assert num_to_words(7)=='seven'\n"
                "assert num_to_words(13)=='thirteen'\n"
                "assert num_to_words(20)=='twenty'\n"
                "assert num_to_words(21)=='twenty-one'"
            ),
            (
                "assert num_to_words(100)=='one hundred'\n"
                "assert num_to_words(101)=='one hundred one'\n"
                "assert num_to_words(999)=='nine hundred ninety-nine'\n"
                "assert num_to_words(1000)=='one thousand'\n"
                "assert num_to_words(1000000)=='one million'"
            ),
            (
                "assert num_to_words(1234567)=='one million two hundred thirty-four thousand fi"
                "ve hundred sixty-seven'\n"
                "assert num_to_words(1000010)=='one million ten'\n"
                "assert num_to_words(1000000000)=='one billion'"
            ),
            (
                "assert num_to_words(-42)=='negative forty-two'\n"
                "assert num_to_words(999999999999)=='nine hundred ninety-nine billion nine hund"
                "red ninety-nine million nine hundred ninety-nine thousand nine hundred ninety-"
                "nine'"
            ),
            (
                "for bad in [10**12, -10**12]:\n"
                "    try:\n"
                "        num_to_words(bad)\n"
                "        raise AssertionError('expected ValueError')\n"
                "    except ValueError:\n"
                "        pass"
            ),
        ],
    },
    {
        "id": "base_convert",
        "prompt": (
            "Implement `base_convert(s, from_base, to_base)` converting the numeral string `s` from "
            "`from_base` to `to_base`. Do not use `int(s, base)`.\n"
            "\n"
            "- Bases run from 2 to 36 inclusive; digits are `0-9a-z`, case-insensitive on input and "
            "LOWERCASE on output.\n"
            "- A single leading `-` is allowed. Leading zeros in the input are allowed but must not "
            "appear in the output, and negative zero normalises to `'0'`.\n"
            "- Raise `ValueError` for a base out of range, an empty numeral, a digit not valid in "
            "`from_base`, or a misplaced sign."
        ),
        "entry": "base_convert",
        "tests": [
            (
                "assert base_convert('255',10,16)=='ff'\n"
                "assert base_convert('ff',16,10)=='255'\n"
                "assert base_convert('FF',16,2)=='11111111'\n"
                "assert base_convert('1010',2,10)=='10'"
            ),
            (
                "assert base_convert('0',10,2)=='0'\n"
                "assert base_convert('0000',10,16)=='0'\n"
                "assert base_convert('-0',10,10)=='0'\n"
                "assert base_convert('7',8,8)=='7'"
            ),
            (
                "assert base_convert('-1a',16,10)=='-26'\n"
                "assert base_convert('zz',36,10)=='1295'\n"
                "assert base_convert('1295',10,36)=='zz'"
            ),
            (
                "for bad in [('',10,2),('12',1,10),('12',10,37),('19',8,10),('-',10,10),('1-2',"
                "10,10),('g',16,10)]:\n"
                "    try:\n"
                "        base_convert(*bad)\n"
                "        raise AssertionError('expected ValueError for %r' % (bad,))\n"
                "    except ValueError:\n"
                "        pass"
            ),
        ],
    },
    {
        "id": "simplify_path",
        "prompt": (
            "Implement `simplify_path(path)` normalising a POSIX-style path LEXICALLY. Do not touch "
            "the filesystem and do not use `os.path`.\n"
            "\n"
            "- Collapse repeated slashes and resolve `.` and `..` textually.\n"
            "- For an ABSOLUTE path (leading `/`), a `..` that would escape the root is dropped; the "
            "result always starts with `/` and never ends with a trailing slash, except the root "
            "itself which is `'/'`.\n"
            "- For a RELATIVE path, leading `..` components are preserved, and a result that reduces "
            "to nothing is `'.'`.\n"
            "- An empty string raises `ValueError`."
        ),
        "entry": "simplify_path",
        "tests": [
            (
                "assert simplify_path('/home/')=='/home'\n"
                "assert simplify_path('/a/./b/../../c/')=='/c'\n"
                "assert simplify_path('/home//foo/')=='/home/foo'\n"
                "assert simplify_path('/')=='/'"
            ),
            (
                "assert simplify_path('/../')=='/'\n"
                "assert simplify_path('/a/../..')=='/'"
            ),
            (
                "assert simplify_path('a/b/../c')=='a/c'\n"
                "assert simplify_path('../a')=='../a'\n"
                "assert simplify_path('a/..')=='.'\n"
                "assert simplify_path('.')=='.'\n"
                "assert simplify_path('./a/./')=='a'"
            ),
            (
                "assert simplify_path('../../x/..')=='../..'\n"
                "assert simplify_path('a/../../b')=='../b'"
            ),
            (
                "try:\n"
                "    simplify_path('')\n"
                "    raise AssertionError('expected ValueError')\n"
                "except ValueError:\n"
                "    pass"
            ),
        ],
    },
    {
        "id": "json_pointer",
        "prompt": (
            "Implement `json_pointer(doc, pointer)` resolving an RFC 6901 JSON Pointer against `doc` "
            "(nested dicts and lists).\n"
            "\n"
            "- `''` returns the whole document.\n"
            "- Otherwise the pointer must start with `/`; tokens are the `/`-separated parts. In each "
            "token, decode `~1` to `/` FIRST and then `~0` to `~` -- the order matters.\n"
            "- A list index must be a non-negative decimal integer with no leading zeros (except "
            "`'0'` itself); `'-'` never resolves.\n"
            "- Raise `KeyError` for a missing dict key, `IndexError` for a bad or out-of-range list "
            "index, and `ValueError` if the pointer is malformed or a token is applied to a scalar."
        ),
        "entry": "json_pointer",
        "tests": [
            (
                "doc={'foo':['bar','baz'],'':0,'a/b':1,'m~n':8,'nested':{'x':[{'y':9}]}}\n"
                "assert json_pointer(doc,'') is doc\n"
                "assert json_pointer(doc,'/foo')==['bar','baz']\n"
                "assert json_pointer(doc,'/foo/0')=='bar'\n"
                "assert json_pointer(doc,'/foo/1')=='baz'\n"
                "assert json_pointer(doc,'/')==0"
            ),
            (
                "doc={'foo':['bar','baz'],'':0,'a/b':1,'m~n':8,'nested':{'x':[{'y':9}]}}\n"
                "assert json_pointer(doc,'/a~1b')==1\n"
                "assert json_pointer(doc,'/m~0n')==8\n"
                "assert json_pointer(doc,'/nested/x/0/y')==9\n"
                "assert json_pointer({'~1':'t'},'/~01')=='t'"
            ),
            (
                "doc={'foo':['bar','baz']}\n"
                "try:\n"
                "    json_pointer(doc,'/nope')\n"
                "    raise AssertionError('expected KeyError')\n"
                "except KeyError:\n"
                "    pass"
            ),
            (
                "doc={'foo':['bar','baz']}\n"
                "for p in ['/foo/9','/foo/01','/foo/-','/foo/x']:\n"
                "    try:\n"
                "        json_pointer(doc,p)\n"
                "        raise AssertionError('expected IndexError for %r' % p)\n"
                "    except IndexError:\n"
                "        pass"
            ),
            (
                "doc={'foo':['bar','baz']}\n"
                "for p in ['foo','/foo/0/x']:\n"
                "    try:\n"
                "        json_pointer(doc,p)\n"
                "        raise AssertionError('expected ValueError for %r' % p)\n"
                "    except ValueError:\n"
                "        pass"
            ),
        ],
    },
    {
        "id": "regex_match",
        "prompt": (
            "Implement `regex_match(pattern, text)` performing FULL-string matching for a minimal "
            "regex language. Do not use the `re` module.\n"
            "\n"
            "- `.` matches any single character.\n"
            "- `*` matches zero or more of the single preceding element (a literal character or `.`).\n"
            "- Every other character is a literal.\n"
            "- Raise `ValueError` if `*` has nothing to repeat, i.e. a pattern starting with `*` or "
            "containing `**`."
        ),
        "entry": "regex_match",
        "tests": [
            (
                "assert regex_match('','') is True\n"
                "assert regex_match('','a') is False\n"
                "assert regex_match('a','a') is True\n"
                "assert regex_match('.','a') is True\n"
                "assert regex_match('.','') is False"
            ),
            (
                "assert regex_match('a*','') is True\n"
                "assert regex_match('a*','aaa') is True\n"
                "assert regex_match('a*','aab') is False\n"
                "assert regex_match('.*','anything') is True"
            ),
            (
                "assert regex_match('ab*c','ac') is True\n"
                "assert regex_match('ab*c','abbbc') is True\n"
                "assert regex_match('mis*is*p*.','mississippi') is False\n"
                "assert regex_match('mis*is*ip*.','mississippi') is True"
            ),
            (
                "assert regex_match('a*a*a*b','a'*21+'b') is True\n"
                "assert regex_match('a*a*a*a*b','a'*21) is False"
            ),
            (
                "for bad in ['*a','a**']:\n"
                "    try:\n"
                "        regex_match(bad,'a')\n"
                "        raise AssertionError('expected ValueError for %r' % bad)\n"
                "    except ValueError:\n"
                "        pass"
            ),
        ],
    },
    {
        "id": "chunk_utf8",
        "prompt": (
            "Implement `chunk_utf8(text, max_bytes)` splitting a string into a list of chunks such "
            "that each chunk's UTF-8 encoding is at most `max_bytes` bytes and no character is ever "
            "split across chunks.\n"
            "\n"
            "- Chunks are greedy: each one is as long as it can be.\n"
            "- Concatenating the chunks reproduces `text` exactly, and empty text returns `[]`.\n"
            "- Raise `ValueError` if `max_bytes` is below 4, since a single UTF-8 character can be 4 "
            "bytes."
        ),
        "entry": "chunk_utf8",
        "tests": [
            (
                "assert chunk_utf8('',10)==[]\n"
                "assert chunk_utf8('abcdef',4)==['abcd','ef']"
            ),
            (
                "out=chunk_utf8('h\\u00e9llo w\\u00f6rld',5)\n"
                "assert ''.join(out)=='h\\u00e9llo w\\u00f6rld'\n"
                "assert all(len(c.encode('utf-8'))<=5 for c in out), out"
            ),
            (
                "assert chunk_utf8('\\u00e9'*3,4)==['\\u00e9\\u00e9','\\u00e9']\n"
                "assert chunk_utf8('\\u65e5\\u672c\\u8a9e',4)==['\\u65e5','\\u672c','\\u8a9e']\n"
                "assert chunk_utf8('a\\u65e5',4)==['a\\u65e5']"
            ),
            "assert chunk_utf8('\\U0001f600\\U0001f600',4)==['\\U0001f600','\\U0001f600']",
            (
                "for bad in [3,0,-1]:\n"
                "    try:\n"
                "        chunk_utf8('x',bad)\n"
                "        raise AssertionError('expected ValueError for %r' % bad)\n"
                "    except ValueError:\n"
                "        pass"
            ),
        ],
    },
    {
        "id": "count_inversions",
        "prompt": (
            "Implement `count_inversions(arr)` returning the number of pairs `(i, j)` with `i < j` "
            "and `arr[i] > arr[j]`.\n"
            "\n"
            "- It must run in O(n log n). The input can hold 100000 elements, so a nested loop over "
            "all pairs will not finish in time."
        ),
        "entry": "count_inversions",
        "tests": [
            (
                "assert count_inversions([])==0\n"
                "assert count_inversions([1,2,3])==0\n"
                "assert count_inversions([3,2,1])==3\n"
                "assert count_inversions([2,4,1,3,5])==3\n"
                "assert count_inversions([1,1,1])==0"
            ),
            (
                "import random\n"
                "rnd=random.Random(11)\n"
                "a=[rnd.randint(0,1000) for _ in range(300)]\n"
                "brute=sum(1 for i in range(len(a)) for j in range(i+1,len(a)) if a[i]>a[j])\n"
                "assert count_inversions(a)==brute"
            ),
            (
                "big=list(range(100000,0,-1))\n"
                "assert count_inversions(big)==100000*99999//2"
            ),
        ],
    },
    {
        "id": "shortest_path",
        "prompt": (
            "Implement `shortest_path(graph, src, dst)` where `graph` maps a node to a dict of "
            "neighbour -> non-negative edge weight.\n"
            "\n"
            "- Return `(cost, path)` where `path` lists the nodes from `src` to `dst` inclusive.\n"
            "- If several shortest paths tie on cost, return the lexicographically smallest path "
            "(comparing the node lists).\n"
            "- If `dst` is unreachable return `(None, [])`, and `shortest_path(g, x, x)` is `(0, "
            "[x])`.\n"
            "- Raise `ValueError` if `src` or `dst` is not a node of the graph, or if any edge weight "
            "is negative."
        ),
        "entry": "shortest_path",
        "tests": [
            (
                "g={'a':{'b':1,'c':4},'b':{'c':2,'d':5},'c':{'d':1},'d':{}}\n"
                "assert shortest_path(g,'a','d')==(4,['a','b','c','d'])\n"
                "assert shortest_path(g,'a','a')==(0,['a'])\n"
                "assert shortest_path(g,'d','a')==(None,[])"
            ),
            (
                "g2={'a':{'b':1,'c':1},'b':{'d':1},'c':{'d':1},'d':{}}\n"
                "assert shortest_path(g2,'a','d')==(2,['a','b','d'])"
            ),
            (
                "g3={'x':{'y':1},'y':{},'z':{}}\n"
                "assert shortest_path(g3,'x','z')==(None,[])\n"
                "try:\n"
                "    shortest_path(g3,'x','q')\n"
                "    raise AssertionError('expected ValueError')\n"
                "except ValueError:\n"
                "    pass"
            ),
            (
                "try:\n"
                "    shortest_path({'a':{'b':-1},'b':{}},'a','b')\n"
                "    raise AssertionError('expected ValueError')\n"
                "except ValueError:\n"
                "    pass"
            ),
        ],
    },
    {
        "id": "free_slots",
        "prompt": (
            "Implement `free_slots(busy_a, busy_b, work_window, duration)`.\n"
            "\n"
            "- `busy_a` and `busy_b` are lists of `[start, end]` integer-minute intervals, possibly "
            "unsorted and overlapping.\n"
            "- `work_window` is a single `[start, end]` interval and `duration` is a positive int.\n"
            "- Return every MAXIMAL window `[start, end]` inside `work_window` during which both "
            "people are free and whose length is at least `duration`, sorted by start.\n"
            "- Intervals are half-open, so `[0,30]` and `[30,60]` do not overlap.\n"
            "- Raise `ValueError` if `duration` is not positive or if the window start is not before "
            "its end."
        ),
        "entry": "free_slots",
        "tests": [
            (
                "a=[[0,30],[60,90]]\n"
                "b=[[15,45]]\n"
                "assert free_slots(a,b,[0,120],10)==[[45,60],[90,120]]\n"
                "assert free_slots(a,b,[0,120],20)==[[90,120]]"
            ),
            (
                "assert free_slots([],[],[0,60],60)==[[0,60]]\n"
                "assert free_slots([[0,60]],[],[0,60],1)==[]"
            ),
            (
                "assert free_slots([[10,20],[15,25]],[],[0,40],5)==[[0,10],[25,40]]\n"
                "assert free_slots([[100,200]],[[0,50]],[30,150],10)==[[50,100]]"
            ),
            (
                "for bad in [([],[],[0,60],0), ([],[],[60,0],5), ([],[],[5,5],5)]:\n"
                "    try:\n"
                "        free_slots(*bad)\n"
                "        raise AssertionError('expected ValueError for %r' % (bad,))\n"
                "    except ValueError:\n"
                "        pass"
            ),
        ],
    },
]
