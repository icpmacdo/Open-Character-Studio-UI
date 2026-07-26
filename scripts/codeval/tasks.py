"""The codeval task registry: two executable tiers plus the qualitative set.

Executable tasks name an entry point so grading is objective (does the code
parse, does it pass the hidden tests). Qualitative tasks have no runnable answer
and are scored for register/leakage only.

    CEILING CONTROL (this file, 20 tasks)
        Interview-standard problems. Base Inkling scored 96.7% pass@1 on these,
        which is a ceiling: there is no room below it to observe a regression, so
        these CANNOT answer "did character training cost coding ability". They
        are retained for one job only -- proving the model under test is not
        simply broken. A trained arm that collapses here has a gross problem; a
        trained arm that matches base here has told you nothing about capability.

    HARD (tasks_hard.py, 30 tasks)
        The tier the degradation estimate actually runs on, calibrated for a
        40-70% base pass@1. See README.md "Pre-registration".

Importing this module requires the codeval directory on ``sys.path`` (the same
convention ``run_sample.py`` and ``grade.py`` already use).
"""

from tasks_hard import HARD_TASKS

CEILING_TASKS = [
    {
        "id": "two_sum",
        "prompt": "Implement a Python function `two_sum(nums, target)` that returns the indices of the two numbers in `nums` that add up to `target`, as a tuple `(i, j)` with `i < j`. Exactly one solution exists.",
        "entry": "two_sum",
        "tests": [
            "assert tuple(two_sum([2,7,11,15], 9)) == (0,1)",
            "assert tuple(two_sum([3,2,4], 6)) == (1,2)",
            "assert tuple(two_sum([3,3], 6)) == (0,1)",
            "assert tuple(two_sum([-1,-2,-3,-4,-5], -8)) == (2,4)",
        ],
    },
    {
        "id": "is_balanced",
        "prompt": "Implement a Python function `is_balanced(s)` that returns True if the brackets in the string `s` are balanced. It must handle `()`, `[]`, and `{}`, and ignore all other characters.",
        "entry": "is_balanced",
        "tests": [
            "assert is_balanced('()') is True",
            "assert is_balanced('([{}])') is True",
            "assert is_balanced('(]') is False",
            "assert is_balanced('([)]') is False",
            "assert is_balanced('') is True",
            "assert is_balanced('a(b)c[d]') is True",
            "assert is_balanced('(') is False",
        ],
    },
    {
        "id": "merge_intervals",
        "prompt": "Implement a Python function `merge_intervals(intervals)` that takes a list of `[start, end]` pairs and returns a new list with all overlapping intervals merged, sorted by start.",
        "entry": "merge_intervals",
        "tests": [
            "assert [list(x) for x in merge_intervals([[1,3],[2,6],[8,10],[15,18]])] == [[1,6],[8,10],[15,18]]",
            "assert [list(x) for x in merge_intervals([[1,4],[4,5]])] == [[1,5]]",
            "assert [list(x) for x in merge_intervals([])] == []",
            "assert [list(x) for x in merge_intervals([[5,6],[1,2]])] == [[1,2],[5,6]]",
        ],
    },
    {
        "id": "longest_common_prefix",
        "prompt": "Implement a Python function `longest_common_prefix(strs)` that returns the longest common prefix string among a list of strings, or an empty string if there is none.",
        "entry": "longest_common_prefix",
        "tests": [
            "assert longest_common_prefix(['flower','flow','flight']) == 'fl'",
            "assert longest_common_prefix(['dog','racecar','car']) == ''",
            "assert longest_common_prefix([]) == ''",
            "assert longest_common_prefix(['abc']) == 'abc'",
            "assert longest_common_prefix(['','abc']) == ''",
        ],
    },
    {
        "id": "group_anagrams",
        "prompt": "Implement a Python function `group_anagrams(words)` that groups a list of words into lists of anagrams. Return a list of groups; each group sorted alphabetically, and the outer list sorted by the first element of each group.",
        "entry": "group_anagrams",
        "tests": [
            "assert group_anagrams(['eat','tea','tan','ate','nat','bat']) == [['ate','eat','tea'],['bat'],['nat','tan']]",
            "assert group_anagrams([]) == []",
            "assert group_anagrams(['a']) == [['a']]",
        ],
    },
    {
        "id": "binary_search",
        "prompt": "Implement a Python function `binary_search(arr, target)` that returns the index of `target` in the sorted list `arr`, or -1 if not present.",
        "entry": "binary_search",
        "tests": [
            "assert binary_search([1,3,5,7,9], 7) == 3",
            "assert binary_search([1,3,5,7,9], 1) == 0",
            "assert binary_search([1,3,5,7,9], 10) == -1",
            "assert binary_search([], 1) == -1",
            "assert binary_search([5], 5) == 0",
        ],
    },
    {
        "id": "rotate_matrix",
        "prompt": "Implement a Python function `rotate_matrix(m)` that rotates an n x n matrix (list of lists) 90 degrees clockwise and returns the result.",
        "entry": "rotate_matrix",
        "tests": [
            "assert [list(r) for r in rotate_matrix([[1,2],[3,4]])] == [[3,1],[4,2]]",
            "assert [list(r) for r in rotate_matrix([[1,2,3],[4,5,6],[7,8,9]])] == [[7,4,1],[8,5,2],[9,6,3]]",
            "assert [list(r) for r in rotate_matrix([[1]])] == [[1]]",
        ],
    },
    {
        "id": "word_frequency",
        "prompt": "Implement a Python function `word_frequency(text, n)` that returns the `n` most common words in `text` as a list of `(word, count)` tuples, sorted by count descending then word ascending. Words are case-insensitive and split on non-alphanumeric characters.",
        "entry": "word_frequency",
        "tests": [
            "assert word_frequency('the cat the dog THE bird cat', 2) == [('the',3),('cat',2)]",
            "assert word_frequency('a b c', 5) == [('a',1),('b',1),('c',1)]",
            "assert word_frequency('', 3) == []",
        ],
    },
    {
        "id": "flatten",
        "prompt": "Implement a Python function `flatten(nested)` that fully flattens an arbitrarily nested list of integers into a single flat list, preserving order.",
        "entry": "flatten",
        "tests": [
            "assert flatten([1,[2,[3,[4]]],5]) == [1,2,3,4,5]",
            "assert flatten([]) == []",
            "assert flatten([[],[[]]]) == []",
            "assert flatten([1,2,3]) == [1,2,3]",
        ],
    },
    {
        "id": "roman_to_int",
        "prompt": "Implement a Python function `roman_to_int(s)` that converts a Roman numeral string to an integer. Handle subtractive notation (IV, IX, XL, XC, CD, CM).",
        "entry": "roman_to_int",
        "tests": [
            "assert roman_to_int('III') == 3",
            "assert roman_to_int('IV') == 4",
            "assert roman_to_int('IX') == 9",
            "assert roman_to_int('LVIII') == 58",
            "assert roman_to_int('MCMXCIV') == 1994",
        ],
    },
    {
        "id": "valid_ipv4",
        "prompt": "Implement a Python function `valid_ipv4(s)` that returns True if `s` is a valid IPv4 address: four decimal octets 0-255 separated by dots, with no leading zeros (except '0' itself).",
        "entry": "valid_ipv4",
        "tests": [
            "assert valid_ipv4('192.168.1.1') is True",
            "assert valid_ipv4('0.0.0.0') is True",
            "assert valid_ipv4('255.255.255.255') is True",
            "assert valid_ipv4('256.1.1.1') is False",
            "assert valid_ipv4('01.1.1.1') is False",
            "assert valid_ipv4('1.1.1') is False",
            "assert valid_ipv4('1.1.1.1.1') is False",
            "assert valid_ipv4('a.b.c.d') is False",
        ],
    },
    {
        "id": "quicksort",
        "prompt": "Implement a Python function `quicksort(arr)` that sorts a list of numbers using the quicksort algorithm and returns a new sorted list. Do not use the built-in sort.",
        "entry": "quicksort",
        "tests": [
            "assert quicksort([3,1,2]) == [1,2,3]",
            "assert quicksort([]) == []",
            "assert quicksort([5,5,5]) == [5,5,5]",
            "import random; r=[random.randint(-50,50) for _ in range(60)]; assert quicksort(list(r)) == sorted(r)",
        ],
    },
    {
        "id": "max_subarray",
        "prompt": "Implement a Python function `max_subarray(nums)` that returns the largest sum of any contiguous subarray of `nums` (Kadane's algorithm). The array is non-empty and may be all negative.",
        "entry": "max_subarray",
        "tests": [
            "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6",
            "assert max_subarray([1]) == 1",
            "assert max_subarray([-3,-1,-2]) == -1",
            "assert max_subarray([5,4,-1,7,8]) == 23",
        ],
    },
    {
        "id": "count_islands",
        "prompt": "Implement a Python function `count_islands(grid)` that counts connected groups of 1s (4-directionally connected) in a 2D grid of 0s and 1s.",
        "entry": "count_islands",
        "tests": [
            "assert count_islands([[1,1,0],[0,1,0],[0,0,1]]) == 2",
            "assert count_islands([[0,0],[0,0]]) == 0",
            "assert count_islands([]) == 0",
            "assert count_islands([[1,0,1],[0,0,0],[1,0,1]]) == 4",
        ],
    },
    {
        "id": "spiral_order",
        "prompt": "Implement a Python function `spiral_order(matrix)` that returns all elements of a 2D matrix in spiral (clockwise, starting top-left) order as a flat list.",
        "entry": "spiral_order",
        "tests": [
            "assert list(spiral_order([[1,2,3],[4,5,6],[7,8,9]])) == [1,2,3,6,9,8,7,4,5]",
            "assert list(spiral_order([[1,2],[3,4]])) == [1,2,4,3]",
            "assert list(spiral_order([])) == []",
            "assert list(spiral_order([[1,2,3]])) == [1,2,3]",
        ],
    },
    {
        "id": "levenshtein",
        "prompt": "Implement a Python function `levenshtein(a, b)` that returns the edit distance between two strings (insertions, deletions, substitutions each cost 1).",
        "entry": "levenshtein",
        "tests": [
            "assert levenshtein('kitten','sitting') == 3",
            "assert levenshtein('','abc') == 3",
            "assert levenshtein('same','same') == 0",
            "assert levenshtein('flaw','lawn') == 2",
        ],
    },
    {
        "id": "topo_sort",
        "prompt": "Implement a Python function `topo_sort(graph)` that takes a dict mapping node -> list of nodes it depends on, and returns a list of nodes in dependency order (dependencies first). Raise ValueError if there is a cycle.",
        "entry": "topo_sort",
        "tests": [
            "r = topo_sort({'a':[], 'b':['a'], 'c':['b']}); assert r.index('a') < r.index('b') < r.index('c')",
            "r = topo_sort({'a':[], 'b':['a'], 'c':['a']}); assert r.index('a') == 0 and len(r) == 3",
            "try:\n    topo_sort({'a':['b'], 'b':['a']})\n    raise AssertionError('expected ValueError')\nexcept ValueError:\n    pass",
        ],
    },
    {
        "id": "parse_query",
        "prompt": "Implement a Python function `parse_query(qs)` that parses a URL query string like 'a=1&b=2&a=3' into a dict mapping key -> list of values, preserving order of appearance.",
        "entry": "parse_query",
        "tests": [
            "assert parse_query('a=1&b=2&a=3') == {'a':['1','3'],'b':['2']}",
            "assert parse_query('') == {}",
            "assert parse_query('x=') == {'x':['']}",
        ],
    },
    {
        "id": "chunk_list",
        "prompt": "Implement a Python function `chunk_list(items, size)` that splits a list into consecutive chunks of length `size` (the final chunk may be shorter). Raise ValueError if `size` is less than 1.",
        "entry": "chunk_list",
        "tests": [
            "assert [list(c) for c in chunk_list([1,2,3,4,5], 2)] == [[1,2],[3,4],[5]]",
            "assert [list(c) for c in chunk_list([], 3)] == []",
            "assert [list(c) for c in chunk_list([1,2,3], 5)] == [[1,2,3]]",
            "try:\n    list(chunk_list([1], 0))\n    raise AssertionError('expected ValueError')\nexcept ValueError:\n    pass",
        ],
    },
    {
        "id": "retry_decorator",
        "prompt": "Implement a Python decorator `retry(times)` that retries the wrapped function up to `times` total attempts if it raises an exception, re-raising the last exception if all attempts fail. It must preserve the wrapped function's return value on success.",
        "entry": "retry",
        "tests": [
            "calls = []\n@retry(3)\ndef f():\n    calls.append(1)\n    if len(calls) < 3:\n        raise ValueError('boom')\n    return 'ok'\nassert f() == 'ok' and len(calls) == 3",
            "n = []\n@retry(2)\ndef g():\n    n.append(1)\n    raise RuntimeError('always')\ntry:\n    g()\n    raise AssertionError('expected RuntimeError')\nexcept RuntimeError:\n    pass\nassert len(n) == 2",
        ],
    },
]

QUAL_TASKS = [
    {
        "id": "review_sql_injection",
        "prompt": "Review this code and tell me what's wrong with it:\n\n```python\ndef get_user(conn, name):\n    q = \"SELECT * FROM users WHERE name = '\" + name + \"'\"\n    return conn.execute(q).fetchall()\n```",
    },
    {
        "id": "debug_offbyone",
        "prompt": "This function is supposed to return the last element but it crashes. Why?\n\n```python\ndef last(xs):\n    return xs[len(xs)]\n```",
    },
    {
        "id": "explain_gil",
        "prompt": "Explain Python's GIL and when it actually matters for performance.",
    },
    {
        "id": "sql_query",
        "prompt": "Write a SQL query that returns the top 5 customers by total order value in 2024, from tables `customers(id, name)` and `orders(id, customer_id, total, created_at)`.",
    },
    {
        "id": "regex_email",
        "prompt": "Write a regex that matches an email address, and explain each part of it.",
    },
    {
        "id": "git_conflict",
        "prompt": "I have a merge conflict in git and I'm not sure how to resolve it safely. Walk me through it.",
    },
    {
        "id": "design_ratelimit",
        "prompt": "Design a rate limiter for an API that allows 100 requests per minute per user. Describe the approach and the tradeoffs.",
    },
    {
        "id": "refactor_nested",
        "prompt": "Refactor this to be more readable:\n\n```python\ndef process(data):\n    result = []\n    for item in data:\n        if item is not None:\n            if 'value' in item:\n                if item['value'] > 0:\n                    result.append(item['value'] * 2)\n    return result\n```",
    },
    {
        "id": "bash_oneliner",
        "prompt": "Give me a bash one-liner that finds the 10 largest files under the current directory.",
    },
    {
        "id": "test_advice",
        "prompt": "What should I actually test in a function that parses user-uploaded CSV files?",
    },
    {
        "id": "async_vs_thread",
        "prompt": "When should I use asyncio versus threads in Python?",
    },
    {
        "id": "prod_incident",
        "prompt": "Production is down, the error is 'connection pool exhausted'. What do I check first?",
    },
]

# Executable tiers, in the order they are reported. "ceiling" is the not-broken
# control; "hard" carries the estimate.
TIERS = {"ceiling": CEILING_TASKS, "hard": HARD_TASKS}

for _tier, _group in TIERS.items():
    for _task in _group:
        _task["tier"] = _tier
del _tier, _group, _task

# Every executable task, both tiers. Consumers that only need "is this task id
# executable, and what are its hidden tests" (grade.py) want this one.
EXEC_TASKS = CEILING_TASKS + HARD_TASKS


def exec_tasks_for(tiers):
    """Executable tasks for the named tiers, in TIERS order."""
    unknown = [t for t in tiers if t not in TIERS]
    if unknown:
        raise ValueError(f"unknown tier(s) {', '.join(unknown)}; choose from {', '.join(TIERS)}")
    return [t for tier, group in TIERS.items() if tier in tiers for t in group]
