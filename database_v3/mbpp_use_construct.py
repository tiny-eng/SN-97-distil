#!/usr/bin/env python3

import argparse
import json
import random
import sys
import traceback
from pathlib import Path


MBPP_DATABASE_VERSION = "synthetic_mbpp_database_v1_pod_eval_compatible"
MBPP_STREAM_SEED = 0x0BBF2026 & 0xFFFFFFFF


TASK_KINDS = [
    "sum_list",
    "product_list",
    "count_even",
    "count_vowels",
    "reverse_string",
    "is_palindrome",
    "factorial",
    "fibonacci",
    "max_in_list",
    "min_in_list",
    "remove_duplicates",
    "square_list",
    "filter_positive",
    "common_elements",
    "string_lengths",
    "count_occurrences",
    "merge_dicts_sum",
    "is_prime",
    "gcd",
    "flatten_once",
]


CONSONANTS = [
    "b", "c", "d", "f", "g", "h", "j", "k", "l", "m",
    "n", "p", "r", "s", "t", "v", "w", "z",
    "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr",
    "pl", "pr", "sk", "sl", "sm", "sn", "sp", "st", "sw", "tr",
    "th", "sh", "ch",
]

VOWELS = ["a", "e", "i", "o", "u", "ai", "ea", "ee", "ie", "oa", "ou", "ay"]

CODAS = ["", "n", "r", "l", "s", "t", "ck", "rd", "rt", "ng", "st", "ld"]


def synthetic_syllable(r: random.Random) -> str:
    return r.choice(CONSONANTS) + r.choice(VOWELS) + r.choice(CODAS)


def synthetic_word(r: random.Random, n_syllables: int = 1) -> str:
    return "".join(synthetic_syllable(r) for _ in range(n_syllables))


def rename_entry(original_entry: str, r: random.Random) -> str:
    suffix = "_" + synthetic_word(r, 1)
    return original_entry + suffix


def normalize_body_completion(completion: str) -> str:
    """
    Normalize a Python function-body completion.

    Rules:
    - completion is body only, not full function
    - multi-line body is allowed
    - every non-empty line must start with 4 spaces
    - no markdown fences
    - must end with newline
    """
    completion = completion.replace("\r\n", "\n").replace("\r", "\n")
    completion = completion.rstrip() + "\n"

    nonempty_lines = [line for line in completion.splitlines() if line.strip()]

    if not nonempty_lines:
        raise ValueError("Completion must not be empty.")

    for line in nonempty_lines:
        stripped = line.strip()

        if stripped.startswith("def "):
            raise ValueError("Completion must be function body only, not a full def.")

        if "```" in line:
            raise ValueError("Completion must not contain markdown fences.")

        if not line.startswith("    "):
            raise ValueError(
                "Every non-empty completion line must start with 4 spaces. "
                f"Bad line: {line!r}"
            )

    return completion


def make_prompt(
    problem: str,
    signature_with_docstring: str,
    examples: list[str] | None = None,
) -> str:
    examples = examples or []

    prompt = (
        "# Write a Python function to solve the following problem.\n"
        "# Output only the function body.\n"
        "# Do not include extra explanation or markdown fences.\n"
        "#\n"
        f"# Problem: {problem}\n"
    )

    if examples:
        prompt += "#\n"
        prompt += "# Examples:\n"
        for example in examples:
            prompt += f"# {example}\n"

    prompt += "#\n"
    prompt += "# Complete the function below.\n"
    prompt += signature_with_docstring

    return prompt


def make_test_block(tests: list[str]) -> str:
    return "def check(candidate):\n" + "\n".join(tests) + "\n"


def make_record(
    kind: str,
    index: int,
    entry_point: str,
    prompt: str,
    completion: str,
    test: str,
    problem: str,
    seed: int,
    metadata: dict | None = None,
) -> dict:
    normalized_completion = normalize_body_completion(completion)

    return {
        "database_version": MBPP_DATABASE_VERSION,

        # HumanEval / pod_eval_vllm.py compatible fields
        "task_id": f"mbpp/{kind}/{index:05d}",
        "prompt": prompt,
        "completion": normalized_completion,
        "test": test,
        "entry_point": entry_point,

        # Compatibility aliases
        "canonical_solution": normalized_completion,
        "gold": normalized_completion,

        # MBPP-style fields
        "text": problem,
        "code": normalized_completion,
        "test_list": [
            line.strip()
            for line in test.splitlines()
            if line.strip().startswith("assert ")
        ],

        # Dataset bookkeeping
        "src": f"synthetic_mbpp/{kind}",
        "kind": kind,
        "status": "gold_mbpp",
        "seed": seed,
        "metadata": metadata or {},
    }


def generate_item(kind: str, index: int, seed: int) -> dict:
    r = random.Random(seed)
    metadata = {}

    if kind == "sum_list":
        original_entry = "sum_list"
        entry = rename_entry(original_entry, r)

        problem = "Given a list of integers, return the sum of all elements."

        sig = (
            f"def {entry}(arr: list[int]) -> int:\n"
            '    """Return the sum of all integers in arr."""\n'
        )

        tests = [
            "    assert candidate([1, 2, 3]) == 6",
            "    assert candidate([]) == 0",
            "    assert candidate([-1, 5, -2]) == 2",
            "    assert candidate([10]) == 10",
        ]

        completion = "    return sum(arr)\n"

    elif kind == "product_list":
        original_entry = "product_list"
        entry = rename_entry(original_entry, r)

        problem = "Given a list of integers, return the product of all elements. Return 1 for an empty list."

        sig = (
            f"def {entry}(arr: list[int]) -> int:\n"
            '    """Return the product of all integers in arr. Empty list returns 1."""\n'
        )

        tests = [
            "    assert candidate([2, 3, 4]) == 24",
            "    assert candidate([]) == 1",
            "    assert candidate([5]) == 5",
            "    assert candidate([-2, 3]) == -6",
        ]

        completion = (
            "    total = 1\n"
            "    for x in arr:\n"
            "        total *= x\n"
            "    return total\n"
        )

    elif kind == "count_even":
        original_entry = "count_even"
        entry = rename_entry(original_entry, r)

        problem = "Given a list of integers, count how many values are even."

        sig = (
            f"def {entry}(arr: list[int]) -> int:\n"
            '    """Return the number of even integers in arr."""\n'
        )

        tests = [
            "    assert candidate([1, 2, 3, 4]) == 2",
            "    assert candidate([2, 4, 6]) == 3",
            "    assert candidate([1, 3, 5]) == 0",
            "    assert candidate([]) == 0",
            "    assert candidate([0, -2, -3]) == 2",
        ]

        completion = "    return sum(1 for x in arr if x % 2 == 0)\n"

    elif kind == "count_vowels":
        original_entry = "count_vowels"
        entry = rename_entry(original_entry, r)

        problem = "Given a string, count the number of vowels in it."

        sig = (
            f"def {entry}(s: str) -> int:\n"
            '    """Return the number of vowels in s. Treat a, e, i, o, u as vowels."""\n'
        )

        tests = [
            "    assert candidate('hello') == 2",
            "    assert candidate('sky') == 0",
            "    assert candidate('AEIOU') == 5",
            "    assert candidate('programming') == 3",
            "    assert candidate('') == 0",
        ]

        completion = "    return sum(1 for ch in s.lower() if ch in 'aeiou')\n"

    elif kind == "reverse_string":
        original_entry = "reverse_string"
        entry = rename_entry(original_entry, r)

        problem = "Given a string, return the string reversed."

        sig = (
            f"def {entry}(s: str) -> str:\n"
            '    """Return s reversed."""\n'
        )

        tests = [
            "    assert candidate('abc') == 'cba'",
            "    assert candidate('racecar') == 'racecar'",
            "    assert candidate('') == ''",
            "    assert candidate('hello world') == 'dlrow olleh'",
        ]

        completion = "    return s[::-1]\n"

    elif kind == "is_palindrome":
        original_entry = "is_palindrome"
        entry = rename_entry(original_entry, r)

        problem = "Given a string, return True if it is a palindrome, otherwise return False."

        sig = (
            f"def {entry}(s: str) -> bool:\n"
            '    """Return True if s reads the same forward and backward."""\n'
        )

        tests = [
            "    assert candidate('racecar') is True",
            "    assert candidate('level') is True",
            "    assert candidate('python') is False",
            "    assert candidate('') is True",
            "    assert candidate('aa') is True",
        ]

        completion = "    return s == s[::-1]\n"

    elif kind == "factorial":
        original_entry = "factorial"
        entry = rename_entry(original_entry, r)

        problem = "Given a non-negative integer n, return its factorial."

        sig = (
            f"def {entry}(n: int) -> int:\n"
            '    """Return n factorial. Assume n is non-negative."""\n'
        )

        tests = [
            "    assert candidate(0) == 1",
            "    assert candidate(1) == 1",
            "    assert candidate(5) == 120",
            "    assert candidate(7) == 5040",
        ]

        completion = (
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
        )

    elif kind == "fibonacci":
        original_entry = "fibonacci"
        entry = rename_entry(original_entry, r)

        problem = "Given a non-negative integer n, return the nth Fibonacci number where fibonacci(0) is 0 and fibonacci(1) is 1."

        sig = (
            f"def {entry}(n: int) -> int:\n"
            '    """Return the nth Fibonacci number with F(0)=0 and F(1)=1."""\n'
        )

        tests = [
            "    assert candidate(0) == 0",
            "    assert candidate(1) == 1",
            "    assert candidate(2) == 1",
            "    assert candidate(6) == 8",
            "    assert candidate(10) == 55",
        ]

        completion = (
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        )

    elif kind == "max_in_list":
        original_entry = "max_in_list"
        entry = rename_entry(original_entry, r)

        problem = "Given a non-empty list of integers, return the largest element."

        sig = (
            f"def {entry}(arr: list[int]) -> int:\n"
            '    """Return the maximum value in a non-empty list."""\n'
        )

        tests = [
            "    assert candidate([1, 5, 3]) == 5",
            "    assert candidate([-10, -2, -30]) == -2",
            "    assert candidate([7]) == 7",
            "    assert candidate([4, 4, 4]) == 4",
        ]

        completion = "    return max(arr)\n"

    elif kind == "min_in_list":
        original_entry = "min_in_list"
        entry = rename_entry(original_entry, r)

        problem = "Given a non-empty list of integers, return the smallest element."

        sig = (
            f"def {entry}(arr: list[int]) -> int:\n"
            '    """Return the minimum value in a non-empty list."""\n'
        )

        tests = [
            "    assert candidate([1, 5, 3]) == 1",
            "    assert candidate([-10, -2, -30]) == -30",
            "    assert candidate([7]) == 7",
            "    assert candidate([4, 4, 4]) == 4",
        ]

        completion = "    return min(arr)\n"

    elif kind == "remove_duplicates":
        original_entry = "remove_duplicates"
        entry = rename_entry(original_entry, r)

        problem = "Given a list, remove duplicate values while preserving their first occurrence order."

        sig = (
            f"def {entry}(arr: list) -> list:\n"
            '    """Return arr with duplicates removed, preserving first occurrence order."""\n'
        )

        tests = [
            "    assert candidate([1, 2, 1, 3, 2]) == [1, 2, 3]",
            "    assert candidate([]) == []",
            "    assert candidate(['a', 'b', 'a']) == ['a', 'b']",
            "    assert candidate([1, 1, 1]) == [1]",
        ]

        completion = (
            "    seen = set()\n"
            "    result = []\n"
            "    for x in arr:\n"
            "        if x not in seen:\n"
            "            seen.add(x)\n"
            "            result.append(x)\n"
            "    return result\n"
        )

    elif kind == "square_list":
        original_entry = "square_list"
        entry = rename_entry(original_entry, r)

        problem = "Given a list of numbers, return a new list containing the square of each number."

        sig = (
            f"def {entry}(arr: list[int]) -> list[int]:\n"
            '    """Return a list containing x*x for each x in arr."""\n'
        )

        tests = [
            "    assert candidate([1, 2, 3]) == [1, 4, 9]",
            "    assert candidate([]) == []",
            "    assert candidate([-2, 0, 3]) == [4, 0, 9]",
        ]

        completion = "    return [x * x for x in arr]\n"

    elif kind == "filter_positive":
        original_entry = "filter_positive"
        entry = rename_entry(original_entry, r)

        problem = "Given a list of integers, return only the positive integers."

        sig = (
            f"def {entry}(arr: list[int]) -> list[int]:\n"
            '    """Return the positive integers from arr, preserving order."""\n'
        )

        tests = [
            "    assert candidate([-1, 0, 2, 3]) == [2, 3]",
            "    assert candidate([-5, -1]) == []",
            "    assert candidate([1, 2, 3]) == [1, 2, 3]",
            "    assert candidate([]) == []",
        ]

        completion = "    return [x for x in arr if x > 0]\n"

    elif kind == "common_elements":
        original_entry = "common_elements"
        entry = rename_entry(original_entry, r)

        problem = "Given two lists, return a list of elements from the first list that also appear in the second list, preserving order and avoiding duplicates."

        sig = (
            f"def {entry}(a: list, b: list) -> list:\n"
            '    """Return unique elements from a that also occur in b, preserving order from a."""\n'
        )

        tests = [
            "    assert candidate([1, 2, 3, 2], [2, 3, 4]) == [2, 3]",
            "    assert candidate(['a', 'b'], ['c']) == []",
            "    assert candidate([], [1, 2]) == []",
            "    assert candidate([1, 1, 2], [1]) == [1]",
        ]

        completion = (
            "    b_set = set(b)\n"
            "    seen = set()\n"
            "    result = []\n"
            "    for x in a:\n"
            "        if x in b_set and x not in seen:\n"
            "            seen.add(x)\n"
            "            result.append(x)\n"
            "    return result\n"
        )

    elif kind == "string_lengths":
        original_entry = "string_lengths"
        entry = rename_entry(original_entry, r)

        problem = "Given a list of strings, return a list containing the length of each string."

        sig = (
            f"def {entry}(words: list[str]) -> list[int]:\n"
            '    """Return the length of each string in words."""\n'
        )

        tests = [
            "    assert candidate(['a', 'bb', 'ccc']) == [1, 2, 3]",
            "    assert candidate([]) == []",
            "    assert candidate(['', 'hi']) == [0, 2]",
        ]

        completion = "    return [len(word) for word in words]\n"

    elif kind == "count_occurrences":
        original_entry = "count_occurrences"
        entry = rename_entry(original_entry, r)

        problem = "Given a list and a target value, count how many times the target appears."

        sig = (
            f"def {entry}(arr: list, target) -> int:\n"
            '    """Return the number of times target appears in arr."""\n'
        )

        tests = [
            "    assert candidate([1, 2, 1, 3], 1) == 2",
            "    assert candidate(['a', 'b', 'a'], 'a') == 2",
            "    assert candidate([], 5) == 0",
            "    assert candidate([1, 2, 3], 4) == 0",
        ]

        completion = "    return sum(1 for x in arr if x == target)\n"

    elif kind == "merge_dicts_sum":
        original_entry = "merge_dicts_sum"
        entry = rename_entry(original_entry, r)

        problem = "Given two dictionaries with numeric values, merge them by summing values for matching keys."

        sig = (
            f"def {entry}(a: dict, b: dict) -> dict:\n"
            '    """Merge two dictionaries by summing values for duplicate keys."""\n'
        )

        tests = [
            "    assert candidate({'x': 1}, {'x': 2, 'y': 3}) == {'x': 3, 'y': 3}",
            "    assert candidate({}, {'a': 5}) == {'a': 5}",
            "    assert candidate({'a': 1}, {}) == {'a': 1}",
            "    assert candidate({'a': -1}, {'a': 1}) == {'a': 0}",
        ]

        completion = (
            "    result = dict(a)\n"
            "    for key, value in b.items():\n"
            "        result[key] = result.get(key, 0) + value\n"
            "    return result\n"
        )

    elif kind == "is_prime":
        original_entry = "is_prime"
        entry = rename_entry(original_entry, r)

        problem = "Given an integer n, return True if n is prime, otherwise return False."

        sig = (
            f"def {entry}(n: int) -> bool:\n"
            '    """Return True if n is a prime number."""\n'
        )

        tests = [
            "    assert candidate(2) is True",
            "    assert candidate(3) is True",
            "    assert candidate(4) is False",
            "    assert candidate(1) is False",
            "    assert candidate(17) is True",
            "    assert candidate(0) is False",
        ]

        completion = (
            "    if n < 2:\n"
            "        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n"
        )

    elif kind == "gcd":
        original_entry = "gcd"
        entry = rename_entry(original_entry, r)

        problem = "Given two non-negative integers, return their greatest common divisor."

        sig = (
            f"def {entry}(a: int, b: int) -> int:\n"
            '    """Return the greatest common divisor of a and b."""\n'
        )

        tests = [
            "    assert candidate(12, 18) == 6",
            "    assert candidate(7, 13) == 1",
            "    assert candidate(0, 5) == 5",
            "    assert candidate(21, 0) == 21",
            "    assert candidate(100, 25) == 25",
        ]

        completion = (
            "    while b:\n"
            "        a, b = b, a % b\n"
            "    return abs(a)\n"
        )

    elif kind == "flatten_once":
        original_entry = "flatten_once"
        entry = rename_entry(original_entry, r)

        problem = "Given a list of lists, flatten it by one level."

        sig = (
            f"def {entry}(nested: list[list]) -> list:\n"
            '    """Flatten a list of lists by one level."""\n'
        )

        tests = [
            "    assert candidate([[1, 2], [3], []]) == [1, 2, 3]",
            "    assert candidate([]) == []",
            "    assert candidate([['a'], ['b', 'c']]) == ['a', 'b', 'c']",
            "    assert candidate([[1], [2], [3]]) == [1, 2, 3]",
        ]

        completion = (
            "    result = []\n"
            "    for group in nested:\n"
            "        result.extend(group)\n"
            "    return result\n"
        )

    else:
        raise ValueError(f"Unknown MBPP task kind: {kind}")

    examples = [
        test.strip().replace("candidate", entry)
        for test in tests[:2]
    ]

    prompt = make_prompt(
        problem=problem,
        signature_with_docstring=sig,
        examples=examples,
    )

    test_block = make_test_block(tests)

    metadata.update(
        {
            "original_entry": original_entry,
            "renamed_entry": entry,
            "num_tests": len(tests),
            "completion_lines": len(
                [line for line in completion.splitlines() if line.strip()]
            ),
        }
    )

    return make_record(
        kind=kind,
        index=index,
        entry_point=entry,
        prompt=prompt,
        completion=completion,
        test=test_block,
        problem=problem,
        seed=seed,
        metadata=metadata,
    )


def build_records(seed: int, n_per_kind: int, shuffle: bool = True) -> list[dict]:
    main_rng = random.Random((int(seed) ^ MBPP_STREAM_SEED) & 0xFFFFFFFF)
    records = []

    index = 0

    for kind in TASK_KINDS:
        for _ in range(n_per_kind):
            item_seed = main_rng.randint(0, 2**31 - 1)

            record = generate_item(
                kind=kind,
                index=index,
                seed=item_seed,
            )

            records.append(record)
            index += 1

    if shuffle:
        main_rng.shuffle(records)

    return records


def write_jsonl(records: list[dict], output_path: Path, append: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"

    with output_path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def verify_record(record: dict) -> tuple[bool, str]:
    namespace = {}

    solution_code = record["prompt"] + record["completion"]
    test_code = record["test"]

    try:
        exec(solution_code, namespace)
        exec(test_code, namespace)

        entry_point = record["entry_point"]

        if entry_point not in namespace:
            return False, f"entry_point {entry_point!r} not defined after exec."

        candidate = namespace[entry_point]

        if "check" not in namespace:
            return False, "check(candidate) was not defined by test code."

        namespace["check"](candidate)

        return True, ""

    except Exception:
        return False, traceback.format_exc()


def verify_records(records: list[dict], max_failures: int = 10) -> bool:
    failures = []

    for record in records:
        ok, err = verify_record(record)

        record["verified"] = bool(ok)

        if not ok:
            record["verify_error"] = err
            failures.append((record["task_id"], record["kind"], err))

            if len(failures) >= max_failures:
                break

    if not failures:
        print("Local verification: all generated MBPP-style records passed tests.")
        return True

    print("Local verification failed.")
    print(f"Failures shown: {len(failures)}")

    for task_id, kind, err in failures:
        print("=" * 80)
        print(f"Task: {task_id}")
        print(f"Kind: {kind}")
        print(err)

    return False


def assert_required_fields(records: list[dict]) -> None:
    required = [
        "database_version",
        "task_id",
        "prompt",
        "completion",
        "test",
        "entry_point",
        "canonical_solution",
        "gold",
        "text",
        "code",
        "test_list",
        "src",
        "kind",
        "status",
        "seed",
        "metadata",
    ]

    bad = []

    for record in records:
        for field in required:
            if field not in record:
                bad.append((record.get("task_id"), field))

    if bad:
        print("Found records with missing required fields:", file=sys.stderr)

        for task_id, field in bad[:20]:
            print(f"Task: {task_id}, missing: {field}", file=sys.stderr)

        raise SystemExit(1)


def assert_completion_format(records: list[dict]) -> None:
    bad = []

    for record in records:
        completion = record.get("completion", "")

        try:
            normalized = normalize_body_completion(completion)
        except Exception as e:
            bad.append(
                (
                    record.get("task_id"),
                    record.get("kind"),
                    str(e),
                    completion,
                )
            )
            continue

        if normalized != completion:
            bad.append(
                (
                    record.get("task_id"),
                    record.get("kind"),
                    "completion is not normalized",
                    completion,
                )
            )

        if record.get("canonical_solution") != completion:
            bad.append(
                (
                    record.get("task_id"),
                    record.get("kind"),
                    "canonical_solution does not match completion",
                    completion,
                )
            )

        if record.get("gold") != completion:
            bad.append(
                (
                    record.get("task_id"),
                    record.get("kind"),
                    "gold does not match completion",
                    completion,
                )
            )

        if record.get("code") != completion:
            bad.append(
                (
                    record.get("task_id"),
                    record.get("kind"),
                    "code does not match completion",
                    completion,
                )
            )

    if bad:
        print("Found bad completion formatting:", file=sys.stderr)

        for task_id, kind, reason, completion in bad[:10]:
            print("=" * 80, file=sys.stderr)
            print(f"Task: {task_id}", file=sys.stderr)
            print(f"Kind: {kind}", file=sys.stderr)
            print(f"Reason: {reason}", file=sys.stderr)
            print(repr(completion), file=sys.stderr)

        raise SystemExit(1)


def print_counts(records: list[dict]) -> None:
    counts = {}

    for record in records:
        kind = record["kind"]
        counts[kind] = counts.get(kind, 0) + 1

    print("\nKind counts:")
    for kind in sorted(counts):
        print(f"  {kind}: {counts[kind]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic synthetic MBPP-style JSONL database "
            "compatible with updated pod_eval_vllm.py / HumanEval-style checking."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset/mbpp_database_all_cases.jsonl",
        help="Output JSONL file path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260505,
        help="Random seed.",
    )

    parser.add_argument(
        "--n-per-kind",
        type=int,
        default=10,
        help="Number of MBPP-style records to generate per task kind.",
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle final records.",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of overwriting it.",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run local verification that prompt + completion passes tests.",
    )

    args = parser.parse_args()

    records = build_records(
        seed=args.seed,
        n_per_kind=args.n_per_kind,
        shuffle=not args.no_shuffle,
    )

    assert_required_fields(records)
    assert_completion_format(records)

    if args.verify:
        ok = verify_records(records)

        if not ok:
            print("Not writing JSONL because local verification failed.", file=sys.stderr)
            raise SystemExit(1)

    output_path = Path(args.output)
    write_jsonl(records, output_path, append=args.append)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Database version: {MBPP_DATABASE_VERSION}")
    print(f"Total records: {len(records)}")
    print(f"Task kinds: {len(TASK_KINDS)}")
    print(f"Records per kind: {args.n_per_kind}")
    print(f"Append mode: {args.append}")
    print(f"Local verified: {args.verify}")
    print("Completion format: multi-line indented function body only")
    print("Core fields: prompt, completion, test, entry_point")
    print("Compatibility fields: canonical_solution, gold, text, code, test_list, metadata")

    print_counts(records)


if __name__ == "__main__":
    main()
