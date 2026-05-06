#!/usr/bin/env python3

import argparse
import json
import random
import sys
import traceback
from pathlib import Path


DEBUG_DATABASE_VERSION = "debug_database_v2_pod_eval_compatible"
DEBUG_STREAM_SEED = 0xD06BEE29


BUG_KINDS = [
    "off_by_one_range",
    "swap_subtract",
    "wrong_comparator",
    "wrong_init",
    "early_break",
    "wrong_index",
    "wrong_modulo",
    "missing_edge_case",
    "wrong_boolean_operator",
    "integer_division",
    "reverse_condition",
    "wrong_accumulator_update",
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


def comment_block(text: str) -> str:
    return "\n".join("# " + line if line else "#" for line in text.splitlines())


def comment_tests(tests: list[str]) -> str:
    return "\n".join(
        ("# " + line.lstrip()) if line.startswith("    ") else ("# " + line)
        for line in tests
    )


def normalize_single_line_completion(completion: str) -> str:
    """
    Keep completion compatible with pod_eval_vllm.py / humaneval_sandbox.

    Rules:
    - Completion is function body only.
    - Completion is exactly one physical non-empty line.
    - Completion is indented because the prompt already contains the function signature.
    - No markdown fences.
    - No full function definition.
    """
    completion = completion.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line for line in completion.splitlines() if line.strip()]

    if len(lines) != 1:
        raise ValueError(
            "Completion must be exactly one non-empty line for this database. "
            f"Got {len(lines)} non-empty lines: {completion!r}"
        )

    line = lines[0].strip()

    if line.startswith("def "):
        raise ValueError("Completion must be function body only, not a full def.")

    if "```" in line:
        raise ValueError("Completion must not contain markdown fences.")

    return "    " + line + "\n"


def make_prompt(
    buggy_code: str,
    tests: list[str],
    signature_with_docstring: str,
) -> str:
    buggy_commented = comment_block(buggy_code)
    commented_tests = comment_tests(tests)

    return (
        "# Fix the bug in the following Python function.\n"
        "# The intended behavior is described in the corrected docstring.\n"
        "# Output only the corrected function body.\n"
        "# Do not include extra explanation or markdown fences.\n"
        "#\n"
        "# Buggy version, DO NOT include in your output:\n"
        f"{buggy_commented}\n"
        "#\n"
        "# Tests the corrected version must pass:\n"
        f"{commented_tests}\n"
        "#\n"
        "# Now complete the corrected version below.\n"
        f"{signature_with_docstring}"
    )


def make_test_block(tests: list[str]) -> str:
    return "def check(candidate):\n" + "\n".join(tests) + "\n"


def make_record(
    kind: str,
    index: int,
    entry_point: str,
    prompt: str,
    completion: str,
    test: str,
    buggy_code: str,
    seed: int,
    metadata: dict | None = None,
) -> dict:
    normalized_completion = normalize_single_line_completion(completion)

    return {
        "database_version": DEBUG_DATABASE_VERSION,

        # HumanEval / pod_eval_vllm core fields
        "task_id": f"debug/{kind}/{index:05d}",
        "prompt": prompt,
        "completion": normalized_completion,
        "test": test,
        "entry_point": entry_point,

        # Compatibility aliases used by some evaluation/training paths
        "canonical_solution": normalized_completion,
        "gold": normalized_completion,

        # Dataset bookkeeping
        "src": f"procedural_debug/{kind}",
        "kind": kind,
        "status": "gold_debug",
        "buggy_code": buggy_code,
        "seed": seed,
        "metadata": metadata or {},
    }


def rename_entry(
    original_entry: str,
    buggy: str,
    sig: str,
    r: random.Random,
) -> tuple[str, str, str]:
    suffix = "_" + synthetic_word(r, 1)
    entry = original_entry + suffix

    buggy = buggy.replace(f"def {original_entry}(", f"def {entry}(")
    sig = sig.replace(f"def {original_entry}(", f"def {entry}(")

    return entry, buggy, sig


def generate_item(kind: str, index: int, seed: int) -> dict:
    r = random.Random(seed)
    metadata = {}

    if kind == "off_by_one_range":
        original_entry = "sum_to"
        n_t = r.randint(5, 15)

        buggy = (
            "def sum_to(n: int) -> int:\n"
            "    total = 0\n"
            "    for i in range(1, n):\n"
            "        total += i\n"
            "    return total\n"
        )

        sig = (
            "def sum_to(n: int) -> int:\n"
            '    """Return the sum of integers from 1 to n inclusive.\n'
            "    Examples: sum_to(3) == 6; sum_to(5) == 15.\n"
            '    """\n'
        )

        tests = [
            f"    assert candidate({n_t}) == {sum(range(1, n_t + 1))}",
            f"    assert candidate({n_t + 1}) == {sum(range(1, n_t + 2))}",
            "    assert candidate(1) == 1",
            "    assert candidate(2) == 3",
        ]

        completion = "return sum(range(1, n + 1))"
        metadata = {"n_t": n_t}

    elif kind == "swap_subtract":
        original_entry = "first_minus_second"

        buggy = (
            "def first_minus_second(a: int, b: int) -> int:\n"
            "    return b - a\n"
        )

        sig = (
            "def first_minus_second(a: int, b: int) -> int:\n"
            '    """Return a minus b, i.e. a - b."""\n'
        )

        tests = [
            "    assert candidate(10, 3) == 7",
            "    assert candidate(0, 5) == -5",
            "    assert candidate(-4, -2) == -2",
            "    assert candidate(100, 99) == 1",
        ]

        completion = "return a - b"

    elif kind == "wrong_comparator":
        original_entry = "at_least"
        threshold = r.randint(3, 9)

        buggy = (
            f"def at_least(arr, threshold={threshold}):\n"
            "    return sum(1 for x in arr if x > threshold)\n"
        )

        sig = (
            f"def at_least(arr, threshold={threshold}):\n"
            '    """Count elements in arr that are >= threshold, inclusive."""\n'
        )

        tests = [
            f"    assert candidate([1, {threshold}, {threshold - 1}, {threshold + 1}, {threshold}], threshold={threshold}) == 3",
            f"    assert candidate([{threshold}, {threshold}, {threshold}], threshold={threshold}) == 3",
            f"    assert candidate([0, 1, 2], threshold={threshold}) == 0",
            "    assert candidate([], threshold=1) == 0",
        ]

        completion = "return sum(1 for x in arr if x >= threshold)"
        metadata = {"threshold": threshold}

    elif kind == "wrong_init":
        original_entry = "product_of_list"

        buggy = (
            "def product_of_list(arr):\n"
            "    total = 0\n"
            "    for x in arr:\n"
            "        total *= x\n"
            "    return total\n"
        )

        sig = (
            "def product_of_list(arr: list[int]) -> int:\n"
            '    """Return the product of integers in arr. Empty list returns 1."""\n'
        )

        tests = [
            "    assert candidate([2, 3, 4]) == 24",
            "    assert candidate([5]) == 5",
            "    assert candidate([]) == 1",
            "    assert candidate([1, 2, 3, 4, 5]) == 120",
            "    assert candidate([-2, 3]) == -6",
        ]

        # Avoid import dependency inside the generated function body.
        completion = "return 1 if not arr else __import__('functools').reduce(lambda a, b: a * b, arr, 1)"

    elif kind == "early_break":
        original_entry = "find_largest"

        buggy = (
            "def find_largest(arr):\n"
            "    largest = arr[0]\n"
            "    for x in arr:\n"
            "        if x > largest:\n"
            "            largest = x\n"
            "            break\n"
            "    return largest\n"
        )

        sig = (
            "def find_largest(arr: list[int]) -> int:\n"
            '    """Return the largest integer in arr. Assume non-empty."""\n'
        )

        tests = [
            "    assert candidate([3, 7, 2, 9, 4]) == 9",
            "    assert candidate([1, 2, 3, 4, 5, 6]) == 6",
            "    assert candidate([5, 4, 3, 2, 1]) == 5",
            "    assert candidate([42]) == 42",
            "    assert candidate([-10, -2, -30]) == -2",
        ]

        completion = "return max(arr)"

    elif kind == "wrong_index":
        original_entry = "second_last"
        sample_arr = [r.randint(1, 99) for _ in range(r.randint(5, 9))]

        buggy = (
            "def second_last(arr):\n"
            "    return arr[-1]\n"
        )

        sig = (
            "def second_last(arr: list):\n"
            '    """Return the second-to-last element of arr."""\n'
        )

        tests = [
            f"    assert candidate({sample_arr!r}) == {sample_arr[-2]}",
            "    assert candidate([1, 2]) == 1",
            "    assert candidate(['a', 'b', 'c']) == 'b'",
            "    assert candidate([10, 20, 30, 40, 50]) == 40",
        ]

        completion = "return arr[-2]"
        metadata = {"sample_arr": sample_arr}

    elif kind == "wrong_modulo":
        original_entry = "mod_then_double"
        mod = r.choice([7, 11, 13])
        wrong_mod = mod + 1

        buggy = (
            f"def mod_then_double(n, modulus={mod}):\n"
            f"    return 2 * (n % {wrong_mod})\n"
        )

        sig = (
            f"def mod_then_double(n: int, modulus: int = {mod}) -> int:\n"
            '    """Return 2 * (n % modulus)."""\n'
        )

        tests = [
            f"    assert candidate(10, modulus={mod}) == {2 * (10 % mod)}",
            f"    assert candidate({mod * 2 + 3}, modulus={mod}) == {2 * ((mod * 2 + 3) % mod)}",
            f"    assert candidate(0, modulus={mod}) == 0",
            f"    assert candidate({mod - 1}, modulus={mod}) == {2 * (mod - 1)}",
            "    assert candidate(100, modulus=9) == 2 * (100 % 9)",
        ]

        completion = "return 2 * (n % modulus)"
        metadata = {"mod": mod, "wrong_mod": wrong_mod}

    elif kind == "missing_edge_case":
        original_entry = "first_or_default"

        buggy = (
            "def first_or_default(arr, default=None):\n"
            "    return arr[0]\n"
        )

        sig = (
            "def first_or_default(arr: list, default=None):\n"
            '    """Return arr[0] if non-empty, else return default."""\n'
        )

        tests = [
            "    assert candidate([7, 8, 9]) == 7",
            "    assert candidate([]) is None",
            "    assert candidate([], default=-1) == -1",
            "    assert candidate(['a']) == 'a'",
            "    assert candidate([], default='empty') == 'empty'",
        ]

        completion = "return arr[0] if arr else default"

    elif kind == "wrong_boolean_operator":
        original_entry = "in_closed_interval"
        low = r.randint(1, 10)
        high = low + r.randint(5, 15)

        buggy = (
            f"def in_closed_interval(x, low={low}, high={high}):\n"
            "    return x >= low or x <= high\n"
        )

        sig = (
            f"def in_closed_interval(x: int, low: int = {low}, high: int = {high}) -> bool:\n"
            '    """Return True if x is between low and high inclusive."""\n'
        )

        tests = [
            f"    assert candidate({low}, low={low}, high={high}) is True",
            f"    assert candidate({high}, low={low}, high={high}) is True",
            f"    assert candidate({low - 1}, low={low}, high={high}) is False",
            f"    assert candidate({high + 1}, low={low}, high={high}) is False",
            f"    assert candidate({(low + high) // 2}, low={low}, high={high}) is True",
        ]

        completion = "return low <= x <= high"
        metadata = {"low": low, "high": high}

    elif kind == "integer_division":
        original_entry = "average_pair"

        buggy = (
            "def average_pair(a, b):\n"
            "    return (a + b) // 2\n"
        )

        sig = (
            "def average_pair(a: float, b: float) -> float:\n"
            '    """Return the arithmetic mean of a and b as a float."""\n'
        )

        tests = [
            "    assert candidate(1, 2) == 1.5",
            "    assert candidate(10, 20) == 15.0",
            "    assert candidate(-1, 1) == 0.0",
            "    assert candidate(2.5, 3.5) == 3.0",
        ]

        completion = "return (a + b) / 2"

    elif kind == "reverse_condition":
        original_entry = "clip_nonnegative"

        buggy = (
            "def clip_nonnegative(x):\n"
            "    if x >= 0:\n"
            "        return 0\n"
            "    return x\n"
        )

        sig = (
            "def clip_nonnegative(x: int) -> int:\n"
            '    """Return x if x is nonnegative, otherwise return 0."""\n'
        )

        tests = [
            "    assert candidate(5) == 5",
            "    assert candidate(0) == 0",
            "    assert candidate(-3) == 0",
            "    assert candidate(100) == 100",
        ]

        completion = "return x if x >= 0 else 0"

    elif kind == "wrong_accumulator_update":
        original_entry = "count_even"

        buggy = (
            "def count_even(arr):\n"
            "    count = 0\n"
            "    for x in arr:\n"
            "        if x % 2 == 0:\n"
            "            count += x\n"
            "    return count\n"
        )

        sig = (
            "def count_even(arr: list[int]) -> int:\n"
            '    """Return the number of even integers in arr."""\n'
        )

        tests = [
            "    assert candidate([1, 2, 3, 4]) == 2",
            "    assert candidate([2, 4, 6]) == 3",
            "    assert candidate([1, 3, 5]) == 0",
            "    assert candidate([]) == 0",
            "    assert candidate([0, -2, -3]) == 2",
        ]

        completion = "return sum(1 for x in arr if x % 2 == 0)"

    else:
        raise ValueError(f"Unknown debug kind: {kind}")

    entry_point, buggy, sig = rename_entry(
        original_entry=original_entry,
        buggy=buggy,
        sig=sig,
        r=r,
    )

    prompt = make_prompt(
        buggy_code=buggy,
        tests=tests,
        signature_with_docstring=sig,
    )

    test_block = make_test_block(tests)

    metadata.update(
        {
            "original_entry": original_entry,
            "renamed_entry": entry_point,
            "num_tests": len(tests),
        }
    )

    return make_record(
        kind=kind,
        index=index,
        entry_point=entry_point,
        prompt=prompt,
        completion=completion,
        test=test_block,
        buggy_code=buggy,
        seed=seed,
        metadata=metadata,
    )


def build_records(seed: int, n_per_kind: int, shuffle: bool = True) -> list[dict]:
    main_rng = random.Random((int(seed) ^ DEBUG_STREAM_SEED) & 0xFFFFFFFF)
    records = []

    index = 0

    for kind in BUG_KINDS:
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
        print("Local verification: all generated debug fixes passed tests.")
        return True

    print("Local verification failed.")
    print(f"Failures shown: {len(failures)}")

    for task_id, kind, err in failures:
        print("=" * 80)
        print(f"Task: {task_id}")
        print(f"Kind: {kind}")
        print(err)

    return False


def assert_all_completions_one_line(records: list[dict]) -> None:
    bad = []

    for record in records:
        completion = record["completion"]
        nonempty_lines = [line for line in completion.splitlines() if line.strip()]

        if len(nonempty_lines) != 1:
            bad.append(
                (
                    record.get("task_id"),
                    record.get("kind"),
                    len(nonempty_lines),
                    completion,
                )
            )

    if bad:
        print("Found non-one-line completions:", file=sys.stderr)

        for task_id, kind, n_lines, completion in bad[:10]:
            print("=" * 80, file=sys.stderr)
            print(f"Task: {task_id}", file=sys.stderr)
            print(f"Kind: {kind}", file=sys.stderr)
            print(f"Non-empty lines: {n_lines}", file=sys.stderr)
            print(repr(completion), file=sys.stderr)

        raise SystemExit(1)


def assert_required_fields(records: list[dict]) -> None:
    required = [
        "task_id",
        "prompt",
        "completion",
        "test",
        "entry_point",
        "canonical_solution",
        "src",
        "kind",
        "status",
        "seed",
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
            "Build a deterministic debug JSONL database with one-line "
            "gold fixed function bodies compatible with updated pod_eval_vllm.py / "
            "HumanEval-style checking."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset/debug_database_all_cases.jsonl",
        help="Output JSONL file path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260501,
        help="Random seed.",
    )

    parser.add_argument(
        "--n-per-kind",
        type=int,
        default=10,
        help="Number of debug records to generate per bug kind.",
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
    assert_all_completions_one_line(records)

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
    print(f"Database version: {DEBUG_DATABASE_VERSION}")
    print(f"Total records: {len(records)}")
    print(f"Bug kinds: {len(BUG_KINDS)}")
    print(f"Records per kind: {args.n_per_kind}")
    print(f"Append mode: {args.append}")
    print(f"Local verified: {args.verify}")
    print("Completion format: one-line indented function body only")
    print("Core fields: prompt, completion, test, entry_point")
    print("Compatibility fields: canonical_solution, gold, metadata")

    print_counts(records)


if __name__ == "__main__":
    main()
