#!/usr/bin/env python3

import argparse
import json
import re
import traceback
from pathlib import Path


DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                yield line_no, {
                    "_json_error": str(e),
                    "_raw": line[:500],
                }


def count_nonempty_lines(text: str) -> int:
    return len([line for line in str(text).splitlines() if line.strip()])


def extract_def_names(code: str) -> list[str]:
    names = []

    for line in str(code).splitlines():
        match = DEF_RE.match(line)

        if match:
            names.append(match.group(1))

    return names


def check_completion_format(completion: str) -> list[str]:
    """
    MBPP-style completions may be multi-line, but they must be function body only.

    Required:
    - completion is a string
    - completion is non-empty
    - completion ends with newline
    - no markdown fences
    - no full def
    - every non-empty line starts with 4 spaces
    """
    problems = []

    if not isinstance(completion, str):
        problems.append("completion is not a string")
        return problems

    if not completion.strip():
        problems.append("completion is empty")
        return problems

    if not completion.endswith("\n"):
        problems.append("completion does not end with newline")

    if "```" in completion:
        problems.append("completion contains markdown fence")

    nonempty_lines = [line for line in completion.splitlines() if line.strip()]

    for i, line in enumerate(nonempty_lines, start=1):
        stripped = line.strip()

        if stripped.startswith("def "):
            problems.append(
                f"completion line {i} contains full function definition; expected body only"
            )

        if not line.startswith("    "):
            problems.append(
                f"completion line {i} is not indented with 4 spaces: {line!r}"
            )

    return problems


def check_test_list(record: dict) -> list[str]:
    problems = []

    test = record.get("test")
    test_list = record.get("test_list")

    if "test_list" not in record:
        return problems

    if not isinstance(test_list, list):
        problems.append("test_list exists but is not a list")
        return problems

    for i, item in enumerate(test_list):
        if not isinstance(item, str):
            problems.append(f"test_list[{i}] is not a string")

    if isinstance(test, str):
        extracted = [
            line.strip()
            for line in test.splitlines()
            if line.strip().startswith("assert ")
        ]

        if test_list != extracted:
            problems.append("test_list does not match assert lines extracted from test")

    return problems


def check_schema(record: dict) -> list[str]:
    problems = []

    if "_json_error" in record:
        problems.append(f"JSON decode error: {record['_json_error']}")
        return problems

    required_fields = [
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

    for field in required_fields:
        if field not in record:
            problems.append(f"missing field: {field}")

    string_fields = [
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
        "src",
        "kind",
        "status",
    ]

    for field in string_fields:
        if field in record and not isinstance(record.get(field), str):
            problems.append(f"{field} exists but is not a string")

    if "seed" in record and not isinstance(record.get("seed"), int):
        problems.append("seed exists but is not an int")

    if "metadata" in record and not isinstance(record.get("metadata"), dict):
        problems.append("metadata exists but is not a dict")

    prompt = record.get("prompt", "")
    test = record.get("test", "")
    completion = record.get("completion", "")

    if isinstance(prompt, str):
        if "def " not in prompt:
            problems.append("prompt does not appear to contain a function signature")

        if "Output only the function body" not in prompt:
            problems.append("prompt missing expected body-only instruction")

        if "Problem:" not in prompt:
            problems.append("prompt missing Problem section")

        if "Complete the function below" not in prompt:
            problems.append("prompt missing completion section")

    if isinstance(test, str):
        if "def check(candidate):" not in test:
            problems.append("test does not define check(candidate)")

        if "assert " not in test:
            problems.append("test does not contain any assert statements")

    if "completion" in record:
        problems.extend(check_completion_format(completion))

    if "canonical_solution" in record:
        canonical_solution = record.get("canonical_solution")

        if not isinstance(canonical_solution, str):
            problems.append("canonical_solution exists but is not a string")
        elif canonical_solution != record.get("completion"):
            problems.append("canonical_solution does not match completion")

    if "gold" in record:
        gold = record.get("gold")

        if not isinstance(gold, str):
            problems.append("gold exists but is not a string")
        elif gold != record.get("completion"):
            problems.append("gold does not match completion")

    if "code" in record:
        code = record.get("code")

        if not isinstance(code, str):
            problems.append("code exists but is not a string")
        elif code != record.get("completion"):
            problems.append("code does not match completion")

    problems.extend(check_test_list(record))

    return problems


def verify_record_execution(record: dict) -> tuple[bool, str, dict]:
    namespace = {}
    verification = {}

    prompt = record.get("prompt", "")
    completion = record.get("completion", "")
    test = record.get("test", "")
    entry_point = record.get("entry_point", "")

    solution_code = prompt + completion
    test_code = test

    verification["entry_point"] = entry_point
    verification["prompt_nonempty_lines"] = count_nonempty_lines(prompt)
    verification["completion_nonempty_lines"] = count_nonempty_lines(completion)
    verification["solution_nonempty_lines"] = count_nonempty_lines(solution_code)
    verification["test_nonempty_lines"] = count_nonempty_lines(test_code)
    verification["defined_functions_before_exec"] = extract_def_names(solution_code)

    try:
        exec(solution_code, namespace)

        defined_after_solution = [
            name
            for name, value in namespace.items()
            if callable(value) and not name.startswith("__")
        ]

        verification["defined_callables_after_solution"] = sorted(defined_after_solution)

        if entry_point not in namespace:
            return (
                False,
                f"entry_point {entry_point!r} not defined after executing prompt + completion",
                verification,
            )

        candidate = namespace[entry_point]

        exec(test_code, namespace)

        if "check" not in namespace:
            return False, "test code did not define check(candidate)", verification

        namespace["check"](candidate)

        verification["passed_tests"] = True

        return True, "", verification

    except Exception:
        verification["passed_tests"] = False
        return False, traceback.format_exc(), verification


def check_record(line_no: int, record: dict) -> tuple[list[str], dict]:
    problems = []
    verification = {}

    schema_problems = check_schema(record)
    problems.extend(schema_problems)

    if "_json_error" in record:
        return problems, verification

    core_fields = ["prompt", "completion", "test", "entry_point"]
    core_ok = all(isinstance(record.get(field), str) for field in core_fields)

    if not core_ok:
        return problems, verification

    ok, err, verification = verify_record_execution(record)

    if not ok:
        problems.append("local execution verification failed")
        verification["execution_error"] = err

    return problems, verification


def short_json(obj, max_chars: int = 3000) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        text = repr(obj)

    if len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>"

    return text


def format_record_for_log(
    line_no: int,
    record: dict,
    problems: list[str],
    verification: dict,
) -> str:
    prompt = record.get("prompt", "")
    completion = record.get("completion", "")
    test = record.get("test", "")

    lines = []

    lines.append("=" * 100)
    lines.append(f"LINE: {line_no}")
    lines.append(f"TASK ID: {record.get('task_id', 'unknown')}")
    lines.append(f"DATABASE VERSION: {record.get('database_version', 'unknown')}")
    lines.append(f"SRC: {record.get('src', 'unknown')}")
    lines.append(f"KIND: {record.get('kind', 'unknown')}")
    lines.append(f"STATUS: {record.get('status', 'unknown')}")
    lines.append(f"ENTRY POINT: {record.get('entry_point', 'unknown')}")
    lines.append(f"SEED: {record.get('seed', 'unknown')}")
    lines.append("")

    lines.append("[TEXT / PROBLEM]")
    lines.append(str(record.get("text", "")).strip())
    lines.append("")

    lines.append("[PROMPT]")
    lines.append(str(prompt).strip() if prompt else "None")
    lines.append("")

    lines.append("[COMPLETION REPR]")
    lines.append(repr(completion))
    lines.append("")

    lines.append("[COMPLETION TEXT]")
    lines.append(str(completion).rstrip() if completion else "None")
    lines.append("")

    lines.append("[PROMPT + COMPLETION TAIL]")
    if prompt or completion:
        combined = str(prompt) + str(completion)
        lines.append(combined[-2000:].rstrip())
    else:
        lines.append("None")
    lines.append("")

    lines.append("[TEST]")
    lines.append(str(test).strip() if test else "None")
    lines.append("")

    if "test_list" in record:
        lines.append("[TEST LIST]")
        lines.append(short_json(record.get("test_list")))
        lines.append("")

    if "canonical_solution" in record:
        lines.append("[CANONICAL SOLUTION REPR]")
        lines.append(repr(record.get("canonical_solution")))
        lines.append("")

    if "gold" in record:
        lines.append("[GOLD REPR]")
        lines.append(repr(record.get("gold")))
        lines.append("")

    if "code" in record:
        lines.append("[CODE REPR]")
        lines.append(repr(record.get("code")))
        lines.append("")

    if "metadata" in record:
        lines.append("[METADATA]")
        lines.append(short_json(record.get("metadata")))
        lines.append("")

    lines.append("[VERIFICATION]")
    lines.append(short_json(verification))
    lines.append("")

    lines.append("[PROBLEMS]")
    if problems:
        for problem in problems:
            lines.append(f"- {problem}")
    else:
        lines.append("None")
    lines.append("")

    return "\n".join(lines)


def print_terminal_record(
    line_no: int,
    record: dict,
    problems: list[str],
    verification: dict,
    show_prompt: bool = False,
    show_completion: bool = False,
    show_test: bool = False,
    show_error: bool = False,
    show_text: bool = False,
) -> None:
    print("=" * 80)
    print(f"Line: {line_no}")
    print(f"Task ID: {record.get('task_id', 'unknown')}")
    print(f"Kind: {record.get('kind', 'unknown')}")
    print(f"Status: {record.get('status', 'unknown')}")
    print(f"Entry point: {record.get('entry_point', 'unknown')}")
    print(f"Completion non-empty lines: {count_nonempty_lines(record.get('completion', ''))}")
    print(f"Passed tests: {verification.get('passed_tests')}")

    if problems:
        print("Problems:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("Problems: none")

    if show_text:
        print("")
        print("[TEXT / PROBLEM]")
        print("-" * 40)
        print(str(record.get("text", "")).rstrip())
        print("-" * 40)

    if show_prompt:
        print("")
        print("[PROMPT]")
        print("-" * 40)
        print(str(record.get("prompt", "")).rstrip())
        print("-" * 40)

    if show_completion:
        print("")
        print("[COMPLETION]")
        print("-" * 40)
        print(repr(record.get("completion", "")))
        print(str(record.get("completion", "")).rstrip())
        print("-" * 40)

    if show_test:
        print("")
        print("[TEST]")
        print("-" * 40)
        print(str(record.get("test", "")).rstrip())
        print("-" * 40)

    if show_error and verification.get("execution_error"):
        print("")
        print("[EXECUTION ERROR]")
        print("-" * 40)
        print(verification["execution_error"])
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check synthetic MBPP-style JSONL records and verify that "
            "prompt + completion passes the HumanEval-style test block."
        )
    )

    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default="dataset/mbpp_database_all_cases.jsonl",
        help="Path to MBPP database JSONL.",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="database_v3/mbpp_dataset_check.log",
        help="Path to output readable log file.",
    )

    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print problem text to terminal.",
    )

    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print prompt text to terminal.",
    )

    parser.add_argument(
        "--show-completion",
        action="store_true",
        help="Print completion text to terminal.",
    )

    parser.add_argument(
        "--show-test",
        action="store_true",
        help="Print test block to terminal.",
    )

    parser.add_argument(
        "--show-error",
        action="store_true",
        help="Print execution traceback to terminal for failing records.",
    )

    parser.add_argument(
        "--show-errors-only",
        action="store_true",
        help="Only print/log records with problems.",
    )

    parser.add_argument(
        "--kind",
        type=str,
        default=None,
        help="Only inspect records with this kind.",
    )

    parser.add_argument(
        "--status",
        type=str,
        default=None,
        help="Only inspect records with this status.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of records to inspect after filters. 0 means all.",
    )

    args = parser.parse_args()

    path = Path(args.path)
    log_path = Path(args.log)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    total_seen = 0
    total_checked = 0
    logged_count = 0
    ok_count = 0
    bad_count = 0

    status_counts = {}
    kind_counts = {}
    version_counts = {}

    with log_path.open("w", encoding="utf-8") as log_f:
        for line_no, record in load_jsonl(path):
            total_seen += 1

            if args.kind is not None and record.get("kind") != args.kind:
                continue

            if args.status is not None and record.get("status") != args.status:
                continue

            total_checked += 1

            if args.limit and total_checked > args.limit:
                total_checked -= 1
                break

            status = record.get("status", "unknown")
            kind = record.get("kind", "unknown")
            version = record.get("database_version", "unknown")

            status_counts[status] = status_counts.get(status, 0) + 1
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            version_counts[version] = version_counts.get(version, 0) + 1

            problems, verification = check_record(
                line_no=line_no,
                record=record,
            )

            if problems:
                bad_count += 1
            else:
                ok_count += 1

            if args.show_errors_only and not problems:
                continue

            log_text = format_record_for_log(
                line_no=line_no,
                record=record,
                problems=problems,
                verification=verification,
            )

            log_f.write(log_text)
            log_f.write("\n")
            log_f.flush()

            logged_count += 1

            print_terminal_record(
                line_no=line_no,
                record=record,
                problems=problems,
                verification=verification,
                show_text=args.show_text,
                show_prompt=args.show_prompt,
                show_completion=args.show_completion,
                show_test=args.show_test,
                show_error=args.show_error,
            )

        summary_lines = []
        summary_lines.append("\n" + "=" * 100)
        summary_lines.append("SUMMARY")
        summary_lines.append("=" * 100)
        summary_lines.append(f"Input file: {path}")
        summary_lines.append(f"Log file: {log_path}")
        summary_lines.append(f"Total JSONL records seen: {total_seen}")
        summary_lines.append(f"Records checked after filters: {total_checked}")
        summary_lines.append(f"Records written to log: {logged_count}")
        summary_lines.append(f"OK records: {ok_count}")
        summary_lines.append(f"Bad records: {bad_count}")
        summary_lines.append("")

        summary_lines.append("Database version counts:")
        for version, count in sorted(version_counts.items()):
            summary_lines.append(f"  {version}: {count}")

        summary_lines.append("")
        summary_lines.append("Status counts:")
        for status, count in sorted(status_counts.items()):
            summary_lines.append(f"  {status}: {count}")

        summary_lines.append("")
        summary_lines.append("Kind counts:")
        for kind, count in sorted(kind_counts.items()):
            summary_lines.append(f"  {kind}: {count}")

        summary_text = "\n".join(summary_lines)

        log_f.write(summary_text)
        log_f.write("\n")

    print(summary_text)
    print(f"\nWrote readable log to: {log_path}")


if __name__ == "__main__":
    main()
