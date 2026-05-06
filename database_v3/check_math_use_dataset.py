#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


# Matches integers, decimals, and comma-separated numbers.
MATH_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Optional boxed-answer support.
BOXED_START_RE = re.compile(r"\\boxed\s*\{")

ANSWER_PHRASE_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*\$?([^\s\n\.]+)",
    re.IGNORECASE,
)

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
THINK_TRAIL_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)
THINK_NARRATIVE_RE = re.compile(
    r"^\s*Thinking Process:.*?(?=\n\n[A-Z0-9]|\Z)",
    re.DOTALL,
)


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


def strip_thinking(text: str) -> str:
    text = str(text or "")

    if "<think>" in text:
        text = THINK_RE.sub("", text, count=1)
    elif "</think>" in text:
        text = THINK_TRAIL_RE.sub("", text, count=1)

    if text.lstrip().startswith("Thinking Process:"):
        text = THINK_NARRATIVE_RE.sub("", text, count=1)

    return text.strip()


def strip_markdown_fences(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    stripped = text.strip()

    if not stripped.startswith("```"):
        return text

    lines = text.splitlines()

    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]

    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip("\n")


def clean_completion(completion: str) -> str:
    completion = str(completion or "")
    completion = strip_thinking(completion)
    completion = strip_markdown_fences(completion)

    return completion.strip()


def extract_boxed(text: str):
    """
    Extract contents of the last \\boxed{...}, with nested-brace support.
    """
    last = None

    for match in BOXED_START_RE.finditer(text):
        i = match.end()
        depth = 1
        j = i

        while j < len(text) and depth > 0:
            ch = text[j]

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1

                if depth == 0:
                    last = text[i:j].strip()
                    break

            j += 1

    return last


def extract_math_answer(completion: str):
    """
    Extract final answer from completion.

    Priority:
      1. #### marker
      2. \\boxed{...}
      3. "answer is ..."
      4. last number in text
      5. last non-empty line
    """
    text = clean_completion(completion)

    if not text:
        return ""

    if "####" in text:
        match = re.search(r"####\s*([^\n]+)", text)

        if match:
            tail = match.group(1).strip().rstrip(".")
            num_match = MATH_NUMBER_RE.search(tail)

            if num_match:
                return num_match.group(0)

            return tail

    boxed = extract_boxed(text)

    if boxed:
        boxed = boxed.strip().rstrip(".")
        num_match = MATH_NUMBER_RE.search(boxed)

        if num_match:
            return num_match.group(0)

        return boxed

    match = ANSWER_PHRASE_RE.search(text)

    if match:
        fragment = match.group(1).strip().rstrip(".,")
        num_match = MATH_NUMBER_RE.search(fragment)

        if num_match:
            return num_match.group(0)

        return fragment

    numbers = MATH_NUMBER_RE.findall(text)

    if numbers:
        return numbers[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if lines:
        return lines[-1].rstrip(".")

    return ""


def normalize_answer(value) -> str:
    value = str(value or "")
    value = value.strip()
    value = value.replace(",", "")
    value = value.replace("$", "")
    value = value.rstrip(".")

    return value


def score_math_answer(pred, gold) -> bool:
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    if not pred_norm or not gold_norm:
        return False

    if pred_norm == gold_norm:
        return True

    try:
        return abs(float(pred_norm) - float(gold_norm)) < 1e-6
    except (TypeError, ValueError):
        return False


def get_gold_answer(record: dict):
    """
    Support both schemas:
      - answer
      - gold
    """
    if "answer" in record:
        return record.get("answer")

    return record.get("gold")


def get_kind(record: dict) -> str:
    if record.get("kind"):
        return str(record["kind"])

    src = str(record.get("src", ""))

    if "/" in src:
        return src.rsplit("/", 1)[-1]

    return "unknown"


def check_record(line_no: int, record: dict):
    problems = []

    if "_json_error" in record:
        problems.append(f"JSON decode error: {record['_json_error']}")
        return problems

    required_fields = [
        "prompt",
        "completion",
        "status",
    ]

    for field in required_fields:
        if field not in record:
            problems.append(f"missing field: {field}")

    if "answer" not in record and "gold" not in record:
        problems.append("missing field: answer or gold")

    prompt = record.get("prompt")
    completion = record.get("completion")
    gold = get_gold_answer(record)
    status = record.get("status")

    if "prompt" in record and not isinstance(prompt, str):
        problems.append("prompt exists but is not a string")

    if "completion" in record and not isinstance(completion, str):
        problems.append("completion exists but is not a string")

    if "status" in record and not isinstance(status, str):
        problems.append("status exists but is not a string")

    if gold is not None and not isinstance(gold, (str, int, float)):
        problems.append("gold/answer exists but is not string/int/float")

    if isinstance(prompt, str):
        if not prompt.strip():
            problems.append("prompt is empty")

    if isinstance(completion, str):
        cleaned = clean_completion(completion)

        if not cleaned:
            problems.append("completion is empty after cleanup")

        if completion.strip().startswith("```"):
            problems.append("completion contains markdown fence")

        # Warning-style problem: expected for this database, but not always fatal.
        if "####" not in cleaned and "\\boxed" not in cleaned:
            problems.append("completion has no #### marker or boxed answer")

        extracted = extract_math_answer(cleaned)

        if not extracted:
            problems.append("could not extract math answer from completion")

        if gold is not None and extracted:
            if not score_math_answer(extracted, gold):
                problems.append(
                    f"extracted answer does not match gold: pred={extracted!r}, gold={gold!r}"
                )

    if record.get("status") == "gold":
        if gold is None:
            problems.append("status=gold but no answer/gold field")

    if record.get("status") in {"self_check_failed", "bad", "failed"}:
        problems.append(f"record status is non-passing: {record.get('status')}")

    return problems


def format_record_for_log(line_no: int, record: dict, problems: list[str]) -> str:
    prompt = record.get("prompt", "")
    completion = record.get("completion", "")
    cleaned_completion = clean_completion(completion)
    extracted = extract_math_answer(cleaned_completion)
    gold = get_gold_answer(record)

    lines = []

    lines.append("=" * 100)
    lines.append(f"LINE: {line_no}")
    lines.append(f"TASK ID: {record.get('task_id')}")
    lines.append(f"STATUS: {record.get('status', 'unknown')}")
    lines.append(f"KIND: {get_kind(record)}")
    lines.append(f"SRC: {record.get('src')}")
    lines.append(f"DIFFICULTY: {record.get('difficulty')}")
    lines.append(f"BLOCK SEED: {record.get('block_seed')}")
    lines.append(f"GENERATOR SEED: {record.get('generator_seed')}")
    lines.append(f"INDEX: {record.get('index')}")
    lines.append("")

    lines.append("[PROMPT]")
    lines.append(str(prompt).strip())
    lines.append("")

    lines.append("[GOLD ANSWER]")
    lines.append(str(gold).strip())
    lines.append("")

    lines.append("[EXTRACTED ANSWER]")
    lines.append(str(extracted).strip() if extracted else "None")
    lines.append("")

    lines.append("[SCORE]")
    if gold is not None and extracted:
        lines.append(f"match={score_math_answer(extracted, gold)}")
    else:
        lines.append("match=False")
    lines.append("")

    lines.append("[PROBLEMS]")
    if problems:
        for p in problems:
            lines.append(f"- {p}")
    else:
        lines.append("None")
    lines.append("")

    lines.append("[COMPLETION TAIL]")
    if completion:
        lines.append(str(completion)[-1500:].strip())
    else:
        lines.append("None")
    lines.append("")

    lines.append("[CLEANED COMPLETION TAIL]")
    if cleaned_completion:
        lines.append(cleaned_completion[-1500:].strip())
    else:
        lines.append("None")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Check math_database JSONL records and log prompts, answers, completions, and extraction results."
    )

    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default="dataset/math_database_all_cases.jsonl",
        help="Path to math database JSONL.",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="database_v3/math_dataset_check.log",
        help="Path to output log file.",
    )

    parser.add_argument(
        "--show-completion",
        action="store_true",
        help="Print completion tail to terminal.",
    )

    parser.add_argument(
        "--show-errors-only",
        action="store_true",
        help="Only print/log records with problems.",
    )

    parser.add_argument(
        "--kind",
        type=str,
        default="",
        help="Only inspect records of this kind.",
    )

    parser.add_argument(
        "--status",
        type=str,
        default="",
        help="Only inspect records with this status.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of records to inspect. 0 means all.",
    )

    parser.add_argument(
        "--require-marker",
        action="store_true",
        help="Treat missing ####/boxed final-answer marker as a problem.",
    )

    args = parser.parse_args()

    path = Path(args.path)
    log_path = Path(args.log)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    inspected = 0
    logged_count = 0
    ok_count = 0
    bad_count = 0
    extracted_count = 0
    marker_count = 0
    status_counts = {}
    kind_counts = {}
    problem_counts = {}

    with log_path.open("w", encoding="utf-8") as log_f:
        for line_no, record in load_jsonl(path):
            total += 1

            if args.limit and inspected >= args.limit:
                break

            if "_json_error" not in record:
                kind = get_kind(record)
                status = str(record.get("status", "unknown"))
            else:
                kind = "unknown"
                status = "json_error"

            if args.kind and kind != args.kind:
                continue

            if args.status and status != args.status:
                continue

            inspected += 1

            status_counts[status] = status_counts.get(status, 0) + 1
            kind_counts[kind] = kind_counts.get(kind, 0) + 1

            completion = record.get("completion", "")
            cleaned = clean_completion(completion)
            extracted = extract_math_answer(cleaned)

            if extracted:
                extracted_count += 1

            if "####" in cleaned or "\\boxed" in cleaned:
                marker_count += 1

            problems = check_record(line_no, record)

            if not args.require_marker:
                # Downgrade missing marker to non-fatal / informational by removing it.
                problems = [
                    p for p in problems
                    if p != "completion has no #### marker or boxed answer"
                ]

            if problems:
                bad_count += 1

                for p in problems:
                    problem_counts[p] = problem_counts.get(p, 0) + 1
            else:
                ok_count += 1

            if args.show_errors_only and not problems:
                continue

            log_text = format_record_for_log(line_no, record, problems)
            log_f.write(log_text)
            log_f.write("\n")
            log_f.flush()

            logged_count += 1

            print("=" * 80)
            print(f"Line: {line_no}")
            print(f"Task ID: {record.get('task_id')}")
            print(f"Status: {status}")
            print(f"Kind: {kind}")
            print(f"Prompt: {str(record.get('prompt', ''))[:180]}")
            print(f"Gold: {get_gold_answer(record)}")
            print(f"Extracted: {extracted}")

            if problems:
                print("Problems:")
                for p in problems:
                    print(f"  - {p}")
            else:
                print("Problems: none")

            if args.show_completion:
                print("\nCompletion tail:")
                print("-" * 40)
                print(str(completion)[-1200:].strip())
                print("-" * 40)

        summary_lines = []
        summary_lines.append("\n" + "=" * 100)
        summary_lines.append("SUMMARY")
        summary_lines.append("=" * 100)
        summary_lines.append(f"Input file: {path}")
        summary_lines.append(f"Log file: {log_path}")
        summary_lines.append(f"Total records read: {total}")
        summary_lines.append(f"Records inspected after filters: {inspected}")
        summary_lines.append(f"Records written to log: {logged_count}")
        summary_lines.append(f"OK records: {ok_count}")
        summary_lines.append(f"Bad records: {bad_count}")
        summary_lines.append(f"Records with extractable answer: {extracted_count}")
        summary_lines.append(f"Records with #### or boxed marker: {marker_count}")
        summary_lines.append("")
        summary_lines.append("Status counts:")

        for status, count in sorted(status_counts.items()):
            summary_lines.append(f"  {status}: {count}")

        summary_lines.append("")
        summary_lines.append("Kind counts:")

        for kind, count in sorted(kind_counts.items()):
            summary_lines.append(f"  {kind}: {count}")

        if problem_counts:
            summary_lines.append("")
            summary_lines.append("Problem counts:")

            for problem, count in sorted(problem_counts.items()):
                summary_lines.append(f"  {problem}: {count}")

        summary_text = "\n".join(summary_lines)

        log_f.write(summary_text)
        log_f.write("\n")

    print(summary_text)
    print(f"\nWrote readable log to: {log_path}")

    if bad_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
