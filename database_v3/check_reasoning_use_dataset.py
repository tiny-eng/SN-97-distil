#!/usr/bin/env python3

import argparse
import json
import re
import traceback
from pathlib import Path


FINAL_RE = re.compile(r"Final answer:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)


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


def normalize_answer(text: str) -> str:
    text = str(text).strip()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def extract_final_answer(completion: str):
    if not completion:
        return None

    match = FINAL_RE.search(completion.strip())

    if not match:
        return None

    return match.group(1).strip()


def check_record(
    line_no: int,
    record: dict,
    answer_only: bool = False,
):
    problems = []
    verification = {}

    if "_json_error" in record:
        problems.append(f"JSON decode error: {record['_json_error']}")
        verification["json_error"] = record["_json_error"]
        return problems, verification

    required_fields = [
        "prompt",
        "completion",
        "question",
        "gold",
        "status",
    ]

    for field in required_fields:
        if field not in record:
            problems.append(f"missing field: {field}")

    prompt = record.get("prompt")
    completion = record.get("completion")
    question = record.get("question")
    gold = record.get("gold")

    if prompt is not None and not isinstance(prompt, str):
        problems.append("prompt exists but is not a string")

    if completion is not None and not isinstance(completion, str):
        problems.append("completion exists but is not a string")

    if question is not None and not isinstance(question, str):
        problems.append("question exists but is not a string")

    if gold is None:
        problems.append("gold is null")

    if isinstance(completion, str) and not completion.strip():
        problems.append("completion is empty")

    if problems:
        return problems, verification

    try:
        if answer_only:
            predicted = completion.strip()
            has_final_answer_line = FINAL_RE.search(completion.strip()) is not None
        else:
            predicted = extract_final_answer(completion)
            has_final_answer_line = predicted is not None

            if predicted is None:
                problems.append("no 'Final answer:' line found")

        verification["answer_only"] = answer_only
        verification["has_final_answer_line"] = has_final_answer_line
        verification["predicted"] = predicted
        verification["gold"] = gold
        verification["normalized_predicted"] = normalize_answer(predicted or "")
        verification["normalized_gold"] = normalize_answer(gold)

        if predicted is not None:
            if normalize_answer(predicted) != normalize_answer(gold):
                problems.append(
                    f"final answer mismatch: predicted={predicted!r}, gold={gold!r}"
                )

    except Exception:
        problems.append(traceback.format_exc())

    return problems, verification


def short_json(obj, max_chars: int = 2000) -> str:
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
    completion = record.get("completion", "")

    lines = []

    lines.append("=" * 100)
    lines.append(f"LINE: {line_no}")
    lines.append(f"TASK ID: {record.get('task_id', 'unknown')}")
    lines.append(f"SRC: {record.get('src', 'unknown')}")
    lines.append(f"KIND: {record.get('kind', 'unknown')}")
    lines.append(f"STATUS: {record.get('status', 'unknown')}")
    lines.append(f"SEED: {record.get('seed', 'unknown')}")
    lines.append("")

    lines.append("[QUESTION]")
    lines.append(str(record.get("question", "")).strip())
    lines.append("")

    lines.append("[PROMPT]")
    lines.append(str(record.get("prompt", "")).strip())
    lines.append("")

    lines.append("[COMPLETION]")
    if completion:
        lines.append(str(completion).strip())
    else:
        lines.append("None")
    lines.append("")

    lines.append("[GOLD ANSWER]")
    lines.append(str(record.get("gold", "")).strip())
    lines.append("")

    lines.append("[EXTRACTED FINAL ANSWER]")
    extracted = extract_final_answer(completion) if isinstance(completion, str) else None
    lines.append(str(extracted) if extracted is not None else "None")
    lines.append("")

    lines.append("[VERIFICATION]")
    lines.append(short_json(verification))
    lines.append("")

    if "metadata" in record:
        lines.append("[METADATA]")
        lines.append(short_json(record.get("metadata")))
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
        lines.append(str(completion)[-1000:].strip())
    else:
        lines.append("None")

    lines.append("")

    return "\n".join(lines)


def print_terminal_record(
    line_no: int,
    record: dict,
    problems: list[str],
    verification: dict,
    show_completion: bool = False,
    show_metadata: bool = False,
) -> None:
    print("=" * 80)
    print(f"Line: {line_no}")
    print(f"Task ID: {record.get('task_id', 'unknown')}")
    print(f"Kind: {record.get('kind', 'unknown')}")
    print(f"Status: {record.get('status', 'unknown')}")
    print(f"Question: {str(record.get('question', ''))[:220]}")
    print(f"Gold: {record.get('gold')}")
    print(f"Predicted: {verification.get('predicted')}")

    if problems:
        print("Problems:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("Problems: none")

    if show_completion:
        print("")
        print("[COMPLETION]")
        print("-" * 40)
        print(str(record.get("completion", "")).strip())
        print("-" * 40)

    if show_metadata:
        print("")
        print("[METADATA]")
        print(short_json(record.get("metadata"), max_chars=1200))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check reasoning database JSONL records and log questions, "
            "gold answers, completions, and extracted final answers."
        )
    )

    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default="dataset/reasoning_database_all_cases.jsonl",
        help="Path to reasoning database JSONL.",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="database_v3/reasoning_dataset_check.log",
        help="Path to output readable log file.",
    )

    parser.add_argument(
        "--show-completion",
        action="store_true",
        help="Print completion text to terminal.",
    )

    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Print metadata to terminal.",
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
        "--answer-only",
        action="store_true",
        help="Check completion as direct answer only, without requiring 'Final answer:'.",
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

    with log_path.open("w", encoding="utf-8") as log_f:
        for line_no, record in load_jsonl(path):
            total_seen += 1

            if args.kind is not None and record.get("kind") != args.kind:
                continue

            total_checked += 1

            if args.limit and total_checked > args.limit:
                total_checked -= 1
                break

            status = record.get("status", "unknown")
            kind = record.get("kind", "unknown")

            status_counts[status] = status_counts.get(status, 0) + 1
            kind_counts[kind] = kind_counts.get(kind, 0) + 1

            problems, verification = check_record(
                line_no=line_no,
                record=record,
                answer_only=args.answer_only,
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
                show_completion=args.show_completion,
                show_metadata=args.show_metadata,
            )

        summary_lines = []
        summary_lines.append("\n" + "=" * 100)
        summary_lines.append("SUMMARY")
        summary_lines.append("=" * 100)
        summary_lines.append(f"Input file: {path}")
        summary_lines.append(f"Log file: {log_path}")
        summary_lines.append(f"Answer-only mode: {args.answer_only}")
        summary_lines.append(f"Total JSONL records seen: {total_seen}")
        summary_lines.append(f"Records checked after filters: {total_checked}")
        summary_lines.append(f"Records written to log: {logged_count}")
        summary_lines.append(f"OK records: {ok_count}")
        summary_lines.append(f"Bad records: {bad_count}")
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
