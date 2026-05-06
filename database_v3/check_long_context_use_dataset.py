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


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_answer(text: str) -> str:
    text = normalize_text(text)
    text = text.lower()
    text = text.strip(" .")
    return text


def extract_final_answer(completion: str):
    if not isinstance(completion, str):
        return None

    match = FINAL_RE.search(completion.strip())

    if not match:
        return None

    return match.group(1).strip()


def count_words(text: str) -> int:
    if not isinstance(text, str):
        return 0

    return len(text.split())


def count_paragraphs(text: str) -> int:
    if not isinstance(text, str):
        return 0

    parts = [p for p in text.split("\n\n") if p.strip()]
    return len(parts)


def short_text(text: str, max_chars: int = 500) -> str:
    text = str(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "...<truncated>"


def short_json(obj, max_chars: int = 3000) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        text = repr(obj)

    if len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>"

    return text


def check_schema(record: dict, answer_only: bool = False) -> list[str]:
    problems = []

    if "_json_error" in record:
        problems.append(f"JSON decode error: {record['_json_error']}")
        return problems

    required_fields = [
        "database_version",
        "task_id",
        "src",
        "kind",
        "context",
        "question",
        "prompt",
        "completion",
        "answer",
        "gold",
        "answer_type",
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
        "src",
        "kind",
        "context",
        "question",
        "prompt",
        "completion",
        "answer",
        "gold",
        "answer_type",
        "status",
    ]

    for field in string_fields:
        if field in record and not isinstance(record.get(field), str):
            problems.append(f"{field} exists but is not a string")

    if "seed" in record and not isinstance(record.get("seed"), int):
        problems.append("seed exists but is not an int")

    if "metadata" in record and not isinstance(record.get("metadata"), dict):
        problems.append("metadata exists but is not a dict")

    context = record.get("context", "")
    question = record.get("question", "")
    prompt = record.get("prompt", "")
    completion = record.get("completion", "")
    answer = record.get("answer", "")
    gold = record.get("gold", "")
    metadata = record.get("metadata", {})

    if isinstance(context, str):
        if not context.strip():
            problems.append("context is empty")

        if "Paragraph" not in context:
            problems.append("context does not appear to contain paragraph markers")

        if "FACT" not in context:
            problems.append("context does not appear to contain a FACT paragraph")

    if isinstance(question, str):
        if not question.strip():
            problems.append("question is empty")

        if not question.strip().endswith("?"):
            problems.append("question does not end with '?'")

    if isinstance(prompt, str):
        if "Context:" not in prompt:
            problems.append("prompt missing Context section")

        if "Question:" not in prompt:
            problems.append("prompt missing Question section")

        if "Answer:" not in prompt:
            problems.append("prompt missing Answer section")

        if isinstance(context, str) and context.strip() and context not in prompt:
            problems.append("prompt does not contain exact context text")

        if isinstance(question, str) and question.strip() and question not in prompt:
            problems.append("prompt does not contain exact question text")

        if answer_only:
            if "Output only the answer" not in prompt:
                problems.append("answer-only prompt missing output-only instruction")
        else:
            if "Final answer:" not in prompt:
                problems.append("prompt missing Final answer instruction")

            if "Only use information found in the context" not in prompt:
                problems.append("prompt missing context-grounding instruction")

    if isinstance(completion, str):
        if not completion.strip():
            problems.append("completion is empty")

        if not completion.endswith("\n"):
            problems.append("completion does not end with newline")

        if "```" in completion:
            problems.append("completion contains markdown fence")

        if not answer_only and "Final answer:" not in completion:
            problems.append("completion missing Final answer line")

    if isinstance(answer, str) and isinstance(gold, str):
        if normalize_answer(answer) != normalize_answer(gold):
            problems.append("answer does not match gold after normalization")

    if isinstance(metadata, dict):
        if "context_chars" in metadata:
            try:
                meta_chars = int(metadata["context_chars"])
                actual_chars = len(context)

                if meta_chars != actual_chars:
                    problems.append(
                        f"metadata context_chars mismatch: metadata={meta_chars}, actual={actual_chars}"
                    )
            except Exception:
                problems.append("metadata context_chars is not an int-compatible value")

        if "context_words" in metadata:
            try:
                meta_words = int(metadata["context_words"])
                actual_words = count_words(context)

                if meta_words != actual_words:
                    problems.append(
                        f"metadata context_words mismatch: metadata={meta_words}, actual={actual_words}"
                    )
            except Exception:
                problems.append("metadata context_words is not an int-compatible value")

        if "total_paragraphs" in metadata:
            try:
                meta_paragraphs = int(metadata["total_paragraphs"])
                actual_paragraphs = count_paragraphs(context)

                if meta_paragraphs != actual_paragraphs:
                    problems.append(
                        f"metadata total_paragraphs mismatch: metadata={meta_paragraphs}, actual={actual_paragraphs}"
                    )
            except Exception:
                problems.append("metadata total_paragraphs is not an int-compatible value")

        if "normalized_gold" in metadata:
            if metadata["normalized_gold"] != normalize_answer(gold):
                problems.append("metadata normalized_gold does not match normalized gold")

    return problems


def verify_answer(record: dict, answer_only: bool = False) -> tuple[bool, str, dict]:
    verification = {}

    try:
        completion = record.get("completion", "")
        gold = record.get("gold", "")
        context = record.get("context", "")
        prompt = record.get("prompt", "")

        if answer_only:
            predicted = completion.strip()
            has_final_answer_line = extract_final_answer(completion) is not None
        else:
            predicted = extract_final_answer(completion)
            has_final_answer_line = predicted is not None

            if predicted is None:
                verification["predicted"] = None
                verification["gold"] = gold
                verification["has_final_answer_line"] = False
                return False, "No 'Final answer:' line found.", verification

        normalized_predicted = normalize_answer(predicted)
        normalized_gold = normalize_answer(gold)

        verification["answer_only"] = answer_only
        verification["has_final_answer_line"] = has_final_answer_line
        verification["predicted"] = predicted
        verification["gold"] = gold
        verification["normalized_predicted"] = normalized_predicted
        verification["normalized_gold"] = normalized_gold
        verification["matched"] = normalized_predicted == normalized_gold
        verification["context_chars"] = len(context) if isinstance(context, str) else 0
        verification["context_words"] = count_words(context)
        verification["context_paragraphs"] = count_paragraphs(context)
        verification["prompt_chars"] = len(prompt) if isinstance(prompt, str) else 0

        if normalized_predicted != normalized_gold:
            return (
                False,
                f"Answer mismatch: predicted={predicted!r}, gold={gold!r}",
                verification,
            )

        return True, "", verification

    except Exception:
        verification["exception"] = traceback.format_exc()
        return False, traceback.format_exc(), verification


def check_grounding(record: dict) -> list[str]:
    """
    Lightweight grounding check.

    For this synthetic dataset, the gold answer should usually appear somewhere
    in the context. This is expected for needle, lookup, timeline, and most count
    examples. Constraint-count answers may be numeric and appear many times or not
    as an explicit final count, so we only warn softly using a problem message.
    """
    problems = []

    context = record.get("context", "")
    gold = record.get("gold", "")
    kind = record.get("kind", "")

    if not isinstance(context, str) or not isinstance(gold, str):
        return problems

    if not gold.strip():
        return problems

    normalized_context = normalize_text(context).lower()
    normalized_gold = normalize_answer(gold)

    if kind == "constraint_count":
        # In constraint_count tasks, the answer is computed from rows.
        # The final count may not explicitly appear as text in the context.
        return problems

    if normalized_gold not in normalized_context:
        problems.append("gold answer does not appear directly in context")

    return problems


def check_record(
    line_no: int,
    record: dict,
    answer_only: bool = False,
) -> tuple[list[str], dict]:
    problems = []
    verification = {}

    schema_problems = check_schema(record, answer_only=answer_only)
    problems.extend(schema_problems)

    if "_json_error" in record:
        return problems, verification

    core_ok = all(
        isinstance(record.get(field), str)
        for field in ["context", "question", "prompt", "completion", "gold"]
    )

    if not core_ok:
        return problems, verification

    problems.extend(check_grounding(record))

    ok, err, verification = verify_answer(
        record=record,
        answer_only=answer_only,
    )

    if not ok:
        problems.append(err)

    return problems, verification


def format_record_for_log(
    line_no: int,
    record: dict,
    problems: list[str],
    verification: dict,
) -> str:
    context = record.get("context", "")
    prompt = record.get("prompt", "")
    completion = record.get("completion", "")

    lines = []

    lines.append("=" * 100)
    lines.append(f"LINE: {line_no}")
    lines.append(f"TASK ID: {record.get('task_id', 'unknown')}")
    lines.append(f"DATABASE VERSION: {record.get('database_version', 'unknown')}")
    lines.append(f"SRC: {record.get('src', 'unknown')}")
    lines.append(f"KIND: {record.get('kind', 'unknown')}")
    lines.append(f"STATUS: {record.get('status', 'unknown')}")
    lines.append(f"ANSWER TYPE: {record.get('answer_type', 'unknown')}")
    lines.append(f"SEED: {record.get('seed', 'unknown')}")
    lines.append("")

    lines.append("[QUESTION]")
    lines.append(str(record.get("question", "")).strip())
    lines.append("")

    lines.append("[CONTEXT STATS]")
    lines.append(f"Characters: {len(context) if isinstance(context, str) else 0}")
    lines.append(f"Words: {count_words(context)}")
    lines.append(f"Paragraphs: {count_paragraphs(context)}")
    lines.append("")

    lines.append("[CONTEXT HEAD]")
    lines.append(short_text(context[:2000], max_chars=2000))
    lines.append("")

    lines.append("[CONTEXT TAIL]")
    lines.append(short_text(context[-2000:], max_chars=2000))
    lines.append("")

    lines.append("[PROMPT HEAD]")
    lines.append(short_text(prompt[:2000], max_chars=2000))
    lines.append("")

    lines.append("[COMPLETION REPR]")
    lines.append(repr(completion))
    lines.append("")

    lines.append("[COMPLETION TEXT]")
    lines.append(str(completion).rstrip() if completion else "None")
    lines.append("")

    lines.append("[ANSWER]")
    lines.append(str(record.get("answer", "")).strip())
    lines.append("")

    lines.append("[GOLD]")
    lines.append(str(record.get("gold", "")).strip())
    lines.append("")

    lines.append("[EXTRACTED FINAL ANSWER]")
    extracted = extract_final_answer(completion)
    lines.append(str(extracted) if extracted is not None else "None")
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
    show_context: bool = False,
    show_prompt: bool = False,
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
    print(f"Matched: {verification.get('matched')}")
    print(f"Context words: {verification.get('context_words')}")
    print(f"Context paragraphs: {verification.get('context_paragraphs')}")

    if problems:
        print("Problems:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("Problems: none")

    if show_context:
        print("")
        print("[CONTEXT HEAD]")
        print("-" * 40)
        print(short_text(record.get("context", "")[:3000], max_chars=3000))
        print("-" * 40)
        print("[CONTEXT TAIL]")
        print("-" * 40)
        print(short_text(record.get("context", "")[-3000:], max_chars=3000))
        print("-" * 40)

    if show_prompt:
        print("")
        print("[PROMPT HEAD]")
        print("-" * 40)
        print(short_text(record.get("prompt", "")[:3000], max_chars=3000))
        print("-" * 40)

    if show_completion:
        print("")
        print("[COMPLETION]")
        print("-" * 40)
        print(repr(record.get("completion", "")))
        print(str(record.get("completion", "")).rstrip())
        print("-" * 40)

    if show_metadata:
        print("")
        print("[METADATA]")
        print("-" * 40)
        print(short_json(record.get("metadata"), max_chars=1600))
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check synthetic long-context JSONL records and verify that "
            "completion answers match gold answers."
        )
    )

    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default="dataset/long_context_database_all_cases.jsonl",
        help="Path to long-context database JSONL.",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="database_v3/long_context_dataset_check.log",
        help="Path to output readable log file.",
    )

    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Check completions as direct answer-only outputs.",
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print context head/tail to terminal.",
    )

    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print prompt head to terminal.",
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

    parser.add_argument(
        "--min-context-words",
        type=int,
        default=0,
        help="Require each checked record to have at least this many context words.",
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
    answer_type_counts = {}

    context_word_counts = []
    context_char_counts = []
    context_paragraph_counts = []

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
            answer_type = record.get("answer_type", "unknown")

            status_counts[status] = status_counts.get(status, 0) + 1
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            version_counts[version] = version_counts.get(version, 0) + 1
            answer_type_counts[answer_type] = answer_type_counts.get(answer_type, 0) + 1

            problems, verification = check_record(
                line_no=line_no,
                record=record,
                answer_only=args.answer_only,
            )

            context_words = count_words(record.get("context", ""))
            context_chars = len(record.get("context", "")) if isinstance(record.get("context", ""), str) else 0
            context_paragraphs = count_paragraphs(record.get("context", ""))

            context_word_counts.append(context_words)
            context_char_counts.append(context_chars)
            context_paragraph_counts.append(context_paragraphs)

            if args.min_context_words and context_words < args.min_context_words:
                problems.append(
                    f"context has fewer words than required: {context_words} < {args.min_context_words}"
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
                show_context=args.show_context,
                show_prompt=args.show_prompt,
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
        summary_lines.append(f"Minimum context words required: {args.min_context_words}")
        summary_lines.append(f"Total JSONL records seen: {total_seen}")
        summary_lines.append(f"Records checked after filters: {total_checked}")
        summary_lines.append(f"Records written to log: {logged_count}")
        summary_lines.append(f"OK records: {ok_count}")
        summary_lines.append(f"Bad records: {bad_count}")
        summary_lines.append("")

        if context_word_counts:
            summary_lines.append("Context word stats:")
            summary_lines.append(f"  Min words: {min(context_word_counts)}")
            summary_lines.append(f"  Max words: {max(context_word_counts)}")
            summary_lines.append(f"  Avg words: {sum(context_word_counts) // len(context_word_counts)}")
            summary_lines.append("")

            summary_lines.append("Context char stats:")
            summary_lines.append(f"  Min chars: {min(context_char_counts)}")
            summary_lines.append(f"  Max chars: {max(context_char_counts)}")
            summary_lines.append(f"  Avg chars: {sum(context_char_counts) // len(context_char_counts)}")
            summary_lines.append("")

            summary_lines.append("Context paragraph stats:")
            summary_lines.append(f"  Min paragraphs: {min(context_paragraph_counts)}")
            summary_lines.append(f"  Max paragraphs: {max(context_paragraph_counts)}")
            summary_lines.append(f"  Avg paragraphs: {sum(context_paragraph_counts) // len(context_paragraph_counts)}")
            summary_lines.append("")

        summary_lines.append("Database version counts:")
        for version, count in sorted(version_counts.items()):
            summary_lines.append(f"  {version}: {count}")

        summary_lines.append("")
        summary_lines.append("Status counts:")
        for status, count in sorted(status_counts.items()):
            summary_lines.append(f"  {status}: {count}")

        summary_lines.append("")
        summary_lines.append("Answer type counts:")
        for answer_type, count in sorted(answer_type_counts.items()):
            summary_lines.append(f"  {answer_type}: {count}")

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
