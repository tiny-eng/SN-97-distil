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


def format_docs_for_prompt(docs: list[dict]) -> str:
    chunks = []

    for doc in docs:
        chunks.append(
            f"[Document ID: {doc['doc_id']}]\n"
            f"Title: {doc['title']}\n"
            f"{doc['text']}"
        )

    return "\n\n---\n\n".join(chunks)


def count_words(text: str) -> int:
    if not isinstance(text, str):
        return 0

    return len(text.split())


def short_text(text: str, max_chars: int = 1200) -> str:
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


def check_doc_schema(record: dict) -> list[str]:
    problems = []

    docs = record.get("docs")
    documents = record.get("documents")
    doc_count = record.get("doc_count")
    docs_text = record.get("docs_text")

    if not isinstance(docs, list):
        problems.append("docs is not a list")
        return problems

    if not docs:
        problems.append("docs is empty")
        return problems

    if documents != docs:
        problems.append("documents does not exactly match docs")

    if not isinstance(doc_count, int):
        problems.append("doc_count is not an int")
    elif doc_count != len(docs):
        problems.append(f"doc_count mismatch: doc_count={doc_count}, len(docs)={len(docs)}")

    seen_doc_ids = set()

    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            problems.append(f"docs[{i}] is not a dict")
            continue

        for field in ["doc_id", "title", "text"]:
            if field not in doc:
                problems.append(f"docs[{i}] missing field: {field}")
            elif not isinstance(doc[field], str):
                problems.append(f"docs[{i}][{field}] is not a string")
            elif not doc[field].strip():
                problems.append(f"docs[{i}][{field}] is empty")

        doc_id = doc.get("doc_id")

        if isinstance(doc_id, str):
            if doc_id in seen_doc_ids:
                problems.append(f"duplicate doc_id found: {doc_id}")
            seen_doc_ids.add(doc_id)

    if isinstance(docs_text, str):
        expected_docs_text = format_docs_for_prompt(docs)

        if docs_text != expected_docs_text:
            problems.append("docs_text does not match format_docs_for_prompt(docs)")

        for doc in docs:
            doc_id = doc.get("doc_id", "")
            title = doc.get("title", "")
            text = doc.get("text", "")

            if isinstance(doc_id, str) and f"[Document ID: {doc_id}]" not in docs_text:
                problems.append(f"docs_text missing document ID marker for {doc_id}")

            if isinstance(title, str) and title not in docs_text:
                problems.append(f"docs_text missing title for doc {doc_id}")

            if isinstance(text, str) and text not in docs_text:
                problems.append(f"docs_text missing text for doc {doc_id}")

    else:
        problems.append("docs_text is not a string")

    return problems


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
        "docs",
        "documents",
        "doc_count",
        "docs_text",
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
        "docs_text",
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

    problems.extend(check_doc_schema(record))

    docs_text = record.get("docs_text", "")
    question = record.get("question", "")
    prompt = record.get("prompt", "")
    completion = record.get("completion", "")
    answer = record.get("answer", "")
    gold = record.get("gold", "")
    metadata = record.get("metadata", {})

    if isinstance(question, str):
        if not question.strip():
            problems.append("question is empty")

        if not question.strip().endswith("?"):
            problems.append("question does not end with '?'")

    if isinstance(prompt, str):
        if "Documents:" not in prompt:
            problems.append("prompt missing Documents section")

        if "Question:" not in prompt:
            problems.append("prompt missing Question section")

        if "Answer:" not in prompt:
            problems.append("prompt missing Answer section")

        if isinstance(docs_text, str) and docs_text.strip() and docs_text not in prompt:
            problems.append("prompt does not contain exact docs_text")

        if isinstance(question, str) and question.strip() and question not in prompt:
            problems.append("prompt does not contain exact question")

        if answer_only:
            if "Output only the answer" not in prompt:
                problems.append("answer-only prompt missing output-only instruction")
        else:
            if "Final answer:" not in prompt:
                problems.append("prompt missing Final answer instruction")

            if "Only use information found in the documents" not in prompt:
                problems.append("prompt missing document-grounding instruction")

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
        docs = record.get("docs", [])

        if isinstance(docs, list):
            if "num_docs" in metadata:
                try:
                    meta_num_docs = int(metadata["num_docs"])

                    if meta_num_docs != len(docs):
                        problems.append(
                            f"metadata num_docs mismatch: metadata={meta_num_docs}, actual={len(docs)}"
                        )
                except Exception:
                    problems.append("metadata num_docs is not int-compatible")

            if "doc_ids" in metadata:
                actual_doc_ids = [
                    doc.get("doc_id")
                    for doc in docs
                    if isinstance(doc, dict)
                ]

                if metadata["doc_ids"] != actual_doc_ids:
                    problems.append("metadata doc_ids does not match docs doc_id order")

            if "total_doc_chars" in metadata:
                try:
                    meta_total_doc_chars = int(metadata["total_doc_chars"])
                    actual_total_doc_chars = sum(
                        len(doc.get("text", ""))
                        for doc in docs
                        if isinstance(doc, dict)
                    )

                    if meta_total_doc_chars != actual_total_doc_chars:
                        problems.append(
                            "metadata total_doc_chars mismatch: "
                            f"metadata={meta_total_doc_chars}, actual={actual_total_doc_chars}"
                        )
                except Exception:
                    problems.append("metadata total_doc_chars is not int-compatible")

            if "total_doc_words" in metadata:
                try:
                    meta_total_doc_words = int(metadata["total_doc_words"])
                    actual_total_doc_words = sum(
                        count_words(doc.get("text", ""))
                        for doc in docs
                        if isinstance(doc, dict)
                    )

                    if meta_total_doc_words != actual_total_doc_words:
                        problems.append(
                            "metadata total_doc_words mismatch: "
                            f"metadata={meta_total_doc_words}, actual={actual_total_doc_words}"
                        )
                except Exception:
                    problems.append("metadata total_doc_words is not int-compatible")

        if "docs_text_chars" in metadata:
            try:
                meta_docs_text_chars = int(metadata["docs_text_chars"])
                actual_docs_text_chars = len(docs_text)

                if meta_docs_text_chars != actual_docs_text_chars:
                    problems.append(
                        "metadata docs_text_chars mismatch: "
                        f"metadata={meta_docs_text_chars}, actual={actual_docs_text_chars}"
                    )
            except Exception:
                problems.append("metadata docs_text_chars is not int-compatible")

        if "docs_text_words" in metadata:
            try:
                meta_docs_text_words = int(metadata["docs_text_words"])
                actual_docs_text_words = count_words(docs_text)

                if meta_docs_text_words != actual_docs_text_words:
                    problems.append(
                        "metadata docs_text_words mismatch: "
                        f"metadata={meta_docs_text_words}, actual={actual_docs_text_words}"
                    )
            except Exception:
                problems.append("metadata docs_text_words is not int-compatible")

        if "normalized_gold" in metadata:
            if metadata["normalized_gold"] != normalize_answer(gold):
                problems.append("metadata normalized_gold does not match normalized gold")

    return problems


def verify_answer(record: dict, answer_only: bool = False) -> tuple[bool, str, dict]:
    verification = {}

    try:
        completion = record.get("completion", "")
        gold = record.get("gold", "")
        docs_text = record.get("docs_text", "")
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
        verification["doc_count"] = record.get("doc_count")
        verification["docs_text_chars"] = len(docs_text) if isinstance(docs_text, str) else 0
        verification["docs_text_words"] = count_words(docs_text)
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

    For lookup/join/latest/compare tasks, the final gold usually appears directly
    somewhere in the documents. For count tasks, the answer is computed and may
    not appear as an explicit final number, so we skip direct answer lookup.
    """
    problems = []

    kind = record.get("kind", "")
    gold = record.get("gold", "")
    docs_text = record.get("docs_text", "")

    if kind == "count_across_docs":
        return problems

    if not isinstance(gold, str) or not isinstance(docs_text, str):
        return problems

    if not gold.strip():
        return problems

    normalized_gold = normalize_answer(gold)
    normalized_docs_text = normalize_text(docs_text).lower()

    if normalized_gold not in normalized_docs_text:
        problems.append("gold answer does not appear directly in docs_text")

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
        for field in ["docs_text", "question", "prompt", "completion", "gold"]
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
    docs = record.get("docs", [])
    docs_text = record.get("docs_text", "")
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

    lines.append("[DOCUMENT STATS]")
    lines.append(f"Doc count: {len(docs) if isinstance(docs, list) else 0}")
    lines.append(f"docs_text chars: {len(docs_text) if isinstance(docs_text, str) else 0}")
    lines.append(f"docs_text words: {count_words(docs_text)}")
    lines.append("")

    lines.append("[DOC IDS]")
    if isinstance(docs, list):
        doc_ids = [
            doc.get("doc_id", "unknown")
            for doc in docs
            if isinstance(doc, dict)
        ]
        lines.append(", ".join(doc_ids))
    else:
        lines.append("docs is not a list")
    lines.append("")

    lines.append("[DOCS_TEXT HEAD]")
    lines.append(short_text(docs_text[:2500], max_chars=2500))
    lines.append("")

    lines.append("[DOCS_TEXT TAIL]")
    lines.append(short_text(docs_text[-2500:], max_chars=2500))
    lines.append("")

    lines.append("[PROMPT HEAD]")
    lines.append(short_text(prompt[:2500], max_chars=2500))
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
    show_docs: bool = False,
    show_prompt: bool = False,
    show_completion: bool = False,
    show_metadata: bool = False,
) -> None:
    print("=" * 80)
    print(f"Line: {line_no}")
    print(f"Task ID: {record.get('task_id', 'unknown')}")
    print(f"Kind: {record.get('kind', 'unknown')}")
    print(f"Status: {record.get('status', 'unknown')}")
    print(f"Doc count: {record.get('doc_count')}")
    print(f"Question: {str(record.get('question', ''))[:220]}")
    print(f"Gold: {record.get('gold')}")
    print(f"Predicted: {verification.get('predicted')}")
    print(f"Matched: {verification.get('matched')}")
    print(f"docs_text words: {verification.get('docs_text_words')}")

    if problems:
        print("Problems:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("Problems: none")

    if show_docs:
        print("")
        print("[DOCS_TEXT HEAD]")
        print("-" * 40)
        print(short_text(record.get("docs_text", "")[:3000], max_chars=3000))
        print("-" * 40)
        print("[DOCS_TEXT TAIL]")
        print("-" * 40)
        print(short_text(record.get("docs_text", "")[-3000:], max_chars=3000))
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
            "Check synthetic multi-document JSONL records and verify that "
            "completion answers match gold answers."
        )
    )

    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default="dataset/multi_doc_database_all_cases.jsonl",
        help="Path to multi-doc database JSONL.",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="database_v3/multi_doc_dataset_check.log",
        help="Path to output readable log file.",
    )

    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Check completions as direct answer-only outputs.",
    )

    parser.add_argument(
        "--show-docs",
        action="store_true",
        help="Print docs_text head/tail to terminal.",
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
        "--min-docs",
        type=int,
        default=0,
        help="Require each checked record to have at least this many documents.",
    )

    parser.add_argument(
        "--min-docs-text-words",
        type=int,
        default=0,
        help="Require docs_text to have at least this many words.",
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

    doc_counts = []
    docs_text_word_counts = []
    docs_text_char_counts = []

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

            docs = record.get("docs", [])
            docs_text = record.get("docs_text", "")

            actual_doc_count = len(docs) if isinstance(docs, list) else 0
            docs_text_words = count_words(docs_text)
            docs_text_chars = len(docs_text) if isinstance(docs_text, str) else 0

            doc_counts.append(actual_doc_count)
            docs_text_word_counts.append(docs_text_words)
            docs_text_char_counts.append(docs_text_chars)

            if args.min_docs and actual_doc_count < args.min_docs:
                problems.append(
                    f"doc count below required minimum: {actual_doc_count} < {args.min_docs}"
                )

            if args.min_docs_text_words and docs_text_words < args.min_docs_text_words:
                problems.append(
                    "docs_text has fewer words than required: "
                    f"{docs_text_words} < {args.min_docs_text_words}"
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
                show_docs=args.show_docs,
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
        summary_lines.append(f"Minimum docs required: {args.min_docs}")
        summary_lines.append(f"Minimum docs_text words required: {args.min_docs_text_words}")
        summary_lines.append(f"Total JSONL records seen: {total_seen}")
        summary_lines.append(f"Records checked after filters: {total_checked}")
        summary_lines.append(f"Records written to log: {logged_count}")
        summary_lines.append(f"OK records: {ok_count}")
        summary_lines.append(f"Bad records: {bad_count}")
        summary_lines.append("")

        if doc_counts:
            summary_lines.append("Document count stats:")
            summary_lines.append(f"  Min docs: {min(doc_counts)}")
            summary_lines.append(f"  Max docs: {max(doc_counts)}")
            summary_lines.append(f"  Avg docs: {sum(doc_counts) // len(doc_counts)}")
            summary_lines.append("")

            summary_lines.append("docs_text word stats:")
            summary_lines.append(f"  Min words: {min(docs_text_word_counts)}")
            summary_lines.append(f"  Max words: {max(docs_text_word_counts)}")
            summary_lines.append(f"  Avg words: {sum(docs_text_word_counts) // len(docs_text_word_counts)}")
            summary_lines.append("")

            summary_lines.append("docs_text char stats:")
            summary_lines.append(f"  Min chars: {min(docs_text_char_counts)}")
            summary_lines.append(f"  Max chars: {max(docs_text_char_counts)}")
            summary_lines.append(f"  Avg chars: {sum(docs_text_char_counts) // len(docs_text_char_counts)}")
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
