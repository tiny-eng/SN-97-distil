#!/usr/bin/env python3

import argparse
import json
import re
import traceback
from pathlib import Path


WORD_RE = re.compile(r"\b[\w'-]+\b")


SUPPORTED_INSTRUCTION_IDS = {
    "length_constraints:number_words",
    "length_constraints:number_sentences",
    "change_case:english_lowercase",
    "change_case:english_capital",
    "startend:end_checker",
    "keywords:frequency",
    "keywords:forbidden_words",
    "detectable_format:json_format",
    "punctuation:no_comma",
    "detectable_format:title",
    "detectable_format:number_bullet_lists",
}


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


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def count_sentences(text: str) -> int:
    text = (text or "").strip()

    if not text:
        return 0

    return len(re.findall(r"[.!?]+(?:\s|$)", text))


def count_keyword(text: str, keyword: str) -> int:
    return len(
        re.findall(
            rf"\b{re.escape(keyword)}\b",
            text or "",
            flags=re.IGNORECASE,
        )
    )


def has_forbidden_word(text: str, forbidden: str) -> bool:
    return (
        re.search(
            rf"\b{re.escape(forbidden)}\b",
            text or "",
            flags=re.IGNORECASE,
        )
        is not None
    )


def verify_instruction_local(
    completion: str,
    instruction_id: str,
    kwargs: dict,
) -> tuple[bool, str]:
    text = (completion or "").strip()

    try:
        if instruction_id == "length_constraints:number_words":
            n_words = count_words(text)
            target = int(kwargs["num_words"])
            relation = kwargs["relation"]

            if relation == "exactly":
                ok = n_words == target
            elif relation == "at least":
                ok = n_words >= target
            elif relation == "at most":
                ok = n_words <= target
            else:
                return False, f"unknown word relation: {relation}"

            return ok, f"word_count={n_words}, target={relation} {target}"

        if instruction_id == "length_constraints:number_sentences":
            n_sentences = count_sentences(text)
            target = int(kwargs["num_sentences"])
            relation = kwargs["relation"]

            if relation == "exactly":
                ok = n_sentences == target
            elif relation == "at least":
                ok = n_sentences >= target
            elif relation == "at most":
                ok = n_sentences <= target
            else:
                return False, f"unknown sentence relation: {relation}"

            return ok, f"sentence_count={n_sentences}, target={relation} {target}"

        if instruction_id == "change_case:english_lowercase":
            for ch in text:
                if ch.isalpha() and ch != ch.lower():
                    return False, f"uppercase_letter_found={ch!r}"

            return True, "lowercase_ok"

        if instruction_id == "change_case:english_capital":
            for ch in text:
                if ch.isalpha() and ch != ch.upper():
                    return False, f"lowercase_letter_found={ch!r}"

            return True, "uppercase_ok"

        if instruction_id == "startend:end_checker":
            phrase = kwargs["end_phrase"]
            ok = text.endswith(phrase)

            return ok, f"endswith={phrase!r}"

        if instruction_id == "keywords:frequency":
            keyword = kwargs["keyword"]
            frequency = int(kwargs["frequency"])
            relation = kwargs.get("relation", "at least")
            actual = count_keyword(text, keyword)

            if relation == "at least":
                ok = actual >= frequency
            elif relation == "exactly":
                ok = actual == frequency
            elif relation == "at most":
                ok = actual <= frequency
            else:
                return False, f"unknown keyword relation: {relation}"

            return (
                ok,
                f"keyword={keyword!r}, count={actual}, target={relation} {frequency}",
            )

        if instruction_id == "keywords:forbidden_words":
            forbidden_words = kwargs["forbidden_words"]

            if not isinstance(forbidden_words, list):
                return False, "forbidden_words is not a list"

            for forbidden in forbidden_words:
                if has_forbidden_word(text, forbidden):
                    return False, f"forbidden_word_found={forbidden!r}"

            return True, "forbidden_words_ok"

        if instruction_id == "detectable_format:json_format":
            parsed = json.loads(text)

            if not isinstance(parsed, dict):
                return False, "json_is_not_object"

            return True, "json_ok"

        if instruction_id == "punctuation:no_comma":
            ok = "," not in text

            return ok, "no_comma_ok" if ok else "comma_found"

        if instruction_id == "detectable_format:title":
            ok = re.match(r"^\s*<<[^>\n]+>>", text) is not None

            return ok, "title_ok" if ok else "missing_double_angle_title"

        if instruction_id == "detectable_format:number_bullet_lists":
            target = int(kwargs["num_bullets"])

            bullet_lines = [
                line
                for line in text.splitlines()
                if line.strip().startswith("- ") or line.strip().startswith("* ")
            ]

            actual = len(bullet_lines)
            ok = actual == target

            return ok, f"bullet_count={actual}, target={target}"

        return False, f"unsupported instruction_id: {instruction_id}"

    except Exception:
        return False, traceback.format_exc()


def verify_record_local(record: dict) -> tuple[bool, list[dict]]:
    completion = record.get("completion", "")
    instruction_ids = record.get("instruction_ids") or []
    kwargs_list = record.get("kwargs") or []

    results = []
    all_ok = True

    if len(instruction_ids) != len(kwargs_list):
        return False, [
            {
                "instruction_id": None,
                "ok": False,
                "detail": (
                    f"instruction_ids length {len(instruction_ids)} "
                    f"!= kwargs length {len(kwargs_list)}"
                ),
            }
        ]

    for instruction_id, kw in zip(instruction_ids, kwargs_list):
        ok, detail = verify_instruction_local(
            completion=completion,
            instruction_id=instruction_id,
            kwargs=kw,
        )

        results.append(
            {
                "instruction_id": instruction_id,
                "ok": ok,
                "detail": detail,
            }
        )

        if not ok:
            all_ok = False

    return all_ok, results


def verify_record_vendor(record: dict) -> tuple[bool, object]:
    try:
        import ifeval_vendor as ifev
    except ImportError:
        return False, "ifeval_vendor not importable"

    try:
        ok, per = ifev.evaluate_item(
            record.get("completion", ""),
            record.get("instruction_ids") or [],
            record.get("kwargs") or [],
        )

        return bool(ok), per

    except Exception:
        return False, traceback.format_exc()


def check_schema(record: dict) -> list[str]:
    problems = []

    if "_json_error" in record:
        problems.append(f"JSON decode error: {record['_json_error']}")
        return problems

    required_fields = [
        "prompt",
        "completion",
        "instruction_ids",
        "kwargs",
        "status",
    ]

    for field in required_fields:
        if field not in record:
            problems.append(f"missing field: {field}")

    prompt = record.get("prompt")
    completion = record.get("completion")
    instruction_ids = record.get("instruction_ids")
    kwargs_list = record.get("kwargs")

    if prompt is not None and not isinstance(prompt, str):
        problems.append("prompt exists but is not a string")

    if completion is not None and not isinstance(completion, str):
        problems.append("completion exists but is not a string")

    if isinstance(completion, str) and not completion.strip():
        problems.append("completion is empty")

    if instruction_ids is not None and not isinstance(instruction_ids, list):
        problems.append("instruction_ids exists but is not a list")

    if kwargs_list is not None and not isinstance(kwargs_list, list):
        problems.append("kwargs exists but is not a list")

    if isinstance(instruction_ids, list) and isinstance(kwargs_list, list):
        if len(instruction_ids) != len(kwargs_list):
            problems.append(
                f"instruction_ids length {len(instruction_ids)} "
                f"!= kwargs length {len(kwargs_list)}"
            )

    if isinstance(instruction_ids, list):
        for idx, instruction_id in enumerate(instruction_ids):
            if not isinstance(instruction_id, str):
                problems.append(f"instruction_ids[{idx}] is not a string")
                continue

            if instruction_id not in SUPPORTED_INSTRUCTION_IDS:
                problems.append(f"unsupported instruction_id: {instruction_id}")

    if isinstance(kwargs_list, list):
        for idx, kw in enumerate(kwargs_list):
            if not isinstance(kw, dict):
                problems.append(f"kwargs[{idx}] is not a dict")

    return problems


def check_record(
    line_no: int,
    record: dict,
    use_vendor: bool = False,
) -> tuple[list[str], list[dict] | object | None]:
    problems = check_schema(record)

    if problems:
        return problems, None

    if use_vendor:
        verified, verification = verify_record_vendor(record)
    else:
        verified, verification = verify_record_local(record)

    if not verified:
        problems.append("completion failed IFEval verification")

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
    verification,
) -> str:
    lines = []

    completion = record.get("completion", "")

    lines.append("=" * 100)
    lines.append(f"LINE: {line_no}")
    lines.append(f"TASK ID: {record.get('task_id', 'unknown')}")
    lines.append(f"SRC: {record.get('src', 'unknown')}")
    lines.append(f"KIND: {record.get('kind', 'unknown')}")
    lines.append(f"DIFFICULTY: {record.get('difficulty', 'unknown')}")
    lines.append(f"STATUS: {record.get('status', 'unknown')}")
    lines.append("")

    lines.append("[PROMPT]")
    lines.append(str(record.get("prompt", "")).strip())
    lines.append("")

    lines.append("[COMPLETION]")
    lines.append(str(completion).strip() if completion else "None")
    lines.append("")

    lines.append("[INSTRUCTION IDS]")
    lines.append(short_json(record.get("instruction_ids")))
    lines.append("")

    lines.append("[KWARGS]")
    lines.append(short_json(record.get("kwargs")))
    lines.append("")

    if "specs" in record:
        lines.append("[SPECS]")
        lines.append(short_json(record.get("specs")))
        lines.append("")

    lines.append("[LOCAL METRICS]")
    if completion:
        lines.append(f"word_count: {count_words(completion)}")
        lines.append(f"sentence_count: {count_sentences(completion)}")
        lines.append(f"line_count: {len(completion.splitlines())}")
        lines.append(f"has_comma: {',' in completion}")
        lines.append(f"starts_with_title: {bool(re.match(r'^\\s*<<[^>\\n]+>>', completion.strip()))}")
    else:
        lines.append("None")
    lines.append("")

    lines.append("[VERIFICATION]")
    if verification is None:
        lines.append("None")
    else:
        lines.append(short_json(verification))
    lines.append("")

    lines.append("[PROBLEMS]")
    if problems:
        for problem in problems:
            lines.append(f"- {problem}")
    else:
        lines.append("None")
    lines.append("")

    lines.append("[COMPLETION TAIL]")
    if completion:
        lines.append(completion[-1000:].strip())
    else:
        lines.append("None")

    lines.append("")

    return "\n".join(lines)


def print_terminal_record(
    line_no: int,
    record: dict,
    problems: list[str],
    verification,
    show_completion: bool = False,
    show_verification: bool = False,
) -> None:
    completion = record.get("completion", "")

    print("=" * 80)
    print(f"Line: {line_no}")
    print(f"Task ID: {record.get('task_id', 'unknown')}")
    print(f"Kind: {record.get('kind', 'unknown')}")
    print(f"Difficulty: {record.get('difficulty', 'unknown')}")
    print(f"Status: {record.get('status', 'unknown')}")
    print(f"Prompt: {str(record.get('prompt', ''))[:220]}")
    print(f"Completion words: {count_words(completion)}")
    print(f"Completion sentences: {count_sentences(completion)}")

    if problems:
        print("Problems:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("Problems: none")

    if show_completion:
        print("\nCompletion:")
        print("-" * 40)
        print(str(completion).strip())
        print("-" * 40)

    if show_verification:
        print("\nVerification:")
        print(short_json(verification, max_chars=1200))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check IFEval database JSONL records and log prompts, "
            "completions, instruction_ids, kwargs, and verification results."
        )
    )

    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default="dataset/ifeval_database_all_cases.jsonl",
        help="Path to IFEval JSONL database.",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="database_v3/ifeval_dataset_check.log",
        help="Path to output readable log file.",
    )

    parser.add_argument(
        "--show-completion",
        action="store_true",
        help="Print completion text to terminal.",
    )

    parser.add_argument(
        "--show-verification",
        action="store_true",
        help="Print verification details to terminal.",
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
        "--difficulty",
        type=str,
        default=None,
        help="Only inspect records with this difficulty.",
    )

    parser.add_argument(
        "--vendor-verify",
        action="store_true",
        help="Use ifeval_vendor.evaluate_item instead of the built-in local verifier.",
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
    difficulty_counts = {}
    instruction_counts = {}

    with log_path.open("w", encoding="utf-8") as log_f:
        for line_no, record in load_jsonl(path):
            total_seen += 1

            if args.kind is not None and record.get("kind") != args.kind:
                continue

            if args.difficulty is not None and record.get("difficulty") != args.difficulty:
                continue

            total_checked += 1

            if args.limit and total_checked > args.limit:
                total_checked -= 1
                break

            status = record.get("status", "unknown")
            kind = record.get("kind", "unknown")
            difficulty = record.get("difficulty", "unknown")

            status_counts[status] = status_counts.get(status, 0) + 1
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

            for instruction_id in record.get("instruction_ids") or []:
                instruction_counts[instruction_id] = instruction_counts.get(instruction_id, 0) + 1

            problems, verification = check_record(
                line_no=line_no,
                record=record,
                use_vendor=args.vendor_verify,
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
                show_verification=args.show_verification,
            )

        summary_lines = []
        summary_lines.append("\n" + "=" * 100)
        summary_lines.append("SUMMARY")
        summary_lines.append("=" * 100)
        summary_lines.append(f"Input file: {path}")
        summary_lines.append(f"Log file: {log_path}")
        summary_lines.append(f"Verifier: {'ifeval_vendor' if args.vendor_verify else 'local'}")
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

        summary_lines.append("")
        summary_lines.append("Difficulty counts:")
        for difficulty, count in sorted(difficulty_counts.items()):
            summary_lines.append(f"  {difficulty}: {count}")

        summary_lines.append("")
        summary_lines.append("Instruction counts:")
        for instruction_id, count in sorted(instruction_counts.items()):
            summary_lines.append(f"  {instruction_id}: {count}")

        summary_text = "\n".join(summary_lines)

        log_f.write(summary_text)
        log_f.write("\n")

    print(summary_text)
    print(f"\nWrote readable log to: {log_path}")


if __name__ == "__main__":
    main()
