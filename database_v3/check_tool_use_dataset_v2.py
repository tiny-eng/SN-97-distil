import argparse
import json
import re
from pathlib import Path


TOOL_CALL_RE = re.compile(r"<python>\s*(.*?)\s*</python>", re.DOTALL)


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


def extract_python_from_completion(completion: str):
    if not completion:
        return None

    match = TOOL_CALL_RE.search(completion)

    if not match:
        return None

    return match.group(1).strip()


def get_messages_content(record: dict, role: str, index: int = 0) -> str | None:
    """Extract content from messages list for a specific role."""
    messages = record.get("messages", [])
    
    if not isinstance(messages, list):
        return None
    
    # Filter messages by role
    role_messages = [m for m in messages if isinstance(m, dict) and m.get("role") == role]
    
    if index >= len(role_messages):
        return None
    
    return role_messages[index].get("content", "")


def check_record(line_no: int, record: dict):
    problems = []

    if "_json_error" in record:
        problems.append(f"JSON decode error: {record['_json_error']}")
        return problems

    # New required fields for the two-pass format
    required_fields = [
        "kind",
        "question",
        "gold",
        "used_tool",
        "tool_call",
        "tool_result",
        "pass_id",
        "status",
        "messages",
    ]

    for field in required_fields:
        if field not in record:
            problems.append(f"missing field: {field}")

    # Check messages structure
    messages = record.get("messages", [])
    
    if not isinstance(messages, list):
        problems.append("messages is not a list")
    else:
        if len(messages) != 2:
            problems.append(f"messages should have exactly 2 items, got {len(messages)}")
        
        # Check first message (user)
        if len(messages) >= 1:
            msg1 = messages[0]
            if not isinstance(msg1, dict):
                problems.append("first message is not a dict")
            elif msg1.get("role") != "user":
                problems.append(f"first message role should be 'user', got '{msg1.get('role')}'")
            elif not msg1.get("content"):
                problems.append("first message has no content")
        
        # Check second message (assistant)
        if len(messages) >= 2:
            msg2 = messages[1]
            if not isinstance(msg2, dict):
                problems.append("second message is not a dict")
            elif msg2.get("role") != "assistant":
                problems.append(f"second message role should be 'assistant', got '{msg2.get('role')}'")
            elif not msg2.get("content"):
                problems.append("second message has no content")

    # Get assistant's completion from messages
    completion = get_messages_content(record, "assistant", 0) or ""
    tool_call_field = record.get("tool_call", "")
    used_tool = record.get("used_tool", False)
    pass_id = record.get("pass_id")
    status = record.get("status", "")
    
    extracted_code = extract_python_from_completion(completion)

    # Check tool usage consistency
    if used_tool and not tool_call_field and not extracted_code:
        problems.append("used_tool=true but no tool_call and no <python> block found")

    if tool_call_field and not isinstance(tool_call_field, str):
        problems.append("tool_call exists but is not a string")

    if extracted_code and tool_call_field:
        if extracted_code.strip() != tool_call_field.strip():
            problems.append("tool_call field does not match <python>...</python> block")

    # Pass-specific checks
    if pass_id == 1:
        # Pass 1 should have tool call but no output yet
        if status == "generated_with_tool_pass1":
            if not used_tool:
                problems.append("pass1 status but used_tool=false")
            if not tool_call_field and not extracted_code:
                problems.append("pass1 but no Python code found")
            if "<output>" in completion:
                problems.append("pass1 completion contains <output> tag (should not have tool result yet)")
        
    elif pass_id == 2:
        # Pass 2 should have the final answer in box
        if status == "generated_with_tool_pass2":
            if not used_tool:
                problems.append("pass2 status but used_tool=false")
            
            # Check for boxed answer
            if "\\boxed{" not in completion:
                problems.append("pass2 completion missing \\boxed{ANSWER}")
            
            # Check for tool result in user message (pass2 prompt includes it)
            user_content = get_messages_content(record, "user", 0) or ""
            if "<output>" not in user_content:
                problems.append("pass2 user message missing <output> tag")
    
    else:
        problems.append(f"invalid pass_id: {pass_id} (should be 1 or 2)")

    # Check tool_result is present
    tool_result = record.get("tool_result")
    if tool_result is None:
        problems.append("tool_result is null")
    elif used_tool and not isinstance(tool_result, str):
        problems.append("tool_result should be a string")

    # Check gold answer matches
    gold = record.get("gold", "")
    if gold and not isinstance(gold, str):
        problems.append("gold should be a string")

    return problems


def is_trusted_record(record: dict, problems: list[str]) -> bool:
    """Determine if a record is trusted for training."""
    if problems:
        return False
    
    # Basic validation
    if not record.get("used_tool", False):
        return False
    
    status = record.get("status", "")
    pass_id = record.get("pass_id")
    
    # For pass 1: must have tool call
    if pass_id == 1:
        if not record.get("tool_call") and not extract_python_from_completion(
            get_messages_content(record, "assistant", 0) or ""
        ):
            return False
        return "generated_with_tool_pass1" in status
    
    # For pass 2: must have boxed answer
    elif pass_id == 2:
        completion = get_messages_content(record, "assistant", 0) or ""
        if "\\boxed{" not in completion:
            return False
        return "generated_with_tool_pass2" in status
    
    return False


def format_record_for_log(line_no: int, record: dict, problems: list[str]) -> str:
    completion = get_messages_content(record, "assistant", 0) or ""
    user_content = get_messages_content(record, "user", 0) or ""
    extracted_code = extract_python_from_completion(completion)
    code = record.get("tool_call") or extracted_code

    lines = []

    lines.append("=" * 100)
    lines.append(f"LINE: {line_no}")
    lines.append(f"KIND: {record.get('kind', 'unknown')}")
    lines.append(f"PASS: {record.get('pass_id', 'unknown')}")
    lines.append(f"STATUS: {record.get('status', 'unknown')}")
    lines.append(f"USED TOOL: {record.get('used_tool')}")
    lines.append("")

    lines.append("[QUESTION]")
    lines.append(str(record.get("question", "")).strip())
    lines.append("")

    lines.append("[GOLD ANSWER]")
    lines.append(str(record.get("gold", "")).strip())
    lines.append("")

    lines.append("[PYTHON CODE]")
    if code:
        lines.append(code.strip())
    else:
        lines.append("None")
    lines.append("")

    lines.append("[TOOL RESULT]")
    tool_result = record.get("tool_result")
    if tool_result is None:
        lines.append("None")
    else:
        lines.append(repr(tool_result))
    lines.append("")

    lines.append("[ASSISTANT COMPLETION]")
    if completion:
        lines.append(completion[:500])
    else:
        lines.append("None")
    lines.append("")

    lines.append("[PROBLEMS]")
    if problems:
        for p in problems:
            lines.append(f"- {p}")
    else:
        lines.append("None")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Check tool_use_dataset.jsonl records with two-pass format."
    )

    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default="dataset/tool_use_database_all_cases_v2.jsonl",
        help="Path to tool_use_dataset.jsonl",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="database_v3/tool_use_dataset_check.log",
        help="Path to output log file.",
    )

    parser.add_argument(
        "--trusted-data",
        type=str,
        default="database_v3/trusted_tool_use_data.jsonl",
        help="Path to output trusted records (for training).",
    )

    parser.add_argument(
        "--show-python",
        action="store_true",
        help="Print extracted Python code block to terminal.",
    )

    parser.add_argument(
        "--show-errors-only",
        action="store_true",
        help="Only print/log records with problems.",
    )

    parser.add_argument(
        "--tool-only",
        action="store_true",
        help="Only print/log records that used a Python tool.",
    )

    parser.add_argument(
        "--pass-filter",
        type=int,
        choices=[1, 2],
        help="Filter by pass_id (1 or 2).",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of records to inspect. 0 means all.",
    )

    parser.add_argument(
        "--extract-trusted",
        action="store_true",
        help="Extract trusted records for training (pass2 only).",
    )

    args = parser.parse_args()

    path = Path(args.path)
    log_path = Path(args.log)
    trusted_path = Path(args.trusted_data)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    trusted_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    logged_count = 0
    ok_count = 0
    bad_count = 0
    used_tool_count = 0
    python_block_count = 0
    trusted_count = 0
    status_counts = {}
    pass_counts = {1: 0, 2: 0}
    trusted_records = []

    with log_path.open("w", encoding="utf-8") as log_f:
        for line_no, record in load_jsonl(path):
            total += 1

            if args.limit and total > args.limit:
                break

            status = record.get("status", "unknown")
            pass_id = record.get("pass_id")
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if pass_id in [1, 2]:
                pass_counts[pass_id] = pass_counts.get(pass_id, 0) + 1

            completion = get_messages_content(record, "assistant", 0) or ""
            extracted_code = extract_python_from_completion(completion)
            used_tool = record.get("used_tool", False)

            if used_tool:
                used_tool_count += 1

            if extracted_code:
                python_block_count += 1

            problems = check_record(line_no, record)

            if problems:
                bad_count += 1
            else:
                ok_count += 1
                # Check if this is a trusted record (pass 2, no problems)
                if is_trusted_record(record, problems):
                    trusted_count += 1
                    if args.extract_trusted and pass_id == 2:
                        # Store only the user/assistant messages for training
                        training_record = {
                            "messages": record.get("messages", []),
                            "kind": record.get("kind"),
                            "gold": record.get("gold"),
                            "pass_id": record.get("pass_id"),
                        }
                        trusted_records.append(training_record)

            # Apply filters
            if args.show_errors_only and not problems:
                continue

            if args.tool_only and not used_tool:
                continue

            if args.pass_filter and pass_id != args.pass_filter:
                continue

            log_text = format_record_for_log(line_no, record, problems)
            log_f.write(log_text)
            log_f.write("\n")
            log_f.flush()

            logged_count += 1

            print("=" * 80)
            print(f"Line: {line_no}")
            print(f"Kind: {record.get('kind')}")
            print(f"Pass: {pass_id}")
            print(f"Status: {status}")
            print(f"Used tool: {used_tool}")
            print(f"Question: {str(record.get('question', ''))[:180]}")
            print(f"Gold: {record.get('gold')}")

            if problems:
                print("Problems:")
                for p in problems:
                    print(f"  - {p}")
            else:
                print("Problems: none")
                if is_trusted_record(record, problems):
                    print("✓ TRUSTED RECORD")

            if args.show_python:
                code = record.get("tool_call") or extracted_code

                if code:
                    print("\nPython code:")
                    print("-" * 40)
                    print(code)
                    print("-" * 40)
                    print(f"Tool result: {repr(record.get('tool_result'))}")
                else:
                    print("\nPython code: none")

        summary_lines = []
        summary_lines.append("\n" + "=" * 100)
        summary_lines.append("SUMMARY")
        summary_lines.append("=" * 100)
        summary_lines.append(f"Input file: {path}")
        summary_lines.append(f"Log file: {log_path}")
        if args.extract_trusted:
            summary_lines.append(f"Trusted data file: {trusted_path}")
        summary_lines.append(f"Total records checked: {total}")
        summary_lines.append(f"Records written to log: {logged_count}")
        summary_lines.append(f"OK records: {ok_count}")
        summary_lines.append(f"Bad records: {bad_count}")
        summary_lines.append(f"Trusted records (pass2, no errors): {trusted_count}")
        summary_lines.append(f"Records with used_tool=true: {used_tool_count}")
        summary_lines.append(f"Records with <python> block: {python_block_count}")
        summary_lines.append("")
        summary_lines.append("Pass counts:")
        summary_lines.append(f"  Pass 1: {pass_counts.get(1, 0)}")
        summary_lines.append(f"  Pass 2: {pass_counts.get(2, 0)}")
        summary_lines.append("")
        summary_lines.append("Status counts:")

        for status, count in sorted(status_counts.items()):
            summary_lines.append(f"  {status}: {count}")

        summary_text = "\n".join(summary_lines)

        log_f.write(summary_text)
        log_f.write("\n")

    # Write trusted records to file
    if args.extract_trusted and trusted_records:
        with trusted_path.open("w", encoding="utf-8") as trusted_f:
            for record in trusted_records:
                trusted_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(trusted_records)} trusted records to: {trusted_path}")
    elif args.extract_trusted:
        print("\nNo trusted records found to extract.")

    print(summary_text)
    print(f"\nWrote readable log to: {log_path}")


if __name__ == "__main__":
    main()