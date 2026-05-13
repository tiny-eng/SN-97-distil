#!/usr/bin/env python3

import argparse
import json
import re
import shutil
import traceback
from datetime import datetime
from pathlib import Path


SYNTHESIS_KINDS = [
    "sum",
    "difference",
    "compare",
    "ratio",
    "sum_three",
    "difference_three",
]


INTEGER_KINDS = {
    "sum",
    "difference",
    "ratio",
    "sum_three",
    "difference_three",
}


NAME_ANSWER_KINDS = {
    "compare",
}


REQUIRED_TRAINING_FIELDS = [
    "src",
    "context",
    "question",
    "answer",
    "confuser_answers",
    "involved_topics",
    "kind",
    "messages",
    "pred",
    "passed",
]


OPTIONAL_LEGACY_FIELDS = [
    "database_version",
    "task_id",
    "gold",
    "metadata",
]


def load_jsonl_with_raw(path: Path):
    """
    Yield:
      line_no, record, raw_line, json_error

    If JSON parsing fails:
      record is {"_json_error": "...", "_raw": "..."}
      json_error is the exception text
    """
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()

            if not stripped:
                continue

            try:
                yield line_no, json.loads(stripped), raw_line, None
            except json.JSONDecodeError as e:
                yield line_no, {
                    "_json_error": str(e),
                    "_raw": stripped[:1000],
                }, raw_line, str(e)


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


def starts_with_the(text: str) -> bool:
    return isinstance(text, str) and text.startswith("the ")


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


def standalone_int_pattern(value: str) -> re.Pattern:
    return re.compile(rf"(?<!\d){re.escape(str(value))}(?!\d)")


def prediction_passed(pred: str, answer: str) -> bool:
    return normalize_answer(pred) == normalize_answer(answer)


def extract_document_blocks(context: str) -> list[tuple[int, str]]:
    """
    Extract document blocks from context.

    Returns:
      [(doc_number, doc_text), ...]
    """
    if not isinstance(context, str):
        return []

    pattern = re.compile(
        r"--- Document\s+(\d+)\s+---\n(.*?)(?=\n\n--- Document\s+\d+\s+---|\Z)",
        re.DOTALL,
    )

    blocks = []

    for match in pattern.finditer(context):
        doc_no = int(match.group(1))
        doc_text = match.group(2).strip()
        blocks.append((doc_no, doc_text))

    return blocks


def extract_value_from_doc_text(doc_text: str) -> int | None:
    """
    Extract the numeric attribute from one document text.

    Supports templates like:
      - membership of 122
      - catalogs 371 unique entries
      - yield of 503 units
      - stands at 728 active members
      - list 960 distinct artefacts
    """
    patterns = [
        r"membership of\s+(-?\d+)",
        r"catalogs\s+(-?\d+)\s+unique entries",
        r"yield of\s+(-?\d+)\s+units",
        r"stands at\s+(-?\d+)\s+active members",
        r"list\s+(-?\d+)\s+distinct artefacts",
        r"lists\s+(-?\d+)\s+distinct artefacts",
    ]

    for pattern in patterns:
        match = re.search(pattern, doc_text)

        if match:
            return int(match.group(1))

    # Fallback: first standalone integer in the document text.
    match = re.search(r"(?<!\d)(-?\d+)(?!\d)", doc_text)

    if match:
        return int(match.group(1))

    return None


def build_topic_value_map_from_context(context: str, known_topics: list[str]) -> dict[str, int]:
    """
    Build topic -> value using context and known topic names.

    This does not require metadata.
    """
    topic_to_value: dict[str, int] = {}

    blocks = extract_document_blocks(context)

    for _, doc_text in blocks:
        value = extract_value_from_doc_text(doc_text)

        if value is None:
            continue

        normalized_doc = normalize_text(doc_text).lower()

        for topic in known_topics:
            if not isinstance(topic, str):
                continue

            topic_norm = normalize_text(topic).lower()

            if topic_norm in normalized_doc:
                topic_to_value[topic] = value

    return topic_to_value


def collect_known_topics(record: dict) -> list[str]:
    """
    Collect all topic-like strings from:
      - involved_topics
      - compare confuser_answers
      - metadata topics if present
    """
    topics = []
    seen = set()

    def add_topic(value):
        if isinstance(value, str) and value not in seen:
            seen.add(value)
            topics.append(value)

    for topic in record.get("involved_topics", []):
        add_topic(topic)

    if record.get("kind") == "compare":
        for topic in record.get("confuser_answers", []):
            add_topic(topic)

    metadata = record.get("metadata")

    if isinstance(metadata, dict):
        for topic in metadata.get("topics", []):
            add_topic(topic)

    return topics


def check_required_schema(record: dict) -> list[str]:
    problems = []

    if "_json_error" in record:
        problems.append(f"JSON decode error: {record['_json_error']}")
        return problems

    for field in REQUIRED_TRAINING_FIELDS:
        if field not in record:
            problems.append(f"missing field: {field}")

    string_fields = [
        "src",
        "context",
        "question",
        "answer",
        "kind",
        "pred",
    ]

    for field in string_fields:
        if field in record and not isinstance(record.get(field), str):
            problems.append(f"{field} exists but is not a string")

    if "gold" in record and not isinstance(record.get("gold"), str):
        problems.append("gold exists but is not a string")

    if "database_version" in record and not isinstance(record.get("database_version"), str):
        problems.append("database_version exists but is not a string")

    if "task_id" in record and not isinstance(record.get("task_id"), str):
        problems.append("task_id exists but is not a string")

    if "confuser_answers" in record and not isinstance(record.get("confuser_answers"), list):
        problems.append("confuser_answers exists but is not a list")

    if "involved_topics" in record and not isinstance(record.get("involved_topics"), list):
        problems.append("involved_topics exists but is not a list")

    if "messages" in record and not isinstance(record.get("messages"), list):
        problems.append("messages exists but is not a list")

    if "passed" in record and not isinstance(record.get("passed"), bool):
        problems.append("passed exists but is not a bool")

    if "metadata" in record and not isinstance(record.get("metadata"), dict):
        problems.append("metadata exists but is not a dict")

    return problems


def check_basic_fields(record: dict) -> list[str]:
    problems = []

    kind = record.get("kind")
    src = record.get("src")
    context = record.get("context", "")
    question = record.get("question", "")
    answer = record.get("answer", "")
    pred = record.get("pred", "")
    passed = record.get("passed")
    confuser_answers = record.get("confuser_answers", [])
    involved_topics = record.get("involved_topics", [])

    if kind not in SYNTHESIS_KINDS:
        problems.append(f"invalid kind: {kind!r}")

    if isinstance(kind, str):
        expected_src = f"multi_doc_synthesis/{kind}"

        if src != expected_src:
            problems.append(f"src mismatch: expected={expected_src!r}, actual={src!r}")

        task_id = record.get("task_id")

        if isinstance(task_id, str) and not task_id.startswith(expected_src + "/"):
            problems.append(
                f"task_id does not start with expected prefix: {expected_src + '/'}"
            )

    if isinstance(context, str):
        if not context.strip():
            problems.append("context is empty")

        if "--- Document " not in context:
            problems.append("context does not contain document markers")

        doc_count = len(extract_document_blocks(context))

        if doc_count == 0:
            problems.append("could not parse document blocks from context")

    if isinstance(question, str):
        if not question.strip():
            problems.append("question is empty")

        if "Reply with" not in question:
            problems.append("question missing explicit reply instruction")

        if kind in INTEGER_KINDS and "integer only" not in question:
            problems.append("integer kind question missing 'integer only' instruction")

        if kind in NAME_ANSWER_KINDS and "full name" not in question:
            problems.append("compare question missing 'full name' instruction")

    if isinstance(answer, str):
        if not answer.strip():
            problems.append("answer is empty")

    if isinstance(pred, str):
        if not pred.strip():
            problems.append("pred is empty")

    if isinstance(answer, str) and isinstance(pred, str):
        expected_passed = prediction_passed(pred, answer)

        if passed != expected_passed:
            problems.append(
                f"passed mismatch: expected={expected_passed}, actual={passed}"
            )

        if expected_passed is False:
            problems.append(f"pred does not match answer: pred={pred!r}, answer={answer!r}")

    if "gold" in record:
        gold = record.get("gold", "")

        if isinstance(answer, str) and isinstance(gold, str):
            if normalize_answer(answer) != normalize_answer(gold):
                problems.append("answer does not match gold after normalization")

    if isinstance(confuser_answers, list):
        answer_norm = normalize_answer(answer)

        for confuser in confuser_answers:
            if normalize_answer(confuser) == answer_norm:
                problems.append(f"confuser matches answer: {confuser!r}")

    if isinstance(involved_topics, list):
        if kind in {"sum", "difference", "compare", "ratio"}:
            if len(involved_topics) != 2:
                problems.append(
                    f"{kind} should have exactly 2 involved_topics, got {len(involved_topics)}"
                )

        if kind in {"sum_three", "difference_three"}:
            if len(involved_topics) != 3:
                problems.append(
                    f"{kind} should have exactly 3 involved_topics, got {len(involved_topics)}"
                )

    return problems


def check_messages(record: dict) -> list[str]:
    problems = []

    messages = record.get("messages")
    context = record.get("context", "")
    question = record.get("question", "")
    answer = record.get("answer", "")
    pred = record.get("pred", "")

    if not isinstance(messages, list):
        return ["messages is not a list"]

    if len(messages) != 2:
        problems.append(f"messages should contain exactly 2 messages, got {len(messages)}")
        return problems

    user_msg = messages[0]
    assistant_msg = messages[1]

    if not isinstance(user_msg, dict):
        problems.append("messages[0] is not a dict")
        return problems

    if not isinstance(assistant_msg, dict):
        problems.append("messages[1] is not a dict")
        return problems

    if user_msg.get("role") != "user":
        problems.append(f"messages[0].role should be user, got {user_msg.get('role')!r}")

    if assistant_msg.get("role") != "assistant":
        problems.append(
            f"messages[1].role should be assistant, got {assistant_msg.get('role')!r}"
        )

    user_content = user_msg.get("content")
    assistant_content = assistant_msg.get("content")

    if not isinstance(user_content, str):
        problems.append("messages[0].content is not a string")
    else:
        if "Read the documents below" not in user_content:
            problems.append("user message missing instruction prefix")

        if context not in user_content:
            problems.append("user message does not contain exact context")

        if question not in user_content:
            problems.append("user message does not contain exact question")

        if "Answer:" not in user_content:
            problems.append("user message missing Answer: suffix")

    if not isinstance(assistant_content, str):
        problems.append("messages[1].content is not a string")
    else:
        if normalize_answer(assistant_content) != normalize_answer(answer):
            problems.append(
                "assistant message content does not match answer after normalization"
            )

        if normalize_answer(assistant_content) != normalize_answer(pred):
            problems.append(
                "assistant message content does not match pred after normalization"
            )

    return problems


def check_compare_the_rule(record: dict) -> list[str]:
    problems = []

    if record.get("kind") != "compare":
        return problems

    answer = record.get("answer", "")
    pred = record.get("pred", "")
    messages = record.get("messages", [])
    involved_topics = record.get("involved_topics", [])
    confuser_answers = record.get("confuser_answers", [])

    if not starts_with_the(answer):
        problems.append(f"compare answer must start with 'the ': {answer!r}")

    if not starts_with_the(pred):
        problems.append(f"compare pred must start with 'the ': {pred!r}")

    if isinstance(messages, list) and len(messages) >= 2 and isinstance(messages[1], dict):
        assistant_content = str(messages[1].get("content", "")).lstrip()

        if not starts_with_the(assistant_content):
            problems.append(
                "compare assistant content must start with 'the ' after lstrip: "
                f"{messages[1].get('content', '')!r}"
            )

    for topic in involved_topics:
        if not starts_with_the(topic):
            problems.append(f"compare involved topic must start with 'the ': {topic!r}")

    for confuser in confuser_answers:
        if not starts_with_the(confuser):
            problems.append(f"compare confuser must start with 'the ': {confuser!r}")

    metadata = record.get("metadata")

    if isinstance(metadata, dict):
        for topic in metadata.get("topics", []):
            if not starts_with_the(topic):
                problems.append(f"compare metadata topic must start with 'the ': {topic!r}")

    return problems


def check_optional_metadata_schema(record: dict) -> list[str]:
    """
    Metadata is optional in training records.

    If metadata exists, validate it strongly.
    """
    problems = []

    if "metadata" not in record:
        return problems

    metadata = record.get("metadata")
    kind = record.get("kind")

    if not isinstance(metadata, dict):
        return ["metadata exists but is not a dict"]

    required_metadata_fields = [
        "global_index",
        "kind_index",
        "seed",
        "n_cards",
        "topics",
        "values",
        "involved_indices",
        "involved_topics",
        "confuser_answers",
        "answer_type",
        "normalized_gold",
    ]

    for field in required_metadata_fields:
        if field not in metadata:
            problems.append(f"metadata missing field: {field}")

    if "topics" in metadata and not isinstance(metadata["topics"], list):
        problems.append("metadata topics is not a list")

    if "values" in metadata and not isinstance(metadata["values"], list):
        problems.append("metadata values is not a list")

    if "involved_indices" in metadata and not isinstance(metadata["involved_indices"], list):
        problems.append("metadata involved_indices is not a list")

    if "involved_topics" in metadata:
        if metadata["involved_topics"] != record.get("involved_topics"):
            problems.append("metadata involved_topics does not match record involved_topics")

    if "confuser_answers" in metadata:
        if metadata["confuser_answers"] != record.get("confuser_answers"):
            problems.append("metadata confuser_answers does not match record confuser_answers")

    if "answer_type" in metadata:
        expected_answer_type = "organization_name" if kind == "compare" else "integer"

        if metadata["answer_type"] != expected_answer_type:
            problems.append(
                f"metadata answer_type mismatch: expected={expected_answer_type!r}, "
                f"actual={metadata['answer_type']!r}"
            )

    answer = record.get("answer", "")

    if metadata.get("normalized_gold") != normalize_answer(answer):
        problems.append("metadata normalized_gold mismatch")

    topics = metadata.get("topics")
    values = metadata.get("values")
    involved_indices = metadata.get("involved_indices")
    n_cards = metadata.get("n_cards")

    if isinstance(topics, list):
        if len(topics) != len(set(topics)):
            problems.append("metadata topics contains duplicates")

    if isinstance(values, list):
        if len(values) != len(set(values)):
            problems.append("metadata values contains duplicates")

        for i, value in enumerate(values):
            if not isinstance(value, int):
                problems.append(f"metadata values[{i}] is not an int")

    if isinstance(n_cards, int):
        if isinstance(topics, list) and len(topics) != n_cards:
            problems.append(
                f"metadata topics length mismatch: len(topics)={len(topics)}, n_cards={n_cards}"
            )

        if isinstance(values, list) and len(values) != n_cards:
            problems.append(
                f"metadata values length mismatch: len(values)={len(values)}, n_cards={n_cards}"
            )

    if isinstance(involved_indices, list):
        expected_len = 3 if kind in {"sum_three", "difference_three"} else 2

        if len(involved_indices) != expected_len:
            problems.append(
                f"metadata involved_indices length mismatch: "
                f"expected={expected_len}, actual={len(involved_indices)}"
            )

        if isinstance(n_cards, int):
            for idx in involved_indices:
                if not isinstance(idx, int):
                    problems.append(f"metadata involved index is not int: {idx!r}")
                elif idx < 0 or idx >= n_cards:
                    problems.append(
                        f"metadata involved index out of range: idx={idx}, n_cards={n_cards}"
                    )

    return problems


def recompute_gold_from_metadata(record: dict) -> tuple[str | None, str | None]:
    try:
        kind = record.get("kind")
        metadata = record.get("metadata", {})

        if kind not in SYNTHESIS_KINDS:
            return None, f"cannot recompute unknown kind: {kind!r}"

        if not isinstance(metadata, dict):
            return None, "metadata is not a dict"

        topics = metadata.get("topics")
        values = metadata.get("values")
        involved_indices = metadata.get("involved_indices")

        if not isinstance(topics, list):
            return None, "metadata topics is not a list"

        if not isinstance(values, list):
            return None, "metadata values is not a list"

        if not isinstance(involved_indices, list):
            return None, "metadata involved_indices is not a list"

        if len(topics) != len(values):
            return None, "metadata topics and values length mismatch"

        involved = []

        for idx in involved_indices:
            if not isinstance(idx, int):
                return None, f"involved index is not int: {idx!r}"

            if idx < 0 or idx >= len(values):
                return None, f"involved index out of range: {idx}"

            involved.append(
                {
                    "idx": idx,
                    "topic": topics[idx],
                    "value": values[idx],
                }
            )

        return recompute_gold_from_involved(kind, involved)

    except Exception:
        return None, traceback.format_exc()


def recompute_gold_from_context(record: dict) -> tuple[str | None, str | None, dict]:
    """
    Recompute answer from context and involved_topics.

    Works without metadata.
    """
    verification_extra = {}

    try:
        kind = record.get("kind")
        context = record.get("context", "")
        involved_topics = record.get("involved_topics", [])

        if kind not in SYNTHESIS_KINDS:
            return None, f"cannot recompute unknown kind: {kind!r}", verification_extra

        if not isinstance(context, str):
            return None, "context is not a string", verification_extra

        if not isinstance(involved_topics, list):
            return None, "involved_topics is not a list", verification_extra

        known_topics = collect_known_topics(record)
        topic_to_value = build_topic_value_map_from_context(context, known_topics)

        verification_extra["topic_to_value"] = topic_to_value

        involved = []

        for topic in involved_topics:
            if topic not in topic_to_value:
                return None, f"could not find value for involved topic: {topic!r}", verification_extra

            involved.append(
                {
                    "topic": topic,
                    "value": topic_to_value[topic],
                }
            )

        return_value, err = recompute_gold_from_involved(kind, involved)

        return return_value, err, verification_extra

    except Exception:
        return None, traceback.format_exc(), verification_extra


def recompute_gold_from_involved(
    kind: str,
    involved: list[dict],
) -> tuple[str | None, str | None]:
    if kind == "sum":
        if len(involved) != 2:
            return None, "sum requires exactly 2 involved items"

        return str(involved[0]["value"] + involved[1]["value"]), None

    if kind == "difference":
        if len(involved) != 2:
            return None, "difference requires exactly 2 involved items"

        vals = [involved[0]["value"], involved[1]["value"]]
        return str(max(vals) - min(vals)), None

    if kind == "compare":
        if len(involved) != 2:
            return None, "compare requires exactly 2 involved items"

        first = involved[0]
        second = involved[1]

        if first["value"] > second["value"]:
            return str(first["topic"]), None

        return str(second["topic"]), None

    if kind == "ratio":
        if len(involved) != 2:
            return None, "ratio requires exactly 2 involved items"

        vals = [involved[0]["value"], involved[1]["value"]]
        larger = max(vals)
        smaller = min(vals)

        if smaller == 0:
            return None, "ratio division by zero"

        return str(larger // smaller), None

    if kind == "sum_three":
        if len(involved) != 3:
            return None, "sum_three requires exactly 3 involved items"

        return str(sum(item["value"] for item in involved)), None

    if kind == "difference_three":
        if len(involved) != 3:
            return None, "difference_three requires exactly 3 involved items"

        vals = sorted([item["value"] for item in involved], reverse=True)
        return str(vals[0] - vals[1] - vals[2]), None

    return None, f"unhandled kind: {kind!r}"


def check_recomputed_answer(record: dict) -> tuple[list[str], dict]:
    problems = []
    verification = {}

    answer = record.get("answer", "")

    # Prefer metadata when available because it is exact.
    if isinstance(record.get("metadata"), dict):
        computed_gold, err = recompute_gold_from_metadata(record)
        verification["recompute_source"] = "metadata"
    else:
        computed_gold, err, extra = recompute_gold_from_context(record)
        verification.update(extra)
        verification["recompute_source"] = "context"

    verification["computed_gold"] = computed_gold
    verification["recompute_error"] = err
    verification["answer"] = answer
    verification["pred"] = record.get("pred")

    if "gold" in record:
        verification["gold"] = record.get("gold")

    if err is not None:
        problems.append(f"could not recompute answer: {err}")
        verification["matched_recomputed_answer"] = False
        return problems, verification

    normalized_computed = normalize_answer(computed_gold)
    normalized_answer = normalize_answer(answer)

    verification["normalized_computed_gold"] = normalized_computed
    verification["normalized_answer"] = normalized_answer
    verification["matched_recomputed_answer"] = normalized_computed == normalized_answer

    if normalized_computed != normalized_answer:
        problems.append(
            f"answer does not match recomputed value: "
            f"computed={computed_gold!r}, answer={answer!r}"
        )

    if "gold" in record:
        normalized_gold = normalize_answer(record.get("gold", ""))
        verification["normalized_gold"] = normalized_gold

        if normalized_computed != normalized_gold:
            problems.append(
                f"gold does not match recomputed value: "
                f"computed={computed_gold!r}, gold={record.get('gold')!r}"
            )

    return problems, verification


def check_context_grounding(record: dict) -> list[str]:
    problems = []

    context = record.get("context", "")
    kind = record.get("kind")
    answer = record.get("answer", "")

    if not isinstance(context, str):
        return ["context is not a string"]

    involved_topics = record.get("involved_topics", [])
    confuser_answers = record.get("confuser_answers", [])

    if isinstance(involved_topics, list):
        normalized_context = normalize_text(context).lower()

        for topic in involved_topics:
            if isinstance(topic, str) and normalize_text(topic).lower() not in normalized_context:
                problems.append(f"context missing involved topic: {topic!r}")

    if kind == "compare":
        normalized_context = normalize_text(context).lower()

        if isinstance(answer, str) and normalize_text(answer).lower() not in normalized_context:
            problems.append("compare answer topic does not appear in context")

        if isinstance(confuser_answers, list):
            for confuser in confuser_answers:
                if isinstance(confuser, str) and normalize_text(confuser).lower() not in normalized_context:
                    problems.append(f"compare confuser topic does not appear in context: {confuser!r}")

    # If metadata exists, check all metadata topics and values appear.
    metadata = record.get("metadata")

    if isinstance(metadata, dict):
        topics = metadata.get("topics", [])
        values = metadata.get("values", [])

        normalized_context = normalize_text(context).lower()

        if isinstance(topics, list):
            for topic in topics:
                if isinstance(topic, str) and normalize_text(topic).lower() not in normalized_context:
                    problems.append(f"context missing metadata topic: {topic!r}")

        if isinstance(values, list):
            for value in values:
                if isinstance(value, int):
                    pattern = standalone_int_pattern(str(value))

                    if not pattern.search(context):
                        problems.append(f"context missing standalone numeric value: {value}")

    return problems


def check_confuser_behavior(record: dict) -> list[str]:
    """
    If metadata exists, perform exact confuser check.
    If metadata does not exist, perform lighter checks.
    """
    problems = []

    kind = record.get("kind")
    confuser_answers = record.get("confuser_answers", [])

    if not isinstance(confuser_answers, list):
        return ["confuser_answers is not a list"]

    metadata = record.get("metadata")

    if isinstance(metadata, dict):
        topics = metadata.get("topics", [])
        values = metadata.get("values", [])
        involved_indices = metadata.get("involved_indices", [])

        if not isinstance(topics, list) or not isinstance(values, list) or not isinstance(involved_indices, list):
            return problems

        involved_set = set(involved_indices)

        if kind == "compare":
            answer = record.get("answer")
            expected_confusers = [
                topic
                for idx, topic in enumerate(topics)
                if idx not in involved_set
            ]

            involved_topics = [
                topics[idx]
                for idx in involved_indices
                if isinstance(idx, int) and 0 <= idx < len(topics)
            ]

            for topic in involved_topics:
                if topic != answer:
                    expected_confusers.append(topic)

            if sorted(confuser_answers) != sorted(expected_confusers):
                problems.append("compare confuser_answers do not match expected topic confusers")

        elif kind in INTEGER_KINDS:
            expected_confusers = [
                str(value)
                for idx, value in enumerate(values)
                if idx not in involved_set
            ]

            if sorted(confuser_answers) != sorted(expected_confusers):
                problems.append("integer-kind confuser_answers do not match expected value confusers")

        return problems

    # Lightweight no-metadata checks.
    if kind == "compare":
        answer_norm = normalize_answer(record.get("answer", ""))

        for confuser in confuser_answers:
            if normalize_answer(confuser) == answer_norm:
                problems.append(f"compare confuser matches answer: {confuser!r}")

    elif kind in INTEGER_KINDS:
        for confuser in confuser_answers:
            if not isinstance(confuser, str):
                problems.append(f"integer-kind confuser is not string: {confuser!r}")
            elif not re.fullmatch(r"-?\d+", confuser.strip()):
                problems.append(f"integer-kind confuser is not integer-like: {confuser!r}")

    return problems


def check_record(line_no: int, record: dict) -> tuple[list[str], dict]:
    problems = []
    verification = {}

    schema_problems = check_required_schema(record)
    problems.extend(schema_problems)

    if "_json_error" in record:
        verification["line_no"] = line_no
        verification["json_error"] = record.get("_json_error")
        return problems, verification

    problems.extend(check_basic_fields(record))
    problems.extend(check_messages(record))
    problems.extend(check_compare_the_rule(record))
    problems.extend(check_optional_metadata_schema(record))
    problems.extend(check_context_grounding(record))
    problems.extend(check_confuser_behavior(record))

    recompute_problems, recompute_verification = check_recomputed_answer(record)
    problems.extend(recompute_problems)
    verification.update(recompute_verification)

    context = record.get("context", "")
    messages = record.get("messages", [])

    verification["line_no"] = line_no
    verification["kind"] = record.get("kind")
    verification["context_chars"] = len(context) if isinstance(context, str) else 0
    verification["context_words"] = count_words(context)
    verification["doc_count"] = len(extract_document_blocks(context)) if isinstance(context, str) else 0
    verification["involved_topics"] = record.get("involved_topics")
    verification["confuser_count"] = (
        len(record.get("confuser_answers", []))
        if isinstance(record.get("confuser_answers"), list)
        else 0
    )
    verification["messages_count"] = len(messages) if isinstance(messages, list) else None

    metadata = record.get("metadata")

    if isinstance(metadata, dict):
        verification["n_cards"] = metadata.get("n_cards")
    else:
        verification["n_cards"] = verification["doc_count"]

    return problems, verification


def format_record_for_log(
    line_no: int,
    record: dict,
    problems: list[str],
    verification: dict,
) -> str:
    context = record.get("context", "")

    lines = []

    lines.append("=" * 100)
    lines.append(f"LINE: {line_no}")
    lines.append(f"SRC: {record.get('src', 'unknown')}")
    lines.append(f"KIND: {record.get('kind', 'unknown')}")

    if "task_id" in record:
        lines.append(f"TASK ID: {record.get('task_id', 'unknown')}")

    if "database_version" in record:
        lines.append(f"DATABASE VERSION: {record.get('database_version', 'unknown')}")

    lines.append("")

    lines.append("[QUESTION]")
    lines.append(str(record.get("question", "")).strip())
    lines.append("")

    lines.append("[CONTEXT STATS]")
    lines.append(f"Context chars: {len(context) if isinstance(context, str) else 0}")
    lines.append(f"Context words: {count_words(context)}")
    lines.append(f"Document count: {len(extract_document_blocks(context)) if isinstance(context, str) else 0}")
    lines.append("")

    lines.append("[CONTEXT HEAD]")
    lines.append(short_text(context[:3000], max_chars=3000))
    lines.append("")

    lines.append("[CONTEXT TAIL]")
    lines.append(short_text(context[-3000:], max_chars=3000))
    lines.append("")

    lines.append("[ANSWER]")
    lines.append(str(record.get("answer", "")).strip())
    lines.append("")

    lines.append("[PRED]")
    lines.append(str(record.get("pred", "")).strip())
    lines.append("")

    lines.append("[PASSED]")
    lines.append(str(record.get("passed")))
    lines.append("")

    if "gold" in record:
        lines.append("[GOLD]")
        lines.append(str(record.get("gold", "")).strip())
        lines.append("")

    lines.append("[INVOLVED TOPICS]")
    lines.append(short_json(record.get("involved_topics")))
    lines.append("")

    lines.append("[CONFUSER ANSWERS]")
    lines.append(short_json(record.get("confuser_answers")))
    lines.append("")

    lines.append("[MESSAGES]")
    lines.append(short_json(record.get("messages"), max_chars=4000))
    lines.append("")

    if "metadata" in record:
        lines.append("[METADATA]")
        lines.append(short_json(record.get("metadata"), max_chars=4000))
        lines.append("")

    if "_json_error" in record:
        lines.append("[RAW JSON ERROR LINE HEAD]")
        lines.append(short_text(record.get("_raw", ""), max_chars=2000))
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
    show_messages: bool = False,
    show_metadata: bool = False,
) -> None:
    print("=" * 80)
    print(f"Line: {line_no}")
    print(f"Kind: {record.get('kind', 'unknown')}")
    print(f"Src: {record.get('src', 'unknown')}")
    print(f"Question: {str(record.get('question', ''))[:240]}")
    print(f"Answer: {record.get('answer')}")
    print(f"Pred: {record.get('pred')}")
    print(f"Passed: {record.get('passed')}")
    print(f"Computed answer: {verification.get('computed_gold')}")
    print(f"Matched recomputed answer: {verification.get('matched_recomputed_answer')}")
    print(f"Recompute source: {verification.get('recompute_source')}")
    print(f"Context words: {verification.get('context_words')}")
    print(f"Document count: {verification.get('doc_count')}")
    print(f"Confuser count: {verification.get('confuser_count')}")

    if problems:
        print("Problems:")
        for problem in problems:
            print(f"  - {problem}")
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

    if show_messages:
        print("")
        print("[MESSAGES]")
        print("-" * 40)
        print(short_json(record.get("messages"), max_chars=5000))
        print("-" * 40)

    if show_metadata:
        print("")
        print("[METADATA]")
        print("-" * 40)
        print(short_json(record.get("metadata"), max_chars=5000))
        print("-" * 40)


def write_cleaned_jsonl(
    *,
    original_path: Path,
    kept_records: list[dict],
    make_backup: bool = True,
) -> Path | None:
    """
    Atomically rewrite original_path using only kept_records.

    Returns:
      backup_path if backup was created, else None
    """
    backup_path = None

    if make_backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = original_path.with_name(
            f"{original_path.name}.bak_{timestamp}"
        )
        shutil.copy2(original_path, backup_path)

    tmp_path = original_path.with_name(f"{original_path.name}.tmp_cleaned")

    with tmp_path.open("w", encoding="utf-8") as f:
        for record in kept_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    tmp_path.replace(original_path)

    return backup_path


def print_summary(
    *,
    path: Path,
    log_path: Path,
    total_seen: int,
    total_checked: int,
    logged_count: int,
    ok_count: int,
    bad_count: int,
    removed_count: int,
    kept_count: int,
    kind_counts: dict,
    src_counts: dict,
    passed_counts: dict,
    context_word_counts: list[int],
    context_char_counts: list[int],
    doc_counts: list[int],
    compare_the_stats: dict,
    rewritten: bool,
    backup_path: Path | None,
    dry_run: bool,
) -> str:
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
    summary_lines.append(f"Records kept for rewritten file: {kept_count}")
    summary_lines.append(f"Records removed from rewritten file: {removed_count}")
    summary_lines.append(f"Dry run: {dry_run}")
    summary_lines.append(f"File rewritten: {rewritten}")

    if backup_path is not None:
        summary_lines.append(f"Backup file: {backup_path}")

    summary_lines.append("")

    if context_word_counts:
        summary_lines.append("Context word stats:")
        summary_lines.append(f"  Min words: {min(context_word_counts)}")
        summary_lines.append(f"  Max words: {max(context_word_counts)}")
        summary_lines.append(f"  Avg words: {sum(context_word_counts) // len(context_word_counts)}")
        summary_lines.append("")

    if context_char_counts:
        summary_lines.append("Context char stats:")
        summary_lines.append(f"  Min chars: {min(context_char_counts)}")
        summary_lines.append(f"  Max chars: {max(context_char_counts)}")
        summary_lines.append(f"  Avg chars: {sum(context_char_counts) // len(context_char_counts)}")
        summary_lines.append("")

    if doc_counts:
        summary_lines.append("Document count stats:")
        summary_lines.append(f"  Min docs: {min(doc_counts)}")
        summary_lines.append(f"  Max docs: {max(doc_counts)}")
        summary_lines.append(f"  Avg docs: {sum(doc_counts) // len(doc_counts)}")
        summary_lines.append("")

    summary_lines.append("Passed counts:")
    for key, count in sorted(passed_counts.items()):
        summary_lines.append(f"  {key}: {count}")

    summary_lines.append("")
    summary_lines.append("Kind counts:")
    for kind in SYNTHESIS_KINDS:
        summary_lines.append(f"  {kind}: {kind_counts.get(kind, 0)}")

    summary_lines.append("")
    summary_lines.append("Src counts:")
    for src, count in sorted(src_counts.items()):
        summary_lines.append(f"  {src}: {count}")

    summary_lines.append("")
    summary_lines.append("Compare 'the' rule stats:")
    summary_lines.append(f"  compare records checked: {compare_the_stats.get('compare_total', 0)}")
    summary_lines.append(f"  answer starts with 'the ': {compare_the_stats.get('answer_the', 0)}")
    summary_lines.append(f"  pred starts with 'the ': {compare_the_stats.get('pred_the', 0)}")
    summary_lines.append(f"  all involved topics start with 'the ': {compare_the_stats.get('involved_the', 0)}")
    summary_lines.append(f"  all confusers start with 'the ': {compare_the_stats.get('confuser_the', 0)}")

    summary_text = "\n".join(summary_lines)
    return summary_text


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check multi_doc_synthesis training/eval JSONL records. "
            "Bad records are removed and the JSONL file is rewritten by default."
        )
    )

    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default="dataset/multi_doc_training_all_cases.jsonl",
        help="Path to multi_doc_synthesis training/eval JSONL file.",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="database_v3/multi_doc_training_check.log",
        help="Path to output readable log file.",
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print context head/tail to terminal.",
    )

    parser.add_argument(
        "--show-messages",
        action="store_true",
        help="Print messages to terminal.",
    )

    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Print metadata to terminal if present.",
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
        choices=SYNTHESIS_KINDS,
        help=(
            "Only inspect records with this synthesis kind. "
            "Records outside this filter are kept unchanged."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Limit number of records to inspect after filters. "
            "Unchecked records after the limit are kept unchanged."
        ),
    )

    parser.add_argument(
        "--min-context-words",
        type=int,
        default=0,
        help="Require context to have at least this many words.",
    )

    parser.add_argument(
        "--min-docs",
        type=int,
        default=0,
        help="Require each checked record to have at least this many documents.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check and log only. Do not rewrite the JSONL file.",
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Rewrite without creating a backup file.",
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
    removed_count = 0

    kind_counts: dict[str, int] = {}
    src_counts: dict[str, int] = {}
    passed_counts: dict[str, int] = {}

    context_word_counts: list[int] = []
    context_char_counts: list[int] = []
    doc_counts: list[int] = []

    compare_the_stats = {
        "compare_total": 0,
        "answer_the": 0,
        "pred_the": 0,
        "involved_the": 0,
        "confuser_the": 0,
    }

    kept_records: list[dict] = []

    with log_path.open("w", encoding="utf-8") as log_f:
        for line_no, record, raw_line, json_error in load_jsonl_with_raw(path):
            total_seen += 1

            if json_error is not None and args.kind is not None:
                kept_records.append(record)
                continue

            if json_error is None and args.kind is not None and record.get("kind") != args.kind:
                kept_records.append(record)
                continue

            if args.limit and total_checked >= args.limit:
                kept_records.append(record)
                continue

            total_checked += 1

            kind = record.get("kind", "unknown")
            src = record.get("src", "unknown")
            passed_value = record.get("passed", "unknown")

            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            src_counts[src] = src_counts.get(src, 0) + 1
            passed_counts[str(passed_value)] = passed_counts.get(str(passed_value), 0) + 1

            if kind == "compare":
                compare_the_stats["compare_total"] += 1

                if starts_with_the(record.get("answer", "")):
                    compare_the_stats["answer_the"] += 1

                if starts_with_the(record.get("pred", "")):
                    compare_the_stats["pred_the"] += 1

                involved_topics = record.get("involved_topics", [])
                if isinstance(involved_topics, list) and all(starts_with_the(x) for x in involved_topics):
                    compare_the_stats["involved_the"] += 1

                confuser_answers = record.get("confuser_answers", [])
                if isinstance(confuser_answers, list) and all(starts_with_the(x) for x in confuser_answers):
                    compare_the_stats["confuser_the"] += 1

            problems, verification = check_record(
                line_no=line_no,
                record=record,
            )

            context = record.get("context", "")
            context_words = count_words(context)
            context_chars = len(context) if isinstance(context, str) else 0
            doc_count = len(extract_document_blocks(context)) if isinstance(context, str) else 0

            context_word_counts.append(context_words)
            context_char_counts.append(context_chars)
            doc_counts.append(doc_count)

            if args.min_context_words and context_words < args.min_context_words:
                problems.append(
                    f"context has fewer words than required: "
                    f"{context_words} < {args.min_context_words}"
                )

            if args.min_docs and doc_count < args.min_docs:
                problems.append(
                    f"document count below required minimum: {doc_count} < {args.min_docs}"
                )

            if problems:
                bad_count += 1
                removed_count += 1
            else:
                ok_count += 1
                kept_records.append(record)

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
                show_messages=args.show_messages,
                show_metadata=args.show_metadata,
            )

        rewritten = False
        backup_path = None

        if removed_count > 0 and not args.dry_run:
            backup_path = write_cleaned_jsonl(
                original_path=path,
                kept_records=kept_records,
                make_backup=not args.no_backup,
            )
            rewritten = True

        summary_text = print_summary(
            path=path,
            log_path=log_path,
            total_seen=total_seen,
            total_checked=total_checked,
            logged_count=logged_count,
            ok_count=ok_count,
            bad_count=bad_count,
            removed_count=removed_count,
            kept_count=len(kept_records),
            kind_counts=kind_counts,
            src_counts=src_counts,
            passed_counts=passed_counts,
            context_word_counts=context_word_counts,
            context_char_counts=context_char_counts,
            doc_counts=doc_counts,
            compare_the_stats=compare_the_stats,
            rewritten=rewritten,
            backup_path=backup_path,
            dry_run=args.dry_run,
        )

        log_f.write(summary_text)
        log_f.write("\n")

    print(summary_text)
    print(f"\nWrote readable log to: {log_path}")

    if removed_count > 0 and not args.dry_run:
        print(f"Updated JSONL file by removing {removed_count} bad record(s): {path}")

    elif removed_count > 0 and args.dry_run:
        print(f"Dry run only. Found {removed_count} bad record(s), but file was not changed.")

    else:
        print("No bad records found. JSONL file was not changed.")


if __name__ == "__main__":
    main()
