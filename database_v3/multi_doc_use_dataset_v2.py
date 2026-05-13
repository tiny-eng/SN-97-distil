#!/usr/bin/env python3

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path


DATABASE_VERSION = "multi_doc_synthesis_bench_v30_5"
MULTI_DOC_STREAM_SEED = 0x4D0C2026

SYNTHESIS_KINDS = [
    "sum",
    "difference",
    "compare",
    "ratio",
    "sum_three",
    "difference_three",
]

BENCH_MULTI_DOC_N_CARDS = int(os.environ.get("BENCH_MULTI_DOC_N_CARDS", "7"))


ORG_PREFIXES = [
    "Azure", "Bronze", "Cedar", "Dawn", "Ember", "Frost", "Golden",
    "Harbor", "Ivory", "Juniper", "Keystone", "Lantern", "Meadow",
    "North", "Orchid", "Pioneer", "Quartz", "River", "Silver", "Thorn",
    "Violet", "Willow", "Amber", "Copper", "Echo", "Marble", "Opal",
    "Sable", "Timber", "Verdant", "Crimson", "Indigo", "Maple",
    "Hollow", "Briar", "Cloud", "Stone", "Glass", "Fern", "Bright",
]

ORG_NOUNS = [
    "Archive", "Circle", "Collective", "Guild", "Institute", "League",
    "Museum", "Observatory", "Registry", "Society", "Trust", "Workshop",
    "Foundation", "Council", "Library", "Bureau", "Network", "Consortium",
    "Cabinet", "Association", "Assembly", "Depot", "Forum", "Center",
]

ORG_SUFFIXES = [
    "of Cartographers",
    "of Field Notes",
    "of Quiet Records",
    "of Seasonal Studies",
    "of Coastal Surveys",
    "of Public Works",
    "of Lantern Keepers",
    "of Meadow Science",
    "of Archive Stewards",
    "of River Histories",
    "of Northern Maps",
    "of Civic Gardens",
    "of Weather Logs",
    "of Harbor Studies",
    "of Stone Markers",
    "of Orchard Records",
    "of Valley Instruments",
    "of Public Catalogues",
    "of Surveyed Paths",
    "of Old Registers",
]


PSEUDO_PREFIXES = [
    "Veeldleanish",
    "Smoapoanese",
    "Pesprostan",
    "Ceertgloackish",
    "Sningnoanese",
    "Staystcleangian",
    "Spackdretese",
    "Flarnockian",
    "Dreelvostish",
    "Quenbralian",
    "Marnstovese",
    "Gleptorian",
    "Trandivish",
    "Yorbclanian",
    "Prellmorian",
    "Zindlewickish",
    "Brindlefearnish",
    "Croamvintese",
    "Trelshavian",
    "Mornclatterish",
    "Driftplorean",
    "Snarthwoldish",
    "Glimmerstackian",
    "Prondalese",
]

PSEUDO_CORES = [
    "Jitdroangcrest",
    "Grercrearburn",
    "Glealjaisfield",
    "Gourtflircrest",
    "Meengrarridge",
    "Smeeckprartcrest",
    "Droatmockhaven",
    "Varnplookmere",
    "Traxwoldfen",
    "Brelstairnook",
    "Clomperridge",
    "Narthgleamford",
    "Ploamwinter",
    "Draskmeadow",
    "Vrelnorthgate",
    "Shornapplecrest",
    "Flemporhaven",
    "Groshmarblefield",
    "Trawnduskport",
    "Meckslateburn",
    "Clarnhollowmere",
    "Sproatfieldgate",
    "Drelvintercrest",
    "Plimstairhaven",
]

PSEUDO_NOUNS = [
    "Mill",
    "Foundation",
    "Society",
    "Guild",
    "Workshop",
    "Atelier",
    "Depot",
    "Registry",
    "League",
    "Institute",
    "Archive",
    "Circle",
    "Council",
    "Bureau",
    "Cabinet",
    "Collective",
    "Forum",
    "Center",
    "Observatory",
    "Museum",
]


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


def build_user_message(context: str, question: str) -> str:
    return (
        "Read the documents below and answer the question that follows. "
        "Reply with just the answer (no extra text).\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def prediction_passed(pred: str, answer: str) -> bool:
    return normalize_answer(pred) == normalize_answer(answer)


def _synthetic_org_topic(r: random.Random) -> str:
    """
    Generate a normal fictional organization name.

    Example:
        Azure Archive of Coastal Surveys
    """
    return (
        f"{r.choice(ORG_PREFIXES)} "
        f"{r.choice(ORG_NOUNS)} "
        f"{r.choice(ORG_SUFFIXES)}"
    )


def _pseudo_org_topic(r: random.Random, use_article: bool = True) -> str:
    """
    Generate a pseudo organization name.

    Example:
        the Staystcleangian Smeeckprartcrest Workshop
    """
    name = (
        f"{r.choice(PSEUDO_PREFIXES)} "
        f"{r.choice(PSEUDO_CORES)} "
        f"{r.choice(PSEUDO_NOUNS)}"
    )

    if use_article:
        return f"the {name}"

    return name


def generate_topics(
    r: random.Random,
    n_cards: int,
    topic_style: str = "pseudo",
    use_article: bool = True,
) -> list[str]:
    topics: list[str] = []
    seen_topics: set[str] = set()
    synth_attempts = 0

    while len(topics) < n_cards and synth_attempts < n_cards * 128:
        synth_attempts += 1

        if topic_style == "pseudo":
            topic = _pseudo_org_topic(r, use_article=use_article)
        elif topic_style == "normal":
            topic = _synthetic_org_topic(r)
        else:
            raise ValueError(f"Unknown topic_style: {topic_style!r}")

        if topic in seen_topics:
            continue

        seen_topics.add(topic)
        topics.append(topic)

    if len(topics) < n_cards:
        raise RuntimeError(
            f"Could not generate enough distinct organization topics: "
            f"needed={n_cards}, got={len(topics)}"
        )

    return topics


def generate_values(r: random.Random, n_cards: int) -> list[int]:
    """
    Generate per-card numeric attributes.

    Distinct ranges reduce accidental substring collisions:
      card 0: 100-180
      card 1: 300-380
      card 2: 500-580
      card 3: 700-780
      card 4: 900-980
      card 5: 1100-1180
      card 6: 1300-1380
    """
    values: list[int] = []
    used: set[int] = set()

    for c in range(n_cards):
        lo = 100 * (2 * c + 1)
        hi = lo + 80

        value = r.randint(lo, hi)

        while value in used:
            value += 7

        used.add(value)
        values.append(value)

    return values


def build_context(topics: list[str], values: list[int]) -> str:
    attribute_templates = [
        "Founded a long time ago, {topic} reports a current membership of {n}.",
        "{topic} catalogs {n} unique entries in its public archive.",
        "An annual yield of {n} units is recorded by {topic} each season.",
        "The roster of {topic} stands at {n} active members this year.",
        "Records from {topic} list {n} distinct artefacts on display.",
    ]

    cards_text: list[str] = []

    for c, (topic, value) in enumerate(zip(topics, values)):
        template = attribute_templates[c % len(attribute_templates)]

        card_text = (
            f"--- Document {c + 1} ---\n"
            + template.format(topic=topic, n=value)
            + " Visitors describe its hall as quiet and orderly. "
            + "Its committee meets quarterly to review activities."
        )

        cards_text.append(card_text)

    return "\n\n".join(cards_text)


def make_multi_doc_synthesis_item(
    *,
    r: random.Random,
    item_seed: int,
    global_index: int,
    kind: str,
    kind_index: int,
    n_cards: int,
    topic_style: str,
    use_article: bool,
    include_metadata: bool,
    include_gold: bool,
    include_ids: bool,
    assistant_leading_space: bool,
) -> dict:
    """
    Generate one multi-document synthesis item in model-training/eval format.

    Required output fields:
      - src
      - context
      - question
      - answer
      - confuser_answers
      - involved_topics
      - kind
      - messages
      - pred
      - passed

    Important:
      For compare records, all topic names are forced to pseudo style
      and forced to begin with "the ".
    """

    if kind not in SYNTHESIS_KINDS:
        raise ValueError(f"Unknown synthesis kind: {kind}")

    # Critical rule:
    # Compare records must always use topic names beginning with "the ".
    if kind == "compare":
        effective_topic_style = "pseudo"
        effective_use_article = True
    else:
        effective_topic_style = topic_style
        effective_use_article = use_article

    topics = generate_topics(
        r,
        n_cards=n_cards,
        topic_style=effective_topic_style,
        use_article=effective_use_article,
    )

    values = generate_values(r, n_cards=n_cards)
    context = build_context(topics=topics, values=values)

    a_idx, b_idx = r.sample(range(n_cards), 2)

    a_topic = topics[a_idx]
    b_topic = topics[b_idx]

    a_val = values[a_idx]
    b_val = values[b_idx]

    if kind == "sum":
        gold = str(a_val + b_val)
        question = (
            f"Considering only {a_topic} and {b_topic}, what is the "
            f"COMBINED total of the numeric attribute reported in "
            f"each of their documents? Reply with the integer only."
        )

        involved_indices = {a_idx, b_idx}
        involved_topics = [a_topic, b_topic]

    elif kind == "difference":
        larger_t, smaller_t = (
            (a_topic, b_topic) if a_val > b_val else (b_topic, a_topic)
        )
        larger, smaller = (
            (a_val, b_val) if a_val > b_val else (b_val, a_val)
        )

        gold = str(larger - smaller)
        question = (
            f"How many more does {larger_t} have than {smaller_t} "
            f"on the numeric attribute reported in their documents? "
            f"Reply with the integer only."
        )

        involved_indices = {a_idx, b_idx}
        involved_topics = [a_topic, b_topic]

    elif kind == "compare":
        larger_t = a_topic if a_val > b_val else b_topic

        gold = larger_t
        question = (
            f"Comparing the numeric attribute reported by {a_topic} "
            f"and {b_topic}, which one has the LARGER value? Reply "
            f"with the full name of the larger one."
        )

        involved_indices = {a_idx, b_idx}
        involved_topics = [a_topic, b_topic]

    elif kind == "ratio":
        larger, smaller = (a_val, b_val) if a_val >= b_val else (b_val, a_val)

        gold = str(larger // smaller)
        question = (
            f"How many times larger (rounded down to integer) is the "
            f"numeric attribute of {a_topic} compared to {b_topic}? "
            f"If {a_topic} is smaller, swap the order. Reply with the integer only."
        )

        involved_indices = {a_idx, b_idx}
        involved_topics = [a_topic, b_topic]

    elif kind == "sum_three":
        third_pool = [
            card_idx
            for card_idx in range(n_cards)
            if card_idx not in (a_idx, b_idx)
        ]

        c_idx = r.choice(third_pool)
        c_topic = topics[c_idx]
        c_val = values[c_idx]

        gold = str(a_val + b_val + c_val)
        question = (
            f"Considering only {a_topic}, {b_topic}, and {c_topic}, "
            f"what is the COMBINED total of the numeric attribute "
            f"reported in their three documents? Reply with the "
            f"integer only."
        )

        involved_indices = {a_idx, b_idx, c_idx}
        involved_topics = [a_topic, b_topic, c_topic]

    elif kind == "difference_three":
        third_pool = [
            card_idx
            for card_idx in range(n_cards)
            if card_idx not in (a_idx, b_idx)
        ]

        c_idx = r.choice(third_pool)
        c_topic = topics[c_idx]
        c_val = values[c_idx]

        three = sorted(
            [
                (a_val, a_topic),
                (b_val, b_topic),
                (c_val, c_topic),
            ],
            reverse=True,
        )

        gold = str(three[0][0] - three[1][0] - three[2][0])
        question = (
            f"Considering only {a_topic}, {b_topic}, and {c_topic}: "
            f"take the LARGEST of the three numeric attributes, "
            f"subtract the MIDDLE one, then subtract the SMALLEST. "
            f"Reply with the integer only."
        )

        involved_indices = {a_idx, b_idx, c_idx}
        involved_topics = [a_topic, b_topic, c_topic]

    else:
        raise ValueError(f"Unhandled kind: {kind!r}")

    if kind == "compare":
        confuser_answers = [
            topic
            for topic_idx, topic in enumerate(topics)
            if topic_idx not in involved_indices
        ]

        loser_t = b_topic if a_val > b_val else a_topic
        confuser_answers.append(loser_t)

    else:
        confuser_answers = [
            str(values[card_idx])
            for card_idx in range(n_cards)
            if card_idx not in involved_indices
        ]

    user_message = build_user_message(context=context, question=question)

    assistant_content = gold

    if assistant_leading_space:
        assistant_content = " " + assistant_content

    pred = assistant_content.strip()
    passed = prediction_passed(pred=pred, answer=gold)

    item = {
        "src": f"multi_doc_synthesis/{kind}",
        "context": context,
        "question": question,
        "answer": gold,
        "confuser_answers": confuser_answers,
        "involved_topics": involved_topics,
        "kind": kind,
        "messages": [
            {
                "role": "user",
                "content": user_message,
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ],
        "pred": pred,
        "passed": passed,
    }

    if include_gold:
        item["gold"] = gold

    if include_ids:
        item["database_version"] = DATABASE_VERSION
        item["task_id"] = f"multi_doc_synthesis/{kind}/{kind_index:05d}"

    if include_metadata:
        item["metadata"] = {
            "global_index": global_index,
            "kind_index": kind_index,
            "seed": item_seed,
            "n_cards": n_cards,
            "topics": topics,
            "values": values,
            "involved_indices": sorted(involved_indices),
            "involved_topics": involved_topics,
            "confuser_answers": confuser_answers,
            "answer_type": "organization_name" if kind == "compare" else "integer",
            "normalized_gold": normalize_answer(gold),
        }

    return item


def build_items(
    *,
    seed: int,
    n_per_kind: int,
    n_cards: int,
    shuffle: bool,
    topic_style: str,
    use_article: bool,
    include_metadata: bool,
    include_gold: bool,
    include_ids: bool,
    assistant_leading_space: bool,
) -> list[dict]:
    """
    Build records classified directly by synthesis kind.

    This produces:
      n_per_kind sum
      n_per_kind difference
      n_per_kind compare
      n_per_kind ratio
      n_per_kind sum_three
      n_per_kind difference_three
    """

    main_rng = random.Random((int(seed) ^ MULTI_DOC_STREAM_SEED) & 0xFFFFFFFF)

    items: list[dict] = []
    global_index = 0

    for kind in SYNTHESIS_KINDS:
        for kind_index in range(n_per_kind):
            item_seed = main_rng.randint(0, 2**31 - 1)
            item_rng = random.Random(item_seed)

            item = make_multi_doc_synthesis_item(
                r=item_rng,
                item_seed=item_seed,
                global_index=global_index,
                kind=kind,
                kind_index=kind_index,
                n_cards=n_cards,
                topic_style=topic_style,
                use_article=use_article,
                include_metadata=include_metadata,
                include_gold=include_gold,
                include_ids=include_ids,
                assistant_leading_space=assistant_leading_space,
            )

            items.append(item)
            global_index += 1

    if shuffle:
        main_rng.shuffle(items)

    return items


def verify_compare_the_rule(item: dict) -> tuple[bool, str]:
    """
    Strict check for compare records.

    For compare data:
      - answer must start with "the "
      - pred must start with "the "
      - assistant content, after lstrip, must start with "the "
      - involved topics must start with "the "
      - confuser answers must start with "the "
      - metadata topics must start with "the " if metadata exists
    """

    answer = item.get("answer", "")
    pred = item.get("pred", "")
    messages = item.get("messages", [])

    if not starts_with_the(answer):
        return False, f"compare answer must start with 'the ': {answer!r}"

    if not starts_with_the(pred):
        return False, f"compare pred must start with 'the ': {pred!r}"

    if isinstance(messages, list) and len(messages) >= 2:
        assistant_content = str(messages[1].get("content", "")).lstrip()

        if not starts_with_the(assistant_content):
            return False, (
                "compare assistant message content must start with 'the ' "
                f"after lstrip: {messages[1].get('content', '')!r}"
            )

    for topic in item.get("involved_topics", []):
        if not starts_with_the(topic):
            return False, f"compare involved topic must start with 'the ': {topic!r}"

    for confuser in item.get("confuser_answers", []):
        if not starts_with_the(confuser):
            return False, f"compare confuser must start with 'the ': {confuser!r}"

    metadata = item.get("metadata")

    if isinstance(metadata, dict):
        metadata_topics = metadata.get("topics", [])

        if isinstance(metadata_topics, list):
            for topic in metadata_topics:
                if not starts_with_the(topic):
                    return False, f"compare metadata topic must start with 'the ': {topic!r}"

    return True, ""


def verify_item(item: dict) -> tuple[bool, str]:
    required_fields = [
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

    for field in required_fields:
        if field not in item:
            return False, f"Missing required field: {field}"

    kind = item.get("kind")

    if kind not in SYNTHESIS_KINDS:
        return False, f"Invalid kind: {kind!r}"

    if item.get("src") != f"multi_doc_synthesis/{kind}":
        return False, "src does not match kind"

    if not isinstance(item.get("context"), str) or not item["context"].strip():
        return False, "context is empty or not a string"

    if not isinstance(item.get("question"), str) or not item["question"].strip():
        return False, "question is empty or not a string"

    if not isinstance(item.get("answer"), str) or not item["answer"].strip():
        return False, "answer is empty or not a string"

    if not isinstance(item.get("confuser_answers"), list):
        return False, "confuser_answers is not a list"

    if not isinstance(item.get("involved_topics"), list):
        return False, "involved_topics is not a list"

    if not isinstance(item.get("messages"), list):
        return False, "messages is not a list"

    if len(item["messages"]) != 2:
        return False, "messages should contain exactly user and assistant messages"

    if item["messages"][0].get("role") != "user":
        return False, "messages[0].role should be user"

    if item["messages"][1].get("role") != "assistant":
        return False, "messages[1].role should be assistant"

    user_content = item["messages"][0].get("content", "")

    if "Read the documents below" not in user_content:
        return False, "user message missing instruction prefix"

    if item["context"] not in user_content:
        return False, "user message does not contain context"

    if item["question"] not in user_content:
        return False, "user message does not contain question"

    pred = item.get("pred")
    answer = item.get("answer")

    expected_passed = prediction_passed(pred=pred, answer=answer)

    if item.get("passed") != expected_passed:
        return False, "passed field does not match pred-vs-answer normalization"

    if not expected_passed:
        return False, f"pred does not match answer: pred={pred!r}, answer={answer!r}"

    answer_norm = normalize_answer(answer)

    for confuser in item["confuser_answers"]:
        if normalize_answer(confuser) == answer_norm:
            return False, f"confuser matches answer: {confuser!r}"

    if kind in {"sum", "difference", "ratio"}:
        if len(item["involved_topics"]) != 2:
            return False, f"{kind} should involve exactly 2 topics"

    if kind in {"sum_three", "difference_three"}:
        if len(item["involved_topics"]) != 3:
            return False, f"{kind} should involve exactly 3 topics"

    if kind == "compare":
        if len(item["involved_topics"]) != 2:
            return False, "compare should involve exactly 2 topics"

        ok, err = verify_compare_the_rule(item)

        if not ok:
            return False, err

    if "gold" in item:
        if item["gold"] != item["answer"]:
            return False, "gold exists but does not match answer"

    if "metadata" in item:
        metadata = item["metadata"]

        if not isinstance(metadata, dict):
            return False, "metadata exists but is not a dict"

        if metadata.get("normalized_gold") != normalize_answer(item["answer"]):
            return False, "metadata normalized_gold mismatch"

    return True, ""


def verify_items(items: list[dict], max_failures: int = 20) -> bool:
    failures = []

    for idx, item in enumerate(items):
        ok, err = verify_item(item)

        if not ok:
            failures.append((idx, item.get("src"), item.get("kind"), err))

            if len(failures) >= max_failures:
                break

    if not failures:
        print("Local verification: all generated training/eval items passed.")
        return True

    print("Local verification failed.", file=sys.stderr)
    print(f"Failures shown: {len(failures)}", file=sys.stderr)

    for idx, src, kind, err in failures:
        print("=" * 80, file=sys.stderr)
        print(f"Item index: {idx}", file=sys.stderr)
        print(f"Src: {src}", file=sys.stderr)
        print(f"Kind: {kind}", file=sys.stderr)
        print(err, file=sys.stderr)

    return False


def write_jsonl(items: list[dict], output_path: Path, append: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"

    with output_path.open(mode, encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_counts(items: list[dict]) -> None:
    counts: dict[str, int] = {}

    for item in items:
        kind = item["kind"]
        counts[kind] = counts.get(kind, 0) + 1

    print("\nKind counts:")

    for kind in SYNTHESIS_KINDS:
        print(f"  {kind}: {counts.get(kind, 0)}")


def print_passed_counts(items: list[dict]) -> None:
    passed = sum(1 for item in items if item.get("passed") is True)
    failed = sum(1 for item in items if item.get("passed") is False)

    print("\nPrediction status:")
    print(f"  passed true: {passed}")
    print(f"  passed false: {failed}")


def print_context_stats(items: list[dict]) -> None:
    if not items:
        return

    word_counts = [
        len(str(item.get("context", "")).split())
        for item in items
    ]

    char_counts = [
        len(str(item.get("context", "")))
        for item in items
    ]

    print("\nContext stats:")
    print(f"  Min context words: {min(word_counts)}")
    print(f"  Max context words: {max(word_counts)}")
    print(f"  Avg context words: {sum(word_counts) // len(word_counts)}")
    print(f"  Min context chars: {min(char_counts)}")
    print(f"  Max context chars: {max(char_counts)}")
    print(f"  Avg context chars: {sum(char_counts) // len(char_counts)}")


def print_compare_the_stats(items: list[dict]) -> None:
    compare_items = [
        item
        for item in items
        if item.get("kind") == "compare"
    ]

    total_compare = len(compare_items)

    answer_ok = sum(
        1
        for item in compare_items
        if starts_with_the(item.get("answer", ""))
    )

    pred_ok = sum(
        1
        for item in compare_items
        if starts_with_the(item.get("pred", ""))
    )

    involved_ok = sum(
        1
        for item in compare_items
        if all(starts_with_the(topic) for topic in item.get("involved_topics", []))
    )

    confuser_ok = sum(
        1
        for item in compare_items
        if all(starts_with_the(confuser) for confuser in item.get("confuser_answers", []))
    )

    print("\nCompare 'the' rule stats:")
    print(f"  compare records: {total_compare}")
    print(f"  answers starting with 'the ': {answer_ok}/{total_compare}")
    print(f"  preds starting with 'the ': {pred_ok}/{total_compare}")
    print(f"  involved topics all starting with 'the ': {involved_ok}/{total_compare}")
    print(f"  confusers all starting with 'the ': {confuser_ok}/{total_compare}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic multi_doc_synthesis training/eval JSONL items. "
            "Output schema includes messages, pred, and passed. "
            "For compare records, all organization names start with 'the '."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset/multi_doc_training_all_cases.jsonl",
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
        default=400,
        help="Number of records to generate for each synthesis kind.",
    )

    parser.add_argument(
        "--n-cards",
        type=int,
        default=BENCH_MULTI_DOC_N_CARDS,
        help=(
            "Number of document cards per item. "
            "The effective value is max(3, --n-cards, 7)."
        ),
    )

    parser.add_argument(
        "--topic-style",
        type=str,
        choices=["pseudo", "normal"],
        default="pseudo",
        help=(
            "Topic naming style for non-compare records. "
            "Compare records always use pseudo topics with leading 'the '."
        ),
    )

    parser.add_argument(
        "--no-article",
        action="store_true",
        help=(
            "For non-compare pseudo topic style, do not prefix names with 'the'. "
            "Compare records ignore this and always use 'the '."
        ),
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
        help="Run local structural verification before writing.",
    )

    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata field in each item.",
    )

    parser.add_argument(
        "--include-gold",
        action="store_true",
        help="Include gold field in each item.",
    )

    parser.add_argument(
        "--include-ids",
        action="store_true",
        help="Include database_version and task_id fields in each item.",
    )

    parser.add_argument(
        "--assistant-leading-space",
        action="store_true",
        help=(
            "Write assistant message content with a leading space, "
            "matching outputs like ' answer'. The pred field is stripped."
        ),
    )

    args = parser.parse_args()

    if args.n_per_kind < 1:
        print("--n-per-kind must be at least 1.", file=sys.stderr)
        raise SystemExit(1)

    effective_n_cards = max(3, args.n_cards, 7)

    items = build_items(
        seed=args.seed,
        n_per_kind=args.n_per_kind,
        n_cards=effective_n_cards,
        shuffle=not args.no_shuffle,
        topic_style=args.topic_style,
        use_article=not args.no_article,
        include_metadata=args.include_metadata,
        include_gold=args.include_gold,
        include_ids=args.include_ids,
        assistant_leading_space=args.assistant_leading_space,
    )

    if args.verify:
        ok = verify_items(items)

        if not ok:
            print("Not writing JSONL because verification failed.", file=sys.stderr)
            raise SystemExit(1)

    output_path = Path(args.output)
    write_jsonl(items, output_path, append=args.append)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Total records: {len(items)}")
    print(f"Synthesis kinds: {len(SYNTHESIS_KINDS)}")
    print(f"Records per kind: {args.n_per_kind}")
    print(f"Cards per item: {effective_n_cards}")
    print(f"Seed: {args.seed}")
    print(f"Topic style for non-compare records: {args.topic_style}")
    print(f"Use article for non-compare pseudo records: {not args.no_article}")
    print("Compare records force article: True")
    print(f"Append mode: {args.append}")
    print(f"Local verified: {args.verify}")
    print(f"Include metadata: {args.include_metadata}")
    print(f"Include gold: {args.include_gold}")
    print(f"Include IDs: {args.include_ids}")
    print(f"Assistant leading space: {args.assistant_leading_space}")
    print("")
    print("Schema:")
    print(
        "  src, context, question, answer, confuser_answers, "
        "involved_topics, kind, messages, pred, passed"
    )

    print_counts(items)
    print_passed_counts(items)
    print_compare_the_stats(items)
    print_context_stats(items)


if __name__ == "__main__":
    main()
