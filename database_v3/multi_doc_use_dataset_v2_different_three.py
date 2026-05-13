#!/usr/bin/env python3

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path


DATABASE_VERSION = "multi_doc_synthesis_bench_difference_three_v1"
MULTI_DOC_STREAM_SEED = 0x4D0C2026

# Only difference_three database construction
SYNTHESIS_KINDS = [
    "difference_three",
]

BENCH_MULTI_DOC_N_CARDS = int(os.environ.get("BENCH_MULTI_DOC_N_CARDS", "7"))


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


def build_user_message(context: str, question: str) -> str:
    return (
        "Read the documents below and answer the question that follows. "
        "Reply with the calculation process and final answer.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def prediction_passed(pred: str, answer: str) -> bool:
    return normalize_answer(pred) == normalize_answer(answer)


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
    use_article: bool = True,
) -> list[str]:
    topics: list[str] = []
    seen_topics: set[str] = set()
    synth_attempts = 0

    while len(topics) < n_cards and synth_attempts < n_cards * 128:
        synth_attempts += 1

        topic = _pseudo_org_topic(r, use_article=use_article)

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

    Distinct ranges reduce accidental collisions:
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


def build_difference_three_answer(
    largest_val: int,
    middle_val: int,
    smallest_val: int,
) -> tuple[str, int]:
    """
    Build the worked-solution answer.

    Format:
        1161, 910, 701

        Largest: 1161
        Middle: 910
        Smallest: 701

        1161 - 910 - 701 = -450

        -450
    """
    final_value = largest_val - middle_val - smallest_val

    answer = (
        f"{largest_val}, {middle_val}, {smallest_val}\n\n"
        f"Largest: {largest_val}\n"
        f"Middle: {middle_val}\n"
        f"Smallest: {smallest_val}\n\n"
        f"{largest_val} - {middle_val} - {smallest_val} = {final_value}\n\n"
        f"{final_value}"
    )

    return answer, final_value


def make_difference_three_item(
    *,
    r: random.Random,
    item_seed: int,
    global_index: int,
    kind_index: int,
    n_cards: int,
    use_article: bool,
    include_metadata: bool,
    include_gold: bool,
    include_ids: bool,
    assistant_leading_space: bool,
) -> dict:
    """
    Generate one difference_three item.

    Logic:
      1. Select 3 organizations.
      2. Read their numeric attributes.
      3. Sort values descending.
      4. Compute:

            largest - middle - smallest

      5. Store answer as a worked calculation process.
    """

    kind = "difference_three"

    topics = generate_topics(
        r,
        n_cards=n_cards,
        use_article=use_article,
    )

    values = generate_values(r, n_cards=n_cards)
    context = build_context(topics=topics, values=values)

    selected_indices = r.sample(range(n_cards), 3)

    a_idx, b_idx, c_idx = selected_indices

    a_topic = topics[a_idx]
    b_topic = topics[b_idx]
    c_topic = topics[c_idx]

    a_val = values[a_idx]
    b_val = values[b_idx]
    c_val = values[c_idx]

    three = sorted(
        [
            (a_val, a_topic, a_idx),
            (b_val, b_topic, b_idx),
            (c_val, c_topic, c_idx),
        ],
        reverse=True,
    )

    largest_val, largest_topic, largest_idx = three[0]
    middle_val, middle_topic, middle_idx = three[1]
    smallest_val, smallest_topic, smallest_idx = three[2]

    gold, final_value = build_difference_three_answer(
        largest_val=largest_val,
        middle_val=middle_val,
        smallest_val=smallest_val,
    )

    question = (
        f"Considering only {a_topic}, {b_topic}, and {c_topic}: "
        f"take the LARGEST of the three numeric attributes, "
        f"subtract the MIDDLE one, then subtract the SMALLEST. "
        f"Include the calculation process and final answer."
    )

    involved_indices = set(selected_indices)
    involved_topics = [a_topic, b_topic, c_topic]

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
        "src": "multi_doc_synthesis/difference_three",
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
        item["task_id"] = f"multi_doc_synthesis/difference_three/{kind_index:05d}"

    if include_metadata:
        item["metadata"] = {
            "global_index": global_index,
            "kind_index": kind_index,
            "seed": item_seed,
            "n_cards": n_cards,
            "topics": topics,
            "values": values,
            "selected_indices": selected_indices,
            "involved_indices": sorted(involved_indices),
            "involved_topics": involved_topics,
            "ordered_values": {
                "largest": largest_val,
                "middle": middle_val,
                "smallest": smallest_val,
            },
            "ordered_topics": {
                "largest": largest_topic,
                "middle": middle_topic,
                "smallest": smallest_topic,
            },
            "ordered_indices": {
                "largest": largest_idx,
                "middle": middle_idx,
                "smallest": smallest_idx,
            },
            "calculation": f"{largest_val} - {middle_val} - {smallest_val} = {final_value}",
            "final_answer": str(final_value),
            "confuser_answers": confuser_answers,
            "answer_type": "worked_integer_solution",
            "normalized_gold": normalize_answer(gold),
        }

    return item


def build_items(
    *,
    seed: int,
    n_records: int,
    n_cards: int,
    shuffle: bool,
    use_article: bool,
    include_metadata: bool,
    include_gold: bool,
    include_ids: bool,
    assistant_leading_space: bool,
) -> list[dict]:
    """
    Build only difference_three records.
    """

    main_rng = random.Random((int(seed) ^ MULTI_DOC_STREAM_SEED) & 0xFFFFFFFF)

    items: list[dict] = []

    for kind_index in range(n_records):
        item_seed = main_rng.randint(0, 2**31 - 1)
        item_rng = random.Random(item_seed)

        item = make_difference_three_item(
            r=item_rng,
            item_seed=item_seed,
            global_index=kind_index,
            kind_index=kind_index,
            n_cards=n_cards,
            use_article=use_article,
            include_metadata=include_metadata,
            include_gold=include_gold,
            include_ids=include_ids,
            assistant_leading_space=assistant_leading_space,
        )

        items.append(item)

    if shuffle:
        main_rng.shuffle(items)

    return items


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

    if item.get("kind") != "difference_three":
        return False, f"Invalid kind for this database: {item.get('kind')!r}"

    if item.get("src") != "multi_doc_synthesis/difference_three":
        return False, "src must be multi_doc_synthesis/difference_three"

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

    if len(item["involved_topics"]) != 3:
        return False, "difference_three should involve exactly 3 topics"

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

    if "Largest:" not in answer:
        return False, "answer missing Largest line"

    if "Middle:" not in answer:
        return False, "answer missing Middle line"

    if "Smallest:" not in answer:
        return False, "answer missing Smallest line"

    if "=" not in answer:
        return False, "answer missing calculation equation"

    if "gold" in item:
        if item["gold"] != item["answer"]:
            return False, "gold exists but does not match answer"

    if "metadata" in item:
        metadata = item["metadata"]

        if not isinstance(metadata, dict):
            return False, "metadata exists but is not a dict"

        if metadata.get("normalized_gold") != normalize_answer(item["answer"]):
            return False, "metadata normalized_gold mismatch"

        if metadata.get("answer_type") != "worked_integer_solution":
            return False, "metadata answer_type should be worked_integer_solution"

        if "final_answer" not in metadata:
            return False, "metadata missing final_answer"

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
        print("Local verification: all generated difference_three items passed.")
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
    print(f"  difference_three: {counts.get('difference_three', 0)}")


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


def print_answer_sample(items: list[dict]) -> None:
    if not items:
        return

    print("\nSample answer:")
    print("-" * 80)
    print(items[0]["answer"])
    print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic difference_three multi-document synthesis "
            "training/eval JSONL items. Answers include calculation process."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset/multi_doc_difference_three.jsonl",
        help="Output JSONL file path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260505,
        help="Random seed.",
    )

    parser.add_argument(
        "--n-records",
        type=int,
        default=400,
        help="Number of difference_three records to generate.",
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
        "--no-article",
        action="store_true",
        help="Do not prefix pseudo organization names with 'the'.",
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
        default=True,
        help="Include metadata field in each item.",
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable metadata field in each item.",
    )

    parser.add_argument(
        "--include-gold",
        action="store_true",
        default=True,
        help="Include gold field in each item.",
    )

    parser.add_argument(
        "--no-gold",
        action="store_true",
        help="Disable gold field in each item.",
    )

    parser.add_argument(
        "--include-ids",
        action="store_true",
        default=True,
        help="Include database_version and task_id fields in each item.",
    )

    parser.add_argument(
        "--no-ids",
        action="store_true",
        help="Disable database_version and task_id fields.",
    )

    parser.add_argument(
        "--assistant-leading-space",
        action="store_true",
        help=(
            "Write assistant message content with a leading space. "
            "The pred field is stripped."
        ),
    )

    args = parser.parse_args()

    if args.n_records < 1:
        print("--n-records must be at least 1.", file=sys.stderr)
        raise SystemExit(1)

    effective_n_cards = max(3, args.n_cards, 7)

    include_metadata = args.include_metadata and not args.no_metadata
    include_gold = args.include_gold and not args.no_gold
    include_ids = args.include_ids and not args.no_ids

    items = build_items(
        seed=args.seed,
        n_records=args.n_records,
        n_cards=effective_n_cards,
        shuffle=not args.no_shuffle,
        use_article=not args.no_article,
        include_metadata=include_metadata,
        include_gold=include_gold,
        include_ids=include_ids,
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
    print("Synthesis kinds: 1")
    print(f"Records: {args.n_records}")
    print(f"Cards per item: {effective_n_cards}")
    print(f"Seed: {args.seed}")
    print(f"Use article: {not args.no_article}")
    print(f"Append mode: {args.append}")
    print(f"Local verified: {args.verify}")
    print(f"Include metadata: {include_metadata}")
    print(f"Include gold: {include_gold}")
    print(f"Include IDs: {include_ids}")
    print(f"Assistant leading space: {args.assistant_leading_space}")
    print("")
    print("Schema:")
    print(
        "  src, context, question, answer, confuser_answers, "
        "involved_topics, kind, messages, pred, passed, gold, "
        "database_version, task_id, metadata"
    )

    print_counts(items)
    print_passed_counts(items)
    print_context_stats(items)
    print_answer_sample(items)


if __name__ == "__main__":
    main()
