#!/usr/bin/env python3

import argparse
import json
import random
import re
import sys
import traceback
from pathlib import Path


REASONING_STREAM_SEED = 0xA11CE2026


REASONING_KINDS = [
    "arithmetic_total",
    "compare_difference",
    "unit_price",
    "percentage_discount",
    "sequence_next",
    "age_difference",
    "set_overlap",
    "time_elapsed",
    "ratio_split",
    "logic_ordering",
]


NAMES = [
    "Mina",
    "Leo",
    "Ava",
    "Noah",
    "Iris",
    "Omar",
    "Lina",
    "Theo",
    "Nora",
    "Eli",
    "Sara",
    "Milo",
]


OBJECTS = [
    "apples",
    "books",
    "stickers",
    "marbles",
    "pencils",
    "tickets",
    "coins",
    "cards",
    "shells",
    "buttons",
]


ITEMS = [
    "notebooks",
    "pens",
    "folders",
    "erasers",
    "markers",
    "rulers",
    "mugs",
    "badges",
]


FINAL_RE = re.compile(r"Final answer:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)


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


def normalize_completion(completion: str) -> str:
    completion = completion.replace("\r\n", "\n").replace("\r", "\n")
    completion = completion.strip()

    return completion + "\n"


def make_prompt(question: str, answer_only: bool = False) -> str:
    if answer_only:
        return (
            "Solve the problem. Output only the final answer.\n\n"
            f"{question}"
        )

    return (
        "Solve the problem. Give a short explanation and end with "
        "'Final answer: <answer>'.\n\n"
        f"{question}"
    )


def make_record(
    kind: str,
    index: int,
    question: str,
    prompt: str,
    completion: str,
    gold: str,
    seed: int,
    metadata: dict | None = None,
) -> dict:
    return {
        "src": f"procedural_reasoning/{kind}",
        "kind": kind,
        "task_id": f"reasoning/{kind}/{index:05d}",
        "question": question,
        "prompt": prompt,
        "completion": normalize_completion(completion),
        "gold": str(gold),
        "answer_type": "short_text",
        "status": "gold_reasoning",
        "seed": seed,
        "metadata": metadata or {},
    }


def make_completion(
    explanation: str,
    gold: str,
    answer_only: bool = False,
) -> str:
    if answer_only:
        return str(gold)

    return f"{explanation}\nFinal answer: {gold}"


def generate_arithmetic_total(r: random.Random, answer_only: bool):
    name = r.choice(NAMES)
    obj = r.choice(OBJECTS)

    start = r.randint(5, 30)
    add = r.randint(2, 20)
    remove = r.randint(1, min(start + add - 1, 15))

    gold = start + add - remove

    question = (
        f"{name} has {start} {obj}. {name} gets {add} more and gives away "
        f"{remove}. How many {obj} does {name} have now?"
    )

    explanation = (
        f"{name} starts with {start} {obj}, gets {add} more, and gives away "
        f"{remove}. So {start} + {add} - {remove} = {gold}."
    )

    return question, make_completion(explanation, str(gold), answer_only), str(gold), {
        "start": start,
        "add": add,
        "remove": remove,
    }


def generate_compare_difference(r: random.Random, answer_only: bool):
    name1, name2 = r.sample(NAMES, 2)
    obj = r.choice(OBJECTS)

    a = r.randint(10, 80)
    diff = r.randint(3, 25)
    b = a + diff

    if r.random() < 0.5:
        question = (
            f"{name1} has {a} {obj}. {name2} has {b} {obj}. "
            f"How many more {obj} does {name2} have than {name1}?"
        )
        explanation = (
            f"{name2} has {b} {obj} and {name1} has {a}. "
            f"The difference is {b} - {a} = {diff}."
        )
    else:
        question = (
            f"{name1} has {b} {obj}. {name2} has {a} {obj}. "
            f"How many fewer {obj} does {name2} have than {name1}?"
        )
        explanation = (
            f"{name1} has {b} {obj} and {name2} has {a}. "
            f"The difference is {b} - {a} = {diff}."
        )

    return question, make_completion(explanation, str(diff), answer_only), str(diff), {
        "larger": b,
        "smaller": a,
        "difference": diff,
    }


def generate_unit_price(r: random.Random, answer_only: bool):
    item = r.choice(ITEMS)

    qty = r.choice([2, 3, 4, 5, 6, 8, 10])
    unit = r.randint(2, 15)
    total = qty * unit

    question = (
        f"A pack of {qty} {item} costs {total} dollars. "
        f"What is the cost of one {item[:-1] if item.endswith('s') else item}?"
    )

    explanation = (
        f"The total cost is {total} dollars for {qty} {item}. "
        f"So one costs {total} / {qty} = {unit} dollars."
    )

    gold = f"{unit} dollars"

    return question, make_completion(explanation, gold, answer_only), gold, {
        "quantity": qty,
        "unit_price": unit,
        "total": total,
    }


def generate_percentage_discount(r: random.Random, answer_only: bool):
    item = r.choice(ITEMS)

    original = r.choice([20, 30, 40, 50, 60, 80, 100, 120])
    percent = r.choice([10, 20, 25, 50])
    discount = original * percent // 100
    final = original - discount

    question = (
        f"A {item[:-1] if item.endswith('s') else item} costs {original} dollars. "
        f"It is discounted by {percent} percent. What is the final price?"
    )

    explanation = (
        f"The discount is {percent} percent of {original}, which is {discount}. "
        f"The final price is {original} - {discount} = {final} dollars."
    )

    gold = f"{final} dollars"

    return question, make_completion(explanation, gold, answer_only), gold, {
        "original": original,
        "percent": percent,
        "discount": discount,
        "final": final,
    }


def generate_sequence_next(r: random.Random, answer_only: bool):
    start = r.randint(1, 20)
    step = r.randint(2, 9)
    length = 5

    seq = [start + i * step for i in range(length)]
    gold = seq[-1] + step

    question = (
        "What is the next number in this sequence? "
        + ", ".join(str(x) for x in seq)
    )

    explanation = (
        f"The sequence increases by {step} each time. "
        f"After {seq[-1]}, the next number is {seq[-1]} + {step} = {gold}."
    )

    return question, make_completion(explanation, str(gold), answer_only), str(gold), {
        "start": start,
        "step": step,
        "sequence": seq,
    }


def generate_age_difference(r: random.Random, answer_only: bool):
    older, younger = r.sample(NAMES, 2)

    younger_age = r.randint(6, 30)
    diff = r.randint(2, 20)
    older_age = younger_age + diff

    question = (
        f"{older} is {older_age} years old. {younger} is {younger_age} years old. "
        f"How many years older is {older} than {younger}?"
    )

    explanation = (
        f"{older} is {older_age} and {younger} is {younger_age}. "
        f"The age difference is {older_age} - {younger_age} = {diff}."
    )

    gold = f"{diff} years"

    return question, make_completion(explanation, gold, answer_only), gold, {
        "older_age": older_age,
        "younger_age": younger_age,
        "difference": diff,
    }


def generate_set_overlap(r: random.Random, answer_only: bool):
    group_a = r.randint(15, 50)
    group_b = r.randint(15, 50)
    both = r.randint(3, min(group_a, group_b, 20))

    total = group_a + group_b - both

    question = (
        f"In a club, {group_a} students like chess, {group_b} students like music, "
        f"and {both} students like both. How many students like chess or music?"
    )

    explanation = (
        f"Add both groups and subtract the overlap once. "
        f"So {group_a} + {group_b} - {both} = {total}."
    )

    return question, make_completion(explanation, str(total), answer_only), str(total), {
        "group_a": group_a,
        "group_b": group_b,
        "both": both,
        "union": total,
    }


def generate_time_elapsed(r: random.Random, answer_only: bool):
    start_hour = r.randint(6, 10)
    start_minute = r.choice([0, 5, 10, 15, 20, 30, 45])
    elapsed_minutes = r.choice([35, 40, 45, 50, 55, 65, 75, 90, 105, 120])

    start_total = start_hour * 60 + start_minute
    end_total = start_total + elapsed_minutes

    end_hour = end_total // 60
    end_minute = end_total % 60

    start_text = f"{start_hour}:{start_minute:02d}"
    end_text = f"{end_hour}:{end_minute:02d}"

    question = (
        f"A class starts at {start_text} and lasts {elapsed_minutes} minutes. "
        f"What time does it end?"
    )

    explanation = (
        f"Add {elapsed_minutes} minutes to {start_text}. "
        f"The ending time is {end_text}."
    )

    return question, make_completion(explanation, end_text, answer_only), end_text, {
        "start": start_text,
        "elapsed_minutes": elapsed_minutes,
        "end": end_text,
    }


def generate_ratio_split(r: random.Random, answer_only: bool):
    name1, name2 = r.sample(NAMES, 2)
    obj = r.choice(OBJECTS)

    a = r.randint(1, 5)
    b = r.randint(1, 5)

    multiplier = r.randint(3, 12)
    total = (a + b) * multiplier

    share1 = a * multiplier
    share2 = b * multiplier

    question = (
        f"{name1} and {name2} share {total} {obj} in the ratio {a}:{b}. "
        f"How many {obj} does {name1} get?"
    )

    explanation = (
        f"The ratio has {a} + {b} = {a + b} parts. "
        f"Each part is {total} / {a + b} = {multiplier}. "
        f"{name1} gets {a} parts, so {a} * {multiplier} = {share1}."
    )

    return question, make_completion(explanation, str(share1), answer_only), str(share1), {
        "ratio": [a, b],
        "total": total,
        "share1": share1,
        "share2": share2,
    }


def generate_logic_ordering(r: random.Random, answer_only: bool):
    people = r.sample(NAMES, 3)

    tallest = people[0]
    middle = people[1]
    shortest = people[2]

    question = (
        f"{tallest} is taller than {middle}. "
        f"{middle} is taller than {shortest}. "
        f"Who is the tallest?"
    )

    explanation = (
        f"{tallest} is taller than {middle}, and {middle} is taller than {shortest}. "
        f"So {tallest} is the tallest."
    )

    gold = tallest

    return question, make_completion(explanation, gold, answer_only), gold, {
        "tallest": tallest,
        "middle": middle,
        "shortest": shortest,
    }


def generate_item(
    kind: str,
    index: int,
    seed: int,
    answer_only: bool = False,
) -> dict:
    r = random.Random(seed)

    if kind == "arithmetic_total":
        question, completion, gold, metadata = generate_arithmetic_total(r, answer_only)

    elif kind == "compare_difference":
        question, completion, gold, metadata = generate_compare_difference(r, answer_only)

    elif kind == "unit_price":
        question, completion, gold, metadata = generate_unit_price(r, answer_only)

    elif kind == "percentage_discount":
        question, completion, gold, metadata = generate_percentage_discount(r, answer_only)

    elif kind == "sequence_next":
        question, completion, gold, metadata = generate_sequence_next(r, answer_only)

    elif kind == "age_difference":
        question, completion, gold, metadata = generate_age_difference(r, answer_only)

    elif kind == "set_overlap":
        question, completion, gold, metadata = generate_set_overlap(r, answer_only)

    elif kind == "time_elapsed":
        question, completion, gold, metadata = generate_time_elapsed(r, answer_only)

    elif kind == "ratio_split":
        question, completion, gold, metadata = generate_ratio_split(r, answer_only)

    elif kind == "logic_ordering":
        question, completion, gold, metadata = generate_logic_ordering(r, answer_only)

    else:
        raise ValueError(f"Unknown reasoning kind: {kind}")

    prompt = make_prompt(question=question, answer_only=answer_only)

    return make_record(
        kind=kind,
        index=index,
        question=question,
        prompt=prompt,
        completion=completion,
        gold=gold,
        seed=seed,
        metadata=metadata,
    )


def verify_record(record: dict, answer_only: bool = False) -> tuple[bool, str]:
    completion = record.get("completion", "")
    gold = record.get("gold", "")

    try:
        if answer_only:
            predicted = completion.strip()
        else:
            predicted = extract_final_answer(completion)

            if predicted is None:
                return False, "No 'Final answer:' line found."

        if normalize_answer(predicted) != normalize_answer(gold):
            return (
                False,
                f"Final answer mismatch. predicted={predicted!r}, gold={gold!r}",
            )

        return True, ""

    except Exception:
        return False, traceback.format_exc()


def verify_records(
    records: list[dict],
    answer_only: bool = False,
    max_failures: int = 10,
) -> bool:
    failures = []

    for record in records:
        ok, err = verify_record(record, answer_only=answer_only)

        if not ok:
            failures.append((record["task_id"], record["kind"], err))

            if len(failures) >= max_failures:
                break

    if not failures:
        print("Local verification: all generated reasoning records passed.")
        return True

    print("Local verification failed.")
    print(f"Failures shown: {len(failures)}")

    for task_id, kind, err in failures:
        print("=" * 80)
        print(f"Task: {task_id}")
        print(f"Kind: {kind}")
        print(err)

    return False


def assert_all_completions_nonempty(records: list[dict]) -> None:
    bad = []

    for record in records:
        completion = record.get("completion", "")

        if not isinstance(completion, str) or not completion.strip():
            bad.append(
                (
                    record.get("task_id"),
                    record.get("kind"),
                    completion,
                )
            )

    if bad:
        print("Found empty completions:", file=sys.stderr)

        for task_id, kind, completion in bad[:10]:
            print("=" * 80, file=sys.stderr)
            print(f"Task: {task_id}", file=sys.stderr)
            print(f"Kind: {kind}", file=sys.stderr)
            print(repr(completion), file=sys.stderr)

        raise SystemExit(1)


def build_records(
    seed: int,
    n_per_kind: int,
    shuffle: bool = True,
    answer_only: bool = False,
) -> list[dict]:
    main_rng = random.Random((int(seed) ^ REASONING_STREAM_SEED) & 0xFFFFFFFF)
    records = []

    index = 0

    for kind in REASONING_KINDS:
        for _ in range(n_per_kind):
            item_seed = main_rng.randint(0, 2**31 - 1)

            record = generate_item(
                kind=kind,
                index=index,
                seed=item_seed,
                answer_only=answer_only,
            )

            records.append(record)
            index += 1

    if shuffle:
        main_rng.shuffle(records)

    return records


def write_jsonl(records: list[dict], output_path: Path, append: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"

    with output_path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_counts(records: list[dict]) -> None:
    counts = {}

    for record in records:
        kind = record["kind"]
        counts[kind] = counts.get(kind, 0) + 1

    print("\nKind counts:")
    for kind in sorted(counts):
        print(f"  {kind}: {counts[kind]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic reasoning JSONL database with short "
            "gold explanations and locally verified final answers."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset/reasoning_database_all_cases.jsonl",
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
        default=100,
        help="Number of reasoning records to generate per kind.",
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
        help="Run local verification that final answers match gold answers.",
    )

    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Generate completions containing only the final answer.",
    )

    args = parser.parse_args()

    records = build_records(
        seed=args.seed,
        n_per_kind=args.n_per_kind,
        shuffle=not args.no_shuffle,
        answer_only=args.answer_only,
    )

    assert_all_completions_nonempty(records)

    if args.verify:
        ok = verify_records(
            records,
            answer_only=args.answer_only,
        )

        if not ok:
            print("Not writing JSONL because local verification failed.", file=sys.stderr)
            raise SystemExit(1)

    output_path = Path(args.output)
    write_jsonl(records, output_path, append=args.append)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Total records: {len(records)}")
    print(f"Reasoning kinds: {len(REASONING_KINDS)}")
    print(f"Records per kind: {args.n_per_kind}")
    print(f"Append mode: {args.append}")
    print(f"Local verified: {args.verify}")
    print(f"Answer only: {args.answer_only}")
    print("Completion format: short explanation plus final answer")

    print_counts(records)


if __name__ == "__main__":
    main()
