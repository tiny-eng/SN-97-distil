#!/usr/bin/env python3

import argparse
import json
import random
import re
import sys
from pathlib import Path


LONG_CONTEXT_DATABASE_VERSION = "synthetic_long_context_database_v1"
LONG_CONTEXT_STREAM_SEED = 0x10C02026


LONG_CONTEXT_KINDS = [
    "needle_in_haystack",
    "attribute_lookup",
    "multi_hop_lookup",
    "timeline_lookup",
    "constraint_count",
]


NAMES = [
    "Avery Stone", "Blair Chen", "Casey Morgan", "Drew Patel", "Emery Brooks",
    "Finley Ross", "Gray Rivera", "Harper Quinn", "Indigo Lane", "Jordan Vale",
    "Kai Mercer", "Logan Reed", "Morgan Ellis", "Noel Carter", "Oakley Price",
    "Parker Sloan", "Quinn Avery", "Reese Novak", "Sawyer Kim", "Taylor Brooks",
    "Uma Hart", "Val Morgan", "Winter Cole", "Xen Lee", "Yael Stone", "Zion Park",
    "Mira Wells", "Nico Fox", "Lena Cross", "Owen Hale", "Iris Blake", "Theo Grant",
]

DEPARTMENTS = [
    "Archive", "Security", "Research", "Operations", "Logistics",
    "Planning", "Engineering", "Compliance", "Field Support", "Records",
]

TOOLS = [
    "caliper", "scanner", "notebook", "compass", "tablet",
    "camera", "microscope", "labeler", "spectrometer", "ledger",
]

CITIES = [
    "Oslo", "Nairobi", "Lisbon", "Seoul", "Toronto", "Helsinki",
    "Cairo", "Madrid", "Tokyo", "Sydney", "Berlin", "Athens",
]

PROJECTS = [
    "Project Amber", "Project Beacon", "Project Cedar", "Project Delta",
    "Project Ember", "Project Falcon", "Project Glacier", "Project Harbor",
    "Project Ion", "Project Juniper", "Project Kestrel", "Project Lantern",
]

OFFICES = [
    "North Annex", "South Wing", "East Tower", "West Archive",
    "Central Lab", "Harbor Office", "Garden Suite", "River Room",
]

ARTIFACTS = [
    "bronze astrolabe", "ivory compass", "silver coin", "ceramic tablet",
    "glass pendant", "iron key", "painted scroll", "wooden mask",
    "marble seal", "copper mirror", "linen map", "stone marker",
]

MATERIALS = [
    "bronze", "ivory", "silver", "ceramic", "glass", "iron",
    "painted linen", "wood", "marble", "copper", "linen", "stone",
]

REGIONS = [
    "north", "south", "east", "west", "central", "coastal",
]

STATUSES = [
    "approved", "pending", "delayed", "returned", "archived",
]

ITEMS = [
    "medical kits", "solar lamps", "water filters", "field radios",
    "blankets", "survey tablets", "sample boxes", "battery packs",
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


FINAL_RE = re.compile(r"Final answer:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)


def extract_final_answer(completion: str) -> str | None:
    match = FINAL_RE.search(str(completion).strip())

    if not match:
        return None

    return match.group(1).strip()


def synthetic_word(r: random.Random) -> str:
    starts = [
        "br", "cl", "dr", "fl", "gr", "kr", "pl", "pr", "sl", "tr",
        "v", "m", "n", "s", "t", "z",
    ]
    mids = [
        "a", "e", "i", "o", "u", "ae", "ai", "oa", "ou", "ia",
    ]
    ends = [
        "n", "r", "s", "t", "l", "m", "ck", "nd", "st", "th",
    ]

    return r.choice(starts) + r.choice(mids) + r.choice(ends)


def synthetic_sentence(r: random.Random, min_words: int = 10, max_words: int = 20) -> str:
    words = [synthetic_word(r) for _ in range(r.randint(min_words, max_words))]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def filler_paragraph(r: random.Random, paragraph_id: int, sentences: int = 5) -> str:
    topic = r.choice(
        [
            "daily operations",
            "archive handling",
            "inspection notes",
            "storage policy",
            "equipment routing",
            "field communication",
            "maintenance review",
            "catalog update",
        ]
    )

    lines = [
        f"Paragraph {paragraph_id:04d}: This section contains background notes about {topic}."
    ]

    for _ in range(sentences):
        lines.append(synthetic_sentence(r))

    return " ".join(lines)


def make_long_context(
    r: random.Random,
    fact_paragraphs: list[str],
    total_paragraphs: int,
    filler_sentences: int,
) -> str:
    """
    Build a long context with fact paragraphs inserted among filler paragraphs.
    """
    total_paragraphs = max(total_paragraphs, len(fact_paragraphs) + 2)

    paragraphs = [
        filler_paragraph(r, i + 1, sentences=filler_sentences)
        for i in range(total_paragraphs)
    ]

    insert_positions = r.sample(range(total_paragraphs), k=len(fact_paragraphs))

    for pos, fact in zip(insert_positions, fact_paragraphs):
        paragraphs[pos] = fact

    return "\n\n".join(paragraphs)


def make_prompt(context: str, question: str, answer_style: str = "short") -> str:
    if answer_style == "answer_only":
        return (
            "Use the long context below to answer the question.\n"
            "Output only the answer. Do not include explanation.\n\n"
            "Context:\n"
            f"{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    return (
        "Use the long context below to answer the question.\n"
        "Give a concise answer and end with 'Final answer: <answer>'.\n"
        "Only use information found in the context.\n\n"
        "Context:\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def make_completion(answer: str, evidence: str | None = None, answer_style: str = "short") -> str:
    if answer_style == "answer_only":
        return f" {answer}\n"

    if evidence:
        return f" The relevant context says: {evidence}\nFinal answer: {answer}\n"

    return f" {answer}\nFinal answer: {answer}\n"


def make_record(
    kind: str,
    index: int,
    context: str,
    question: str,
    answer: str,
    prompt: str,
    completion: str,
    seed: int,
    metadata: dict | None = None,
) -> dict:
    return {
        "database_version": LONG_CONTEXT_DATABASE_VERSION,
        "task_id": f"long_context/{kind}/{index:05d}",
        "src": f"synthetic_long_context/{kind}",
        "kind": kind,

        "context": context,
        "question": question,
        "prompt": prompt,
        "completion": completion,

        "answer": answer,
        "gold": answer,
        "answer_type": "short_text",

        "status": "gold_long_context",
        "seed": seed,
        "metadata": metadata or {},
    }


def generate_needle_in_haystack(
    r: random.Random,
    total_paragraphs: int,
    filler_sentences: int,
):
    dossier_id = f"DOS-{r.randint(1000, 9999)}"
    access_code = f"orchid-{r.randint(10000, 99999)}"

    fact = (
        f"Paragraph FACT: Confidential routing memo. "
        f"The access code for dossier {dossier_id} is {access_code}. "
        f"This code is the only valid retrieval key for the dossier."
    )

    context = make_long_context(
        r=r,
        fact_paragraphs=[fact],
        total_paragraphs=total_paragraphs,
        filler_sentences=filler_sentences,
    )

    question = f"What is the access code for dossier {dossier_id}?"
    answer = access_code
    evidence = f"The access code for dossier {dossier_id} is {access_code}."

    metadata = {
        "dossier_id": dossier_id,
        "access_code": access_code,
        "fact_count": 1,
    }

    return context, question, answer, evidence, metadata


def generate_attribute_lookup(
    r: random.Random,
    total_paragraphs: int,
    filler_sentences: int,
):
    selected_names = r.sample(NAMES, k=12)

    rows = []
    people = []

    for i, name in enumerate(selected_names):
        department = r.choice(DEPARTMENTS)
        tool = r.choice(TOOLS)
        badge = f"B-{r.randint(1000, 9999)}"

        people.append(
            {
                "name": name,
                "department": department,
                "tool": tool,
                "badge": badge,
            }
        )

        rows.append(
            f"Employee record {i + 1}: {name} works in {department}, "
            f"uses the {tool}, and has badge {badge}."
        )

    target = r.choice(people)
    ask_field = r.choice(["department", "tool", "badge"])

    if ask_field == "department":
        question = f"Which department does {target['name']} work in?"
        answer = target["department"]
        evidence = f"{target['name']} works in {target['department']}."

    elif ask_field == "tool":
        question = f"What tool does {target['name']} use?"
        answer = target["tool"]
        evidence = f"{target['name']} uses the {target['tool']}."

    else:
        question = f"What is the badge number for {target['name']}?"
        answer = target["badge"]
        evidence = f"{target['name']} has badge {target['badge']}."

    fact = "Paragraph FACT: Staff directory. " + " ".join(rows)

    context = make_long_context(
        r=r,
        fact_paragraphs=[fact],
        total_paragraphs=total_paragraphs,
        filler_sentences=filler_sentences,
    )

    metadata = {
        "target_name": target["name"],
        "ask_field": ask_field,
        "target_record": target,
        "num_employee_records": len(people),
    }

    return context, question, answer, evidence, metadata


def generate_multi_hop_lookup(
    r: random.Random,
    total_paragraphs: int,
    filler_sentences: int,
):
    managers = r.sample(NAMES, k=8)
    projects = r.sample(PROJECTS, k=8)

    project_rows = []
    manager_rows = []
    project_map = {}
    office_map = {}

    for project, manager in zip(projects, managers):
        project_map[project] = manager
        project_rows.append(f"{project} is managed by {manager}.")

    for manager in managers:
        office = r.choice(OFFICES)
        city = r.choice(CITIES)
        office_map[manager] = {
            "office": office,
            "city": city,
        }
        manager_rows.append(f"{manager} is assigned to the {office} in {city}.")

    target_project = r.choice(projects)
    target_manager = project_map[target_project]
    target_office = office_map[target_manager]["office"]

    fact_1 = "Paragraph FACT A: Project ownership list. " + " ".join(project_rows)
    fact_2 = "Paragraph FACT B: Manager office list. " + " ".join(manager_rows)

    context = make_long_context(
        r=r,
        fact_paragraphs=[fact_1, fact_2],
        total_paragraphs=total_paragraphs,
        filler_sentences=filler_sentences,
    )

    question = f"Which office is assigned to the manager of {target_project}?"
    answer = target_office
    evidence = (
        f"{target_project} is managed by {target_manager}; "
        f"{target_manager} is assigned to the {target_office}."
    )

    metadata = {
        "target_project": target_project,
        "target_manager": target_manager,
        "target_office": target_office,
        "num_projects": len(projects),
        "requires_hops": 2,
    }

    return context, question, answer, evidence, metadata


def generate_timeline_lookup(
    r: random.Random,
    total_paragraphs: int,
    filler_sentences: int,
):
    artifacts = r.sample(ARTIFACTS, k=10)

    rows = []
    records = []

    for i, artifact in enumerate(artifacts):
        year = r.randint(1820, 2020)
        material = MATERIALS[i % len(MATERIALS)]
        city = r.choice(CITIES)

        records.append(
            {
                "artifact": artifact,
                "year": year,
                "material": material,
                "city": city,
            }
        )

        rows.append(
            f"The {artifact} was cataloged in {year}, "
            f"is made of {material}, and is stored in {city}."
        )

    target = r.choice(records)
    ask_field = r.choice(["year", "material", "city"])

    if ask_field == "year":
        question = f"In what year was the {target['artifact']} cataloged?"
        answer = str(target["year"])
        evidence = f"The {target['artifact']} was cataloged in {target['year']}."

    elif ask_field == "material":
        question = f"What material is the {target['artifact']} made of?"
        answer = target["material"]
        evidence = f"The {target['artifact']} is made of {target['material']}."

    else:
        question = f"Where is the {target['artifact']} stored?"
        answer = target["city"]
        evidence = f"The {target['artifact']} is stored in {target['city']}."

    fact = "Paragraph FACT: Artifact catalog timeline. " + " ".join(rows)

    context = make_long_context(
        r=r,
        fact_paragraphs=[fact],
        total_paragraphs=total_paragraphs,
        filler_sentences=filler_sentences,
    )

    metadata = {
        "target_artifact": target["artifact"],
        "ask_field": ask_field,
        "target_record": target,
        "num_artifact_records": len(records),
    }

    return context, question, answer, evidence, metadata


def generate_constraint_count(
    r: random.Random,
    total_paragraphs: int,
    filler_sentences: int,
):
    shipments = []
    rows = []

    target_region = r.choice(REGIONS)
    target_status = r.choice(STATUSES)

    for i in range(30):
        region = r.choice(REGIONS)
        status = r.choice(STATUSES)
        item = r.choice(ITEMS)
        quantity = r.randint(5, 90)
        shipment_id = f"S-{r.randint(10000, 99999)}"

        shipment = {
            "shipment_id": shipment_id,
            "region": region,
            "status": status,
            "item": item,
            "quantity": quantity,
        }

        shipments.append(shipment)

        rows.append(
            f"Shipment {shipment_id}: region={region}; status={status}; "
            f"item={item}; quantity={quantity}."
        )

    answer_count = sum(
        1
        for shipment in shipments
        if shipment["region"] == target_region and shipment["status"] == target_status
    )

    fact = "Paragraph FACT: Shipment ledger. " + " ".join(rows)

    context = make_long_context(
        r=r,
        fact_paragraphs=[fact],
        total_paragraphs=total_paragraphs,
        filler_sentences=filler_sentences,
    )

    question = (
        f"How many shipments are in the {target_region} region "
        f"with status {target_status}?"
    )

    answer = str(answer_count)
    evidence = (
        f"Counting shipment rows where region={target_region} "
        f"and status={target_status} gives {answer_count}."
    )

    metadata = {
        "target_region": target_region,
        "target_status": target_status,
        "answer_count": answer_count,
        "num_shipments": len(shipments),
    }

    return context, question, answer, evidence, metadata


def generate_item(
    kind: str,
    index: int,
    seed: int,
    total_paragraphs: int,
    filler_sentences: int,
    answer_style: str,
) -> dict:
    r = random.Random(seed)

    if kind == "needle_in_haystack":
        context, question, answer, evidence, metadata = generate_needle_in_haystack(
            r=r,
            total_paragraphs=total_paragraphs,
            filler_sentences=filler_sentences,
        )

    elif kind == "attribute_lookup":
        context, question, answer, evidence, metadata = generate_attribute_lookup(
            r=r,
            total_paragraphs=total_paragraphs,
            filler_sentences=filler_sentences,
        )

    elif kind == "multi_hop_lookup":
        context, question, answer, evidence, metadata = generate_multi_hop_lookup(
            r=r,
            total_paragraphs=total_paragraphs,
            filler_sentences=filler_sentences,
        )

    elif kind == "timeline_lookup":
        context, question, answer, evidence, metadata = generate_timeline_lookup(
            r=r,
            total_paragraphs=total_paragraphs,
            filler_sentences=filler_sentences,
        )

    elif kind == "constraint_count":
        context, question, answer, evidence, metadata = generate_constraint_count(
            r=r,
            total_paragraphs=total_paragraphs,
            filler_sentences=filler_sentences,
        )

    else:
        raise ValueError(f"Unknown long-context kind: {kind}")

    prompt = make_prompt(
        context=context,
        question=question,
        answer_style=answer_style,
    )

    completion = make_completion(
        answer=answer,
        evidence=evidence,
        answer_style=answer_style,
    )

    context_chars = len(context)
    context_words = len(context.split())

    metadata.update(
        {
            "answer_style": answer_style,
            "context_chars": context_chars,
            "context_words": context_words,
            "total_paragraphs": total_paragraphs,
            "filler_sentences": filler_sentences,
            "normalized_gold": normalize_answer(answer),
        }
    )

    return make_record(
        kind=kind,
        index=index,
        context=context,
        question=question,
        answer=answer,
        prompt=prompt,
        completion=completion,
        seed=seed,
        metadata=metadata,
    )


def build_records(
    seed: int,
    n_per_kind: int,
    total_paragraphs: int,
    filler_sentences: int,
    shuffle: bool = True,
    answer_style: str = "short",
) -> list[dict]:
    main_rng = random.Random((int(seed) ^ LONG_CONTEXT_STREAM_SEED) & 0xFFFFFFFF)
    records = []

    index = 0

    for kind in LONG_CONTEXT_KINDS:
        for _ in range(n_per_kind):
            item_seed = main_rng.randint(0, 2**31 - 1)

            record = generate_item(
                kind=kind,
                index=index,
                seed=item_seed,
                total_paragraphs=total_paragraphs,
                filler_sentences=filler_sentences,
                answer_style=answer_style,
            )

            records.append(record)
            index += 1

    if shuffle:
        main_rng.shuffle(records)

    return records


def verify_record(record: dict, answer_style: str = "short") -> tuple[bool, str]:
    try:
        completion = record.get("completion", "")
        gold = record.get("gold", "")

        if answer_style == "answer_only":
            predicted = completion.strip()
        else:
            predicted = extract_final_answer(completion)

            if predicted is None:
                return False, "No 'Final answer:' line found."

        if normalize_answer(predicted) != normalize_answer(gold):
            return (
                False,
                f"Answer mismatch: predicted={predicted!r}, gold={gold!r}",
            )

        return True, ""

    except Exception as e:
        return False, repr(e)


def verify_records(
    records: list[dict],
    answer_style: str = "short",
    max_failures: int = 10,
) -> bool:
    failures = []

    for record in records:
        ok, err = verify_record(record, answer_style=answer_style)
        record["verified"] = bool(ok)

        if not ok:
            record["verify_error"] = err
            failures.append((record.get("task_id"), record.get("kind"), err))

            if len(failures) >= max_failures:
                break

    if not failures:
        print("Local verification: all generated long-context records passed.")
        return True

    print("Local verification failed.")
    print(f"Failures shown: {len(failures)}")

    for task_id, kind, err in failures:
        print("=" * 80)
        print(f"Task: {task_id}")
        print(f"Kind: {kind}")
        print(err)

    return False


def assert_required_fields(records: list[dict]) -> None:
    required = [
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

    bad = []

    for record in records:
        for field in required:
            if field not in record:
                bad.append((record.get("task_id"), field))

    if bad:
        print("Found records with missing required fields:", file=sys.stderr)

        for task_id, field in bad[:20]:
            print(f"Task: {task_id}, missing: {field}", file=sys.stderr)

        raise SystemExit(1)


def assert_completion_format(records: list[dict], answer_style: str = "short") -> None:
    bad = []

    for record in records:
        completion = record.get("completion", "")

        if not isinstance(completion, str) or not completion.strip():
            bad.append((record.get("task_id"), record.get("kind"), "empty completion"))
            continue

        if not completion.endswith("\n"):
            bad.append(
                (
                    record.get("task_id"),
                    record.get("kind"),
                    "completion does not end with newline",
                )
            )

        if "```" in completion:
            bad.append(
                (
                    record.get("task_id"),
                    record.get("kind"),
                    "completion contains markdown fence",
                )
            )

        if answer_style != "answer_only" and "Final answer:" not in completion:
            bad.append(
                (
                    record.get("task_id"),
                    record.get("kind"),
                    "missing Final answer line",
                )
            )

    if bad:
        print("Found bad completion formatting:", file=sys.stderr)

        for task_id, kind, reason in bad[:20]:
            print(f"Task: {task_id}, kind: {kind}, reason: {reason}", file=sys.stderr)

        raise SystemExit(1)


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


def print_context_stats(records: list[dict]) -> None:
    if not records:
        return

    word_counts = [
        int(record.get("metadata", {}).get("context_words", 0))
        for record in records
    ]

    char_counts = [
        int(record.get("metadata", {}).get("context_chars", 0))
        for record in records
    ]

    print("\nContext stats:")
    print(f"  Min words: {min(word_counts)}")
    print(f"  Max words: {max(word_counts)}")
    print(f"  Avg words: {sum(word_counts) // len(word_counts)}")
    print(f"  Min chars: {min(char_counts)}")
    print(f"  Max chars: {max(char_counts)}")
    print(f"  Avg chars: {sum(char_counts) // len(char_counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic synthetic long-context JSONL database "
            "for long-context QA SFT/evaluation."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset/long_context_database_all_cases.jsonl",
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
        default=10,
        help="Number of records to generate per long-context kind.",
    )

    parser.add_argument(
        "--paragraphs",
        type=int,
        default=80,
        help="Number of paragraphs per context.",
    )

    parser.add_argument(
        "--filler-sentences",
        type=int,
        default=5,
        help="Number of synthetic filler sentences per filler paragraph.",
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
        help="Run local verification that completion answer matches gold.",
    )

    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Generate completions containing only the answer.",
    )

    args = parser.parse_args()

    answer_style = "answer_only" if args.answer_only else "short"

    records = build_records(
        seed=args.seed,
        n_per_kind=args.n_per_kind,
        total_paragraphs=args.paragraphs,
        filler_sentences=args.filler_sentences,
        shuffle=not args.no_shuffle,
        answer_style=answer_style,
    )

    assert_required_fields(records)
    assert_completion_format(records, answer_style=answer_style)

    if args.verify:
        ok = verify_records(records, answer_style=answer_style)

        if not ok:
            print("Not writing JSONL because local verification failed.", file=sys.stderr)
            raise SystemExit(1)

    output_path = Path(args.output)
    write_jsonl(records, output_path, append=args.append)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Database version: {LONG_CONTEXT_DATABASE_VERSION}")
    print(f"Total records: {len(records)}")
    print(f"Long-context kinds: {len(LONG_CONTEXT_KINDS)}")
    print(f"Records per kind: {args.n_per_kind}")
    print(f"Paragraphs per context: {args.paragraphs}")
    print(f"Filler sentences per paragraph: {args.filler_sentences}")
    print(f"Append mode: {args.append}")
    print(f"Answer style: {answer_style}")
    print(f"Local verified: {args.verify}")
    print("Completion format: concise evidence plus Final answer line")
    print("Core fields: context, question, prompt, completion, answer, gold, kind, metadata")

    print_counts(records)
    print_context_stats(records)


if __name__ == "__main__":
    main()
