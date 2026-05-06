#!/usr/bin/env python3

import argparse
import json
import random
import re
import sys
from pathlib import Path


MULTI_DOC_DATABASE_VERSION = "synthetic_multi_doc_database_v1"
MULTI_DOC_STREAM_SEED = 0x4D0C2026


MULTI_DOC_KINDS = [
    "single_doc_lookup",
    "cross_doc_join",
    "latest_doc_resolution",
    "count_across_docs",
    "compare_across_docs",
]


NAMES = [
    "Avery Stone", "Blair Chen", "Casey Morgan", "Drew Patel", "Emery Brooks",
    "Finley Ross", "Gray Rivera", "Harper Quinn", "Indigo Lane", "Jordan Vale",
    "Kai Mercer", "Logan Reed", "Morgan Ellis", "Noel Carter", "Oakley Price",
    "Parker Sloan", "Quinn Avery", "Reese Novak", "Sawyer Kim", "Taylor Brooks",
    "Mira Wells", "Nico Fox", "Lena Cross", "Owen Hale", "Iris Blake", "Theo Grant",
]

DEPARTMENTS = [
    "Archive", "Security", "Research", "Operations", "Logistics",
    "Planning", "Engineering", "Compliance", "Field Support", "Records",
]

TOOLS = [
    "scanner", "tablet", "ledger", "camera", "compass",
    "labeler", "caliper", "notebook", "spectrometer", "microscope",
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

CITIES = [
    "Oslo", "Nairobi", "Lisbon", "Seoul", "Toronto", "Helsinki",
    "Cairo", "Madrid", "Tokyo", "Sydney", "Berlin", "Athens",
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

TICKET_STATUSES = [
    "open", "in review", "approved", "blocked", "closed",
]

DOC_TOPICS = [
    "operations memo",
    "field report",
    "archive note",
    "planning summary",
    "inventory bulletin",
    "compliance update",
]


FINAL_RE = re.compile(r"Final answer:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)


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


def extract_final_answer(completion: str) -> str | None:
    match = FINAL_RE.search(str(completion).strip())

    if not match:
        return None

    return match.group(1).strip()


def synthetic_word(r: random.Random) -> str:
    starts = [
        "br", "cl", "dr", "fl", "gr", "kr", "pl", "pr",
        "sl", "tr", "v", "m", "n", "s", "t", "z",
    ]
    mids = [
        "a", "e", "i", "o", "u", "ae", "ai", "oa", "ou", "ia",
    ]
    ends = [
        "n", "r", "s", "t", "l", "m", "ck", "nd", "st", "th",
    ]

    return r.choice(starts) + r.choice(mids) + r.choice(ends)


def synthetic_sentence(r: random.Random, min_words: int = 8, max_words: int = 16) -> str:
    words = [synthetic_word(r) for _ in range(r.randint(min_words, max_words))]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def filler_text(r: random.Random, sentences: int = 3) -> str:
    topic = r.choice(DOC_TOPICS)
    parts = [
        f"This document also includes background material about {topic}."
    ]

    for _ in range(sentences):
        parts.append(synthetic_sentence(r))

    return " ".join(parts)


def make_doc(doc_id: str, title: str, text: str) -> dict:
    return {
        "doc_id": doc_id,
        "title": title,
        "text": text,
    }


def format_docs_for_prompt(docs: list[dict]) -> str:
    chunks = []

    for doc in docs:
        chunks.append(
            f"[Document ID: {doc['doc_id']}]\n"
            f"Title: {doc['title']}\n"
            f"{doc['text']}"
        )

    return "\n\n---\n\n".join(chunks)


def make_prompt(
    docs: list[dict],
    question: str,
    answer_style: str = "short",
) -> str:
    docs_text = format_docs_for_prompt(docs)

    if answer_style == "answer_only":
        return (
            "Use the document set below to answer the question.\n"
            "Output only the answer. Do not include explanation.\n\n"
            "Documents:\n"
            f"{docs_text}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    return (
        "Use the document set below to answer the question.\n"
        "Give a concise answer and end with 'Final answer: <answer>'.\n"
        "Only use information found in the documents.\n\n"
        "Documents:\n"
        f"{docs_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def make_completion(
    answer: str,
    evidence: str | None = None,
    answer_style: str = "short",
) -> str:
    if answer_style == "answer_only":
        return f" {answer}\n"

    if evidence:
        return f" The relevant document evidence is: {evidence}\nFinal answer: {answer}\n"

    return f" {answer}\nFinal answer: {answer}\n"


def make_record(
    kind: str,
    index: int,
    docs: list[dict],
    question: str,
    answer: str,
    prompt: str,
    completion: str,
    seed: int,
    metadata: dict | None = None,
) -> dict:
    docs_text = format_docs_for_prompt(docs)

    return {
        "database_version": MULTI_DOC_DATABASE_VERSION,
        "task_id": f"multi_doc/{kind}/{index:05d}",
        "src": f"synthetic_multi_doc/{kind}",
        "kind": kind,

        "docs": docs,
        "documents": docs,
        "doc_count": len(docs),
        "docs_text": docs_text,

        "question": question,
        "prompt": prompt,
        "completion": completion,

        "answer": answer,
        "gold": answer,
        "answer_type": "short_text",

        "status": "gold_multi_doc",
        "seed": seed,
        "metadata": metadata or {},
    }


def generate_single_doc_lookup(
    r: random.Random,
    num_docs: int,
    filler_sentences: int,
):
    docs = []
    selected_names = r.sample(NAMES, k=max(8, num_docs))

    target_record = None

    for i in range(num_docs):
        doc_id = f"DOC-{i + 1:03d}"
        name = selected_names[i % len(selected_names)]
        department = r.choice(DEPARTMENTS)
        tool = r.choice(TOOLS)
        badge = f"B-{r.randint(1000, 9999)}"

        record = {
            "name": name,
            "department": department,
            "tool": tool,
            "badge": badge,
            "doc_id": doc_id,
        }

        if target_record is None or r.random() < 0.25:
            target_record = record

        text = (
            f"Staff profile: {name} works in {department}. "
            f"{name} uses the {tool}. "
            f"The badge number for {name} is {badge}. "
            f"{filler_text(r, sentences=filler_sentences)}"
        )

        docs.append(
            make_doc(
                doc_id=doc_id,
                title=f"Staff Profile {i + 1}",
                text=text,
            )
        )

    ask_field = r.choice(["department", "tool", "badge"])

    if ask_field == "department":
        question = f"Which department does {target_record['name']} work in?"
        answer = target_record["department"]
        evidence = (
            f"In {target_record['doc_id']}, {target_record['name']} works in "
            f"{target_record['department']}."
        )

    elif ask_field == "tool":
        question = f"What tool does {target_record['name']} use?"
        answer = target_record["tool"]
        evidence = (
            f"In {target_record['doc_id']}, {target_record['name']} uses the "
            f"{target_record['tool']}."
        )

    else:
        question = f"What is the badge number for {target_record['name']}?"
        answer = target_record["badge"]
        evidence = (
            f"In {target_record['doc_id']}, the badge number for "
            f"{target_record['name']} is {target_record['badge']}."
        )

    metadata = {
        "target_doc_id": target_record["doc_id"],
        "target_name": target_record["name"],
        "ask_field": ask_field,
        "target_record": target_record,
        "requires_docs": 1,
    }

    return docs, question, answer, evidence, metadata


def generate_cross_doc_join(
    r: random.Random,
    num_docs: int,
    filler_sentences: int,
):
    managers = r.sample(NAMES, k=8)
    projects = r.sample(PROJECTS, k=8)

    project_map = {}
    office_map = {}

    project_rows = []
    office_rows = []

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
        office_rows.append(f"{manager} is assigned to the {office} in {city}.")

    docs = [
        make_doc(
            doc_id="DOC-001",
            title="Project Manager Directory",
            text=" ".join(project_rows) + " " + filler_text(r, sentences=filler_sentences),
        ),
        make_doc(
            doc_id="DOC-002",
            title="Manager Office Directory",
            text=" ".join(office_rows) + " " + filler_text(r, sentences=filler_sentences),
        ),
    ]

    for i in range(2, num_docs):
        docs.append(
            make_doc(
                doc_id=f"DOC-{i + 1:03d}",
                title=f"Supplementary Memo {i + 1}",
                text=filler_text(r, sentences=filler_sentences + 2),
            )
        )

    target_project = r.choice(projects)
    target_manager = project_map[target_project]
    ask_field = r.choice(["office", "city"])

    if ask_field == "office":
        answer = office_map[target_manager]["office"]
        question = f"Which office is assigned to the manager of {target_project}?"
        evidence = (
            f"DOC-001 says {target_project} is managed by {target_manager}; "
            f"DOC-002 says {target_manager} is assigned to the {answer}."
        )
    else:
        answer = office_map[target_manager]["city"]
        question = f"Which city is assigned to the manager of {target_project}?"
        evidence = (
            f"DOC-001 says {target_project} is managed by {target_manager}; "
            f"DOC-002 says {target_manager} is assigned in {answer}."
        )

    metadata = {
        "target_project": target_project,
        "target_manager": target_manager,
        "ask_field": ask_field,
        "requires_docs": 2,
        "requires_hops": 2,
    }

    return docs, question, answer, evidence, metadata


def generate_latest_doc_resolution(
    r: random.Random,
    num_docs: int,
    filler_sentences: int,
):
    ticket_id = f"TCK-{r.randint(1000, 9999)}"

    base_year = r.randint(2021, 2025)
    updates = []

    for i in range(num_docs):
        month = i + 1
        day = r.randint(1, 28)
        date = f"{base_year}-{month:02d}-{day:02d}"
        status = r.choice(TICKET_STATUSES)
        owner = r.choice(NAMES)

        updates.append(
            {
                "doc_id": f"DOC-{i + 1:03d}",
                "date": date,
                "status": status,
                "owner": owner,
            }
        )

    latest = max(updates, key=lambda x: x["date"])

    docs = []

    shuffled_updates = list(updates)
    r.shuffle(shuffled_updates)

    for update in shuffled_updates:
        text = (
            f"Update date: {update['date']}. "
            f"Ticket {ticket_id} has status {update['status']}. "
            f"The owner listed for ticket {ticket_id} is {update['owner']}. "
            f"{filler_text(r, sentences=filler_sentences)}"
        )

        docs.append(
            make_doc(
                doc_id=update["doc_id"],
                title=f"Ticket Update {update['date']}",
                text=text,
            )
        )

    ask_field = r.choice(["status", "owner"])

    if ask_field == "status":
        question = f"What is the latest status of ticket {ticket_id}?"
        answer = latest["status"]
        evidence = (
            f"The latest update is dated {latest['date']} in {latest['doc_id']}, "
            f"where ticket {ticket_id} has status {latest['status']}."
        )
    else:
        question = f"Who is the latest listed owner of ticket {ticket_id}?"
        answer = latest["owner"]
        evidence = (
            f"The latest update is dated {latest['date']} in {latest['doc_id']}, "
            f"where ticket {ticket_id} is owned by {latest['owner']}."
        )

    metadata = {
        "ticket_id": ticket_id,
        "latest_doc_id": latest["doc_id"],
        "latest_date": latest["date"],
        "ask_field": ask_field,
        "updates": updates,
        "requires_latest_resolution": True,
    }

    return docs, question, answer, evidence, metadata


def generate_count_across_docs(
    r: random.Random,
    num_docs: int,
    filler_sentences: int,
):
    target_region = r.choice(REGIONS)
    target_status = r.choice(STATUSES)

    docs = []
    shipments = []

    for i in range(num_docs):
        rows = []

        for _ in range(r.randint(5, 9)):
            shipment_id = f"S-{r.randint(10000, 99999)}"
            region = r.choice(REGIONS)
            status = r.choice(STATUSES)
            item = r.choice(ITEMS)
            quantity = r.randint(5, 100)

            shipment = {
                "shipment_id": shipment_id,
                "region": region,
                "status": status,
                "item": item,
                "quantity": quantity,
                "doc_id": f"DOC-{i + 1:03d}",
            }

            shipments.append(shipment)

            rows.append(
                f"Shipment {shipment_id}: region={region}; status={status}; "
                f"item={item}; quantity={quantity}."
            )

        text = (
            "Shipment ledger section. "
            + " ".join(rows)
            + " "
            + filler_text(r, sentences=filler_sentences)
        )

        docs.append(
            make_doc(
                doc_id=f"DOC-{i + 1:03d}",
                title=f"Shipment Ledger Part {i + 1}",
                text=text,
            )
        )

    answer_count = sum(
        1
        for shipment in shipments
        if shipment["region"] == target_region and shipment["status"] == target_status
    )

    question = (
        f"How many shipments across all documents are in the {target_region} "
        f"region with status {target_status}?"
    )

    answer = str(answer_count)
    evidence = (
        f"Counting all shipment rows with region={target_region} and "
        f"status={target_status} gives {answer_count}."
    )

    metadata = {
        "target_region": target_region,
        "target_status": target_status,
        "answer_count": answer_count,
        "num_shipments": len(shipments),
        "requires_docs": num_docs,
        "operation": "count",
    }

    return docs, question, answer, evidence, metadata


def generate_compare_across_docs(
    r: random.Random,
    num_docs: int,
    filler_sentences: int,
):
    selected_projects = r.sample(PROJECTS, k=max(5, min(len(PROJECTS), num_docs + 3)))
    project_records = []

    docs = []

    for i, project in enumerate(selected_projects):
        budget = r.randint(50, 900) * 1000
        owner = r.choice(NAMES)
        city = r.choice(CITIES)

        record = {
            "project": project,
            "budget": budget,
            "owner": owner,
            "city": city,
            "doc_id": f"DOC-{i + 1:03d}",
        }

        project_records.append(record)

        text = (
            f"Budget summary: {project} has allocated budget {budget}. "
            f"The project owner is {owner}. "
            f"The project city is {city}. "
            f"{filler_text(r, sentences=filler_sentences)}"
        )

        docs.append(
            make_doc(
                doc_id=f"DOC-{i + 1:03d}",
                title=f"Budget Note for {project}",
                text=text,
            )
        )

    while len(docs) < num_docs:
        i = len(docs)
        docs.append(
            make_doc(
                doc_id=f"DOC-{i + 1:03d}",
                title=f"General Finance Memo {i + 1}",
                text=filler_text(r, sentences=filler_sentences + 2),
            )
        )

    highest = max(project_records, key=lambda x: x["budget"])
    lowest = min(project_records, key=lambda x: x["budget"])

    ask_mode = r.choice(["highest_project", "lowest_project", "highest_budget"])

    if ask_mode == "highest_project":
        question = "Which project has the highest allocated budget across the documents?"
        answer = highest["project"]
        evidence = (
            f"{highest['doc_id']} lists {highest['project']} with budget "
            f"{highest['budget']}, which is the highest."
        )

    elif ask_mode == "lowest_project":
        question = "Which project has the lowest allocated budget across the documents?"
        answer = lowest["project"]
        evidence = (
            f"{lowest['doc_id']} lists {lowest['project']} with budget "
            f"{lowest['budget']}, which is the lowest."
        )

    else:
        question = "What is the highest allocated budget across the documents?"
        answer = str(highest["budget"])
        evidence = (
            f"{highest['doc_id']} lists {highest['project']} with budget "
            f"{highest['budget']}, which is the highest."
        )

    metadata = {
        "ask_mode": ask_mode,
        "highest_project": highest["project"],
        "highest_budget": highest["budget"],
        "lowest_project": lowest["project"],
        "lowest_budget": lowest["budget"],
        "num_project_records": len(project_records),
        "operation": "compare",
    }

    return docs, question, answer, evidence, metadata


def generate_item(
    kind: str,
    index: int,
    seed: int,
    num_docs: int,
    filler_sentences: int,
    answer_style: str,
) -> dict:
    r = random.Random(seed)

    if kind == "single_doc_lookup":
        docs, question, answer, evidence, metadata = generate_single_doc_lookup(
            r=r,
            num_docs=num_docs,
            filler_sentences=filler_sentences,
        )

    elif kind == "cross_doc_join":
        docs, question, answer, evidence, metadata = generate_cross_doc_join(
            r=r,
            num_docs=num_docs,
            filler_sentences=filler_sentences,
        )

    elif kind == "latest_doc_resolution":
        docs, question, answer, evidence, metadata = generate_latest_doc_resolution(
            r=r,
            num_docs=num_docs,
            filler_sentences=filler_sentences,
        )

    elif kind == "count_across_docs":
        docs, question, answer, evidence, metadata = generate_count_across_docs(
            r=r,
            num_docs=num_docs,
            filler_sentences=filler_sentences,
        )

    elif kind == "compare_across_docs":
        docs, question, answer, evidence, metadata = generate_compare_across_docs(
            r=r,
            num_docs=num_docs,
            filler_sentences=filler_sentences,
        )

    else:
        raise ValueError(f"Unknown multi-doc kind: {kind}")

    prompt = make_prompt(
        docs=docs,
        question=question,
        answer_style=answer_style,
    )

    completion = make_completion(
        answer=answer,
        evidence=evidence,
        answer_style=answer_style,
    )

    docs_text = format_docs_for_prompt(docs)
    total_doc_chars = sum(len(doc["text"]) for doc in docs)
    total_doc_words = sum(len(doc["text"].split()) for doc in docs)

    metadata.update(
        {
            "answer_style": answer_style,
            "num_docs": len(docs),
            "doc_ids": [doc["doc_id"] for doc in docs],
            "total_doc_chars": total_doc_chars,
            "total_doc_words": total_doc_words,
            "docs_text_chars": len(docs_text),
            "docs_text_words": len(docs_text.split()),
            "filler_sentences": filler_sentences,
            "normalized_gold": normalize_answer(answer),
        }
    )

    return make_record(
        kind=kind,
        index=index,
        docs=docs,
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
    num_docs: int,
    filler_sentences: int,
    shuffle: bool = True,
    answer_style: str = "short",
) -> list[dict]:
    main_rng = random.Random((int(seed) ^ MULTI_DOC_STREAM_SEED) & 0xFFFFFFFF)
    records = []

    index = 0

    for kind in MULTI_DOC_KINDS:
        for _ in range(n_per_kind):
            item_seed = main_rng.randint(0, 2**31 - 1)

            record = generate_item(
                kind=kind,
                index=index,
                seed=item_seed,
                num_docs=num_docs,
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
        print("Local verification: all generated multi-doc records passed.")
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


def assert_record_consistency(records: list[dict], answer_style: str = "short") -> None:
    bad = []

    for record in records:
        task_id = record.get("task_id")
        kind = record.get("kind")

        docs = record.get("docs")
        documents = record.get("documents")
        docs_text = record.get("docs_text")
        prompt = record.get("prompt")
        question = record.get("question")
        completion = record.get("completion")
        metadata = record.get("metadata", {})

        if not isinstance(docs, list) or not docs:
            bad.append((task_id, kind, "docs is empty or not a list"))
            continue

        if documents != docs:
            bad.append((task_id, kind, "documents does not match docs"))

        if record.get("doc_count") != len(docs):
            bad.append((task_id, kind, "doc_count does not match len(docs)"))

        for i, doc in enumerate(docs):
            if not isinstance(doc, dict):
                bad.append((task_id, kind, f"docs[{i}] is not a dict"))
                continue

            for field in ["doc_id", "title", "text"]:
                if field not in doc:
                    bad.append((task_id, kind, f"docs[{i}] missing {field}"))
                elif not isinstance(doc[field], str):
                    bad.append((task_id, kind, f"docs[{i}][{field}] is not a string"))

        if isinstance(docs_text, str):
            expected_docs_text = format_docs_for_prompt(docs)

            if docs_text != expected_docs_text:
                bad.append((task_id, kind, "docs_text does not match formatted docs"))

        else:
            bad.append((task_id, kind, "docs_text is not a string"))

        if isinstance(prompt, str):
            if "Documents:" not in prompt:
                bad.append((task_id, kind, "prompt missing Documents section"))

            if "Question:" not in prompt:
                bad.append((task_id, kind, "prompt missing Question section"))

            if "Answer:" not in prompt:
                bad.append((task_id, kind, "prompt missing Answer section"))

            if isinstance(docs_text, str) and docs_text not in prompt:
                bad.append((task_id, kind, "prompt does not contain docs_text"))

            if isinstance(question, str) and question not in prompt:
                bad.append((task_id, kind, "prompt does not contain question"))

            if answer_style != "answer_only" and "Final answer:" not in prompt:
                bad.append((task_id, kind, "prompt missing Final answer instruction"))

        else:
            bad.append((task_id, kind, "prompt is not a string"))

        if not isinstance(completion, str) or not completion.strip():
            bad.append((task_id, kind, "completion is empty or not a string"))
        else:
            if not completion.endswith("\n"):
                bad.append((task_id, kind, "completion does not end with newline"))

            if "```" in completion:
                bad.append((task_id, kind, "completion contains markdown fence"))

            if answer_style != "answer_only" and "Final answer:" not in completion:
                bad.append((task_id, kind, "completion missing Final answer line"))

        if record.get("answer") != record.get("gold"):
            if normalize_answer(record.get("answer", "")) != normalize_answer(record.get("gold", "")):
                bad.append((task_id, kind, "answer does not match gold"))

        if isinstance(metadata, dict):
            if metadata.get("normalized_gold") != normalize_answer(record.get("gold", "")):
                bad.append((task_id, kind, "metadata normalized_gold mismatch"))

            if metadata.get("num_docs") != len(docs):
                bad.append((task_id, kind, "metadata num_docs mismatch"))

    if bad:
        print("Found bad multi-doc record consistency:", file=sys.stderr)

        for task_id, kind, reason in bad[:30]:
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


def print_doc_stats(records: list[dict]) -> None:
    if not records:
        return

    doc_counts = [int(record.get("doc_count", 0)) for record in records]
    word_counts = [
        int(record.get("metadata", {}).get("docs_text_words", 0))
        for record in records
    ]
    char_counts = [
        int(record.get("metadata", {}).get("docs_text_chars", 0))
        for record in records
    ]

    print("\nDocument stats:")
    print(f"  Min docs: {min(doc_counts)}")
    print(f"  Max docs: {max(doc_counts)}")
    print(f"  Avg docs: {sum(doc_counts) // len(doc_counts)}")
    print(f"  Min docs_text words: {min(word_counts)}")
    print(f"  Max docs_text words: {max(word_counts)}")
    print(f"  Avg docs_text words: {sum(word_counts) // len(word_counts)}")
    print(f"  Min docs_text chars: {min(char_counts)}")
    print(f"  Max docs_text chars: {max(char_counts)}")
    print(f"  Avg docs_text chars: {sum(char_counts) // len(char_counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic synthetic multi-document JSONL database "
            "for multi-doc QA SFT/evaluation."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset/multi_doc_database_all_cases.jsonl",
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
        help="Number of records to generate per multi-doc kind.",
    )

    parser.add_argument(
        "--num-docs",
        type=int,
        default=6,
        help="Number of documents per record.",
    )

    parser.add_argument(
        "--filler-sentences",
        type=int,
        default=3,
        help="Synthetic filler sentences per document.",
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

    if args.num_docs < 2:
        print("--num-docs must be at least 2 for multi-doc tasks.", file=sys.stderr)
        raise SystemExit(1)

    answer_style = "answer_only" if args.answer_only else "short"

    records = build_records(
        seed=args.seed,
        n_per_kind=args.n_per_kind,
        num_docs=args.num_docs,
        filler_sentences=args.filler_sentences,
        shuffle=not args.no_shuffle,
        answer_style=answer_style,
    )

    assert_required_fields(records)
    assert_record_consistency(records, answer_style=answer_style)

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
    print(f"Database version: {MULTI_DOC_DATABASE_VERSION}")
    print(f"Total records: {len(records)}")
    print(f"Multi-doc kinds: {len(MULTI_DOC_KINDS)}")
    print(f"Records per kind: {args.n_per_kind}")
    print(f"Documents per record: {args.num_docs}")
    print(f"Filler sentences per document: {args.filler_sentences}")
    print(f"Append mode: {args.append}")
    print(f"Answer style: {answer_style}")
    print(f"Local verified: {args.verify}")
    print("Completion format: concise evidence plus Final answer line")
    print("Core fields: docs, docs_text, question, prompt, completion, answer, gold, kind, metadata")

    print_counts(records)
    print_doc_stats(records)


if __name__ == "__main__":
    main()
