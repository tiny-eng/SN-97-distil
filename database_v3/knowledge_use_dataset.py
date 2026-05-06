#!/usr/bin/env python3

import argparse
import json
import random
import re
import sys
from pathlib import Path


KNOWLEDGE_DATABASE_VERSION = "synthetic_knowledge_database_v1"
KNOWLEDGE_STREAM_SEED = 0xA17E2026


KNOWLEDGE_KINDS = [
    "capital_city",
    "country_currency",
    "planet_fact",
    "chemical_symbol",
    "animal_class",
    "historical_fact",
    "geography_fact",
    "science_definition",
    "technology_definition",
    "literary_author",
    "math_concept",
    "language_fact",
]


CAPITAL_CITY_FACTS = [
    ("France", "Paris"),
    ("Japan", "Tokyo"),
    ("Canada", "Ottawa"),
    ("Australia", "Canberra"),
    ("Brazil", "Brasília"),
    ("Egypt", "Cairo"),
    ("Germany", "Berlin"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("South Korea", "Seoul"),
    ("Thailand", "Bangkok"),
    ("Argentina", "Buenos Aires"),
    ("Kenya", "Nairobi"),
    ("Norway", "Oslo"),
    ("Sweden", "Stockholm"),
    ("Finland", "Helsinki"),
    ("Portugal", "Lisbon"),
    ("Greece", "Athens"),
    ("Mexico", "Mexico City"),
    ("India", "New Delhi"),
]


COUNTRY_CURRENCY_FACTS = [
    ("Japan", "yen"),
    ("United States", "dollar"),
    ("United Kingdom", "pound sterling"),
    ("India", "rupee"),
    ("China", "yuan"),
    ("South Korea", "won"),
    ("Mexico", "peso"),
    ("Brazil", "real"),
    ("Canada", "dollar"),
    ("Australia", "dollar"),
    ("Switzerland", "franc"),
    ("Thailand", "baht"),
    ("South Africa", "rand"),
    ("Norway", "krone"),
    ("Sweden", "krona"),
]


PLANET_FACTS = [
    ("Mercury", "the closest planet to the Sun"),
    ("Venus", "the hottest planet in the Solar System"),
    ("Earth", "the only known planet with abundant liquid surface water"),
    ("Mars", "often called the Red Planet"),
    ("Jupiter", "the largest planet in the Solar System"),
    ("Saturn", "known for its prominent ring system"),
    ("Uranus", "an ice giant that rotates on its side"),
    ("Neptune", "the farthest recognized planet from the Sun"),
]


CHEMICAL_SYMBOL_FACTS = [
    ("hydrogen", "H"),
    ("helium", "He"),
    ("carbon", "C"),
    ("nitrogen", "N"),
    ("oxygen", "O"),
    ("sodium", "Na"),
    ("magnesium", "Mg"),
    ("aluminum", "Al"),
    ("silicon", "Si"),
    ("phosphorus", "P"),
    ("sulfur", "S"),
    ("chlorine", "Cl"),
    ("potassium", "K"),
    ("calcium", "Ca"),
    ("iron", "Fe"),
    ("copper", "Cu"),
    ("zinc", "Zn"),
    ("silver", "Ag"),
    ("gold", "Au"),
    ("lead", "Pb"),
]


ANIMAL_CLASS_FACTS = [
    ("dog", "mammal"),
    ("cat", "mammal"),
    ("dolphin", "mammal"),
    ("eagle", "bird"),
    ("penguin", "bird"),
    ("salmon", "fish"),
    ("frog", "amphibian"),
    ("turtle", "reptile"),
    ("snake", "reptile"),
    ("butterfly", "insect"),
    ("bee", "insect"),
    ("octopus", "mollusk"),
]


HISTORICAL_FACTS = [
    ("the first President of the United States", "George Washington"),
    ("the document adopted in 1776 declaring American independence", "the Declaration of Independence"),
    ("the ancient civilization that built the pyramids at Giza", "ancient Egypt"),
    ("the wall built in northern China for defense", "the Great Wall of China"),
    ("the global conflict that lasted from 1939 to 1945", "World War II"),
    ("the ship that sank in 1912 after hitting an iceberg", "the Titanic"),
    ("the Renaissance artist who painted the Mona Lisa", "Leonardo da Vinci"),
    ("the empire ruled by Julius Caesar", "the Roman Empire"),
]


GEOGRAPHY_FACTS = [
    ("the largest ocean on Earth", "the Pacific Ocean"),
    ("the longest river in Africa", "the Nile River"),
    ("the tallest mountain above sea level", "Mount Everest"),
    ("the largest desert by area", "the Antarctic Desert"),
    ("the continent with the most countries", "Africa"),
    ("the imaginary line at zero degrees latitude", "the Equator"),
    ("the ocean between Africa and Australia", "the Indian Ocean"),
    ("the smallest continent by land area", "Australia"),
]


SCIENCE_DEFINITIONS = [
    ("photosynthesis", "the process by which plants use light energy to make food from carbon dioxide and water"),
    ("gravity", "the force that attracts objects with mass toward each other"),
    ("evaporation", "the process in which a liquid changes into a gas"),
    ("condensation", "the process in which a gas changes into a liquid"),
    ("atom", "the basic unit of ordinary matter"),
    ("ecosystem", "a community of organisms interacting with each other and their environment"),
    ("friction", "a force that resists motion between surfaces in contact"),
    ("density", "mass per unit volume"),
]


TECHNOLOGY_DEFINITIONS = [
    ("CPU", "the central processing unit that executes instructions in a computer"),
    ("RAM", "temporary memory used by a computer while programs are running"),
    ("database", "an organized collection of data"),
    ("algorithm", "a step-by-step procedure for solving a problem"),
    ("encryption", "the process of encoding information to protect it"),
    ("API", "an interface that allows software systems to communicate"),
    ("operating system", "software that manages computer hardware and provides services for programs"),
    ("URL", "an address used to locate a resource on the internet"),
]


LITERARY_AUTHOR_FACTS = [
    ("Romeo and Juliet", "William Shakespeare"),
    ("Pride and Prejudice", "Jane Austen"),
    ("Moby-Dick", "Herman Melville"),
    ("1984", "George Orwell"),
    ("Animal Farm", "George Orwell"),
    ("The Odyssey", "Homer"),
    ("The Great Gatsby", "F. Scott Fitzgerald"),
    ("Jane Eyre", "Charlotte Brontë"),
    ("Frankenstein", "Mary Shelley"),
    ("The Hobbit", "J. R. R. Tolkien"),
]


MATH_CONCEPTS = [
    ("prime number", "a whole number greater than 1 with exactly two positive divisors: 1 and itself"),
    ("even number", "an integer divisible by 2"),
    ("odd number", "an integer not divisible by 2"),
    ("triangle", "a polygon with three sides"),
    ("square", "a quadrilateral with four equal sides and four right angles"),
    ("radius", "the distance from the center of a circle to any point on the circle"),
    ("diameter", "a line segment across a circle through its center"),
    ("mean", "the sum of values divided by the number of values"),
]


LANGUAGE_FACTS = [
    ("a noun", "a word that names a person, place, thing, or idea"),
    ("a verb", "a word that expresses an action or state"),
    ("an adjective", "a word that describes a noun"),
    ("an adverb", "a word that modifies a verb, adjective, or another adverb"),
    ("a synonym", "a word with a similar meaning to another word"),
    ("an antonym", "a word with the opposite meaning of another word"),
    ("a prefix", "a word part added to the beginning of a word"),
    ("a suffix", "a word part added to the end of a word"),
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


def make_prompt(question: str, answer_style: str = "short") -> str:
    if answer_style == "answer_only":
        return (
            "Answer the following knowledge question.\n"
            "Output only the answer, with no explanation.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    return (
        "Answer the following knowledge question.\n"
        "Give a concise answer and end with 'Final answer: <answer>'.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def make_completion(answer: str, explanation: str | None = None, answer_style: str = "short") -> str:
    if answer_style == "answer_only":
        return f" {answer}\n"

    if explanation:
        return f" {explanation}\nFinal answer: {answer}\n"

    return f" {answer}\nFinal answer: {answer}\n"


def make_record(
    kind: str,
    index: int,
    question: str,
    answer: str,
    prompt: str,
    completion: str,
    seed: int,
    metadata: dict | None = None,
) -> dict:
    return {
        "database_version": KNOWLEDGE_DATABASE_VERSION,
        "task_id": f"knowledge/{kind}/{index:05d}",
        "src": f"synthetic_knowledge/{kind}",
        "kind": kind,
        "question": question,
        "prompt": prompt,
        "completion": completion,
        "answer": answer,
        "gold": answer,
        "answer_type": "short_text",
        "status": "gold_knowledge",
        "seed": seed,
        "metadata": metadata or {},
    }


def generate_capital_city(r: random.Random, answer_style: str):
    country, capital = r.choice(CAPITAL_CITY_FACTS)

    question_templates = [
        f"What is the capital city of {country}?",
        f"Which city is the capital of {country}?",
        f"Name the capital of {country}.",
    ]

    question = r.choice(question_templates)
    answer = capital
    explanation = f"The capital city of {country} is {capital}."

    metadata = {
        "country": country,
        "capital": capital,
    }

    return question, answer, explanation, metadata


def generate_country_currency(r: random.Random, answer_style: str):
    country, currency = r.choice(COUNTRY_CURRENCY_FACTS)

    question_templates = [
        f"What currency is used in {country}?",
        f"What is the currency of {country}?",
        f"Name the currency used in {country}.",
    ]

    question = r.choice(question_templates)
    answer = currency
    explanation = f"The currency used in {country} is the {currency}."

    metadata = {
        "country": country,
        "currency": currency,
    }

    return question, answer, explanation, metadata


def generate_planet_fact(r: random.Random, answer_style: str):
    planet, fact = r.choice(PLANET_FACTS)

    question_templates = [
        f"What is {planet} known for?",
        f"Give one common fact about {planet}.",
        f"Which planet is described as {fact}?",
    ]

    question = r.choice(question_templates)

    if question.startswith("Which planet"):
        answer = planet
        explanation = f"The planet described as {fact} is {planet}."
    else:
        answer = fact
        explanation = f"{planet} is known as {fact}."

    metadata = {
        "planet": planet,
        "fact": fact,
    }

    return question, answer, explanation, metadata


def generate_chemical_symbol(r: random.Random, answer_style: str):
    element, symbol = r.choice(CHEMICAL_SYMBOL_FACTS)

    question_templates = [
        f"What is the chemical symbol for {element}?",
        f"Which chemical symbol represents {element}?",
        f"Give the chemical symbol of {element}.",
    ]

    question = r.choice(question_templates)
    answer = symbol
    explanation = f"The chemical symbol for {element} is {symbol}."

    metadata = {
        "element": element,
        "symbol": symbol,
    }

    return question, answer, explanation, metadata


def generate_animal_class(r: random.Random, answer_style: str):
    animal, animal_class = r.choice(ANIMAL_CLASS_FACTS)

    question_templates = [
        f"What class of animal is a {animal}?",
        f"A {animal} belongs to which animal class?",
        f"Classify a {animal} as an animal type.",
    ]

    question = r.choice(question_templates)
    answer = animal_class
    explanation = f"A {animal} is classified as a {animal_class}."

    metadata = {
        "animal": animal,
        "animal_class": animal_class,
    }

    return question, answer, explanation, metadata


def generate_historical_fact(r: random.Random, answer_style: str):
    description, answer = r.choice(HISTORICAL_FACTS)

    question_templates = [
        f"What is {description}?",
        f"Name {description}.",
        f"Identify {description}.",
    ]

    question = r.choice(question_templates)
    explanation = f"The answer is {answer}."

    metadata = {
        "description": description,
        "answer": answer,
    }

    return question, answer, explanation, metadata


def generate_geography_fact(r: random.Random, answer_style: str):
    description, answer = r.choice(GEOGRAPHY_FACTS)

    question_templates = [
        f"What is {description}?",
        f"Name {description}.",
        f"Identify {description}.",
    ]

    question = r.choice(question_templates)
    explanation = f"The answer is {answer}."

    metadata = {
        "description": description,
        "answer": answer,
    }

    return question, answer, explanation, metadata


def generate_science_definition(r: random.Random, answer_style: str):
    term, definition = r.choice(SCIENCE_DEFINITIONS)

    question_templates = [
        f"What is {term}?",
        f"Define {term}.",
        f"What does {term} mean in science?",
    ]

    question = r.choice(question_templates)
    answer = definition
    explanation = f"{term.capitalize()} is {definition}."

    metadata = {
        "term": term,
        "definition": definition,
    }

    return question, answer, explanation, metadata


def generate_technology_definition(r: random.Random, answer_style: str):
    term, definition = r.choice(TECHNOLOGY_DEFINITIONS)

    question_templates = [
        f"What is a {term}?",
        f"Define {term}.",
        f"What does {term} mean in technology?",
    ]

    question = r.choice(question_templates)
    answer = definition
    explanation = f"{term} means {definition}."

    metadata = {
        "term": term,
        "definition": definition,
    }

    return question, answer, explanation, metadata


def generate_literary_author(r: random.Random, answer_style: str):
    work, author = r.choice(LITERARY_AUTHOR_FACTS)

    question_templates = [
        f"Who wrote {work}?",
        f"Name the author of {work}.",
        f"{work} was written by whom?",
    ]

    question = r.choice(question_templates)
    answer = author
    explanation = f"{work} was written by {author}."

    metadata = {
        "work": work,
        "author": author,
    }

    return question, answer, explanation, metadata


def generate_math_concept(r: random.Random, answer_style: str):
    term, definition = r.choice(MATH_CONCEPTS)

    question_templates = [
        f"What is a {term}?",
        f"Define {term}.",
        f"What does {term} mean in mathematics?",
    ]

    question = r.choice(question_templates)
    answer = definition
    explanation = f"A {term} is {definition}."

    metadata = {
        "term": term,
        "definition": definition,
    }

    return question, answer, explanation, metadata


def generate_language_fact(r: random.Random, answer_style: str):
    term, definition = r.choice(LANGUAGE_FACTS)

    question_templates = [
        f"What is {term}?",
        f"Define {term}.",
        f"What does {term} mean in grammar?",
    ]

    question = r.choice(question_templates)
    answer = definition
    explanation = f"{term.capitalize()} is {definition}."

    metadata = {
        "term": term,
        "definition": definition,
    }

    return question, answer, explanation, metadata


def generate_item(
    kind: str,
    index: int,
    seed: int,
    answer_style: str = "short",
) -> dict:
    r = random.Random(seed)

    if kind == "capital_city":
        question, answer, explanation, metadata = generate_capital_city(r, answer_style)

    elif kind == "country_currency":
        question, answer, explanation, metadata = generate_country_currency(r, answer_style)

    elif kind == "planet_fact":
        question, answer, explanation, metadata = generate_planet_fact(r, answer_style)

    elif kind == "chemical_symbol":
        question, answer, explanation, metadata = generate_chemical_symbol(r, answer_style)

    elif kind == "animal_class":
        question, answer, explanation, metadata = generate_animal_class(r, answer_style)

    elif kind == "historical_fact":
        question, answer, explanation, metadata = generate_historical_fact(r, answer_style)

    elif kind == "geography_fact":
        question, answer, explanation, metadata = generate_geography_fact(r, answer_style)

    elif kind == "science_definition":
        question, answer, explanation, metadata = generate_science_definition(r, answer_style)

    elif kind == "technology_definition":
        question, answer, explanation, metadata = generate_technology_definition(r, answer_style)

    elif kind == "literary_author":
        question, answer, explanation, metadata = generate_literary_author(r, answer_style)

    elif kind == "math_concept":
        question, answer, explanation, metadata = generate_math_concept(r, answer_style)

    elif kind == "language_fact":
        question, answer, explanation, metadata = generate_language_fact(r, answer_style)

    else:
        raise ValueError(f"Unknown knowledge kind: {kind}")

    prompt = make_prompt(
        question=question,
        answer_style=answer_style,
    )

    completion = make_completion(
        answer=answer,
        explanation=explanation,
        answer_style=answer_style,
    )

    metadata.update(
        {
            "answer_style": answer_style,
            "normalized_gold": normalize_answer(answer),
        }
    )

    return make_record(
        kind=kind,
        index=index,
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
    shuffle: bool = True,
    answer_style: str = "short",
) -> list[dict]:
    main_rng = random.Random((int(seed) ^ KNOWLEDGE_STREAM_SEED) & 0xFFFFFFFF)
    records = []

    index = 0

    for kind in KNOWLEDGE_KINDS:
        for _ in range(n_per_kind):
            item_seed = main_rng.randint(0, 2**31 - 1)

            record = generate_item(
                kind=kind,
                index=index,
                seed=item_seed,
                answer_style=answer_style,
            )

            records.append(record)
            index += 1

    if shuffle:
        main_rng.shuffle(records)

    return records


FINAL_RE = re.compile(r"Final answer:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)


def extract_final_answer(completion: str) -> str | None:
    match = FINAL_RE.search(str(completion).strip())

    if not match:
        return None

    return match.group(1).strip()


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
        print("Local verification: all generated knowledge records passed.")
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
            bad.append((record.get("task_id"), record.get("kind"), "completion does not end with newline"))

        if answer_style != "answer_only" and "Final answer:" not in completion:
            bad.append((record.get("task_id"), record.get("kind"), "missing Final answer line"))

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic synthetic knowledge JSONL database "
            "for factual QA SFT/evaluation."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset/knowledge_database_all_cases.jsonl",
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
        default=20,
        help="Number of records to generate per knowledge kind.",
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
    print(f"Database version: {KNOWLEDGE_DATABASE_VERSION}")
    print(f"Total records: {len(records)}")
    print(f"Knowledge kinds: {len(KNOWLEDGE_KINDS)}")
    print(f"Records per kind: {args.n_per_kind}")
    print(f"Append mode: {args.append}")
    print(f"Answer style: {answer_style}")
    print(f"Local verified: {args.verify}")
    print("Completion format: concise answer plus Final answer line")
    print("Core fields: question, prompt, completion, answer, gold, kind, metadata")

    print_counts(records)


if __name__ == "__main__":
    main()
