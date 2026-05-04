#!/usr/bin/env python3

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Repo root helper
# ═══════════════════════════════════════════════════════════════════════════════

def find_repo_root(start: Path) -> Path:
    """
    Walk upward until we find the repository root containing ./distil.

    This mirrors your debug database script style, but this math builder
    intentionally does NOT import pod_eval_vllm.py because that imports torch.
    """
    current = start.resolve()

    for path in [current] + list(current.parents):
        if (path / "distil").exists():
            return path

    # Fallback: allow running in smaller local test folders too.
    # We do not strictly need ./distil for this script.
    return current.parent


REPO_ROOT = find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(REPO_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "math_db_v1"

DEFAULT_OUTPUT = "dataset/math_database_all_cases.jsonl"
DEFAULT_FAILURE_LOG = "database_v3/math_database_failures.log"
DEFAULT_SUMMARY = "dataset/math_database_all_cases.summary.json"

REQUIRED_FIELDS = [
    "prompt",
    "completion",
    "answer",
    "kind",
]


NAMES = [
    "Alex", "Sam", "Jordan", "Taylor", "Mira", "Priya", "Leo", "Nora",
    "Kai", "Riley", "Morgan", "Avery", "Quinn", "Zara", "Theo", "Iris",
    "Rowan", "Eden", "Naomi", "Owen", "Luca", "Maya", "Sage", "Drew",
]

OBJECTS = [
    "stickers", "marbles", "notebooks", "pencils", "cards", "shells",
    "buttons", "postcards", "coins", "tickets", "erasers", "markers",
    "books", "candles", "beads", "magnets", "paper clips", "toy cars",
]

SHOPS = [
    "bakery", "bookstore", "market", "stationery shop", "toy store",
    "garden center", "grocery store", "hardware store", "corner shop",
    "farmers market", "candy shop", "art supply store",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Lightweight math extraction / scoring helpers
# No torch. No transformers. No GPU dependencies.
# ═══════════════════════════════════════════════════════════════════════════════

CHAT_PROBE_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
CHAT_PROBE_THINK_TRAIL = re.compile(r"^.*?</think>\s*", re.DOTALL)
CHAT_PROBE_NARRATIVE = re.compile(
    r"^\s*Thinking Process:.*?(?=\n\n[A-Z0-9]|\Z)",
    re.DOTALL,
)

MATH_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
MATH_BOXED_START_RE = re.compile(r"\\boxed\s*\{")
MATH_ANSWER_PHRASE_RE = re.compile(
    r"(?:the\s+)?answer\s*(?:is|=|:)\s*\$?([^\s\n\.]+)",
    re.IGNORECASE,
)


def strip_thinking_probe(text: str) -> str:
    """
    Lightweight copy of the thinking-strip behavior used in pod_eval_vllm.py.
    """
    text = str(text or "")

    if "<think>" in text:
        text = CHAT_PROBE_THINK_RE.sub("", text, count=1)
    elif "</think>" in text:
        text = CHAT_PROBE_THINK_TRAIL.sub("", text, count=1)

    if text.lstrip().startswith("Thinking Process:"):
        text = CHAT_PROBE_NARRATIVE.sub("", text, count=1)

    return text.strip()


def strip_markdown_fences(text: str) -> str:
    """
    Defensive cleanup only. Generated database records should not contain fences.
    """
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    stripped = text.strip()

    if not stripped.startswith("```"):
        return text

    lines = text.splitlines()

    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]

    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip("\n") + "\n"


def clean_completion(completion: str) -> str:
    """
    Cleanup for generated math completions.
    """
    completion = str(completion)
    completion = strip_thinking_probe(completion)
    completion = strip_markdown_fences(completion)

    return completion.strip()


def extract_boxed(text: str) -> str | None:
    """
    Extract the contents of the last \\boxed{...}, supporting nested braces.
    """
    last = None

    for match in MATH_BOXED_START_RE.finditer(text):
        i = match.end()
        depth = 1
        j = i

        while j < len(text) and depth > 0:
            char = text[j]

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

                if depth == 0:
                    last = text[i:j].strip()
                    break

            j += 1

    return last


def math_extract_answer(text: str, src: str = "math") -> str:
    """
    Lightweight copy of pod_eval_vllm._math_extract_answer.

    For this database, src='math' is expected because completions end with:

        #### N
    """
    cleaned = strip_thinking_probe(text or "")

    if not cleaned:
        return ""

    if src == "math500":
        boxed = extract_boxed(cleaned)

        if boxed:
            return boxed.rstrip(".")

    if "####" in cleaned:
        match = re.search(r"####\s*([^\n]+)", cleaned)

        if match:
            tail = match.group(1).strip().rstrip(".")
            num_match = MATH_NUMBER_RE.search(tail)

            if num_match:
                return num_match.group(0)

            return tail

    match = MATH_ANSWER_PHRASE_RE.search(cleaned)

    if match:
        fragment = match.group(1).strip().rstrip(".,")
        num_match = MATH_NUMBER_RE.search(fragment)

        if num_match:
            return num_match.group(0)

        if fragment:
            return fragment

    numbers = MATH_NUMBER_RE.findall(cleaned)

    if numbers:
        return numbers[-1]

    return cleaned.strip().splitlines()[-1].strip() if cleaned else ""


def math_score_one(pred: str, gold: str) -> int:
    """
    Lightweight copy of pod_eval_vllm._math_score_one.
    """
    if not pred:
        return 0

    pred_norm = str(pred).replace(",", "").replace("$", "").strip().rstrip(".")
    gold_norm = str(gold).replace(",", "").replace("$", "").strip().rstrip(".")

    if pred_norm == gold_norm:
        return 1

    try:
        return 1 if abs(float(pred_norm) - float(gold_norm)) < 1e-6 else 0
    except (TypeError, ValueError):
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# JSONL helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.rstrip("\n")

            if not raw.strip():
                continue

            try:
                yield line_no, json.loads(raw)
            except json.JSONDecodeError as e:
                yield line_no, {
                    "_json_error": str(e),
                    "_raw": raw[:1000],
                }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Record construction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def final_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Solve step by step and end with '#### N' where N is the final integer answer."
    )


def make_record(
    *,
    index: int,
    block_seed: int,
    generator_seed: int,
    kind: str,
    difficulty: int,
    prompt: str,
    completion: str,
    answer: int | str,
    metadata: dict | None = None,
) -> dict:
    return {
        "task_id": f"{VERSION}/{kind}/{index:08d}",
        "src": f"procedural_math_database/{kind}",
        "kind": kind,
        "status": "gold",
        "block_seed": block_seed,
        "index": index,
        "difficulty": difficulty,
        "prompt": prompt,
        "completion": completion,
        "answer": str(answer),
        "generator_seed": generator_seed,
        "metadata": {
            "version": VERSION,
            **(metadata or {}),
        },
    }


def get_kind(record: dict) -> str:
    if record.get("kind"):
        return str(record["kind"])

    src = str(record.get("src", ""))

    if "/" in src:
        return src.rsplit("/", 1)[-1]

    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# Math generators
# ═══════════════════════════════════════════════════════════════════════════════

def gen_arithmetic_mixed(rng: random.Random):
    a = rng.randint(10, 99)
    b = rng.randint(10, 99)
    c = rng.randint(2, 12)
    d = rng.randint(1, 50)

    ans = (a + b) * c - d

    question = f"Compute ({a} + {b}) * {c} - {d}."

    solution = (
        f"First add: {a} + {b} = {a + b}. "
        f"Then multiply: {a + b} * {c} = {(a + b) * c}. "
        f"Finally subtract {d}: {(a + b) * c} - {d} = {ans}.\n"
        f"#### {ans}"
    )

    return "arithmetic_mixed", 1, final_prompt(question), solution, ans


def gen_linear_equation(rng: random.Random):
    x = rng.randint(-20, 30)

    while x == 0:
        x = rng.randint(-20, 30)

    a = rng.randint(2, 12)
    b = rng.randint(-40, 40)
    c = a * x + b

    if b >= 0:
        equation = f"{a}x + {b} = {c}"
        first_step = f"Subtract {b} from both sides: {a}x = {c - b}."
    else:
        equation = f"{a}x - {abs(b)} = {c}"
        first_step = f"Add {abs(b)} to both sides: {a}x = {c - b}."

    question = f"Solve for x: {equation}."

    solution = (
        f"Start with {equation}. "
        f"{first_step} "
        f"Divide by {a}: x = {x}.\n"
        f"#### {x}"
    )

    return "linear_equation", 2, final_prompt(question), solution, x


def gen_two_step_word_problem(rng: random.Random):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)

    start = rng.randint(10, 90)
    bought = rng.randint(5, 60)
    gave = rng.randint(3, min(start + bought - 1, 70))

    ans = start + bought - gave

    question = (
        f"{name} had {start} {obj}. "
        f"{name} bought {bought} more and then gave away {gave}. "
        f"How many {obj} does {name} have now?"
    )

    solution = (
        f"Start with {start}. "
        f"After buying {bought}, the total is {start + bought}. "
        f"After giving away {gave}, the remaining number is "
        f"{start + bought} - {gave} = {ans}.\n"
        f"#### {ans}"
    )

    return "word_problem_two_step", 2, final_prompt(question), solution, ans


def gen_shopping_budget(rng: random.Random):
    name = rng.choice(NAMES)
    shop = rng.choice(SHOPS)
    item_a = rng.choice(OBJECTS)
    item_b = rng.choice([x for x in OBJECTS if x != item_a])

    money = rng.choice([50, 60, 75, 80, 100, 120, 150])
    n_a = rng.randint(2, 8)
    p_a = rng.randint(2, 12)
    n_b = rng.randint(2, 6)
    p_b = rng.randint(3, 15)

    spent_a = n_a * p_a
    spent_b = n_b * p_b
    spent = spent_a + spent_b
    ans = money - spent

    question = (
        f"{name} goes to the {shop} with {money} dollars. "
        f"{name} buys {n_a} {item_a} at {p_a} dollars each and "
        f"{n_b} {item_b} at {p_b} dollars each. "
        f"How many dollars does {name} have left?"
    )

    solution = (
        f"The {item_a} cost {n_a} * {p_a} = {spent_a}. "
        f"The {item_b} cost {n_b} * {p_b} = {spent_b}. "
        f"Total spent is {spent_a} + {spent_b} = {spent}. "
        f"Money left is {money} - {spent} = {ans}.\n"
        f"#### {ans}"
    )

    return "shopping_budget", 2, final_prompt(question), solution, ans


def gen_percentage_integer(rng: random.Random):
    base = rng.choice([100, 120, 150, 200, 240, 300, 400, 500, 600, 800])
    pct = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75])

    ans = base * pct // 100

    question = f"What is {pct} percent of {base}?"

    solution = (
        f"{pct} percent means {pct} out of 100. "
        f"So {pct} percent of {base} is {base} * {pct} / 100 = {ans}.\n"
        f"#### {ans}"
    )

    return "percentage_integer", 2, final_prompt(question), solution, ans


def gen_rate_distance(rng: random.Random):
    speed = rng.choice([30, 40, 45, 50, 55, 60, 70, 80])
    hours = rng.randint(2, 8)
    extra = rng.randint(5, 50)

    ans = speed * hours + extra

    question = (
        f"A car travels at {speed} miles per hour for {hours} hours, "
        f"then travels {extra} more miles. "
        f"How many miles does it travel in total?"
    )

    solution = (
        f"Distance for the first part is speed times time: "
        f"{speed} * {hours} = {speed * hours}. "
        f"Then add the extra {extra} miles: {speed * hours} + {extra} = {ans}.\n"
        f"#### {ans}"
    )

    return "rate_distance", 2, final_prompt(question), solution, ans


def gen_sequence(rng: random.Random):
    mode = rng.choice(["arithmetic", "geometric", "square"])

    if mode == "arithmetic":
        start = rng.randint(1, 25)
        diff = rng.randint(2, 12)
        seq = [start + i * diff for i in range(5)]
        ans = start + 5 * diff

        question = (
            f"What is the next number in the sequence "
            f"{', '.join(str(x) for x in seq)}, ?"
        )

        solution = (
            f"The sequence increases by {diff} each time. "
            f"The next term is {seq[-1]} + {diff} = {ans}.\n"
            f"#### {ans}"
        )

        return "sequence_arithmetic", 2, final_prompt(question), solution, ans

    if mode == "geometric":
        start = rng.randint(2, 8)
        ratio = rng.choice([2, 3, 4])
        seq = [start * (ratio ** i) for i in range(5)]
        ans = start * (ratio ** 5)

        question = (
            f"What is the next number in the sequence "
            f"{', '.join(str(x) for x in seq)}, ?"
        )

        solution = (
            f"Each term is multiplied by {ratio}. "
            f"The next term is {seq[-1]} * {ratio} = {ans}.\n"
            f"#### {ans}"
        )

        return "sequence_geometric", 3, final_prompt(question), solution, ans

    start = rng.randint(2, 8)
    seq = [(start + i) ** 2 for i in range(5)]
    ans = (start + 5) ** 2

    question = (
        f"What is the next number in the sequence "
        f"{', '.join(str(x) for x in seq)}, ?"
    )

    solution = (
        f"The terms are consecutive squares starting from {start} squared. "
        f"The next term is {start + 5} squared, which is {ans}.\n"
        f"#### {ans}"
    )

    return "sequence_square", 3, final_prompt(question), solution, ans


def gen_gcd_lcm(rng: random.Random):
    a = rng.randint(12, 240)
    b = rng.randint(12, 240)
    mode = rng.choice(["gcd", "lcm"])

    if mode == "gcd":
        ans = math.gcd(a, b)

        question = f"What is the greatest common divisor of {a} and {b}?"

        solution = (
            f"The greatest common divisor of {a} and {b} is {ans}.\n"
            f"#### {ans}"
        )

        return "number_theory_gcd", 3, final_prompt(question), solution, ans

    ans = abs(a * b) // math.gcd(a, b)

    question = f"What is the least common multiple of {a} and {b}?"

    solution = (
        f"The least common multiple equals abs({a} * {b}) divided by "
        f"gcd({a}, {b}). This gives {ans}.\n"
        f"#### {ans}"
    )

    return "number_theory_lcm", 3, final_prompt(question), solution, ans


def gen_geometry(rng: random.Random):
    mode = rng.choice([
        "rectangle_area",
        "rectangle_perimeter",
        "triangle_area",
        "circle_area_coefficient",
    ])

    if mode == "rectangle_area":
        w = rng.randint(3, 40)
        h = rng.randint(3, 40)
        ans = w * h

        question = f"A rectangle has width {w} and height {h}. What is its area?"

        solution = (
            f"Area is width times height: {w} * {h} = {ans}.\n"
            f"#### {ans}"
        )

        return "geometry_rectangle_area", 2, final_prompt(question), solution, ans

    if mode == "rectangle_perimeter":
        w = rng.randint(3, 40)
        h = rng.randint(3, 40)
        ans = 2 * (w + h)

        question = f"A rectangle has width {w} and height {h}. What is its perimeter?"

        solution = (
            f"Perimeter is 2 times width plus height: "
            f"2 * ({w} + {h}) = {ans}.\n"
            f"#### {ans}"
        )

        return "geometry_rectangle_perimeter", 2, final_prompt(question), solution, ans

    if mode == "triangle_area":
        base = rng.choice([4, 6, 8, 10, 12, 14, 16, 18, 20])
        height = rng.choice([3, 5, 7, 9, 11, 13, 15])
        ans = base * height // 2

        question = f"A triangle has base {base} and height {height}. What is its area?"

        solution = (
            f"Triangle area is base times height divided by 2: "
            f"{base} * {height} / 2 = {ans}.\n"
            f"#### {ans}"
        )

        return "geometry_triangle_area", 2, final_prompt(question), solution, ans

    radius = rng.randint(2, 25)
    ans = radius * radius

    question = (
        f"A circle has radius {radius}. "
        f"What is its area divided by pi? Give the numeric coefficient."
    )

    solution = (
        f"The area of a circle is pi times radius squared. "
        f"The coefficient is {radius} squared, which is {ans}.\n"
        f"#### {ans}"
    )

    return "geometry_circle_area_coefficient", 2, final_prompt(question), solution, ans


def gen_system_equations(rng: random.Random):
    x = rng.randint(-10, 15)
    y = rng.randint(-10, 15)

    while x == y:
        y = rng.randint(-10, 15)

    a1 = rng.randint(1, 7)
    b1 = rng.randint(1, 7)
    a2 = rng.randint(1, 7)
    b2 = rng.randint(1, 7)

    while a1 * b2 - a2 * b1 == 0:
        a2 = rng.randint(1, 7)
        b2 = rng.randint(1, 7)

    c1 = a1 * x + b1 * y
    c2 = a2 * x + b2 * y

    ask = rng.choice(["x", "y", "x_plus_y", "x_minus_y"])

    if ask == "x":
        ans = x
        ask_text = "x"
    elif ask == "y":
        ans = y
        ask_text = "y"
    elif ask == "x_plus_y":
        ans = x + y
        ask_text = "x + y"
    else:
        ans = x - y
        ask_text = "x - y"

    question = (
        f"Solve the system of equations: "
        f"{a1}x + {b1}y = {c1}, and {a2}x + {b2}y = {c2}. "
        f"What is {ask_text}?"
    )

    solution = (
        f"The system has solution x = {x} and y = {y}. "
        f"Therefore {ask_text} = {ans}.\n"
        f"#### {ans}"
    )

    return "system_equations", 4, final_prompt(question), solution, ans


def gen_modular_arithmetic(rng: random.Random):
    a = rng.randint(2, 20)
    b = rng.randint(2, 20)
    c = rng.randint(1, 99)
    m = rng.choice([7, 11, 13, 17, 19, 23, 29, 31])

    raw = a * b + c
    ans = raw % m

    question = f"Compute ({a} * {b} + {c}) mod {m}."

    solution = (
        f"First compute {a} * {b} + {c} = {raw}. "
        f"The remainder when {raw} is divided by {m} is {ans}.\n"
        f"#### {ans}"
    )

    return "modular_arithmetic", 3, final_prompt(question), solution, ans


def gen_unit_conversion(rng: random.Random):
    mode = rng.choice([
        "hours_to_minutes",
        "minutes_to_seconds",
        "kilometers_to_meters",
        "meters_to_centimeters",
        "days_to_hours",
        "weeks_to_days",
    ])

    if mode == "hours_to_minutes":
        hours = rng.randint(2, 24)
        ans = hours * 60
        question = f"How many minutes are in {hours} hours?"
        solution = f"There are 60 minutes in an hour, so {hours} * 60 = {ans}.\n#### {ans}"
        return "unit_conversion_hours_to_minutes", 1, final_prompt(question), solution, ans

    if mode == "minutes_to_seconds":
        minutes = rng.randint(2, 90)
        ans = minutes * 60
        question = f"How many seconds are in {minutes} minutes?"
        solution = f"There are 60 seconds in a minute, so {minutes} * 60 = {ans}.\n#### {ans}"
        return "unit_conversion_minutes_to_seconds", 1, final_prompt(question), solution, ans

    if mode == "kilometers_to_meters":
        km = rng.randint(2, 50)
        ans = km * 1000
        question = f"How many meters are in {km} kilometers?"
        solution = f"There are 1000 meters in a kilometer, so {km} * 1000 = {ans}.\n#### {ans}"
        return "unit_conversion_km_to_m", 1, final_prompt(question), solution, ans

    if mode == "meters_to_centimeters":
        meters = rng.randint(2, 80)
        ans = meters * 100
        question = f"How many centimeters are in {meters} meters?"
        solution = f"There are 100 centimeters in a meter, so {meters} * 100 = {ans}.\n#### {ans}"
        return "unit_conversion_m_to_cm", 1, final_prompt(question), solution, ans

    if mode == "days_to_hours":
        days = rng.randint(2, 30)
        ans = days * 24
        question = f"How many hours are in {days} days?"
        solution = f"There are 24 hours in a day, so {days} * 24 = {ans}.\n#### {ans}"
        return "unit_conversion_days_to_hours", 1, final_prompt(question), solution, ans

    weeks = rng.randint(2, 52)
    ans = weeks * 7
    question = f"How many days are in {weeks} weeks?"
    solution = f"There are 7 days in a week, so {weeks} * 7 = {ans}.\n#### {ans}"
    return "unit_conversion_weeks_to_days", 1, final_prompt(question), solution, ans


def gen_multi_step_table(rng: random.Random):
    name = rng.choice(NAMES)
    item = rng.choice(OBJECTS)

    monday = rng.randint(5, 25)
    tuesday = rng.randint(5, 25)
    wednesday = rng.randint(5, 25)
    lost = rng.randint(1, min(10, monday + tuesday + wednesday - 1))

    total = monday + tuesday + wednesday
    ans = total - lost

    question = (
        f"{name} collected {item} over three days. "
        f"On Monday, {name} collected {monday}. "
        f"On Tuesday, {name} collected {tuesday}. "
        f"On Wednesday, {name} collected {wednesday}. "
        f"Then {name} lost {lost}. "
        f"How many {item} does {name} have left?"
    )

    solution = (
        f"First add the collected amounts: "
        f"{monday} + {tuesday} + {wednesday} = {total}. "
        f"Then subtract the lost amount: {total} - {lost} = {ans}.\n"
        f"#### {ans}"
    )

    return "multi_step_table", 2, final_prompt(question), solution, ans


GENERATORS = [
    gen_arithmetic_mixed,
    gen_linear_equation,
    gen_two_step_word_problem,
    gen_shopping_budget,
    gen_percentage_integer,
    gen_rate_distance,
    gen_sequence,
    gen_gcd_lcm,
    gen_geometry,
    gen_system_equations,
    gen_modular_arithmetic,
    gen_unit_conversion,
    gen_multi_step_table,
]


# ═══════════════════════════════════════════════════════════════════════════════
# Build / self-check
# ═══════════════════════════════════════════════════════════════════════════════

def generate_record(index: int, block_seed: int) -> dict:
    generator_seed = block_seed + index
    rng = random.Random(generator_seed)

    generator = rng.choice(GENERATORS)
    kind, difficulty, prompt, completion, answer = generator(rng)

    return make_record(
        index=index,
        block_seed=block_seed,
        generator_seed=generator_seed,
        kind=kind,
        difficulty=difficulty,
        prompt=prompt,
        completion=completion,
        answer=answer,
        metadata={
            "generator": generator.__name__,
        },
    )


def score_record(record: dict) -> tuple[bool, str, str]:
    completion = clean_completion(record["completion"])
    pred = math_extract_answer(completion, "math")
    gold = str(record.get("answer", ""))

    ok = bool(math_score_one(pred, gold))

    return ok, pred, gold


def self_check_record(record: dict) -> tuple[bool, str]:
    ok, pred, gold = score_record(record)

    if ok:
        return True, ""

    return False, f"self-check failed: pred={pred!r}, gold={gold!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# Validation / checking
# ═══════════════════════════════════════════════════════════════════════════════

def validate_record(record: dict) -> list[str]:
    problems = []

    if "_json_error" in record:
        problems.append(f"JSON decode error: {record['_json_error']}")
        return problems

    for field in REQUIRED_FIELDS:
        if field not in record:
            problems.append(f"missing field: {field}")

    if problems:
        return problems

    prompt = record.get("prompt")
    completion = record.get("completion")
    answer = record.get("answer")
    kind = record.get("kind")

    if not isinstance(prompt, str):
        problems.append("prompt is not a string")

    if not isinstance(completion, str):
        problems.append("completion is not a string")

    if not isinstance(answer, str) and not isinstance(answer, int):
        problems.append("answer is not a string or int")

    if not isinstance(kind, str):
        problems.append("kind is not a string")

    if isinstance(prompt, str):
        if "#### N" not in prompt:
            problems.append("prompt does not contain expected math final-answer marker")

        if "Solve step by step" not in prompt:
            problems.append("prompt does not contain expected solve-step instruction")

    if isinstance(completion, str):
        cleaned = clean_completion(completion)

        if not cleaned.strip():
            problems.append("completion is empty after cleanup")

        if completion.strip().startswith("```"):
            problems.append("completion contains markdown fence")

        if "####" not in cleaned:
            problems.append("completion does not contain #### final-answer marker")

    return problems


def is_fatal_validation_problem(problems: list[str]) -> bool:
    hard_prefixes = [
        "JSON decode error",
        "missing field",
        "prompt is not a string",
        "completion is not a string",
        "answer is not a string or int",
        "kind is not a string",
        "completion is empty",
    ]

    return any(
        any(problem.startswith(prefix) for prefix in hard_prefixes)
        for problem in problems
    )


def write_failure_log(log_path: Path, failures: list[dict]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as f:
        for failure in failures:
            record = failure["record"]

            f.write("=" * 100 + "\n")
            f.write(f"Line: {failure.get('line_no')}\n")
            f.write(f"Task ID: {record.get('task_id')}\n")
            f.write(f"Kind: {get_kind(record)}\n")
            f.write(f"Src: {record.get('src')}\n")
            f.write(f"Status: {record.get('status')}\n")
            f.write(f"Difficulty: {record.get('difficulty')}\n")
            f.write(f"Block seed: {record.get('block_seed')}\n")
            f.write(f"Generator seed: {record.get('generator_seed')}\n")
            f.write(f"Index: {record.get('index')}\n")
            f.write("\n")

            if failure.get("validation_problems"):
                f.write("[VALIDATION PROBLEMS]\n")

                for problem in failure["validation_problems"]:
                    f.write(f"- {problem}\n")

                f.write("\n")

            if failure.get("score_result"):
                f.write("[SCORE RESULT]\n")
                f.write(failure["score_result"])
                f.write("\n\n")

            f.write("[PROMPT]\n")
            f.write(str(record.get("prompt", "")))
            f.write("\n\n")

            f.write("[RAW COMPLETION]\n")
            f.write(str(record.get("completion", "")))
            f.write("\n\n")

            f.write("[CLEANED COMPLETION]\n")
            f.write(clean_completion(record.get("completion", "")))
            f.write("\n\n")

            f.write("[EXTRACTED ANSWER]\n")
            try:
                f.write(math_extract_answer(clean_completion(record.get("completion", "")), "math"))
            except Exception as e:
                f.write(f"<extract failed: {type(e).__name__}: {e}>")
            f.write("\n\n")

            f.write("[GOLD ANSWER]\n")
            f.write(str(record.get("answer", "")))
            f.write("\n\n")


def print_counter(title: str, counter: Counter) -> None:
    print(f"\n{title}:")

    if not counter:
        print("  <none>")
        return

    for key, value in sorted(counter.items()):
        print(f"  {key}: {value}")


# ═══════════════════════════════════════════════════════════════════════════════
# Build mode
# ═══════════════════════════════════════════════════════════════════════════════

def build_database(args) -> None:
    records = []
    failures = []

    kind_counts = Counter()
    difficulty_counts = Counter()

    for offset in range(args.n):
        index = args.start_index + offset
        record = generate_record(index=index, block_seed=args.block_seed)

        kind_counts[record["kind"]] += 1
        difficulty_counts[str(record["difficulty"])] += 1

        if not args.no_self_check:
            ok, reason = self_check_record(record)

            if ok:
                record["status"] = "gold"
            else:
                record["status"] = "self_check_failed"
                failures.append({
                    "line_no": offset + 1,
                    "record": record,
                    "score_result": reason,
                })

        records.append(record)

        if (offset + 1) % args.progress_every == 0:
            print(f"[build] generated {offset + 1}/{args.n}")

    output_path = Path(args.output)
    write_jsonl(output_path, records)

    summary = {
        "version": VERSION,
        "output": str(output_path),
        "n": len(records),
        "block_seed": args.block_seed,
        "start_index": args.start_index,
        "self_check_failures": len(failures),
        "kind_counts": dict(sorted(kind_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
    }

    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    write_json(summary_path, summary)

    if failures:
        failure_log = Path(args.failure_log)
        write_failure_log(failure_log, failures)
        print(f"[build] WARNING: {len(failures)} failures written to {failure_log}")

    print("=" * 80)
    print("MATH DATABASE BUILD SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

    if failures:
        raise SystemExit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Check mode
# ═══════════════════════════════════════════════════════════════════════════════

def check_database(args) -> None:
    input_path = Path(args.path)

    if not input_path.exists():
        raise FileNotFoundError(f"Math database file does not exist: {input_path}")

    total = 0
    validation_bad = 0
    score_passed = 0
    score_failed = 0

    status_counts = Counter()
    kind_counts = Counter()
    kind_pass_counts = Counter()
    kind_fail_counts = Counter()
    validation_problem_counts = Counter()

    failures = []

    def add_failure(
        line_no: int,
        record: dict,
        validation_problems: list[str] | None = None,
        score_result: str | None = None,
    ) -> None:
        failure = {
            "line_no": line_no,
            "record": record,
        }

        if validation_problems:
            failure["validation_problems"] = validation_problems

            for problem in validation_problems:
                validation_problem_counts[problem] += 1

        if score_result:
            failure["score_result"] = score_result

        failures.append(failure)

    for line_no, record in load_jsonl(input_path):
        total += 1

        if args.limit and total > args.limit:
            total -= 1
            break

        if "_json_error" not in record:
            kind = get_kind(record)
            status = str(record.get("status", "unknown"))

            kind_counts[kind] += 1
            status_counts[status] += 1
        else:
            kind = "unknown"

        problems = validate_record(record)

        if problems:
            fatal = args.strict_validation or is_fatal_validation_problem(problems)

            if fatal:
                validation_bad += 1

                add_failure(
                    line_no=line_no,
                    record=record,
                    validation_problems=problems,
                )

                if args.show_failures:
                    print("=" * 80)
                    print(f"VALIDATION FAILED line={line_no}")

                    for problem in problems:
                        print(f"- {problem}")

                continue

            for problem in problems:
                validation_problem_counts[problem] += 1

        try:
            ok, pred, gold = score_record(record)
        except Exception as e:
            score_failed += 1
            kind_fail_counts[kind] += 1

            add_failure(
                line_no=line_no,
                record=record,
                score_result=f"Exception while scoring: {type(e).__name__}: {e}",
            )

            if args.show_failures:
                print("=" * 80)
                print(f"SCORE EXCEPTION line={line_no} kind={kind}")
                print(f"{type(e).__name__}: {e}")

            continue

        if ok:
            score_passed += 1
            kind_pass_counts[kind] += 1
        else:
            score_failed += 1
            kind_fail_counts[kind] += 1

            score_text = (
                f"passed: False\n"
                f"pred: {pred!r}\n"
                f"gold: {gold!r}"
            )

            add_failure(
                line_no=line_no,
                record=record,
                score_result=score_text,
            )

            if args.show_failures:
                print("=" * 80)
                print(
                    f"FAILED line={line_no} "
                    f"task_id={record.get('task_id')} "
                    f"kind={kind}"
                )
                print(score_text)

    failure_log = Path(args.failure_log)

    if failures:
        write_failure_log(failure_log, failures)

    print("=" * 80)
    print("MATH DATABASE CHECK SUMMARY")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Total records checked: {total}")
    print(f"Validation bad records: {validation_bad}")
    print(f"Score passed: {score_passed}")
    print(f"Score failed: {score_failed}")
    print(f"Total failures logged: {len(failures)}")

    if failures:
        print(f"Failure log: {failure_log}")

    print_counter("Status counts", status_counts)

    print("\nKind counts:")
    if not kind_counts:
        print("  <none>")
    else:
        for kind, count in sorted(kind_counts.items()):
            passed = kind_pass_counts.get(kind, 0)
            failed = kind_fail_counts.get(kind, 0)
            print(f"  {kind}: total={count}, passed={passed}, failed={failed}")

    if validation_problem_counts:
        print_counter("Validation problem counts", validation_problem_counts)

    if validation_bad == 0 and score_failed == 0:
        print("\nResult: PASS")
    else:
        print("\nResult: FAIL")
        raise SystemExit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build/check procedural math JSONL database without torch. "
            "Compatible with pod_eval_vllm-style #### math answer extraction."
        )
    )

    sub = parser.add_subparsers(dest="command")

    # Build command
    build = sub.add_parser("build", help="Build procedural math database.")

    build.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path.",
    )

    build.add_argument(
        "--summary",
        type=str,
        default=DEFAULT_SUMMARY,
        help="Output summary JSON path.",
    )

    build.add_argument(
        "--failure-log",
        type=str,
        default=DEFAULT_FAILURE_LOG,
        help="Path to write failure diagnostics.",
    )

    build.add_argument(
        "--n",
        type=int,
        default=10000,
        help="Number of records to generate.",
    )

    build.add_argument(
        "--block-seed",
        type=int,
        default=20260504,
        help="Base seed used to generate deterministic records.",
    )

    build.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for generated task IDs.",
    )

    build.add_argument(
        "--no-self-check",
        action="store_true",
        help="Skip internal extraction/scoring self-check.",
    )

    build.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N generated records.",
    )

    # Check command
    check = sub.add_parser("check", help="Check existing math database.")

    check.add_argument(
        "path",
        nargs="?",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to math JSONL database.",
    )

    check.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of records to check. 0 means all.",
    )

    check.add_argument(
        "--failure-log",
        type=str,
        default=DEFAULT_FAILURE_LOG,
        help="Path to write failure diagnostics.",
    )

    check.add_argument(
        "--show-failures",
        action="store_true",
        help="Print failure details to terminal.",
    )

    check.add_argument(
        "--strict-validation",
        action="store_true",
        help=(
            "Treat all validation problems as fatal. Without this flag, "
            "style problems are logged but scoring still runs where possible."
        ),
    )

    args = parser.parse_args()

    # Default behavior when user runs:
    #
    #     python -m database_v3.math_use_construct
    #
    # is build mode.
    if args.command is None:
        args.command = "build"
        args.output = DEFAULT_OUTPUT
        args.summary = DEFAULT_SUMMARY
        args.failure_log = DEFAULT_FAILURE_LOG
        args.n = 10000
        args.block_seed = 20260504
        args.start_index = 0
        args.no_self_check = False
        args.progress_every = 1000

    if args.command == "build":
        build_database(args)
    elif args.command == "check":
        check_database(args)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
