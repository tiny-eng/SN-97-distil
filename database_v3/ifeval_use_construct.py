#!/usr/bin/env python3

import argparse
import json
import random
import re
import sys
import traceback
from pathlib import Path


IFEVAL_STREAM_SEED = 0x1FEAF001


IFEVAL_KINDS = [
    "exact_words",
    "min_words",
    "max_words",
    "all_lowercase",
    "all_uppercase",
    "ends_with_phrase",
    "include_keyword",
    "forbid_keyword",
    "json_format",
    "exact_sentences",
    "no_comma",
    "title_format",
    "bullet_list",
    "compound2",
    "compound3",
]


TOPICS = [
    "urban gardening",
    "quiet libraries",
    "morning routines",
    "public transportation",
    "home cooking",
    "software documentation",
    "team communication",
    "weather forecasting",
    "recycling habits",
    "learning a new language",
    "bicycle commuting",
    "small business planning",
    "remote work",
    "healthy sleep habits",
    "community volunteering",
    "clear technical writing",
    "weekend hiking",
    "home budgeting",
    "coffee brewing",
    "indoor plants",
    "study planning",
    "open source maintenance",
    "library renovation",
    "neighborhood safety",
]


KEYWORDS = [
    "harbor",
    "compass",
    "garden",
    "library",
    "signal",
    "lantern",
    "bridge",
    "notebook",
    "orchard",
    "window",
    "planet",
    "river",
    "meadow",
    "clock",
    "silver",
    "maple",
]


FORBIDDEN_WORDS = [
    "urgent",
    "perfect",
    "impossible",
    "guarantee",
    "obvious",
    "always",
    "never",
    "failure",
]


END_PHRASES = [
    "Thank you for reading.",
    "End of report.",
    "Is there anything else I can help with?",
    "That is the final note.",
    "This completes the response.",
]


LOWERCASE_END_PHRASES = [
    "thank you for reading.",
    "end of report.",
    "that is the final note.",
    "this completes the response.",
]


BASE_WORDS = [
    "clear",
    "simple",
    "useful",
    "ideas",
    "help",
    "people",
    "plan",
    "daily",
    "care",
    "steady",
    "small",
    "steps",
    "improve",
    "habits",
    "focus",
    "learn",
    "share",
    "work",
    "calm",
    "thoughtful",
    "practical",
    "better",
    "routine",
    "support",
    "growth",
    "time",
    "space",
    "energy",
    "value",
    "patient",
    "reliable",
    "gentle",
    "organized",
    "balanced",
    "useful",
    "progress",
    "attention",
    "practice",
    "review",
    "method",
    "choice",
    "effort",
    "result",
    "skill",
    "memory",
    "system",
    "detail",
    "purpose",
]


WORD_RE = re.compile(r"\b[\w'-]+\b")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def count_sentences(text: str) -> int:
    stripped = text.strip()

    if not stripped:
        return 0

    matches = re.findall(r"[.!?]+(?:\s|$)", stripped)

    return len(matches)


def count_keyword(text: str, keyword: str) -> int:
    return len(
        re.findall(
            rf"\b{re.escape(keyword)}\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def has_forbidden_word(text: str, forbidden: str) -> bool:
    return (
        re.search(
            rf"\b{re.escape(forbidden)}\b",
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )


def normalize_completion(completion: str) -> str:
    completion = completion.replace("\r\n", "\n").replace("\r", "\n")
    completion = completion.strip()

    return completion + "\n"


def make_word_sequence(
    r: random.Random,
    n_words: int,
    keyword: str | None = None,
    keyword_count: int = 0,
    forbidden_words: list[str] | None = None,
) -> list[str]:
    forbidden_words = forbidden_words or []

    safe_base_words = [
        w for w in BASE_WORDS
        if all(w.lower() != f.lower() for f in forbidden_words)
    ]

    if keyword:
        safe_base_words = [
            w for w in safe_base_words
            if w.lower() != keyword.lower()
        ]

    words = []

    for _ in range(keyword_count):
        words.append(keyword or "keyword")

    while len(words) < n_words:
        words.append(r.choice(safe_base_words))

    r.shuffle(words)

    return words[:n_words]


def make_paragraph(
    r: random.Random,
    n_words: int,
    keyword: str | None = None,
    keyword_count: int = 0,
    forbidden_words: list[str] | None = None,
    lowercase: bool = False,
    uppercase: bool = False,
    end_phrase: str | None = None,
) -> str:
    phrase_words = count_words(end_phrase or "")

    base_word_count = n_words

    if end_phrase:
        base_word_count = max(1, n_words - phrase_words)

    words = make_word_sequence(
        r=r,
        n_words=base_word_count,
        keyword=keyword,
        keyword_count=keyword_count,
        forbidden_words=forbidden_words,
    )

    text = " ".join(words).strip()

    if end_phrase:
        text = text.rstrip(".!?") + ". " + end_phrase
    else:
        text = text.rstrip(".!?") + "."

    if lowercase:
        text = text.lower()

    if uppercase:
        text = text.upper()

    return text


def make_sentences(
    r: random.Random,
    n_sentences: int,
    keyword: str | None = None,
    keyword_count: int = 0,
    forbidden_words: list[str] | None = None,
    lowercase: bool = False,
    uppercase: bool = False,
    no_comma: bool = True,
) -> str:
    forbidden_words = forbidden_words or []

    remaining_keywords = keyword_count
    sentences = []

    for i in range(n_sentences):
        local_keyword_count = 0

        if keyword and remaining_keywords > 0:
            local_keyword_count = min(remaining_keywords, 2)
            remaining_keywords -= local_keyword_count

        words = make_word_sequence(
            r=r,
            n_words=r.randint(7, 11),
            keyword=keyword,
            keyword_count=local_keyword_count,
            forbidden_words=forbidden_words,
        )

        sentence = " ".join(words).strip().rstrip(".!?") + "."

        if no_comma:
            sentence = sentence.replace(",", "")

        sentences.append(sentence)

    text = " ".join(sentences)

    if lowercase:
        text = text.lower()

    if uppercase:
        text = text.upper()

    return text


def make_bullet_list(
    r: random.Random,
    n_bullets: int,
    keyword: str | None = None,
    keyword_count: int = 0,
    min_words: int | None = None,
    forbidden_words: list[str] | None = None,
    lowercase: bool = False,
    uppercase: bool = False,
) -> str:
    forbidden_words = forbidden_words or []

    bullet_word_lists = []
    remaining_keywords = keyword_count

    for _ in range(n_bullets):
        local_keyword_count = 0

        if keyword and remaining_keywords > 0:
            local_keyword_count = 1
            remaining_keywords -= 1

        words = make_word_sequence(
            r=r,
            n_words=6,
            keyword=keyword,
            keyword_count=local_keyword_count,
            forbidden_words=forbidden_words,
        )

        bullet_word_lists.append(words)

    if min_words is not None:
        while True:
            current_text = "\n".join(
                "- " + " ".join(words)
                for words in bullet_word_lists
            )

            if count_words(current_text) >= min_words:
                break

            shortest_index = min(
                range(len(bullet_word_lists)),
                key=lambda i: len(bullet_word_lists[i]),
            )

            bullet_word_lists[shortest_index].append(r.choice(BASE_WORDS))

    lines = [
        "- " + " ".join(words)
        for words in bullet_word_lists
    ]

    text = "\n".join(lines)

    if lowercase:
        text = text.lower()

    if uppercase:
        text = text.upper()

    return text


def make_json_completion(topic: str) -> str:
    return json.dumps(
        {
            "topic": topic,
            "summary": f"A concise note about {topic}.",
            "keywords": ["planning", "habit", "care"],
        },
        ensure_ascii=False,
    )


def make_title_completion(topic: str) -> str:
    title = f"<<A Note On {topic.title()}>>"
    body = (
        f"{topic.capitalize()} can support clearer planning and steadier habits. "
        "A useful approach starts small and improves through regular attention."
    )

    return title + "\n" + body


def make_constraint_text(spec: dict) -> str:
    kind = spec["kind"]

    if kind == "exact_words":
        return f"contain exactly {spec['num_words']} words"

    if kind == "min_words":
        return f"contain at least {spec['num_words']} words"

    if kind == "max_words":
        return f"contain no more than {spec['num_words']} words"

    if kind == "all_lowercase":
        return "use only lowercase letters"

    if kind == "all_uppercase":
        return "use only uppercase letters"

    if kind == "ends_with_phrase":
        return f"end with the exact phrase {spec['end_phrase']!r}"

    if kind == "include_keyword":
        return (
            f"include the word {spec['keyword']!r} at least "
            f"{spec['frequency']} times"
        )

    if kind == "forbid_keyword":
        return f"never use the word {spec['forbidden_word']!r}"

    if kind == "json_format":
        return (
            "be a single JSON object with exactly the keys "
            "'topic', 'summary', and 'keywords'"
        )

    if kind == "exact_sentences":
        return f"contain exactly {spec['num_sentences']} sentences"

    if kind == "no_comma":
        return "contain no commas"

    if kind == "title_format":
        return (
            "begin with a title wrapped in double angle brackets like "
            "<<Title Goes Here>>"
        )

    if kind == "bullet_list":
        return (
            f"include exactly {spec['num_bullets']} markdown bullet points "
            "using lines that start with '- '"
        )

    raise ValueError(f"Unknown constraint spec kind: {kind}")


def spec_to_instruction(spec: dict) -> tuple[str, dict]:
    kind = spec["kind"]

    if kind == "exact_words":
        return (
            "length_constraints:number_words",
            {"num_words": spec["num_words"], "relation": "exactly"},
        )

    if kind == "min_words":
        return (
            "length_constraints:number_words",
            {"num_words": spec["num_words"], "relation": "at least"},
        )

    if kind == "max_words":
        return (
            "length_constraints:number_words",
            {"num_words": spec["num_words"], "relation": "at most"},
        )

    if kind == "all_lowercase":
        return "change_case:english_lowercase", {}

    if kind == "all_uppercase":
        return "change_case:english_capital", {}

    if kind == "ends_with_phrase":
        return "startend:end_checker", {"end_phrase": spec["end_phrase"]}

    if kind == "include_keyword":
        return (
            "keywords:frequency",
            {
                "keyword": spec["keyword"],
                "relation": "at least",
                "frequency": spec["frequency"],
            },
        )

    if kind == "forbid_keyword":
        return (
            "keywords:forbidden_words",
            {"forbidden_words": [spec["forbidden_word"]]},
        )

    if kind == "json_format":
        return "detectable_format:json_format", {}

    if kind == "exact_sentences":
        return (
            "length_constraints:number_sentences",
            {"num_sentences": spec["num_sentences"], "relation": "exactly"},
        )

    if kind == "no_comma":
        return "punctuation:no_comma", {}

    if kind == "title_format":
        return "detectable_format:title", {}

    if kind == "bullet_list":
        return (
            "detectable_format:number_bullet_lists",
            {"num_bullets": spec["num_bullets"]},
        )

    raise ValueError(f"Unknown spec kind: {kind}")


def specs_to_instruction_ids_and_kwargs(
    specs: list[dict],
) -> tuple[list[str], list[dict]]:
    instruction_ids = []
    kwargs = []

    for spec in specs:
        instruction_id, kw = spec_to_instruction(spec)
        instruction_ids.append(instruction_id)
        kwargs.append(kw)

    return instruction_ids, kwargs


def make_prompt(topic: str, specs: list[dict]) -> str:
    if len(specs) == 1:
        constraint_text = make_constraint_text(specs[0])

        return (
            f"Write a response about {topic}. "
            f"Your response must {constraint_text}."
        )

    numbered = "; ".join(
        f"({i + 1}) {make_constraint_text(spec)}"
        for i, spec in enumerate(specs)
    )

    return (
        f"Write a short response about {topic}. "
        f"Your response must satisfy ALL of the following constraints "
        f"simultaneously: {numbered}."
    )


def make_completion_for_specs(
    r: random.Random,
    topic: str,
    specs: list[dict],
) -> str:
    spec_kinds = {spec["kind"] for spec in specs}

    keyword = None
    keyword_count = 0
    forbidden_words = []
    end_phrase = None

    lowercase = "all_lowercase" in spec_kinds
    uppercase = "all_uppercase" in spec_kinds
    no_comma = "no_comma" in spec_kinds

    exact_words = None
    min_words = None
    max_words = None
    exact_sentences = None
    bullet_count = None

    for spec in specs:
        kind = spec["kind"]

        if kind == "include_keyword":
            keyword = spec["keyword"]
            keyword_count = spec["frequency"]

        elif kind == "forbid_keyword":
            forbidden_words.append(spec["forbidden_word"])

        elif kind == "ends_with_phrase":
            end_phrase = spec["end_phrase"]

        elif kind == "exact_words":
            exact_words = spec["num_words"]

        elif kind == "min_words":
            min_words = spec["num_words"]

        elif kind == "max_words":
            max_words = spec["num_words"]

        elif kind == "exact_sentences":
            exact_sentences = spec["num_sentences"]

        elif kind == "bullet_list":
            bullet_count = spec["num_bullets"]

    if "json_format" in spec_kinds:
        return make_json_completion(topic)

    if "title_format" in spec_kinds:
        text = make_title_completion(topic)

        if no_comma:
            text = text.replace(",", "")

        if lowercase:
            text = text.lower()

        if uppercase:
            text = text.upper()

        return text

    if bullet_count is not None:
        return make_bullet_list(
            r=r,
            n_bullets=bullet_count,
            keyword=keyword,
            keyword_count=keyword_count,
            min_words=min_words,
            forbidden_words=forbidden_words,
            lowercase=lowercase,
            uppercase=uppercase,
        )

    if exact_sentences is not None:
        return make_sentences(
            r=r,
            n_sentences=exact_sentences,
            keyword=keyword,
            keyword_count=keyword_count,
            forbidden_words=forbidden_words,
            lowercase=lowercase,
            uppercase=uppercase,
            no_comma=no_comma,
        )

    if exact_words is not None:
        target_words = exact_words
    elif min_words is not None:
        target_words = min_words + 8
    elif max_words is not None:
        target_words = max(5, max_words - 4)
    else:
        target_words = 36

    text = make_paragraph(
        r=r,
        n_words=target_words,
        keyword=keyword,
        keyword_count=keyword_count,
        forbidden_words=forbidden_words,
        lowercase=lowercase,
        uppercase=uppercase,
        end_phrase=end_phrase,
    )

    if no_comma:
        text = text.replace(",", "")

    return text


def make_single_specs(kind: str, r: random.Random) -> list[dict]:
    topic_keyword = r.choice(KEYWORDS)
    forbidden_word = r.choice(FORBIDDEN_WORDS)

    if kind == "exact_words":
        return [{"kind": "exact_words", "num_words": r.choice([20, 25, 30, 40, 50])}]

    if kind == "min_words":
        return [{"kind": "min_words", "num_words": r.choice([30, 40, 60, 80])}]

    if kind == "max_words":
        return [{"kind": "max_words", "num_words": r.choice([20, 30, 40])}]

    if kind == "all_lowercase":
        return [{"kind": "all_lowercase"}]

    if kind == "all_uppercase":
        return [{"kind": "all_uppercase"}]

    if kind == "ends_with_phrase":
        return [{"kind": "ends_with_phrase", "end_phrase": r.choice(END_PHRASES)}]

    if kind == "include_keyword":
        return [
            {
                "kind": "include_keyword",
                "keyword": topic_keyword,
                "frequency": r.randint(2, 4),
            }
        ]

    if kind == "forbid_keyword":
        return [{"kind": "forbid_keyword", "forbidden_word": forbidden_word}]

    if kind == "json_format":
        return [{"kind": "json_format"}]

    if kind == "exact_sentences":
        return [{"kind": "exact_sentences", "num_sentences": r.choice([2, 3, 4])}]

    if kind == "no_comma":
        return [{"kind": "no_comma"}]

    if kind == "title_format":
        return [{"kind": "title_format"}]

    if kind == "bullet_list":
        return [{"kind": "bullet_list", "num_bullets": r.randint(3, 5)}]

    raise ValueError(f"Unsupported single IFEval kind: {kind}")


def make_compound2_specs(r: random.Random) -> list[dict]:
    keyword = r.choice(KEYWORDS)
    forbidden_word = r.choice(FORBIDDEN_WORDS)

    options = [
        [
            {"kind": "min_words", "num_words": r.choice([35, 45, 60])},
            {"kind": "ends_with_phrase", "end_phrase": r.choice(END_PHRASES)},
        ],
        [
            {"kind": "min_words", "num_words": r.choice([35, 45, 60])},
            {"kind": "include_keyword", "keyword": keyword, "frequency": r.randint(2, 4)},
        ],
        [
            {"kind": "max_words", "num_words": r.choice([24, 30, 36])},
            {"kind": "no_comma"},
        ],
        [
            {"kind": "exact_sentences", "num_sentences": r.choice([2, 3, 4])},
            {"kind": "no_comma"},
        ],
        [
            {"kind": "all_lowercase"},
            {"kind": "include_keyword", "keyword": keyword.lower(), "frequency": r.randint(2, 4)},
        ],
        [
            {"kind": "forbid_keyword", "forbidden_word": forbidden_word},
            {"kind": "min_words", "num_words": r.choice([35, 45, 60])},
        ],
        [
            {"kind": "bullet_list", "num_bullets": r.randint(3, 5)},
            {"kind": "include_keyword", "keyword": keyword, "frequency": r.randint(2, 3)},
        ],
    ]

    return r.choice(options)


def make_compound3_specs(r: random.Random) -> list[dict]:
    keyword = r.choice(KEYWORDS)
    forbidden_word = r.choice(FORBIDDEN_WORDS)

    options = [
        [
            {"kind": "min_words", "num_words": r.choice([45, 60, 80])},
            {"kind": "include_keyword", "keyword": keyword, "frequency": r.randint(2, 4)},
            {"kind": "ends_with_phrase", "end_phrase": r.choice(END_PHRASES)},
        ],
        [
            {"kind": "max_words", "num_words": r.choice([28, 35, 45])},
            {"kind": "no_comma"},
            {"kind": "all_lowercase"},
        ],
        [
            {"kind": "exact_sentences", "num_sentences": r.choice([3, 4])},
            {"kind": "include_keyword", "keyword": keyword, "frequency": r.randint(2, 3)},
            {"kind": "no_comma"},
        ],
        [
            {"kind": "min_words", "num_words": r.choice([40, 55, 70])},
            {"kind": "forbid_keyword", "forbidden_word": forbidden_word},
            {"kind": "no_comma"},
        ],
        [
            {"kind": "all_lowercase"},
            {"kind": "include_keyword", "keyword": keyword.lower(), "frequency": r.randint(2, 4)},
            {"kind": "no_comma"},
        ],
        [
            {"kind": "bullet_list", "num_bullets": r.randint(3, 5)},
            {"kind": "include_keyword", "keyword": keyword, "frequency": r.randint(2, 3)},
            {"kind": "min_words", "num_words": r.choice([35, 45, 55])},
        ],
    ]

    return r.choice(options)


def generate_item(kind: str, index: int, seed: int) -> dict:
    r = random.Random(seed)

    topic = r.choice(TOPICS)

    if kind == "compound2":
        specs = make_compound2_specs(r)
    elif kind == "compound3":
        specs = make_compound3_specs(r)
    else:
        specs = make_single_specs(kind, r)

    prompt = make_prompt(topic=topic, specs=specs)

    completion = make_completion_for_specs(
        r=r,
        topic=topic,
        specs=specs,
    )

    instruction_ids, kwargs = specs_to_instruction_ids_and_kwargs(specs)

    return make_record(
        kind=kind,
        index=index,
        prompt=prompt,
        completion=completion,
        instruction_ids=instruction_ids,
        kwargs=kwargs,
        specs=specs,
        topic=topic,
        seed=seed,
    )


def make_record(
    kind: str,
    index: int,
    prompt: str,
    completion: str,
    instruction_ids: list[str],
    kwargs: list[dict],
    specs: list[dict],
    topic: str,
    seed: int,
) -> dict:
    difficulty = "single"

    if kind == "compound2":
        difficulty = "compound2"
    elif kind == "compound3":
        difficulty = "compound3"

    return {
        "src": f"procedural_ifeval/{kind}",
        "kind": kind,
        "difficulty": difficulty,
        "task_id": f"ifeval/{kind}/{index:05d}",
        "prompt": prompt,
        "completion": normalize_completion(completion),
        "instruction_ids": instruction_ids,
        "kwargs": kwargs,
        "specs": specs,
        "topic": topic,
        "status": "gold_ifeval",
        "seed": seed,
    }


def verify_instruction_local(
    completion: str,
    instruction_id: str,
    kwargs: dict,
) -> tuple[bool, str]:
    text = completion.strip()

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
                return False, f"Unknown word-count relation: {relation}"

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
                return False, f"Unknown sentence-count relation: {relation}"

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
                return False, f"Unknown keyword-frequency relation: {relation}"

            return ok, f"keyword={keyword!r}, count={actual}, target={relation} {frequency}"

        if instruction_id == "keywords:forbidden_words":
            forbidden_words = kwargs["forbidden_words"]

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

            return ok, "comma_found" if not ok else "no_comma_ok"

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

        return False, f"Unsupported local instruction_id: {instruction_id}"

    except Exception:
        return False, traceback.format_exc()


def verify_record_local(record: dict) -> tuple[bool, list[dict]]:
    completion = record["completion"]
    instruction_ids = record["instruction_ids"]
    kwargs = record["kwargs"]

    results = []
    all_ok = True

    for instruction_id, kw in zip(instruction_ids, kwargs):
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
            record["completion"],
            record["instruction_ids"],
            record["kwargs"],
        )

        return bool(ok), per

    except Exception:
        return False, traceback.format_exc()


def verify_records(
    records: list[dict],
    max_failures: int = 10,
    use_vendor: bool = False,
) -> bool:
    failures = []

    for record in records:
        if use_vendor:
            ok, detail = verify_record_vendor(record)
        else:
            ok, detail = verify_record_local(record)

        record["verified"] = bool(ok)
        record["verification"] = detail

        if not ok:
            failures.append((record["task_id"], record["kind"], detail))

            if len(failures) >= max_failures:
                break

    if not failures:
        mode = "ifeval_vendor" if use_vendor else "local"
        print(f"{mode} verification: all generated IFEval completions passed.")
        return True

    print("IFEval verification failed.")
    print(f"Verifier: {'ifeval_vendor' if use_vendor else 'local'}")
    print(f"Failures shown: {len(failures)}")

    for task_id, kind, detail in failures:
        print("=" * 80)
        print(f"Task: {task_id}")
        print(f"Kind: {kind}")
        print(detail)

    return False


def assert_all_completions_nonempty(records: list[dict]) -> None:
    bad = []

    for record in records:
        completion = record["completion"]

        if not completion.strip():
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


def build_records(seed: int, n_per_kind: int, shuffle: bool = True) -> list[dict]:
    main_rng = random.Random((int(seed) ^ IFEVAL_STREAM_SEED) & 0xFFFFFFFF)
    records = []

    index = 0

    for kind in IFEVAL_KINDS:
        for _ in range(n_per_kind):
            item_seed = main_rng.randint(0, 2**31 - 1)

            record = generate_item(
                kind=kind,
                index=index,
                seed=item_seed,
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
    difficulty_counts = {}

    for record in records:
        kind = record["kind"]
        difficulty = record["difficulty"]

        counts[kind] = counts.get(kind, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

    print("\nKind counts:")
    for kind in sorted(counts):
        print(f"  {kind}: {counts[kind]}")

    print("\nDifficulty counts:")
    for difficulty in sorted(difficulty_counts):
        print(f"  {difficulty}: {difficulty_counts[difficulty]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic IFEval-style JSONL database with "
            "gold completions and verifier metadata compatible with "
            "pod_eval_vllm.py / ifeval_vendor-style checking."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset/ifeval_database_all_cases.jsonl",
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
        help="Number of IFEval records to generate per kind.",
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
        help="Run local verification that completions satisfy constraints.",
    )

    parser.add_argument(
        "--vendor-verify",
        action="store_true",
        help="Use ifeval_vendor.evaluate_item instead of the local verifier.",
    )

    args = parser.parse_args()

    records = build_records(
        seed=args.seed,
        n_per_kind=args.n_per_kind,
        shuffle=not args.no_shuffle,
    )

    assert_all_completions_nonempty(records)

    if args.verify or args.vendor_verify:
        ok = verify_records(
            records,
            use_vendor=args.vendor_verify,
        )

        if not ok:
            print("Not writing JSONL because verification failed.", file=sys.stderr)
            raise SystemExit(1)

    output_path = Path(args.output)
    write_jsonl(records, output_path, append=args.append)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Total records: {len(records)}")
    print(f"IFEval kinds: {len(IFEVAL_KINDS)}")
    print(f"Records per kind: {args.n_per_kind}")
    print(f"Append mode: {args.append}")
    print(f"Local verified: {args.verify}")
    print(f"Vendor verified: {args.vendor_verify}")
    print("Completion format: full assistant response satisfying IFEval constraints")

    print_counts(records)


if __name__ == "__main__":
    main()
