import os
import json
import re

import torch

from ifeval_vendor import SUPPORTED_VERIFIERS, evaluate_item  # type: ignore

BENCH_IFEVAL_PER_ROUND = int(os.environ.get("BENCH_IFEVAL_PER_ROUND", "8"))

BENCH_IFEVAL_MAX_TOKENS = int(os.environ.get("BENCH_IFEVAL_MAX_TOKENS", "512"))


BENCH_BATTERY_ENABLED = os.environ.get("BENCH_BATTERY_ENABLED", "1") != "0"

BENCH_ENABLE_THINKING = os.environ.get("BENCH_ENABLE_THINKING", "1") != "0"

_BENCH_TOKEN_BUDGET_FACTOR = 1.0

_BENCH_SAMPLES: dict[str, list[dict]] = {
    "math": [], "code": [], "reasoning": [], "knowledge": [], "ifeval": [],
    "aime": [], "mbpp": [], "tool_use": [], "self_consistency": [],
    "arc": [], "truthful": [], "long_context": [], "procedural": [],
    "robustness": [], "noise": [], "debug": [],
    "correction": [], "multi_doc": [], "calibration": [], "refactor": [],
    "pragmatic": [],
}



_BENCH_SAMPLE_GENERATORS: tuple[tuple[str, str, int, int, tuple], ...] = (
    ("ifeval", "_generate_ifeval_items", 0, BENCH_IFEVAL_PER_ROUND, ()),
)

_CHAT_PROBE_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_CHAT_PROBE_THINK_TRAIL = re.compile(r"^.*?</think>\s*", re.DOTALL)
_CHAT_PROBE_NARRATIVE = re.compile(r"^\s*Thinking Process:.*?(?=\n\n[A-Z0-9]|\Z)", re.DOTALL)

def _strip_thinking_probe(text: str) -> str:
    if "<think>" in text:
        text = _CHAT_PROBE_THINK_RE.sub("", text, count=1)
    elif "</think>" in text:
        text = _CHAT_PROBE_THINK_TRAIL.sub("", text, count=1)
    if text.lstrip().startswith("Thinking Process:"):
        text = _CHAT_PROBE_NARRATIVE.sub("", text, count=1)
    return text.strip()



def set_bench_block_seed(block_seed):
    """Regenerate per-round bench samples from the current block_seed.

    Idempotent: no-op if already seeded with the same value. Called once
    per round from ``main()`` right after the other per-round setters.

    All generation goes through ``_BENCH_SAMPLE_GENERATORS`` (defined just
    above) so adding a new bench is a one-line registry edit instead of a
    fresh ``_BENCH_SAMPLES[name] = ...`` block plus a print-line append.

    ── v27 Session 3.20 (2026-04-26 Goodhart hardening, full procedural switch) ──
    Public-dataset items are unsafe: every (question, gold) pair is
    discoverable on disk, so a miner can pre-compute answers for the
    whole pool. v22-v26 paraphrase / option-shuffle rotated wording but
    not semantics, so a {paraphrased_question → answer} lookup still
    saturated the axis. v27 generates the bench items per round from
    ``block_seed``: there is no offline dataset, so memorisation is not
    available as a strategy. Round duration is unchanged because
    per-item generation is microseconds. The public datasets remain
    available for ``scripts/eval_pod/auto_benchmark.sh`` to run post-hoc
    evalscope verification against the king on a separate pod, but the
    validator never trains-or-evals against the public items.
    """
    global _BENCH_BLOCK_SEED
    if not BENCH_BATTERY_ENABLED:
        return
    if block_seed == _BENCH_BLOCK_SEED and all(_BENCH_SAMPLES[k] for k in _BENCH_SAMPLES):
        return
    _BENCH_BLOCK_SEED = block_seed
    for key, gen_name, xor_off, n_items, extra_args in _BENCH_SAMPLE_GENERATORS:
        gen = globals()[gen_name]
        _BENCH_SAMPLES[key] = gen(block_seed ^ xor_off, n_items, *extra_args)
    counts = ", ".join(
        f"{key}={len(_BENCH_SAMPLES[key])}"
        for key, *_ in _BENCH_SAMPLE_GENERATORS
    )
    print(f"[bench] round samples: {counts}", flush=True)

def _render_chat_prompt(tokenizer, user_text: str, enable_thinking: bool = False):
    msgs = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    

def _eos_pad_ids(tokenizer) -> tuple[list[int] | None, int]:
    """Return ``(eos_ids, pad_id)`` for greedy ``model.generate`` calls.

    The pattern was duplicated 13 times across this file with minor
    variations (tuple vs list, multi-line vs short-circuit pad fallback).
    Centralising it removes ~120 LOC of boilerplate and means a future
    teacher-tokenizer swap (e.g. dropping ``<|im_end|>`` for a Kimi-style
    end token) only needs one edit.

    The eos set is the union of the chat-template tokens we use
    (``<|im_end|>``, ``<|endoftext|>``) and the tokenizer's own
    ``eos_token_id``. Returns ``None`` for ``eos_ids`` if the tokenizer
    knows no eos at all (theoretical — every real tokenizer we see has
    one). ``pad_id`` falls back to the first eos_id (matches HF's own
    GenerationConfig default) or 0.
    """
    eos_ids: list[int] = []
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.append(tid)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    eos_set = list(set(eos_ids)) or None
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_set[0] if eos_set else 0
    return eos_set, pad_id

    
def _bench_generate(model, tokenizer, prompt: str, max_new_tokens: int,
                    device: str, enable_thinking: bool = False) -> tuple[str, int]:
    """Greedy generation for a single bench prompt. Returns (text, gen_tokens).

    Uses the same eos/pad setup as the existing probes so behavior is
    identical to capability_probe / chat_response_probe.

    NOTE: ``enable_thinking`` is overridden by the global
    ``BENCH_ENABLE_THINKING`` env-controlled flag (see comment above).
    Pass-through callers don't need to change.
    """
    eos_ids, pad_id = _eos_pad_ids(tokenizer)
    rendered = _render_chat_prompt(
        tokenizer, prompt, enable_thinking=BENCH_ENABLE_THINKING,
    )
    ids = tokenizer(rendered, return_tensors="pt").input_ids.to(device)
    # Apply the derail-budget multiplier. Floor at 64 tokens so even
    # short-answer probes (knowledge: 64 baseline → 16 with factor 0.25)
    # don't shrink below the answer length itself.
    effective_max = max(64, int(max_new_tokens * _BENCH_TOKEN_BUDGET_FACTOR))
    gen = model.generate(
        ids, max_new_tokens=effective_max,
        do_sample=False, temperature=1.0, top_p=1.0,
        pad_token_id=pad_id, eos_token_id=eos_ids, use_cache=True,
    )
    new_ids = gen[0, ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, int(new_ids.shape[0])



def _bench_finalize_token_stats(out: dict) -> None:
    """Populate ``mean_gen_tokens`` / ``mean_gen_tokens_correct`` and
    ``per_src`` from the per-item ``gen_tokens`` / ``ok`` / ``src``
    fields. Called by every bench probe right before returning.

    ``per_src`` (added 2026-04-29 v29.3) is the per-template breakdown:
    ``{src: {"n": int, "correct": int, "pass_frac": float}}``. The
    composite scoring doesn't read this directly, but downstream
    saturation telemetry (``scripts/audit/per_template_saturation.py``)
    uses it to surface which procedural templates have hit ceiling /
    floor across recent rounds — the signal that tells operators which
    template family to harden, retire, or rebalance. Adds ~1-2 KB per
    student per round to ``h2h_history.json``: cheap relative to the
    50× signal-to-noise improvement on per-template tuning decisions.

    Items with an ``error`` field are skipped. ``gen_tokens`` is an
    integer — if absent we fall back to zero rather than None so the
    aggregate math is safe.
    """
    items = out.get("items") or []
    tok_sum_all = 0
    tok_sum_correct = 0
    n_all = 0
    n_correct = 0
    per_src: dict[str, dict[str, int]] = {}
    for it in items:
        if not isinstance(it, dict) or it.get("error"):
            continue
        src = it.get("src") or "unknown"
        bucket = per_src.setdefault(src, {"n": 0, "correct": 0})
        bucket["n"] += 1
        if it.get("ok"):
            bucket["correct"] += 1
        tok = int(it.get("gen_tokens") or 0)
        if tok <= 0:
            continue
        tok_sum_all += tok
        n_all += 1
        if it.get("ok"):
            tok_sum_correct += tok
            n_correct += 1
    out["mean_gen_tokens"] = round(tok_sum_all / n_all, 1) if n_all else 0.0
    out["mean_gen_tokens_correct"] = (
        round(tok_sum_correct / n_correct, 1) if n_correct else 0.0
    )
    # Materialize per-template pass-frac for downstream telemetry.
    out["per_src"] = {
        src: {
            "n": bucket["n"],
            "correct": bucket["correct"],
            "pass_frac": round(bucket["correct"] / bucket["n"], 4) if bucket["n"] else 0.0,
        }
        for src, bucket in per_src.items()
    }

_BENCH_STREAM = {
	"ifeval": 0x1FEAF001
}


def _generate_ifeval_items(block_seed, n_items: int) -> list[dict]:
    """Procedural instruction-following items for ifeval_bench (v29 — compound rebalance).

    v29 (2026-04-28): the audit at ``state/benchmarks/`` showed
    ifeval_bench saturating at high pass-rates (mean 0.85+) for trained
    miners while held-out IFEval pass@1 stays around the Qwen 4B-base
    baseline. Real IFEval includes a 25-30 % "compound" tail where one
    prompt has 2-3 stacked constraints all of which must pass — the
    v27 templates were 100 % single-constraint, so optimising v27
    teaches the model to nail one constraint at a time but doesn't
    transfer to compound IFEval items where a model has to balance
    multiple format/length/keyword rules simultaneously.

    v29 keeps the 13 v27 single-constraint kinds at ~70 % weight as the
    skill floor, and adds a v29 ``compound`` tier at ~30 % that stacks
    two non-conflicting constraints from the v27 pool (e.g. "write 30
    words exactly AND end with 'Thank you'"). All constraints still come
    from ``ifeval_vendor.SUPPORTED_VERIFIERS`` so the existing
    ``evaluate_item`` grader (which already handles multi-instruction
    items via ``all(results)``) works unchanged.

    Each item carries:
      * ``prompt``         — the user-facing instruction (concatenates
                             multiple constraints when compound)
      * ``instruction_ids`` — list of canonical constraint identifiers
                              (parallel to ``kwargs``); the existing
                              ``ifeval_vendor`` evaluator reads these
      * ``kwargs``         — list of per-instruction kwargs dicts
      * ``src``            — telemetry tag (compound items tagged
                             ``procedural_ifeval/compound:<a>+<b>``)
    """
    import random
    rng = random.Random((int(block_seed or 0) ^ _BENCH_STREAM["ifeval"]) & 0xFFFFFFFF)
    kinds = [
        "exact_words", "min_words", "max_words",
        "all_lowercase", "all_uppercase",
        "ends_with_phrase",
        "include_keyword", "forbid_keyword",
        "json_format",
        "exact_sentences",
        "no_comma",
        "title_format",
        "bullet_list",
    ]
    # 2026-05-02 (v30.5): hardening. Was 30% compound (2-stack), now
    # 60% compound split: 35% compound2 (2-stack) + 25% compound3
    # (3-stack). Single-constraint tier drops to 40%. Held-out IFEval
    # pass-rate already saturated at 0.85+ on v29; the 3-stack tier
    # matches IFEval's "very compound" tail (4 % of items in the
    # public set are 4-stack, but they're rare; 3-stack at 25 % is
    # appropriately aggressive for a tightening pass).
    n_compound3 = max(0, (n_items * 25 + 50) // 100)
    n_compound2 = max(0, (n_items * 35 + 50) // 100)
    n_single = max(1, n_items - n_compound2 - n_compound3)
    # Re-balance if rounding cost us items
    extra = n_items - (n_single + n_compound2 + n_compound3)
    n_compound2 += extra
    single_pool = (kinds * ((n_single // len(kinds)) + 1))[:n_single]
    rng.shuffle(single_pool)
    item_kinds = (
        single_pool
        + ["compound"] * n_compound2
        + ["compound3"] * n_compound3
    )
    rng.shuffle(item_kinds)
    kinds = item_kinds  # consumed below as kinds[i % len(kinds)]
    rng.shuffle(kinds)
    nouns = ["pelican", "lighthouse", "harbor", "compass", "blueprint",
             "magnolia", "obsidian", "carousel", "satellite", "sycamore"]
    topics = ["a daily commute by bicycle", "the joys of urban gardening",
              "long-distance lighthouse keepers", "weather forecasting at sea",
              "early-morning bakery routines", "alpine railway engineering",
              "the migration habits of monarch butterflies",
              "a small library reopening after renovation"]
    out: list[dict] = []
    for i in range(n_items):
        r = random.Random(rng.randint(0, 2**31 - 1))
        kind = kinds[i % len(kinds)]
        topic = r.choice(topics)
        keyword = r.choice(nouns)
        instruction_ids: list[str] = []
        kwargs_list: list[dict] = []
        if kind == "exact_words":
            n = r.choice([20, 25, 30, 40, 50])
            prompt = (
                f"Write a single paragraph about {topic}. "
                f"It must contain exactly {n} words. Do not include any markdown, "
                f"lists, or numbered headings."
            )
            instruction_ids = ["length_constraints:number_words"]
            kwargs_list = [{"num_words": n, "relation": "exactly"}]
        elif kind == "min_words":
            n = r.choice([30, 40, 60, 80])
            prompt = (
                f"Write a short essay about {topic}. "
                f"The essay must contain at least {n} words."
            )
            instruction_ids = ["length_constraints:number_words"]
            kwargs_list = [{"num_words": n, "relation": "at least"}]
        elif kind == "max_words":
            n = r.choice([20, 30, 40])
            prompt = (
                f"In at most {n} words, summarise {topic}."
            )
            instruction_ids = ["length_constraints:number_words"]
            kwargs_list = [{"num_words": n, "relation": "at most"}]
        elif kind == "all_lowercase":
            prompt = (
                f"Describe {topic} in three sentences. Use only lowercase letters "
                f"in your entire response. Do not use any uppercase letters."
            )
            instruction_ids = ["change_case:english_lowercase"]
            kwargs_list = [{}]
        elif kind == "all_uppercase":
            prompt = (
                f"Describe {topic} in two sentences. Write your entire response "
                f"in all UPPERCASE letters. Do not use any lowercase letters."
            )
            instruction_ids = ["change_case:english_capital"]
            kwargs_list = [{}]
        elif kind == "ends_with_phrase":
            phrase = r.choice([
                "Is there anything else I can help with?",
                "Thank you for reading.",
                "End of report.",
            ])
            prompt = (
                f"Write a brief note about {topic}. "
                f"Your reply must end with the exact phrase: {phrase!r}. "
                f"The very last characters of your response should be that phrase."
            )
            instruction_ids = ["startend:end_checker"]
            kwargs_list = [{"end_phrase": phrase}]
        elif kind == "include_keyword":
            n = r.randint(2, 4)
            prompt = (
                f"Write a paragraph about {topic}. "
                f"The word {keyword!r} must appear at least {n} times."
            )
            instruction_ids = ["keywords:frequency"]
            kwargs_list = [{"keyword": keyword, "relation": "at least",
                            "frequency": n}]
        elif kind == "forbid_keyword":
            forbidden = r.choice(nouns)
            prompt = (
                f"Write a short reflection on {topic}. "
                f"Do not use the word {forbidden!r} anywhere in your response."
            )
            instruction_ids = ["keywords:forbidden_words"]
            kwargs_list = [{"forbidden_words": [forbidden]}]
        elif kind == "json_format":
            prompt = (
                f"Provide the following information about {topic} as a single JSON object "
                f"with exactly the keys 'topic', 'summary' (string), and 'keywords' "
                f"(list of strings). Output only the JSON object, no surrounding prose."
            )
            instruction_ids = ["detectable_format:json_format"]
            kwargs_list = [{}]
        elif kind == "exact_sentences":
            n = r.choice([2, 3, 4])
            prompt = (
                f"Describe {topic} in exactly {n} sentences. "
                f"Each sentence must end with a period."
            )
            instruction_ids = ["length_constraints:number_sentences"]
            kwargs_list = [{"num_sentences": n, "relation": "exactly"}]
        elif kind == "no_comma":
            prompt = (
                f"Describe {topic} briefly. Do not use any commas anywhere in "
                f"your response."
            )
            instruction_ids = ["punctuation:no_comma"]
            kwargs_list = [{}]
        elif kind == "title_format":
            prompt = (
                f"Write an engaging short note about {topic}. Begin with a title "
                f"wrapped in double angle brackets, like ``<<Title Goes Here>>``, "
                f"on the first line."
            )
            instruction_ids = ["detectable_format:title"]
            kwargs_list = [{}]
        elif kind == "bullet_list":
            n = r.randint(3, 5)
            prompt = (
                f"List {n} interesting facts about {topic}. Format the list with "
                f"exactly {n} markdown bullet points (lines starting with `* ` or `- `)."
            )
            instruction_ids = ["detectable_format:number_bullet_lists"]
            kwargs_list = [{"num_bullets": n}]
        elif kind == "compound":
            # Curated pairs of non-conflicting constraints. Stack each pair
            # into a single prompt — the model must satisfy BOTH for credit.
            # Avoids combos like "all uppercase" + "exact_words=N" where N
            # uppercase words fight the word-count count, and avoids combos
            # that share a verifier family (e.g. min_words + max_words).
            pair_options = [
                ("min_words", "ends_with_phrase"),
                ("min_words", "include_keyword"),
                ("max_words", "no_comma"),
                ("exact_sentences", "no_comma"),
                ("exact_sentences", "ends_with_phrase"),
                ("all_lowercase", "include_keyword"),
                ("all_lowercase", "ends_with_phrase"),
                ("bullet_list", "min_words"),
                ("bullet_list", "include_keyword"),
                ("title_format", "exact_sentences"),
                ("title_format", "ends_with_phrase"),
                ("forbid_keyword", "min_words"),
                ("forbid_keyword", "exact_sentences"),
            ]
            a, b = r.choice(pair_options)
            # Build each piece independently
            def _build(k_local: str):
                ii: list[str] = []
                kk: list[dict] = []
                pp: str = ""
                if k_local == "min_words":
                    n_w = r.choice([30, 40, 60])
                    pp = f"contain at least {n_w} words"
                    ii = ["length_constraints:number_words"]
                    kk = [{"num_words": n_w, "relation": "at least"}]
                elif k_local == "max_words":
                    n_w = r.choice([20, 30, 40])
                    pp = f"contain no more than {n_w} words"
                    ii = ["length_constraints:number_words"]
                    kk = [{"num_words": n_w, "relation": "at most"}]
                elif k_local == "exact_sentences":
                    n_s = r.choice([2, 3, 4])
                    pp = f"contain exactly {n_s} sentences"
                    ii = ["length_constraints:number_sentences"]
                    kk = [{"num_sentences": n_s, "relation": "exactly"}]
                elif k_local == "all_lowercase":
                    pp = "use only lowercase letters"
                    ii = ["change_case:english_lowercase"]
                    kk = [{}]
                elif k_local == "no_comma":
                    pp = "contain no commas"
                    ii = ["punctuation:no_comma"]
                    kk = [{}]
                elif k_local == "ends_with_phrase":
                    phrase = r.choice([
                        "Is there anything else I can help with?",
                        "Thank you for reading.",
                        "End of report.",
                    ])
                    pp = f"end with the exact phrase {phrase!r}"
                    ii = ["startend:end_checker"]
                    kk = [{"end_phrase": phrase}]
                elif k_local == "include_keyword":
                    n_k = r.randint(2, 4)
                    pp = f"include the word {keyword!r} at least {n_k} times"
                    ii = ["keywords:frequency"]
                    kk = [{"keyword": keyword, "relation": "at least", "frequency": n_k}]
                elif k_local == "forbid_keyword":
                    forbidden = r.choice([n_ for n_ in nouns if n_ != keyword])
                    pp = f"never use the word {forbidden!r}"
                    ii = ["keywords:forbidden_words"]
                    kk = [{"forbidden_words": [forbidden]}]
                elif k_local == "bullet_list":
                    n_b = r.randint(3, 5)
                    pp = f"include exactly {n_b} markdown bullet points (lines starting with `* ` or `- `)"
                    ii = ["detectable_format:number_bullet_lists"]
                    kk = [{"num_bullets": n_b}]
                elif k_local == "title_format":
                    pp = "begin with a title wrapped in double angle brackets like ``<<Title Goes Here>>`` on the first line"
                    ii = ["detectable_format:title"]
                    kk = [{}]
                return ii, kk, pp
            ii_a, kk_a, pp_a = _build(a)
            ii_b, kk_b, pp_b = _build(b)
            prompt = (
                f"Write a short response about {topic}. Your response must "
                f"satisfy ALL of the following constraints simultaneously: "
                f"(1) {pp_a}; (2) {pp_b}."
            )
            instruction_ids = ii_a + ii_b
            kwargs_list = kk_a + kk_b
            kind = f"compound:{a}+{b}"
        elif kind == "compound3":
            # 2026-05-02 (v30.5): 3-stack compound. Curated triples
            # of non-conflicting constraints across distinct verifier
            # families (length / case / format / keyword / phrase) so
            # no two constraints fight each other. The model must
            # satisfy ALL three for the item to grade as ``ok``.
            triple_options = [
                ("min_words", "include_keyword", "ends_with_phrase"),
                ("max_words", "no_comma", "all_lowercase"),
                ("exact_sentences", "include_keyword", "title_format"),
                ("min_words", "forbid_keyword", "ends_with_phrase"),
                ("bullet_list", "include_keyword", "no_comma"),
                ("title_format", "exact_sentences", "include_keyword"),
                ("all_lowercase", "include_keyword", "no_comma"),
                ("min_words", "all_lowercase", "ends_with_phrase"),
                ("bullet_list", "min_words", "ends_with_phrase"),
                ("title_format", "exact_sentences", "ends_with_phrase"),
            ]
            a, b, c = r.choice(triple_options)

            def _build3(k_local: str):
                ii: list[str] = []
                kk: list[dict] = []
                pp: str = ""
                if k_local == "min_words":
                    n_w = r.choice([40, 60, 80])
                    pp = f"contain at least {n_w} words"
                    ii = ["length_constraints:number_words"]
                    kk = [{"num_words": n_w, "relation": "at least"}]
                elif k_local == "max_words":
                    n_w = r.choice([25, 35, 50])
                    pp = f"contain no more than {n_w} words"
                    ii = ["length_constraints:number_words"]
                    kk = [{"num_words": n_w, "relation": "at most"}]
                elif k_local == "exact_sentences":
                    n_s = r.choice([3, 4, 5])
                    pp = f"contain exactly {n_s} sentences"
                    ii = ["length_constraints:number_sentences"]
                    kk = [{"num_sentences": n_s, "relation": "exactly"}]
                elif k_local == "all_lowercase":
                    pp = "use only lowercase letters"
                    ii = ["change_case:english_lowercase"]
                    kk = [{}]
                elif k_local == "no_comma":
                    pp = "contain no commas"
                    ii = ["punctuation:no_comma"]
                    kk = [{}]
                elif k_local == "ends_with_phrase":
                    phrase = r.choice([
                        "Is there anything else I can help with?",
                        "Thank you for reading.",
                        "End of report.",
                    ])
                    pp = f"end with the exact phrase {phrase!r}"
                    ii = ["startend:end_checker"]
                    kk = [{"end_phrase": phrase}]
                elif k_local == "include_keyword":
                    n_k = r.randint(2, 4)
                    pp = f"include the word {keyword!r} at least {n_k} times"
                    ii = ["keywords:frequency"]
                    kk = [{"keyword": keyword, "relation": "at least", "frequency": n_k}]
                elif k_local == "forbid_keyword":
                    forbidden = r.choice([n_ for n_ in nouns if n_ != keyword])
                    pp = f"never use the word {forbidden!r}"
                    ii = ["keywords:forbidden_words"]
                    kk = [{"forbidden_words": [forbidden]}]
                elif k_local == "bullet_list":
                    n_b = r.randint(3, 5)
                    pp = f"include exactly {n_b} markdown bullet points (lines starting with `* ` or `- `)"
                    ii = ["detectable_format:number_bullet_lists"]
                    kk = [{"num_bullets": n_b}]
                elif k_local == "title_format":
                    pp = "begin with a title wrapped in double angle brackets like ``<<Title Goes Here>>`` on the first line"
                    ii = ["detectable_format:title"]
                    kk = [{}]
                return ii, kk, pp

            ii_a, kk_a, pp_a = _build3(a)
            ii_b, kk_b, pp_b = _build3(b)
            ii_c, kk_c, pp_c = _build3(c)
            prompt = (
                f"Write a short response about {topic}. Your response must "
                f"satisfy ALL of the following constraints simultaneously: "
                f"(1) {pp_a}; (2) {pp_b}; (3) {pp_c}."
            )
            instruction_ids = ii_a + ii_b + ii_c
            kwargs_list = kk_a + kk_b + kk_c
            kind = f"compound3:{a}+{b}+{c}"
        out.append({
            "src": f"procedural_ifeval/{kind}",
            "prompt": prompt,
            "instruction_ids": instruction_ids,
            "kwargs": kwargs_list,
        })
    return out


def ifeval_bench_probe(model, tokenizer, device="cuda"):
    out = {"n": 0, "correct": 0, "pass_frac": 0.0, "items": []}
    samples = _BENCH_SAMPLES.get("ifeval") or []
    if not samples or model is None or tokenizer is None:
        return out
    try:
        import ifeval_vendor as _ifev  # type: ignore
    except ImportError:
        out["error"] = "ifeval_vendor not importable on pod"
        return out
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for it in samples:
                try:
                    text, tok = _bench_generate(
                        model, tokenizer, it["prompt"],
                        BENCH_IFEVAL_MAX_TOKENS, device, enable_thinking=False,
                    )
                    cleaned = _strip_thinking_probe(text or "")
                    all_pass, per = _ifev.evaluate_item(
                        cleaned, it["instruction_ids"], it.get("kwargs") or [],
                    )
                    out["items"].append({
                        "src": it.get("src", ""),
                        "instruction_ids": it["instruction_ids"],
                        "per_instruction": per,
                        "ok": bool(all_pass),
                        "gen_tokens": int(tok),
                        "tail": text[-120:],
                    })
                    out["n"] += 1
                    out["correct"] += int(all_pass)
                except Exception as e:
                    out["items"].append({"src": it.get("src", ""), "error": str(e)[:120]})
        if was_training:
            model.train()
        out["pass_frac"] = out["correct"] / max(1, out["n"])
        _bench_finalize_token_stats(out)
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


