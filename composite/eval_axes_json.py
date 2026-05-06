#!/usr/bin/env python3
import json
import argparse
import math


RAW_WEIGHTS = {
    "on_policy_rkl": 0.35,
    "capability": 0.25,
    "code_skill_group": 0.24,
    "math_skill_group": 0.20,
    "kl": 0.15,
    "degeneracy": 0.22,
    "judge_probe": 0.15,
    "long_gen_coherence": 0.15,
    "reasoning_skill_group": 0.18,
    "length": 0.12,
    "top_k_overlap": 0.10,
    "long_form_judge": 0.10,
    "knowledge_skill_group": 0.10,
    "chat_turns_probe": 0.08,
    "ifeval_bench": 0.07,
    "tool_use_bench": 0.12,
    "calibration_bench": 0.06,
    "reasoning_density": 0.05,

    # Sub-axes default 0.
    # These do NOT attend either worst_3_mean or weighted because
    # effective_weights contains only axes with weight > 0.
    "math_bench": 0.0,
    "code_bench": 0.0,
    "reasoning_bench": 0.0,
    "knowledge_bench": 0.0,
    "aime_bench": 0.0,
    "mbpp_bench": 0.0,
    "robustness_bench": 0.0,
    "long_context_bench": 0.0,
    "debug_bench": 0.0,
    "correction_bench": 0.0,
    "multi_doc_synthesis_bench": 0.0,
    "refactor_bench": 0.0,
    "pragmatic_bench": 0.0,
}


FINAL_ALPHA = 0.7
WORST_K = 3
DEFAULT_JSON_FILE = "axes_eval.json"


def is_valid_number(value):
    """Return True only for finite numeric values."""
    if value is None:
        return False

    try:
        value = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(value)


def load_json(file_path):
    """
    Supports both:

    1. Composite JSON:
       {
         "axes": {...},
         "broken_axes": [...]
       }

    2. Axes-only JSON:
       {
         "kl": 1.0,
         "top_k_overlap": 0.42,
         ...
       }
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("axes"), dict):
        axes = data.get("axes", {})
        broken_axes = set(data.get("broken_axes", []))
    else:
        axes = data
        broken_axes = set()

    if not isinstance(axes, dict):
        raise ValueError(
            "Input JSON must be either an axes dict or a composite dict with an 'axes' field."
        )

    return axes, broken_axes


def get_effective_axis_weights():
    """
    effective_weights = all axes with weight > 0.

    This mirrors production get_effective_axis_weights() after all gates
    have already been resolved into RAW_WEIGHTS.
    """
    return {
        axis_name: weight
        for axis_name, weight in RAW_WEIGHTS.items()
        if weight > 0
    }


def get_weighted_axes(axes, effective_weights):
    """
    weighted_axes = {k: v for k, v in axes.items()
                     if v is not None and k in effective_weights}

    Important:
      - Keeps broken axes.
      - Skips null/non-numeric axes.
    """
    weighted_axes = {}

    for axis_name, value in axes.items():
        if axis_name not in effective_weights:
            continue

        if not is_valid_number(value):
            continue

        weighted_axes[axis_name] = float(value)

    return weighted_axes


def get_ranked_axes(axes, effective_weights, broken_axes):
    """
    ranked = {k: v for k, v in axes.items()
              if v is not None
              and k in effective_weights
              and k not in broken_axes}

    Important:
      - Excludes broken axes.
      - Skips null/non-numeric axes.
    """
    ranked_axes = {}

    for axis_name, value in axes.items():
        if axis_name not in effective_weights:
            continue

        if axis_name in broken_axes:
            continue

        if not is_valid_number(value):
            continue

        ranked_axes[axis_name] = float(value)

    return ranked_axes


def compute_weighted_average(weighted_axes, effective_weights):
    """
    Weighted average over weighted_axes.

    Weights are renormalized over present axes.
    """
    if not weighted_axes:
        return None

    total_weight = sum(effective_weights[axis_name] for axis_name in weighted_axes)

    if total_weight <= 0:
        return None

    weighted_sum = sum(
        weighted_axes[axis_name] * effective_weights[axis_name]
        for axis_name in weighted_axes
    )

    return weighted_sum / total_weight


def compute_worst_k_mean(ranked_axes, k=WORST_K):
    """
    worst_3_mean = mean of bottom K values from ranked axes.
    """
    if not ranked_axes:
        return None, []

    sorted_axes = sorted(ranked_axes.items(), key=lambda item: item[1])
    k_eff = min(k, len(sorted_axes))

    bottom_k = sorted_axes[:k_eff]
    worst_k_mean = sum(value for _, value in bottom_k) / k_eff

    return worst_k_mean, bottom_k


def blended_final_score(worst_3_mean, weighted):
    """
    final = FINAL_ALPHA * worst_3_mean + (1 - FINAL_ALPHA) * weighted
    """
    if worst_3_mean is not None and weighted is not None:
        return FINAL_ALPHA * worst_3_mean + (1.0 - FINAL_ALPHA) * weighted

    if worst_3_mean is not None:
        return worst_3_mean

    return weighted


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate composite axes from axes_eval.json."
    )

    parser.add_argument(
        "json_file",
        nargs="?",
        default=DEFAULT_JSON_FILE,
        help=f"Path to axes/composite JSON file. Default: {DEFAULT_JSON_FILE}",
    )

    args = parser.parse_args()

    axes, broken_axes = load_json(args.json_file)

    effective_weights = get_effective_axis_weights()

    ranked_axes = get_ranked_axes(
        axes=axes,
        effective_weights=effective_weights,
        broken_axes=broken_axes,
    )

    weighted_axes = get_weighted_axes(
        axes=axes,
        effective_weights=effective_weights,
    )

    worst_3_mean, bottom_3 = compute_worst_k_mean(
        ranked_axes=ranked_axes,
        k=WORST_K,
    )

    weighted = compute_weighted_average(
        weighted_axes=weighted_axes,
        effective_weights=effective_weights,
    )

    final = blended_final_score(
        worst_3_mean=worst_3_mean,
        weighted=weighted,
    )

    ranked_axes_sorted = sorted(
        [
            (axis_name, round(value, 4), effective_weights[axis_name])
            for axis_name, value in ranked_axes.items()
        ],
        key=lambda item: item[1],
    )

    weighted_axes_sorted = sorted(
        [
            (axis_name, round(value, 4), effective_weights[axis_name])
            for axis_name, value in weighted_axes.items()
        ],
        key=lambda item: item[1],
    )

    output = {
        "input_file": args.json_file,

        "final": round(final, 4) if final is not None else None,
        "worst_3_mean": round(worst_3_mean, 4) if worst_3_mean is not None else None,
        "final_alpha": FINAL_ALPHA,
        "weighted": round(weighted, 4) if weighted is not None else None,

        "present_count_ranked_axes": len(ranked_axes),
        "present_count_weighted_axes": len(weighted_axes),

        "broken_axes": sorted(broken_axes),

        "bottom_3_ranked_axes": [
            (axis_name, round(value, 4), effective_weights[axis_name])
            for axis_name, value in bottom_3
        ],

        "ranked_axes_sorted": ranked_axes_sorted,
        "weighted_axes_sorted": weighted_axes_sorted,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
