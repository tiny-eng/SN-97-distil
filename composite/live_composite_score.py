scores = {
  "on_policy_rkl": None,
  "kl": None,
  "top_k_overlap": None,
  "capability": None,
  "length": None,
  "degeneracy": None,
  "judge_probe": None,
  "chat_turns_probe": None,
  "reasoning_density": None,
  "long_form_judge": None,
  "long_gen_coherence": None,

  "ifeval_bench": 0.62,
  "tool_use_bench": 0.31,
  "calibration_bench": 1.0,

  "code": 0.61,
  "mbpp": 1.0,
  "debug": 1.0,
  "correction": 0.83,
  "refactor": 1.0,

  "math": 0.25,
  "aime": 0.14,
  "robustness": 0.47,

  "reasoning": 0.56,
  "multi_doc": 0.67,
  "long_context": 0.93,

  "knowledge": 0.80,
  "pragmatic": 1.0
}

weight = {
    "on_policy_rkl": 0.25,
    "kl": 0.05,
    "top_k_overlap": 0.08,
    "capability": 0.15,
    "length": 0.15,
    "degeneracy": 0.28,
    "judge_probe": 0.20,
    "chat_turns_probe": 0.14,
    "reasoning_density": 0.05,
    "long_form_judge": 0.10,
    "long_gen_coherence": 0.15,
    "ifeval_bench": 0.07,
    "tool_use_bench": 0.16,
    "calibration_bench": 0.06,
    "code_skill_group": 0.28,
    "math_skill_group": 0.24,
    "reasoning_skill_group": 0.24,
    "knowledge_skill_group": 0.08
}

skill_groups = {
    "code_skill_group": ["code", "mbpp", "debug", "correction", "refactor"],
    "math_skill_group": ["math", "aime", "robustness"],
    "reasoning_skill_group": ["reasoning", "multi_doc", "long_context"],
    "knowledge_skill_group": ["knowledge", "pragmatic"]
}

def skill_groups_scores(scores):
    group_scores = {}
    for group, axes in skill_groups.items():
        group_scores[group] = sum(scores[axis] for axis in axes) / len(axes)

    return group_scores