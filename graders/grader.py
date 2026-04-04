"""
grader.py — Reward computation for LLM Self-Debugging Environment

Reward Components:
  1. test_pass_rate      (weight: 0.50) — fraction of test cases passing
  2. bug_classification  (weight: 0.20) — correct bug_type label
  3. explanation_quality (weight: 0.20) — semantic similarity to ground truth
  4. code_quality        (weight: 0.10) — idiomatic, clean Python
  5. attempt_penalty     (–0.05 per attempt beyond the first)
"""

import ast
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Tuple


# ── Test Case Checkers ────────────────────────────────────────────────────────

def _check_test_case(check_name: str, fix_code: str) -> bool:
    """
    Static analysis checks — no code execution needed.
    Returns True if the fix_code passes the given check.
    """
    code = fix_code.strip()

    checkers = {
        # Easy checks
        "zero_grad_called_before_backward": lambda c: (
            "zero_grad" in c and
            c.index("zero_grad") < c.index("backward")
        ),
        "loss_decreases": lambda c: (
            "optimizer.step()" in c and "zero_grad" in c
        ),
        "step_called": lambda c: "optimizer.step()" in c,

        "uses_cross_entropy": lambda c: "CrossEntropyLoss" in c,
        "labels_are_long": lambda c: ".float()" not in c or "labels" not in c.split(".float()")[0],
        "accuracy_above_50": lambda c: "CrossEntropyLoss" in c,

        "eval_mode_set": lambda c: "model.eval()" in c,
        "deterministic_output": lambda c: "model.eval()" in c,
        "no_grad_used": lambda c: "torch.no_grad()" in c or "no_grad" in c,

        # Medium checks
        "batch_dim_preserved": lambda c: (
            "x.size(0)" in c or "x.shape[0]" in c or ".view(batch" in c
        ),
        "output_shape_correct": lambda c: (
            "x.size(0)" in c or "x.shape[0]" in c
        ),
        "no_runtime_error": lambda c: _syntax_valid(c),

        "output_shape_2d": lambda c: "mean(dim=1)" in c or "mean(dim=-2)" in c or "[:,0]" in c,
        "sequence_pooled": lambda c: "mean(" in c or "max(" in c or "[:,0]" in c,
        "no_error": lambda c: _syntax_valid(c),

        "key_transposed": lambda c: "transpose(-2, -1)" in c or "transpose(-1, -2)" in c or ".T" in c,
        "scores_shape_correct": lambda c: "transpose" in c,
        "output_shape_matches_value": lambda c: "matmul(weights, value)" in c or "matmul(attn" in c,

        # Hard checks
        "layer1_grad_exists": lambda c: ".detach()" not in c,
        "layer1_weights_update": lambda c: ".detach()" not in c and "optimizer.step()" in c,
        "loss_decreases_consistently": lambda c: ".detach()" not in c,
        "no_detach_in_forward": lambda c: ".detach()" not in c,

        "uses_relu": lambda c: "relu" in c.lower(),
        "uses_kaiming_init": lambda c: "kaiming" in c,
        "first_layer_grad_healthy": lambda c: "kaiming" in c and "relu" in c.lower(),
        "all_layers_learn": lambda c: "kaiming" in c and "relu" in c.lower(),

        "no_inplace_ops": lambda c: _no_inplace_on_leaf(c),
        "backward_succeeds": lambda c: _no_inplace_on_leaf(c) and _syntax_valid(c),
        "embedding_grads_exist": lambda c: "stack" in c or "cat" in c,
        "loss_has_grad_fn": lambda c: "stack" in c or ("losses" in c and "mean" in c),
    }

    checker = checkers.get(check_name)
    if checker is None:
        return False

    try:
        return checker(code)
    except Exception:
        return False


def _syntax_valid(code: str) -> bool:
    """Check if code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _no_inplace_on_leaf(code: str) -> bool:
    """Detect in-place ops on tensors with requires_grad."""
    # Heuristic: check for += on a variable that's initialized with requires_grad=True
    has_inplace = "+=" in code
    has_requires_grad_init = "requires_grad=True" in code
    if has_inplace and has_requires_grad_init:
        return False
    return True


# ── Reward Components ─────────────────────────────────────────────────────────

def _test_pass_rate(fix_code: str, test_cases: list) -> float:
    """
    Score based on fraction of static test checks passing.
    Weight: 0.50
    """
    if not test_cases:
        return 0.0

    passed = sum(
        1 for tc in test_cases
        if _check_test_case(tc["check"], fix_code)
    )
    return passed / len(test_cases)


def _bug_classification_score(submitted_type: str, correct_type: str) -> float:
    """
    Score for correctly identifying the bug category.
    Weight: 0.20
    Full credit: exact match
    Partial credit: related categories
    """
    if submitted_type == correct_type:
        return 1.0

    # Partial credit for related types
    partial_credit = {
        ("logic_error", "runtime_error"): 0.5,
        ("runtime_error", "logic_error"): 0.5,
        ("shape_mismatch", "runtime_error"): 0.4,
        ("gradient_issue", "logic_error"): 0.3,
    }
    return partial_credit.get((submitted_type, correct_type), 0.0)


def _explanation_quality_score(explanation: str, ground_truth_explanation: str) -> float:
    """
    Semantic similarity between agent's explanation and ground truth.
    Uses keyword matching + sequence similarity as a proxy for LLM judge.
    Weight: 0.20
    """
    if not explanation.strip():
        return 0.0

    explanation_lower = explanation.lower()
    gt_lower = ground_truth_explanation.lower()

    # Extract key technical terms from ground truth
    technical_terms = re.findall(r'\b[a-z_]{4,}\b', gt_lower)
    key_terms = [t for t in technical_terms if len(t) > 5][:10]

    # Keyword coverage score
    coverage = sum(1 for term in key_terms if term in explanation_lower)
    keyword_score = coverage / max(len(key_terms), 1)

    # Sequence similarity score
    seq_score = SequenceMatcher(None, explanation_lower, gt_lower).ratio()

    # Minimum length check — too short = poor explanation
    length_bonus = 1.0 if len(explanation.split()) >= 10 else 0.6

    return min((keyword_score * 0.6 + seq_score * 0.4) * length_bonus, 1.0)


def _code_quality_score(fix_code: str) -> float:
    """
    Heuristic code quality check.
    Weight: 0.10
    Checks: valid syntax, has comments, no debug prints, uses type hints
    """
    score = 0.0

    if _syntax_valid(fix_code):
        score += 0.5

    # Has inline comments explaining the fix
    if "#" in fix_code:
        score += 0.2

    # No debug print statements
    if "print(" not in fix_code:
        score += 0.15

    # Follows PyTorch naming conventions
    if not re.search(r'\bx1\b|\btemp\b|\bfoo\b|\bbar\b', fix_code):
        score += 0.15

    return min(score, 1.0)


# ── Main Reward Function ──────────────────────────────────────────────────────

def compute_reward(
    action: Any,
    task: Dict,
    task_level: str,
    attempt_number: int,
) -> Tuple[Dict, float]:
    """
    Compute the full reward for an agent's debugging action.

    Returns:
        (breakdown_dict, total_reward_float)

    Partial progress is always rewarded — even incomplete fixes get credit
    for what they got right.
    """

    # Component scores (each 0.0–1.0)
    test_score   = _test_pass_rate(action.fix_code, task["test_cases"])
    bug_score    = _bug_classification_score(action.bug_type, task["bug_type"])
    expl_score   = _explanation_quality_score(action.explanation, task["explanation"])
    quality_score = _code_quality_score(action.fix_code)

    # Weighted sum
    weighted_total = (
        test_score    * 0.50 +
        bug_score     * 0.20 +
        expl_score    * 0.20 +
        quality_score * 0.10
    )

    # Attempt penalty (encourage solving on first try)
    penalty = max(0, (attempt_number - 1)) * 0.05
    total = max(0.0, weighted_total - penalty)

    breakdown = {
        "test_pass_rate":      round(test_score, 4),
        "bug_classification":  round(bug_score, 4),
        "explanation_quality": round(expl_score, 4),
        "code_quality":        round(quality_score, 4),
        "attempt_penalty":     round(-penalty, 4),
        "weighted_before_penalty": round(weighted_total, 4),
    }

    return breakdown, round(total, 4)
