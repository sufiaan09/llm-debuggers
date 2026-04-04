import argparse
import json
import sys
import time
sys.path.insert(0, ".")

from env.env import LLMDebuggingEnv, Action
from tasks.tasks import TASKS


class HeuristicAgent:
    BUG_KEYWORDS = {
        "syntax_error":   ["SyntaxError", "IndentationError", "invalid syntax"],
        "shape_mismatch": ["shape", "size", "dimension", "mat1", "mat2", "reshape", "view"],
        "gradient_issue": ["gradient", "grad", "detach", "backward", "requires_grad", "None"],
        "logic_error":    ["converge", "accuracy", "deterministic", "oscillate", "zero_grad"],
        "runtime_error":  ["RuntimeError", "TypeError", "AttributeError"],
    }

    def classify_bug(self, error_message, buggy_code):
        combined = error_message + " " + buggy_code
        scores = {}
        for bug_type, keywords in self.BUG_KEYWORDS.items():
            scores[bug_type] = sum(kw.lower() in combined.lower() for kw in keywords)
        return max(scores, key=scores.get)

    def generate_fix(self, observation, bug_type):
        code = observation["buggy_code"]
        error = observation["error_message"]
        fix_code = code
        explanation = f"Detected {bug_type}. "

        if bug_type == "logic_error":
            if "zero_grad" in error or "converge" in error.lower():
                fix_code = "optimizer.zero_grad()\n" + code
                explanation += "Added optimizer.zero_grad() before backward pass."
            elif "CrossEntropy" in error or "accuracy" in error:
                fix_code = code.replace("MSELoss", "CrossEntropyLoss")
                explanation += "Replaced MSELoss with CrossEntropyLoss."
            elif "eval" in error or "deterministic" in error:
                fix_code = code.replace("def predict(", "def predict(\n    model.eval()\n")
                explanation += "Added model.eval() before inference."

        elif bug_type == "shape_mismatch":
            if "view" in code:
                fix_code = code.replace("x.view(32 * 26 * 26, -1)", "x.view(x.size(0), -1)")
                explanation += "Fixed view() to preserve batch dimension."
            elif "embedding" in code.lower():
                fix_code = code.replace(
                    "out = self.fc(embedded)",
                    "pooled = embedded.mean(dim=1)\n        out = self.fc(pooled)"
                )
                explanation += "Added mean pooling over sequence dimension."
            elif "matmul" in code:
                fix_code = code.replace(
                    "torch.matmul(query, key)",
                    "torch.matmul(query, key.transpose(-2, -1))"
                )
                explanation += "Transposed key tensor for attention."

        elif bug_type == "gradient_issue":
            if ".detach()" in code:
                fix_code = code.replace("h = h.detach()", "# removed detach")
                explanation += "Removed .detach() to restore gradient flow."
            elif "sigmoid" in code:
                fix_code = code.replace("torch.sigmoid", "torch.relu").replace(
                    "xavier_uniform_", "kaiming_uniform_"
                )
                explanation += "Replaced sigmoid with ReLU and Kaiming init."
            elif "+=" in code and "requires_grad" in code:
                fix_code = code.replace(
                    "loss = torch.zeros(1, requires_grad=True)", "losses = []"
                ).replace("loss +=", "losses.append(")
                explanation += "Replaced in-place += with list accumulation."

        return fix_code, explanation


def evaluate_level(level, seed=42, verbose=True):
    agent = HeuristicAgent()
    tasks = TASKS[level]
    results = []

    for task_idx, task in enumerate(tasks):
        env = LLMDebuggingEnv(task_level=level, seed=seed)
        obs = env.reset(task_index=task_idx)

        if verbose:
            print(f"\n  Task {task_idx+1}/{len(tasks)}: {task['title']}")

        best_reward = 0.0
        attempt_rewards = []

        for attempt in range(env.MAX_ATTEMPTS):
            bug_type = agent.classify_bug(obs["error_message"], obs["buggy_code"])
            fix_code, explanation = agent.generate_fix(obs, bug_type)

            action = Action(
                fix_code=fix_code,
                explanation=explanation,
                bug_type=bug_type,
            )

            result = env.step(action)
            obs = result.observation.to_dict()
            attempt_rewards.append(result.reward)
            best_reward = max(best_reward, result.reward)

            if verbose:
                print(f"    Attempt {attempt+1}: reward={result.reward:.4f} | bug_type={bug_type}")

            if result.done:
                break

        results.append({
            "task_id": task["id"],
            "task_title": task["title"],
            "best_reward": best_reward,
            "attempt_rewards": attempt_rewards,
        })

    avg_score = sum(r["best_reward"] for r in results) / len(results)
    return {
        "level": level,
        "avg_score": round(avg_score, 4),
        "max_possible": env.MAX_SCORES[level],
        "tasks": results,
        "seed": seed,
    }


def run_all_levels(seed=42, verbose=True):
    print("=" * 60)
    print("  LLM Self-Debugging Env — Baseline Evaluation")
    print(f"  Seed: {seed}")
    print("=" * 60)

    all_results = {}
    start = time.time()

    for level in ["easy", "medium", "hard"]:
        print(f"\n  Level: {level.upper()}")
        print("-" * 60)
        result = evaluate_level(level, seed=seed, verbose=verbose)
        all_results[level] = result

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    for level, res in all_results.items():
        bar = "█" * int(res["avg_score"] / res["max_possible"] * 20)
        pct = res["avg_score"] / res["max_possible"] * 100
        print(f"  {level.upper():8s} | {res['avg_score']:.4f} / {res['max_possible']} | {bar:<20} {pct:.1f}%")

    print(f"\n  Total time: {elapsed:.2f}s")
    print("=" * 60)
    return {"results": all_results, "elapsed_seconds": round(elapsed, 2), "seed": seed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["easy", "medium", "hard"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.level:
        results = evaluate_level(args.level, seed=args.seed, verbose=verbose)
    else:
        results = run_all_levels(seed=args.seed, verbose=verbose)

    if args.json:
        print(json.dumps(results, indent=2))