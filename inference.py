import sys
sys.path.insert(0, ".")
from env.env import LLMDebuggingEnv, Action

env = LLMDebuggingEnv(task_level="easy", seed=42)

def run_inference(task_level="easy", task_index=0):
    env = LLMDebuggingEnv(task_level=task_level, seed=42)
    obs = env.reset(task_index=task_index)
    
    action = Action(
        fix_code=obs["buggy_code"] + "\n# baseline fix",
        explanation="Identified and fixed the bug in the code",
        bug_type="logic_error"
    )
    
    result = env.step(action)
    return {
        "task_id": obs["task_id"],
        "reward": result.reward,
        "done": result.done,
        "breakdown": result.info["reward_breakdown"]
    }

if __name__ == "__main__":
    for level in ["easy", "medium", "hard"]:
        result = run_inference(task_level=level)
        print(f"{level}: {result}")