from fastapi import FastAPI
import sys
sys.path.insert(0, ".")

from env.env import LLMDebuggingEnv, Action

app = FastAPI()   # ✅ REQUIRED

# Global env
env = LLMDebuggingEnv(task_level="easy", seed=42)

@app.post("/reset")
def reset():
    global env
    env = LLMDebuggingEnv(task_level="easy", seed=42)
    return {"status": "ok"}

@app.post("/infer")
def infer(task_level: str = "easy", task_index: int = 0):
    env = LLMDebuggingEnv(task_level=task_level, seed=42)
    obs = env.reset(task_index=task_index)

    action = Action(
        fix_code=obs["buggy_code"] + "\n# baseline fix",
        explanation="Identified and fixed the bug",
        bug_type="logic_error"
    )

    result = env.step(action)

    return {
        "task_id": obs["task_id"],
        "reward": result.reward,
        "done": result.done,
        "breakdown": result.info["reward_breakdown"]
    }