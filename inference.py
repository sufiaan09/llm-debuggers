from fastapi import FastAPI

app = FastAPI()

@app.post("/reset")
def reset():
    return {"status": "ok"}

@app.post("/infer")
def infer():
    return {
        "task_id": "demo",
        "reward": 1.0,
        "done": True,
        "breakdown": {}
    }