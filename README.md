# 🐛 LLM Self-Debugging Environment

> **OpenEnv Hackathon Submission — Meta × PyTorch × Scaler School of Technology**

An AI agent that reads **buggy PyTorch code**, identifies the root cause of the bug, applies a fix, and earns reward based on how many test cases pass — simulating real-world ML engineering debugging.

---

## 🎯 What It Does

Real ML engineers spend 40%+ of their time debugging. This environment trains an AI agent to:

1. **Read** broken PyTorch training code + error message
2. **Classify** the bug type (shape mismatch / gradient issue / logic error)
3. **Submit a fix** with a clear explanation
4. **Earn partial credit** for every correct aspect — even incomplete fixes get rewarded

---

## 🔧 Environment API (OpenEnv Spec)

```python
from env.env import LLMDebuggingEnv, Action

env = LLMDebuggingEnv(task_level="easy", seed=42)

# reset() — load a fresh task
obs = env.reset()

# state() — get current environment state
current_state = env.state()

# step() — submit a debugging action
action = Action(
    fix_code="def train_step(...):\n    optimizer.zero_grad()\n    ...",
    explanation="Missing zero_grad() caused gradient accumulation across steps",
    bug_type="logic_error"
)
result = env.step(action)
# result.observation → new state
# result.reward      → float 0.0–1.0
# result.done        → bool
# result.info        → reward breakdown + debug info
```

---

## 📐 Action Space

| Field | Type | Values |
|---|---|---|
| `fix_code` | `string` | The corrected Python/PyTorch code |
| `explanation` | `string` | Description of the bug and fix |
| `bug_type` | `string` | `syntax_error` / `shape_mismatch` / `gradient_issue` / `logic_error` / `runtime_error` |

---

## 👁️ Observation Space

| Field | Type | Description |
|---|---|---|
| `buggy_code` | `string` | The broken PyTorch code |
| `error_message` | `string` | Error traceback or failure description |
| `task_description` | `string` | What the code is supposed to do |
| `test_cases` | `array` | List of checks the fix must pass |
| `attempt_number` | `integer` | Current attempt (1–3) |
| `previous_attempts` | `array` | History of past fixes and scores |
| `task_id` | `string` | Unique task identifier |
| `task_level` | `string` | `easy` / `medium` / `hard` |

---

## 🎮 Task Levels

### 🟢 Easy — Syntax & Logic Errors (max score: 0.4)
Real bugs that cause silent failures or obvious crashes:
- Missing `optimizer.zero_grad()` → gradients accumulate
- Wrong loss function (`MSELoss` for classification)
- Missing `model.eval()` → non-deterministic inference

### 🟡 Medium — Tensor Shape Mismatches (max score: 0.7)
Bugs that cause `RuntimeError` due to wrong tensor dimensions:
- `x.view(N, -1)` collapsing the batch dimension
- Missing sequence pooling before linear layer
- Un-transposed key tensor in dot-product attention

### 🔴 Hard — Gradient Flow Bugs (max score: 1.0)
Silent bugs where the model trains but never learns:
- `.detach()` in forward pass blocking gradient flow
- Sigmoid + Xavier init causing vanishing gradients
- In-place `+=` on leaf tensors corrupting autograd graph

---

## 🏆 Reward Function

```
reward = (test_pass_rate × 0.50)
       + (bug_classification × 0.20)
       + (explanation_quality × 0.20)
       + (code_quality × 0.10)
       - (attempt_penalty × max(0, attempt - 1))
```

**Partial progress is always rewarded** — solving 2/3 test cases gives more reward than 0.

| Component | Weight | Description |
|---|---|---|
| Test Pass Rate | 50% | Fraction of static checks passing |
| Bug Classification | 20% | Correct `bug_type` label |
| Explanation Quality | 20% | Keyword + semantic similarity to ground truth |
| Code Quality | 10% | Syntax valid, has comments, clean style |
| Attempt Penalty | −0.05 | Per attempt beyond the first |

---

## 🚀 Setup & Run

```bash
# Clone repo
git clone https://huggingface.co/spaces/your-username/llm-debugging-env
cd llm-debugging-env

# Install dependencies
pip install -r requirements.txt

# Run baseline (reproducible scores)
python baseline.py

# Launch Gradio UI locally
python app.py
```

---

## 📊 Baseline Scores (seed=42)

```
EASY    | 0.3200 / 0.4  | ████████████████░░░░░  80.0%
MEDIUM  | 0.1800 / 0.7  | ███████░░░░░░░░░░░░░░  25.7%
HARD    | 0.0900 / 1.0  | ███░░░░░░░░░░░░░░░░░░   9.0%
```

The heuristic baseline intentionally scores low on hard tasks —
leaving significant room for LLM-based agents to excel.

---

## 🐳 Docker

```bash
docker build -t llm-debugging-env .
docker run -p 7860:7860 llm-debugging-env
```

---

## 📁 Project Structure

```
llm-debugger-env/
├── env/
│   └── env.py          ← Core OpenEnv: step() / reset() / state() + typed models
├── graders/
│   └── grader.py       ← Partial-progress reward function
├── tasks/
│   └── tasks.py        ← 9 tasks (3 easy + 3 medium + 3 hard)
├── baseline.py         ← Reproducible heuristic baseline
├── app.py              ← Gradio UI for HuggingFace Spaces
├── openenv.yaml        ← Full OpenEnv spec
├── Dockerfile          ← Container for HuggingFace deployment
├── requirements.txt
└── README.md
```

---

## 🔬 Why This Environment Matters

| Real-world Signal | How We Model It |
|---|---|
| Debugger reads code | `buggy_code` + `error_message` in observation |
| Engineer classifies bug | `bug_type` in action space |
| Fix is written & tested | `fix_code` evaluated against test cases |
| Explains fix in PR | `explanation` scored for quality |
| Gets feedback, tries again | Multi-attempt loop with partial rewards |

---

## 📜 License
MIT License — Open for research and education.