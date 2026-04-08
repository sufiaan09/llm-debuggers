import sys
sys.path.insert(0, ".")
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import gradio as gr
from env.env import LLMDebuggingEnv, Action

# ── Global env ──────────────────────────────────────────────
_env = LLMDebuggingEnv(task_level="easy", seed=42)
_env.reset()

# ── HTTP API Server (for hackathon checks) ───────────────────
class OpenEnvHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress logs

    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == "/health":
            self._json(200, {"status": "ok"})
        
        elif parsed.path == "/state":
            self._json(200, _env.state())
        
        elif parsed.path == "/openenv.yaml":
            with open("openenv.yaml", "r") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-type", "text/yaml")
            self.end_headers()
            self.wfile.write(content.encode())
        
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        
        try:
            data = json.loads(body)
        except Exception:
            data = {}

        if parsed.path == "/reset":
            task_level = data.get("task_level", "easy")
            task_index = data.get("task_index", 0)
            global _env
            _env = LLMDebuggingEnv(task_level=task_level, seed=42)
            obs = _env.reset(task_index=task_index)
            self._json(200, obs)

        elif parsed.path == "/step":
            fix_code   = data.get("fix_code", "# no fix provided")
            explanation = data.get("explanation", "no explanation")
            bug_type   = data.get("bug_type", "logic_error")
            
            action = Action(
                fix_code=fix_code,
                explanation=explanation,
                bug_type=bug_type
            )
            try:
                result = _env.step(action)
                self._json(200, {
                    "observation": result.observation.to_dict(),
                    "reward": result.reward,
                    "done": result.done,
                    "info": result.info
                })
            except Exception as e:
                self._json(400, {"error": str(e)})

        elif parsed.path == "/state":
            self._json(200, _env.state())

        else:
            self._json(404, {"error": "not found"})

    def _json(self, code, data):
        self.send_response(code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

def start_api_server():
    server = HTTPServer(("0.0.0.0", 7861), OpenEnvHandler)
    server.serve_forever()

# ── Gradio UI ────────────────────────────────────────────────
_env_state = {"env": None, "obs": None}
MAX_SCORES = {"easy": 0.4, "medium": 0.7, "hard": 1.0}
LEVEL_COLORS = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}

def start_episode(level, task_idx):
    env = LLMDebuggingEnv(task_level=level, seed=42)
    obs = env.reset(task_index=int(task_idx))
    _env_state["env"] = env
    _env_state["obs"] = obs
    info = f"""### {LEVEL_COLORS[level]} Task: {obs['task_id']} — {level.upper()}
**Description:** {obs['task_description']}
**Error:** {obs['error_message']}
**Max Score:** {MAX_SCORES[level]}"""
    return obs["buggy_code"], info, "Episode started!", ""

def submit_fix(fix_code, explanation, bug_type):
    env = _env_state.get("env")
    if env is None:
        return "❌ Start an episode first!", "", ""
    action = Action(fix_code=fix_code, explanation=explanation, bug_type=bug_type)
    valid, msg = action.validate()
    if not valid:
        return f"❌ Invalid: {msg}", "", ""
    result = env.step(action)
    obs = result.observation.to_dict()
    _env_state["obs"] = obs
    reward = result.reward
    b = result.info["reward_breakdown"]
    bar_len = int(reward / MAX_SCORES[env.task_level] * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    feedback = f"""## {"✅ Solved!" if result.done and reward > 0.3 else "🔄 Try again!" if not result.done else "⏱️ Out of attempts"}
**Reward: {reward:.4f} / {MAX_SCORES[env.task_level]}**
`[{bar}]`
| Component | Score | Weight |
|---|---|---|
| ✅ Test Pass Rate | {b['test_pass_rate']:.3f} | 50% |
| 🐛 Bug Classification | {b['bug_classification']:.3f} | 20% |
| 💬 Explanation | {b['explanation_quality']:.3f} | 20% |
| 🧹 Code Quality | {b['code_quality']:.3f} | 10% |
| ⚠️ Penalty | {b['attempt_penalty']:.3f} | — |
**Attempts remaining:** {result.info['attempts_remaining']}
**Correct bug type:** `{result.info['correct_bug_type']}`"""
    history = ""
    for a in obs["previous_attempts"]:
        history += f"- Attempt {a['attempt']}: **{a['reward']:.4f}** — `{a['bug_type_submitted']}`\n"
    status = "✅ Done!" if result.done else f"Attempt {obs['attempt_number']}/3"
    return feedback, history, status

with gr.Blocks(title="🐛 LLM Self-Debugging Environment") as demo:
    gr.Markdown("""# 🐛 LLM Self-Debugging Environment
### OpenEnv Hackathon — Meta × PyTorch × Scaler
*AI agent reads buggy PyTorch code and fixes it!*""")
    with gr.Row():
        with gr.Column(scale=1):
            level_dd  = gr.Dropdown(["easy","medium","hard"], value="easy", label="Difficulty")
            task_sl   = gr.Slider(0, 2, step=1, value=0, label="Task (0-2)")
            start_btn = gr.Button("🚀 Start Episode", variant="primary")
            status_tx = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=2):
            task_info = gr.Markdown("*Start an episode to load a task.*")
    gr.Markdown("### 🐛 Buggy Code")
    buggy_box = gr.Code(language="python", label="", interactive=False)
    gr.Markdown("### 🛠️ Your Fix")
    with gr.Row():
        with gr.Column(scale=2):
            fix_box  = gr.Code(language="python", label="Fixed Code", lines=12)
        with gr.Column(scale=1):
            expl_box = gr.Textbox(label="Explanation", lines=5)
            bug_dd   = gr.Dropdown(
                ["syntax_error","shape_mismatch","gradient_issue","logic_error","runtime_error"],
                value="logic_error", label="Bug Type"
            )
            sub_btn  = gr.Button("📤 Submit Fix", variant="primary")
    gr.Markdown("### 📊 Score")
    with gr.Row():
        feedback_md = gr.Markdown("*Submit fix to see score.*")
        history_md  = gr.Markdown("*History here.*")
    start_btn.click(fn=start_episode, inputs=[level_dd, task_sl], outputs=[buggy_box, task_info, status_tx, feedback_md])
    sub_btn.click(fn=submit_fix, inputs=[fix_box, expl_box, bug_dd], outputs=[feedback_md, history_md, status_tx])

if __name__ == "__main__":
    # Start API server on port 7861 in background
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    print("API server running on port 7861")
    # Start Gradio on port 7860
    demo.launch(server_name="0.0.0.0", server_port=7860)