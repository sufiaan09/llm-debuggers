import sys
sys.path.insert(0, ".")
import gradio as gr
from env.env import LLMDebuggingEnv, Action

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
    return obs["buggy_code"], info, "Episode started! Submit your fix.", ""

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


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

:root {
    --neon-cyan: #00f5ff;
    --neon-green: #00ff88;
    --neon-orange: #ff6b00;
    --neon-pink: #ff0080;
    --dark-bg: #020408;
    --panel-bg: rgba(0, 20, 40, 0.85);
    --border-glow: rgba(0, 245, 255, 0.4);
    --text-primary: #e0f8ff;
    --text-dim: #5a8a99;
}

/* ── Animated Background ── */
body, .gradio-container {
    background: var(--dark-bg) !important;
    font-family: 'Exo 2', sans-serif !important;
    color: var(--text-primary) !important;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}

.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse at 20% 20%, rgba(0,245,255,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(0,255,136,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(255,107,0,0.03) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: bgPulse 8s ease-in-out infinite alternate;
}

@keyframes bgPulse {
    0%   { opacity: 0.6; }
    100% { opacity: 1.2; }
}

/* ── Grid Lines ── */
.gradio-container::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(0,245,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,245,255,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
    animation: gridScroll 20s linear infinite;
}

@keyframes gridScroll {
    0%   { transform: perspective(500px) rotateX(0deg) translateY(0px); }
    100% { transform: perspective(500px) rotateX(0deg) translateY(50px); }
}

/* ── Main Title ── */
.main-title {
    font-family: 'Orbitron', monospace !important;
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    text-align: center;
    background: linear-gradient(135deg, var(--neon-cyan), var(--neon-green), var(--neon-orange));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    letter-spacing: 4px;
    margin-bottom: 0.5rem !important;
    animation: titleGlow 3s ease-in-out infinite alternate;
    position: relative;
    z-index: 1;
}

@keyframes titleGlow {
    0%   { filter: drop-shadow(0 0 20px rgba(0,245,255,0.5)); }
    100% { filter: drop-shadow(0 0 40px rgba(0,255,136,0.7)); }
}

/* ── Subtitle ── */
.subtitle {
    text-align: center;
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--neon-cyan) !important;
    font-size: 0.9rem !important;
    letter-spacing: 3px !important;
    opacity: 0.8;
    animation: blink 2s step-end infinite;
}

@keyframes blink {
    0%, 100% { opacity: 0.8; }
    50%       { opacity: 0.4; }
}

/* ── Panels ── */
.gr-group, .gr-box, .panel, div[data-testid="block"] {
    background: var(--panel-bg) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 4px !important;
    backdrop-filter: blur(12px) !important;
    position: relative;
    box-shadow:
        0 0 20px rgba(0,245,255,0.08),
        inset 0 1px 0 rgba(0,245,255,0.1) !important;
    transition: all 0.3s ease !important;
    z-index: 1;
}

.gr-group:hover, .gr-box:hover {
    border-color: rgba(0,245,255,0.7) !important;
    box-shadow:
        0 0 30px rgba(0,245,255,0.15),
        inset 0 1px 0 rgba(0,245,255,0.2) !important;
    transform: translateY(-1px);
}

/* ── Corner Decorations ── */
.gr-group::before {
    content: '';
    position: absolute;
    top: -1px; left: -1px;
    width: 20px; height: 20px;
    border-top: 2px solid var(--neon-cyan);
    border-left: 2px solid var(--neon-cyan);
    border-radius: 4px 0 0 0;
    z-index: 2;
}

.gr-group::after {
    content: '';
    position: absolute;
    bottom: -1px; right: -1px;
    width: 20px; height: 20px;
    border-bottom: 2px solid var(--neon-green);
    border-right: 2px solid var(--neon-green);
    border-radius: 0 0 4px 0;
    z-index: 2;
}

/* ── Buttons ── */
button.primary, .gr-button-primary, button[variant="primary"] {
    background: transparent !important;
    border: 2px solid var(--neon-cyan) !important;
    color: var(--neon-cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    padding: 14px 28px !important;
    border-radius: 2px !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 15px rgba(0,245,255,0.3), inset 0 0 15px rgba(0,245,255,0.05) !important;
}

button.primary::before, .gr-button-primary::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,245,255,0.2), transparent);
    transition: left 0.5s ease;
}

button.primary:hover::before {
    left: 100%;
}

button.primary:hover, .gr-button-primary:hover {
    background: rgba(0,245,255,0.1) !important;
    box-shadow: 0 0 30px rgba(0,245,255,0.6), inset 0 0 30px rgba(0,245,255,0.1) !important;
    transform: translateY(-2px) !important;
    color: #fff !important;
}

button.primary:active {
    transform: translateY(0px) !important;
    box-shadow: 0 0 10px rgba(0,245,255,0.4) !important;
}

/* ── Labels ── */
label, .gr-label, span.svelte-1gfkn6j {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--neon-cyan) !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* ── Inputs ── */
input, textarea, select, .gr-input, .gr-textarea {
    background: rgba(0, 10, 20, 0.8) !important;
    border: 1px solid rgba(0,245,255,0.3) !important;
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    border-radius: 2px !important;
    transition: all 0.3s ease !important;
}

input:focus, textarea:focus {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 15px rgba(0,245,255,0.3) !important;
    outline: none !important;
}

/* ── Code Editor ── */
.code-editor, .cm-editor, .cm-content {
    background: rgba(0, 5, 15, 0.95) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    border: 1px solid rgba(0,245,255,0.2) !important;
    color: #00ff88 !important;
}

.cm-line { color: #7fffb2 !important; }
.cm-keyword { color: var(--neon-cyan) !important; }
.cm-string  { color: var(--neon-orange) !important; }
.cm-comment { color: #3a6b7a !important; font-style: italic !important; }
.cm-number  { color: var(--neon-pink) !important; }
.cm-def     { color: #ffee00 !important; }

/* ── Dropdowns ── */
select, .gr-dropdown {
    background: rgba(0,10,25,0.9) !important;
    border: 1px solid rgba(0,245,255,0.3) !important;
    color: var(--neon-cyan) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Markdown Output ── */
.gr-markdown, .prose {
    color: var(--text-primary) !important;
    font-family: 'Exo 2', sans-serif !important;
}

.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    font-family: 'Orbitron', monospace !important;
    color: var(--neon-cyan) !important;
    letter-spacing: 2px !important;
    text-shadow: 0 0 15px rgba(0,245,255,0.5) !important;
}

.gr-markdown table {
    border-collapse: collapse !important;
    width: 100% !important;
}

.gr-markdown th {
    background: rgba(0,245,255,0.1) !important;
    color: var(--neon-cyan) !important;
    border: 1px solid rgba(0,245,255,0.3) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    padding: 8px 12px !important;
}

.gr-markdown td {
    border: 1px solid rgba(0,245,255,0.15) !important;
    padding: 8px 12px !important;
    color: var(--text-primary) !important;
}

.gr-markdown tr:hover td {
    background: rgba(0,245,255,0.05) !important;
}

.gr-markdown code {
    background: rgba(0,245,255,0.1) !important;
    color: var(--neon-green) !important;
    padding: 2px 6px !important;
    border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important;
    border: 1px solid rgba(0,245,255,0.2) !important;
}

/* ── Slider ── */
input[type="range"] {
    accent-color: var(--neon-cyan) !important;
}

/* ── Status Box ── */
.gr-textbox input, .gr-textbox textarea {
    color: var(--neon-green) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--dark-bg); }
::-webkit-scrollbar-thumb {
    background: var(--neon-cyan);
    border-radius: 3px;
    box-shadow: 0 0 6px var(--neon-cyan);
}

/* ── Scan line effect ── */
.gradio-container > div::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
    animation: scanLine 4s linear infinite;
    z-index: 999;
    opacity: 0.5;
    pointer-events: none;
}

@keyframes scanLine {
    0%   { top: 0%; }
    100% { top: 100%; }
}

/* ── Floating particles ── */
@keyframes float1 {
    0%, 100% { transform: translateY(0px) translateX(0px); opacity: 0.3; }
    33%       { transform: translateY(-30px) translateX(15px); opacity: 0.8; }
    66%       { transform: translateY(-10px) translateX(-10px); opacity: 0.5; }
}

/* ── Number counter animation ── */
@keyframes countUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Footer ── */
.footer-text {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--text-dim) !important;
    font-size: 0.7rem !important;
    text-align: center !important;
    letter-spacing: 2px !important;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Orbitron', monospace !important;
    color: var(--neon-orange) !important;
    font-size: 0.8rem !important;
    letter-spacing: 4px !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(255,107,0,0.3) !important;
    padding-bottom: 6px !important;
    margin-bottom: 12px !important;
}

/* ── Tabs ── */
.tab-nav button {
    font-family: 'Orbitron', monospace !important;
    color: var(--text-dim) !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    color: var(--neon-cyan) !important;
    border-bottom-color: var(--neon-cyan) !important;
    text-shadow: 0 0 10px var(--neon-cyan) !important;
}

/* ── Row gaps ── */
.gr-row { gap: 16px !important; }

/* ── Neon divider ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent) !important;
    margin: 20px 0 !important;
    opacity: 0.4 !important;
}
"""

HEADER_HTML = """
<div style="text-align:center; padding: 30px 0 10px; position:relative; z-index:2;">
  <div style="
    font-family: 'Orbitron', monospace;
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00f5ff, #00ff88, #ff6b00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 4px;
    filter: drop-shadow(0 0 25px rgba(0,245,255,0.6));
    animation: titleGlow 3s ease-in-out infinite alternate;
    margin-bottom: 8px;
  ">🐛 LLM DEBUGGER</div>

  <div style="
    font-family: 'Share Tech Mono', monospace;
    color: #00f5ff;
    font-size: 0.8rem;
    letter-spacing: 4px;
    opacity: 0.7;
    margin-bottom: 6px;
  ">▸ OPENENV HACKATHON · META × PYTORCH × SCALER ◂</div>

  <div style="
    font-family: 'Exo 2', sans-serif;
    color: #5a8a99;
    font-size: 0.75rem;
    letter-spacing: 2px;
    font-style: italic;
  ">[ AI AGENT THAT READS BROKEN PYTORCH CODE AND FIXES IT ]</div>

  <!-- Animated stats bar -->
  <div style="
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-top: 20px;
    padding: 12px 30px;
    background: rgba(0,245,255,0.05);
    border: 1px solid rgba(0,245,255,0.2);
    border-radius: 4px;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
  ">
    <div style="text-align:center;">
      <div style="font-family:'Orbitron',monospace; color:#00ff88; font-size:1.4rem; font-weight:700;">9</div>
      <div style="font-family:'Share Tech Mono',monospace; color:#5a8a99; font-size:0.65rem; letter-spacing:2px;">TASKS</div>
    </div>
    <div style="text-align:center;">
      <div style="font-family:'Orbitron',monospace; color:#00f5ff; font-size:1.4rem; font-weight:700;">3</div>
      <div style="font-family:'Share Tech Mono',monospace; color:#5a8a99; font-size:0.65rem; letter-spacing:2px;">LEVELS</div>
    </div>
    <div style="text-align:center;">
      <div style="font-family:'Orbitron',monospace; color:#ff6b00; font-size:1.4rem; font-weight:700;">5</div>
      <div style="font-family:'Share Tech Mono',monospace; color:#5a8a99; font-size:0.65rem; letter-spacing:2px;">METRICS</div>
    </div>
    <div style="text-align:center;">
      <div style="font-family:'Orbitron',monospace; color:#ff0080; font-size:1.4rem; font-weight:700;">1.0</div>
      <div style="font-family:'Share Tech Mono',monospace; color:#5a8a99; font-size:0.65rem; letter-spacing:2px;">MAX SCORE</div>
    </div>
  </div>
</div>
"""

with gr.Blocks(css=CUSTOM_CSS, title="🐛 LLM Self-Debugging Environment") as demo:

    gr.HTML(HEADER_HTML)

    gr.HTML("<hr/>")

    with gr.Row():
        # ── LEFT: Control Panel ──
        with gr.Column(scale=1):
            gr.HTML("""<div style="font-family:'Orbitron',monospace; color:#ff6b00;
                font-size:0.75rem; letter-spacing:4px; border-bottom:1px solid rgba(255,107,0,0.3);
                padding-bottom:6px; margin-bottom:4px;">⚙ MISSION CONTROL</div>""")

            level_dd  = gr.Dropdown(
                ["easy","medium","hard"], value="easy",
                label="DIFFICULTY LEVEL"
            )
            task_sl   = gr.Slider(0, 2, step=1, value=0, label="TASK INDEX (0 → 2)")
            start_btn = gr.Button("⚡ INITIALIZE EPISODE", variant="primary")
            status_tx = gr.Textbox(label="SYSTEM STATUS", interactive=False)

            gr.HTML("""
            <div style="margin-top:16px; padding:12px;
                background:rgba(0,255,136,0.05);
                border:1px solid rgba(0,255,136,0.2); border-radius:4px;">
              <div style="font-family:'Share Tech Mono',monospace; color:#00ff88;
                  font-size:0.7rem; letter-spacing:2px; margin-bottom:8px;">DIFFICULTY GUIDE</div>
              <div style="font-family:'Exo 2',sans-serif; color:#5a8a99; font-size:0.75rem; line-height:1.8;">
                🟢 <b style="color:#00ff88">EASY</b> — Logic errors · Max 0.4<br/>
                🟡 <b style="color:#ffee00">MEDIUM</b> — Shape bugs · Max 0.7<br/>
                🔴 <b style="color:#ff6b00">HARD</b> — Gradient issues · Max 1.0
              </div>
            </div>
            """)

        # ── RIGHT: Task Intel ──
        with gr.Column(scale=2):
            gr.HTML("""<div style="font-family:'Orbitron',monospace; color:#ff6b00;
                font-size:0.75rem; letter-spacing:4px; border-bottom:1px solid rgba(255,107,0,0.3);
                padding-bottom:6px; margin-bottom:4px;">📋 MISSION BRIEFING</div>""")
            task_info = gr.Markdown("*Initialize episode to receive mission briefing...*")

    gr.HTML("<hr/>")

    # ── Buggy Code ──
    gr.HTML("""<div style="font-family:'Orbitron',monospace; color:#ff0080;
        font-size:0.75rem; letter-spacing:4px; margin-bottom:8px;">
        🐛 CORRUPTED CODE — IDENTIFY & NEUTRALIZE THE BUG</div>""")
    buggy_box = gr.Code(language="python", label="", interactive=False)

    gr.HTML("<hr/>")

    # ── Fix Panel ──
    gr.HTML("""<div style="font-family:'Orbitron',monospace; color:#00f5ff;
        font-size:0.75rem; letter-spacing:4px; margin-bottom:8px;">
        🛠 REPAIR STATION — SUBMIT YOUR FIX</div>""")

    with gr.Row():
        with gr.Column(scale=2):
            fix_box = gr.Code(language="python", label="PATCHED CODE", lines=14)
        with gr.Column(scale=1):
            expl_box = gr.Textbox(
                label="DIAGNOSTIC REPORT — explain the bug",
                lines=5,
                placeholder="Describe what caused the bug and how your fix resolves it..."
            )
            bug_dd = gr.Dropdown(
                ["syntax_error","shape_mismatch","gradient_issue","logic_error","runtime_error"],
                value="logic_error",
                label="BUG CLASSIFICATION"
            )
            sub_btn = gr.Button("📤 DEPLOY FIX", variant="primary")

    gr.HTML("<hr/>")

    # ── Results ──
    gr.HTML("""<div style="font-family:'Orbitron',monospace; color:#00ff88;
        font-size:0.75rem; letter-spacing:4px; margin-bottom:8px;">
        📊 PERFORMANCE ANALYTICS</div>""")

    with gr.Row():
        feedback_md = gr.Markdown("*Deploy a fix to see your performance score...*")
        history_md  = gr.Markdown("*Attempt history will appear here...*")

    gr.HTML("""
    <div style="text-align:center; padding:20px 0 10px; font-family:'Share Tech Mono',monospace;
        color:#1a3a4a; font-size:0.65rem; letter-spacing:3px;">
      OPENENV SPEC ✓ · STEP() / RESET() / STATE() ✓ · PARTIAL REWARDS ✓ · HUGGINGFACE READY ✓
    </div>
    """)

    start_btn.click(
        fn=start_episode,
        inputs=[level_dd, task_sl],
        outputs=[buggy_box, task_info, status_tx, feedback_md]
    )
    sub_btn.click(
        fn=submit_fix,
        inputs=[fix_box, expl_box, bug_dd],
        outputs=[feedback_md, history_md, status_tx]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
