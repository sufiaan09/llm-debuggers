import gradio as gr
import pandas as pd
from your_script import run_all_levels, evaluate_level


def run(level, seed):
    if level == "all":
        result = run_all_levels(seed=seed, verbose=False)
    else:
        result = evaluate_level(level, seed=seed, verbose=False)

    # Prepare chart data
    chart_data = []
    if "results" in result:
        for lvl, res in result["results"].items():
            chart_data.append({
                "Level": lvl,
                "Score": res["avg_score"]
            })
    else:
        chart_data.append({
            "Level": result["level"],
            "Score": result["avg_score"]
        })

    df = pd.DataFrame(chart_data)
    return result, df


with gr.Blocks(theme=gr.themes.Soft(), title="LLM Debugger UI") as demo:

    gr.Markdown("""
    # 🧠 LLM Self-Debugging Dashboard
    Analyze how well your agent fixes bugs across tasks.
    """)

    with gr.Row():
        level = gr.Dropdown(
            ["easy", "medium", "hard", "all"],
            value="all",
            label="Select Level"
        )
        seed = gr.Number(value=42, label="Seed")

    run_btn = gr.Button("🚀 Run Evaluation", variant="primary")

    with gr.Row():
        output_json = gr.JSON(label="📊 Detailed Results")
        output_chart = gr.BarPlot(label="📈 Performance by Level")

    run_btn.click(
        fn=run,
        inputs=[level, seed],
        outputs=[output_json, output_chart]
    )

demo.launch()