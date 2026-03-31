import gradio as gr

def hello():
    return "🚀 Snake AI coming soon!"

demo = gr.Interface(
    fn=hello,
    inputs=[],
    outputs="text",
    title="Snake AI (DQN)",
    description="Built from scratch using NumPy"
)

demo.launch()