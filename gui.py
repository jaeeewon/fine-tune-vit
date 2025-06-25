import gradio as gr
from PIL import Image

from infer import Inference
from config import model_path, labels


def predict(image: Image):
    inference = Inference(model_path, labels)
    inf = inference.inferByImage(image)
    return f"{labels[inf]} ({inf})"


demo = gr.Interface(
    fn=predict,
    inputs=["image"],
    outputs=["text"],
)

demo.launch()
# src: https://www.gradio.app/guides/quickstart#building-your-first-demo
