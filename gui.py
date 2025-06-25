import gradio as gr
from PIL import Image
import numpy as np

from infer import Inference
from config import model_path, labels


def predict(image: Image):
    inference = Inference(model_path, labels)
    infProbs = inference.inferByImage(image)
    inf = np.argmax(infProbs)
    prs = [f"{l}: {(float(pr) * 100):.2f}%" for l, pr in zip(labels, infProbs)]
    return f"{labels[inf]} ({inf})\n{", ".join(prs)}"


demo = gr.Interface(
    fn=predict,
    inputs=["image"],
    outputs=["text"],
    description="단순히 가장 높은 확률을 갖는 걸 선택하도록 했습니다. 내일까지 결정이론에서 배운 내용을 적용해 수정하겠습니다.",
)

demo.launch()
# src: https://www.gradio.app/guides/quickstart#building-your-first-demo
