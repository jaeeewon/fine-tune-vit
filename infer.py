from dataset import ds
from config import model_name_or_path
from util import getProcessor, wrappedTransform
from model import getModel
from train import getTrainer

from datasets import Dataset
from PIL import Image
from torch import Tensor, tensor
import numpy as np

TUNED_MODEL_PATH = "./vit-base-beans"
labels = ["angular_leaf_spot", "bean_rust", "healthy"]
sets = ["validation", "test", "train"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def evalDataset(setName="validation"):
    if setName not in sets:
        raise AssertionError("invalid setName " + setName)
    processor = getProcessor(TUNED_MODEL_PATH)
    corr = 0
    for vset in ds[setName]:
        infer = inferByPixel(
            processor(vset["image"], return_tensors="pt")["pixel_values"]
        )
        if infer == vset["labels"]:
            corr += 1
        else:
            print(
                f"[{setName}] misclassificated {labels[vset['labels']]} as {labels[infer]}"
            )
    total = len(ds[setName])
    print(f"accuracy of {setName} set: {corr / total * 100}%")
    print(f"{corr} out of {total}")


def inferByPixel(pixel):
    """pixel_values를 가지고 추론합니다."""
    model = getModel(TUNED_MODEL_PATH, labels)
    result = model.forward(pixel)
    # result: ImageClassifierOutput(loss=None, logits=tensor([[-1.8621,  3.1887, -1.5510]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
    logits = result.logits[0]

    inference = {"index": -1, "score": 0}
    for i, logit in enumerate(logits):
        score = logit.item()
        # prob = sigmoid(score)
        if inference["score"] < score:
            inference["index"] = i
            inference["score"] = score
        # print(f'{i}th {labels[i]} {prob * 100}%', end=', ')
    # print()

    # print(f'inference result: {labels[inference['index']]}({sigmoid(inference['score']) * 100})')
    return inference["index"]


def getPixel(imgPath):
    """pixel_values를 포함하는 어떠한 것을 반환합니다."""
    processor = getProcessor(TUNED_MODEL_PATH)
    image = Image.open(imgPath).convert("RGB")
    ft = processor(image, return_tensors="pt")
    return ft


def inferByPath(imgPath):
    """입력한 경로의 이미지를 inference합니다."""
    ft = getPixel(imgPath)  # {pixel_values}
    # print(ft)
    return inferByPixel(ft["pixel_values"])


def inferCustomData():
    """해당 파일의 inference를 불러옵니다."""
    files = ["./dataset/test.png", "./dataset/test2.jpg"]
    for path in files:
        print(f"{path} infered as {labels[inferByPath(path)]}")


if __name__ == "__main__":
    inferCustomData()

    for setName in sets:
        print(f"{'=' * 10} {setName} set ({len(ds[setName])}개) {'=' * 10}")
        evalDataset(setName)
