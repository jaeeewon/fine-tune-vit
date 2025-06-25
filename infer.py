from dataset import ds
from util import getProcessor
from model import getModel

from PIL import Image
from torch import Tensor
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification


class Inference:
    sets: list[str] = ["validation", "test", "train"]

    def __init__(self, model_path: str, labels: list[str]):
        self.model_path: str = model_path
        self.labels: list[str] = labels
        self.processor: ViTImageProcessor = getProcessor(self.model_path)
        self.model: ViTForImageClassification = getModel(self.model_path, labels)

    def __inferByPixel(self, pixel: Tensor):
        """pixel_values를 가지고 추론합니다."""
        result = self.model.forward(pixel)
        # result: ImageClassifierOutput(loss=None, logits=tensor([[-1.8621,  3.1887, -1.5510]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)

        logits: Tensor = result.logits[0]

        prob = Inference.softmax_log_sum_exp_trick(logits)
        return np.argmax(prob)

    def __getPixel(self, image: Image):
        """pixel_values를 포함하는 어떠한 것을 반환합니다."""
        return self.processor(image, return_tensors="pt")["pixel_values"]
        # transformers/models/vit/image_processing_vit.py l284, l285에서 이미지 픽셀 정보를 반환함을 확인함

    def __getPixelByImgPath(self, imgPath):
        """pixel_values를 포함하는 어떠한 것을 반환합니다."""
        image = Image.open(imgPath).convert("RGB")
        return self.__getPixel(image)

    def inferByPath(self, imgPath):
        """입력한 경로의 이미지를 inference합니다."""
        px = self.__getPixelByImgPath(imgPath)
        return self.__inferByPixel(px)

    def inferCustomData(self, files=["./dataset/test.png", "./dataset/test2.jpg"]):
        """해당 파일의 inference를 불러옵니다."""
        for path in files:
            print(f"{path} infered as {self.labels[self.inferByPath(path)]}")

    def evalDataset(self, setName="validation"):
        if setName not in Inference.sets:
            raise AssertionError(
                f"invalid setName {setName}: {Inference.sets} are the only available!"
            )
        corr = 0
        for vset in ds[setName]:
            infer = self.__inferByPixel(self.__getPixel(vset["image"]))
            if infer == vset["labels"]:
                corr += 1
            else:
                print(
                    f"[{setName}] misclassificated {self.labels[vset['labels']]} as {self.labels[infer]}"
                )
        total = len(ds[setName])
        print(f"accuracy of {setName} set: {corr / total * 100}%")
        print(f"{corr} out of {total}")

    def testInference(self):
        self.inferCustomData()
        for setName in Inference.sets:
            print(f"{'=' * 10} {setName} set ({len(ds[setName])}개) {'=' * 10}")
            self.evalDataset(setName)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(logits: Tensor):
        expScore: np.float64 = [np.exp(logit.item()) for logit in logits]
        expSum = np.sum(expScore)
        probs: list[np.float64] = [(e / expSum) for e in expScore]
        return probs

    @staticmethod
    def softmax_log_sum_exp_trick(logits: Tensor):
        maxScore = logits.max().item()
        lseScore: np.float64 = [np.exp(logit.item() - maxScore) for logit in logits]
        lseSum = np.sum(lseScore)
        probs: list[np.float64] = [l / lseSum for l in lseScore]
        return probs
