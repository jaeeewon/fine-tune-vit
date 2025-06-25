from dataset import ds
from config import model_name_or_path
from util import getProcessor, wrappedTransform
from model import getModel
from train import train
from eval import eval


def main():
    processor = getProcessor(model_name_or_path)
    prepared_ds = ds.with_transform(wrappedTransform(processor))
    labels = ds["train"].features["labels"].names

    model = getModel(model_name_or_path, labels)

    trainer = train(model, prepared_ds, processor)

    eval(trainer, prepared_ds)


if __name__ == "__main__":
    main()
