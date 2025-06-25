from transformers import ViTForImageClassification


def getModel(model_name, labels) -> ViTForImageClassification:
    return ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
    )
