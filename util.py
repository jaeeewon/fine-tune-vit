from evaluate import load
import torch
import numpy as np
from transformers import ViTImageProcessor

processor = {}

def getProcessor(model_name_or_path) -> ViTImageProcessor:
    if model_name_or_path in processor: return processor[model_name_or_path]
    processor[model_name_or_path] = ViTImageProcessor.from_pretrained(model_name_or_path)
    return processor[model_name_or_path]

def wrappedTransform(processor):
    def transform(example_batch):
        # Take a list of PIL images and turn them to pixel values
        inputs = processor([x for x in example_batch['image']], return_tensors='pt')

        # Don't forget to include the labels!
        inputs['labels'] = example_batch['labels']
        return inputs
    return transform

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

metric = load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
