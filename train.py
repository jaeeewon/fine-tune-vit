from transformers import Trainer

from config import training_args
from util import collate_fn, compute_metrics


def getTrainer(model, prepared_ds, processor):
    return Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["validation"],
        tokenizer=processor,
    )


def train(model, prepared_ds, processor):
    trainer = getTrainer(model, prepared_ds, processor)

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    return trainer
