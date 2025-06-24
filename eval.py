
def eval(trainer, prepared_ds):
    metrics = trainer.evaluate(prepared_ds['validation'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
