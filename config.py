from transformers import TrainingArguments

model_name = "google/vit-base-patch16-224-in21k"
model_path="./vit-base-beans"
labels=["angular_leaf_spot", "bean_rust", "healthy"]

training_args = TrainingArguments(
    output_dir="./vit-base-beans",
    per_device_train_batch_size=16,
    eval_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="tensorboard",
    load_best_model_at_end=True,
)
