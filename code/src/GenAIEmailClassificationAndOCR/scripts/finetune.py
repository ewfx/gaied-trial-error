from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch

# Define request types for classification
LABELS = ["Adjustment", "AU Transfer", "Closing Notice", "Commitment Change", "Fee Payment", "Money Movement Inbound", "Money Movement Outbound"]

# Load base model for fine-tuning
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS))

def fine_tune_model():
    dataset = load_dataset("csv", data_files={"train": "data/fine_tune_data.csv"})
    
    def preprocess_data(examples):
        if isinstance(examples["text"], str):
            return TOKENIZER(examples["text"], padding="max_length", truncation=True, max_length=1024)
        return TOKENIZER([str(txt) for txt in examples["text"]], padding="max_length", truncation=True, max_length=1024)
    
    encoded_dataset = dataset.map(preprocess_data, batched=True)
    
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        evaluation_strategy="no",
        save_strategy="no",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True
    )
    
    trainer = Trainer(
        model=MODEL,
        args=training_args,
        train_dataset=encoded_dataset["train"],
    )
    
    trainer.train()
    trainer.save_model("./fine_tuned_model")
    TOKENIZER.save_pretrained("./fine_tuned_model")

if __name__ == "__main__":
    fine_tune_model()
