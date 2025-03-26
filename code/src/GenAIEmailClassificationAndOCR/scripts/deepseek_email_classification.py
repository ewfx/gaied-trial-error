import os
import json
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import re

# Define request types for classification
LABELS = ["Adjustment", "AU Transfer", "Closing Notice", "Commitment Change", "Fee Payment", "Money Movement Inbound", "Money Movement Outbound"]

# Load tokenizer and model from Hugging Face
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
TOKENIZER.pad_token = TOKENIZER.eos_token  # Ensure padding token is set

# Load model with a classification head
MODEL = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    problem_type="single_label_classification"
)

# Reinitialize classification layer
MODEL.classifier = nn.Linear(MODEL.config.hidden_size, len(LABELS))
MODEL.classifier.weight.data.normal_(mean=0.0, std=MODEL.config.initializer_range)
MODEL.classifier.bias.data.zero_()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL.to(device)

def tokenize_function(examples):
    return TOKENIZER(examples["email_text"], padding=True, truncation=True, max_length=1024)

def load_training_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = [item["email_text"].strip()[:1024] for item in data]
    labels = [LABELS.index(item["label"]) if item["label"] in LABELS else -1 for item in data]
    
    dataset = Dataset.from_dict({
        "email_text": texts,
        "labels": labels
    })
    return dataset.map(tokenize_function, batched=True)

def train_model(dataset_path):
    dataset = load_training_data(dataset_path)
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="no",
        per_device_train_batch_size=1,  # Set batch size to 1
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
    )
    
    trainer = Trainer(
        model=MODEL,
        args=training_args,
        train_dataset=dataset,
        tokenizer=TOKENIZER,
    )
    
    trainer.train()
    MODEL.save_pretrained("./trained_model")
    print("Model training complete using DeepSeek Hugging Face model!")

def classify_email(email_text):
    inputs = TOKENIZER(email_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to device
    
    with torch.no_grad():
        outputs = MODEL(**inputs)
    
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return LABELS[predicted_class]

def extract_key_details(text):
    amount = re.search(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', text)
    date = re.search(r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b', text)
    deal_name = re.search(r'Deal Name[:\-]?\s*(\w+)', text, re.IGNORECASE)
    
    return {
        "Amount": amount.group(1) if amount else None,
        "Date": date.group(1) if date else None,
        "Deal Name": deal_name.group(1) if deal_name else None
    }

# Example usage
train_model("data/email_training_data.json")
