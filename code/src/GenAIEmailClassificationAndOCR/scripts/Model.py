import torch
import torch.nn as nn
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer, softmax

# Define request types for classification
LABELS = ["Adjustment", "AU Transfer", "Closing Notice", "Commitment Change", "Fee Payment", "Money Movement Inbound", "Money Movement Outbound"]

# Load tokenizer and model from trained directory
MODEL_PATH ="trained_model"   #"fine_tuned_model"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(LABELS))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL.to(device)

def classify_email_text(email_text):
    inputs = TOKENIZER(email_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to device
    
    with torch.no_grad():
        outputs = MODEL(**inputs)
    
    probabilities = softmax(outputs.logits, dim=1).cpu().numpy()[0]
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    confidence_score = probabilities[predicted_class]
    
    return LABELS[predicted_class], confidence_score

def extract_key_details(text):
    amount = re.search(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', text)
    date = re.search(r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b', text)
    deal_name = re.search(r'Deal Name[:\-]?\s*(\w+)', text, re.IGNORECASE)
    
    return {
        "Amount": amount.group(1) if amount else None,
        "Date": date.group(1) if date else None,
        "Deal Name": deal_name.group(1) if deal_name else None
    }