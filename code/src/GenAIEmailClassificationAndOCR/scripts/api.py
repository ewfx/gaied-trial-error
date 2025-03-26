import os
import extract_msg  # For .msg files
import eml_parser  # For .eml files
import pytesseract  # OCR for images
from pdf2image import convert_from_bytes
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from Model import classify_email_text , extract_key_details  # Import classification & extraction functions

app = FastAPI()

def extract_text_from_msg(file_path):
    msg = extract_msg.Message(file_path)
    return msg.body if msg.body else ""

def extract_text_from_eml(file_content):
    parser = eml_parser.EmlParser()
    parsed_eml = parser.decode_email_bytes(file_content.read())
    return parsed_eml.get("body", "")

def perform_ocr(file_bytes):
    image = Image.open(file_bytes)
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(file_bytes):
    images = convert_from_bytes(file_bytes)
    return "\n".join([pytesseract.image_to_string(img) for img in images])

@app.post("/classify-email")
async def classify_email(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1].lower()
    extracted_text = ""
    
    if file_extension == "msg":
        with open("temp.msg", "wb") as f:
            f.write(await file.read())
        extracted_text = extract_text_from_msg("temp.msg")
    elif file_extension == "eml":
        extracted_text = extract_text_from_eml(file.file)
    else:
        return {"error": "Unsupported file format"}
    
    request_type = classify_email(extracted_text)
    #key_details = extract_key_details(extracted_text)
    
    return {
        "Request Type": request_type,
       # "Extracted Details": key_details
    }
