import pdfplumber
import os
from PyPDF2 import PdfReader

grades = range(11,12)

log_file = "extraction_errors.log"

def log_error(message):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{message}\n")

def extract_with_pypdf2(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            extracted_text = page.extract_text()
            text += extracted_text or ""
        return text
    except Exception as e:
        log_error(f"PyPDF2 failed for {pdf_path}: {str(e)}")
        return ""

for i in grades:

    pdf_dir = f"books/{i}/"
    output_dir = f"books/{i}/"
    os.makedirs(output_dir, exist_ok=True)

    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            text = ""

            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        try:
                            extracted_text = page.extract_text()
                            text += extracted_text or ""
                        except Exception as e:
                            log_error(f"Error extracting text from page {page.page_number} in {pdf_path}: {str(e)}")
                            continue

            except Exception as e:
                log_error(f"pdfplumber failed for {pdf_path}: {str(e)}")
                text = extract_with_pypdf2(pdf_path)

            if text.strip():
                output_path = os.path.join(output_dir, pdf_file.replace(".pdf", ".txt"))
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Processed: {pdf_path}")
            else:
                log_error(f"No text extracted from {pdf_path}")
                print(f"Skipped (no text): {pdf_path}")

