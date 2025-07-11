import os

from pathlib import Path
from text_corrector import correct_ocr_text
from image_processor import extract_text_from_image

def save_text(text, image_path, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        return output_path

    except Exception as e:
        print(f"Error saving text to {output_path}: {str(e)}")
        return ""

def read_text(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading text from {file_path}: {str(e)}")
        return ""