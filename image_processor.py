import os
import base64
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
import io

def load_env():
    """Load environment variables from .env file."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")
    if not api_key or not model_name:
        raise ValueError("GEMINI_API_KEY or GEMINI_MODEL not set in .env")
    return api_key, model_name

def configure_gemini(api_key):
    """Configure Gemini API with API key."""
    genai.configure(api_key=api_key)

def encode_image(image):
    """Encode a PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG") 
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_mime_type(file_path):
    """Determine correct MIME type for the file."""
    extension = os.path.splitext(file_path)[1][1:].lower()
    mime_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "bmp": "image/bmp",
        "gif": "image/gif"
    }
    return mime_types.get(extension, "image/png") 

def extract_text_from_image(image, model, file_path):
    """Extract text from a PIL Image using Gemini API."""
    try:
        image_data = encode_image(image)
        prompt = [
            {
                "mime_type": get_mime_type(file_path),
                "data": image_data
            },
            {"text": "Extract all text from the image, including mathematical expressions."}
        ]
        response = model.generate_content(prompt)
        extracted_text = response.text.strip()
        return extracted_text if extracted_text else None
    except GoogleAPIError as e:
        print(f"Gemini API error for {file_path}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing image {file_path}: {str(e)}")
        return None

def extract_text_from_pdf(pdf_path, model):
    try:
        images = convert_from_path(pdf_path, first_page=1, last_page=5)
        text = ""
        for i, img in enumerate(images, 1):
            print(f"Processing PDF page {i} of {pdf_path}...")
            page_text = extract_text_from_image(img, model, pdf_path)
            if page_text:
                text += page_text + "\n"
                print(f"Extracted text from {pdf_path}:\n{text}\n{'-'*50}")
        return text.strip() if text else None
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return None

def extract_text_from_file(file_path):
    """Extract text from a file (image or PDF) using Gemini API."""
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None

    try:
        api_key, model_name = load_env()
        configure_gemini(api_key)
        model = genai.GenerativeModel(model_name)

        ext = os.path.splitext(file_path)[1].lower()
        if ext in {'.png', '.jpg', '.jpeg'}:
            img = Image.open(file_path)
            text = extract_text_from_image(img, model, file_path)
            return text if text else "No text extracted from image"
        elif ext == '.pdf':
            text = extract_text_from_pdf(file_path, model)
            return text if text else "No text extracted from PDF"
        else:
            print(f"Unsupported file extension: {ext}")
            return "Unsupported file type"
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return f"Error processing file: {str(e)}"