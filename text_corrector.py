import re
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from image_processor import extract_text_from_image

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

def apply_regex_corrections(text):
    corrections = [
        (r'\bv(\d+)\b', r'√\1'),  
        (r'\bsqrt\s*(\d+)', r'√\1'),  
        (r'\bJ(\d+)\b', r'√\1'),  

        (r'\b6\b', r'⋅'),  
        (r'[×xX]', r'⋅'),  
        (r'\*', r'⋅'), 

        (r'÷', r'/'),  
        (r'\\', r'/'), 
        (r'\bl\b', r'/'),  

        (r'(\d+)\^2\b', r'\1²'),  
        (r'(\d+)\^3\b', r'\1³'),  
        (r'(\d+)\^\s*(\d+)', r'\1^\2'), 

        (r'\bpi\b', r'π'), 
        (r'\bn\b', r'π'),  
        (r'[θΘ]', r'θ'),  
        (r'\b0\b', r'θ'), 

        (r'<\s*([A-Z]{1,3})', r'∠\1'),  
        (r'°', r'°'),  
        (r'\bo\b', r'°'), 
        (r'\bO\b', r'°'), 

        (r'\{', r'('), 
        (r'\}', r')'),  
        (r'\[\s*', r'('), 
        (r'\]\s*', r')'), 

        (r'±', r'±'),  
        (r'=', r'='),  
        (r'–', r'-'),  
        (r'−', r'-'), 

        (r'\s+', r' '),  
        (r'^\s+|\s+$', r''),  
    ]
    corrected_text = text
    for pattern, replacement in corrections:
        corrected_text = re.sub(pattern, replacement, corrected_text)
    return corrected_text

def validate_with_gemini(text, model_name, lang="uz"):
    try:
        model = genai.GenerativeModel(model_name)
        prompt = (
            "You are a text correction assistant for student homework across various subjects. "
            "The following text is extracted from a student's homework via OCR and may contain typos, misread characters, or formatting errors. "
            "Your task is to correct only clear OCR errors (e.g., 'TASODHTY' to 'TASODIFIY', 'tshblandi' to 'tashlandi', or misread symbols like 'l' to '1') "
            "and standardize formatting (e.g., consistent spacing, proper punctuation). "
            "DO NOT modify the content, structure, or answers unless they are unreadable or clearly incorrect due to OCR errors. "
            "Preserve the student's questions and answers exactly as written, even if they seem incomplete, incorrect, or subject-specific. "
            "For example, if a probability answer lists only '(1,1), (1,2)', keep it as is, or if a history answer mentions a specific event, do not rephrase it. "
            "If the text includes diagrams or illustrations, describe them clearly and concisely in the output, noting their relevance to the content. "
            f"The text is in {lang} language, so ensure the output is in {lang} with appropriate terminology for the subject. "
            "Return the corrected text with minimal changes, suitable for homework evaluation.\n\nText:\n" + text
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except GoogleAPIError as e:
        print(f"Gemini API error: {str(e)}")
        return text
    except Exception as e:
        print(f"Error validating with Gemini: {str(e)}")
        return text

def correct_ocr_text(raw_text):
    """Correct OCR text, fixing math symbol errors and standardizing output."""
    try:
        if not raw_text:
            return ""
        api_key, model_name = load_env()
        configure_gemini(api_key)
        corrected_text = apply_regex_corrections(raw_text)
        final_text = validate_with_gemini(corrected_text, model_name)
        final_text = re.sub(r'√(\d+)', r'sqrt(\1)', final_text)
        return final_text.strip()
    except Exception as e:
        print(f"Error correcting text: {str(e)}")
        return raw_text
