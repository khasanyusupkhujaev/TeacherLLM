import os
import json
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

# --- Constants ---
JINA_API_KEY = ""
GEMINI_API_KEY = ""
JINA_URL = "https://api.jina.ai/v1/embeddings"
MODEL_NAME = "jina-embeddings-v4"
TOP_K = 1
SIMILARITY_THRESHOLD = 0.7

def load_resources():
    """Load FAISS index and metadata."""
    try:
        index = faiss.read_index("embeddings/index2.faiss")
        with open("embeddings/database2.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata
    except Exception as e:
        print(f"Error loading resources: {e}")
        return None, None

def setup_gemini():
    """Set up Gemini API."""
    try:
        api_key = GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        return model
    except Exception as e:
        print(f"Error setting up Gemini: {e}")
        return None

def setup_jina_session():
    """Set up Jina AI API session with retries."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=5, status_forcelist=[429, 500, 502, 503, 524])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def get_jina_embedding(text, session):
    """Get embedding from Jina."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    payload = {
        "model": MODEL_NAME,
        "input": [{"text": text}]
    }
    try:
        response = session.post(JINA_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            print("❌ Jina error response:", response.text)
            response.raise_for_status()
        embedding = response.json()["data"][0]["embedding"]
        return np.array(embedding, dtype="float32")
    except requests.RequestException as e:
        print(f"Failed to get Jina embedding: {str(e)}")
        return None

def get_context_for_question(question, session, index, metadata):
    """FAISS Search for context."""
    query_vector = get_jina_embedding(question, session)
    if query_vector is None:
        return None, None
    query_vector = query_vector.reshape(1, -1)
    distances, indices = index.search(query_vector, TOP_K)
    distance = distances[0][0]
    if distance > SIMILARITY_THRESHOLD:
        return None, distance
    matched_text = metadata[indices[0][0]]["text"]
    return matched_text, distance

def validate_answer_with_gemini(question, user_answer, correct_context, gemini_model):
    """Use Gemini to check answer."""
    prompt = f"""
Quyida savol, foydalanuvchi javobi va kontekst (to‘g‘ri javob mavjud bo‘lishi mumkin bo‘lgan matn) keltirilgan.

Savol:
{question}

Foydalanuvchi javobi:
{user_answer}

To‘g‘ri kontekst:
{correct_context}

Iltimos, quyidagi formatda javob bering:

1. Verdict: "To‘g‘ri" yoki "Noto‘g‘ri"
2. To‘g‘ri javob: Agar foydalanuvchi javobi noto‘g‘ri bo‘lsa, to‘g‘ri javobni yozing. Agar to‘g‘ri bo‘lsa, shunchaki "Javob to‘g‘ri" deb yozing.
"""
    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        verdict_match = re.search(r"1\.\s*Verdict:\s*(.*)", text)
        correct_answer_match = re.search(r"2\.\s*To‘g‘ri javob:\s*(.*?)(?:\n\d\.|\Z)", text, re.DOTALL)
        verdict = verdict_match.group(1).strip() if verdict_match else "Noma'lum"
        correct_answer = correct_answer_match.group(1).strip() if correct_answer_match else "Noma'lum"
        return verdict, correct_answer
    except Exception as e:
        print(f"Error validating with Gemini: {e}")
        return "Noma'lum", "Noma'lum"

def parse_homework_text(homework_text):
    """Parse homework text into question-answer pairs."""
    qa_pairs = []
    # Match questions and answers using regex
    pattern = r"(\d+\s*savol\.)(.*?)(Javob:.*?)(?=\d+\s*savol\.|$)"
    matches = re.finditer(pattern, homework_text, re.DOTALL)
    
    for match in matches:
        question_part = match.group(2).strip()
        answer_part = match.group(3).replace("Javob:", "").strip()
        qa_pairs.append({
            "question": question_part,
            "answer": answer_part
        })
    
    return qa_pairs

def check_homework(homework_text=None, topic=None, grade=10, lang="uz", question=None, student_answer=None):
    """
    Evaluate homework based on provided text OR a specific question-answer pair.
    Returns evaluations and grades using Gemini.
    """
    gemini_model = setup_gemini()
    if not gemini_model:
        return {"error": "Failed to set up Gemini."}

    lang_map = {"uz": "uzbek", "ru": "russian", "en": "english"}
    if lang not in lang_map:
        return {"error": f"Unsupported language. Choose 'uz', 'ru', or 'en'."}

    try:
        session = setup_jina_session()
        index, metadata = load_resources()
        if not index or not metadata:
            return {"error": "Failed to load FAISS resources."}

        evaluations = []
        overall_grade = 0
        overall_feedback = []

        if question and student_answer:
            # Handle single question-answer pair
            correct_context, distance = get_context_for_question(question, session, index, metadata)
            if not correct_context or distance > SIMILARITY_THRESHOLD:
                return {"error": "Savol bazada topilmadi yoki yetarlicha o‘xshash emas."}

            verdict, correct_answer = validate_answer_with_gemini(question, student_answer, correct_context, gemini_model)
            grade = 5 if verdict == "To‘g‘ri" else 2
            evaluations.append({
                "problem": "1",
                "feedback": f"Foydalanuvchi javobi: {student_answer}\nVerdict: {verdict}",
                "grade": grade
            })
            overall_grade = grade
            overall_feedback.append(correct_answer)

        elif homework_text:
            # Handle OCR-extracted homework text
            qa_pairs = parse_homework_text(homework_text)
            if not qa_pairs:
                return {"error": "No valid question-answer pairs found in homework text."}

            for idx, qa in enumerate(qa_pairs, 1):
                question = qa["question"]
                student_answer = qa["answer"]
                correct_context, distance = get_context_for_question(question, session, index, metadata)
                if not correct_context or distance > SIMILARITY_THRESHOLD:
                    evaluations.append({
                        "problem": str(idx),
                        "feedback": f"Savol: {question}\nFoydalanuvchi javobi: {student_answer}\nVerdict: Savol bazada topilmadi.",
                        "grade": 2
                    })
                    overall_feedback.append("Savol bazada topilmadi.")
                    continue

                verdict, correct_answer = validate_answer_with_gemini(question, student_answer, correct_context, gemini_model)
                grade = 5 if verdict == "To‘g‘ri" else 2
                evaluations.append({
                    "problem": str(idx),
                    "feedback": f"Savol: {question}\nFoydalanuvchi javobi: {student_answer}\nVerdict: {verdict}",
                    "grade": grade
                })
                overall_feedback.append(f"Savol {idx}: {correct_answer}")
                overall_grade += grade

            if evaluations:
                overall_grade = round(overall_grade / len(evaluations))
            else:
                overall_grade = 2

        else:
            return {"error": "You must provide either homework_text or question and student_answer."}

        return {
            "evaluations": evaluations,
            "overall_feedback": "\n".join(overall_feedback),
            "overall_grade": overall_grade
        }

    except Exception as e:
        return {"error": f"Failed to evaluate: {str(e)}"}

if __name__ == "__main__":
    question = "Tanga 3 marta tashlandi. Bunda qanday imkoniyatlar bo'lishi mumkin? Ularni yozib chiqing"
    answer = input("Sizning javobingiz: ")
    result = check_homework(topic="Probability", grade=8, lang="uz", question=question, student_answer=answer)

    print("\n--- NATIJA ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

