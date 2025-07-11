# import os
# import json
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
# from dotenv import load_dotenv
# import re
# from file_handler import read_text

# load_dotenv()

# def load_resources():
#     """Load FAISS index and metadata."""
#     try:
#         index = faiss.read_index("embeddings/index.faiss")
#         with open("embeddings/chunks_metadata.json", "r", encoding="utf-8") as f:
#             metadata = json.load(f)
#         model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
#         return index, metadata, model
#     except Exception as e:
#         print(f"Error loading resources: {e}")
#         return None, None, None

# def setup_gemini():
#     try:
#         api_key = os.getenv("GEMINI_API_KEY")
#         model_name = os.getenv("GEMINI_MODEL")
#         if not api_key:
#             raise ValueError("GEMINI_API_KEY environment variable not set.")
#         genai.configure(api_key=api_key)
#         model = genai.GenerativeModel(model_name)
#         return model
#     except Exception as e:
#         print(f"Error setting up Gemini: {e}")
#         return None

# def check_homework(homework_text, topic, grade, lang="uz"):
#     """Check homework using Gemini, provide feedback and a 1-5 grade for each problem."""
#     if not homework_text or len(homework_text.strip()) < 10:
#         return {"error": "Homework text is empty or too short."}

#     index, metadata, model = load_resources()
#     if not index or not metadata or not model:
#         return {"error": "Failed to load FAISS resources."}

#     gemini_model = setup_gemini()
#     if not gemini_model:
#         return {"error": "Failed to set up Gemini."}

#     lang_map = {"uz": "uzbek", "ru": "russian", "en": "english"}
#     if lang not in lang_map:
#         return {"error": f"Unsupported language. Choose 'uz' for Uzbek, 'ru' for Russian, or 'en' for English."}
#     full_lang = lang_map[lang]

#     try:
#         homework_embedding = model.encode(homework_text)
#         homework_embedding = homework_embedding / np.linalg.norm(homework_embedding)
#         homework_embedding = np.array([homework_embedding]).astype('float32')

#         problem_count = len(re.findall(r'Masala \d+', homework_text)) or 5
#         D, I = index.search(homework_embedding, k=problem_count)

#         print(f"Homework Text: {homework_text[:100]}...")
#         print(f"Distances: {D[0]}")
#         print(f"Indices: {I[0]}")

#         distance_threshold = 0.8
#         if all(d > distance_threshold for d in D[0]):
#             return {"error": f"No relevant content found for topic '{topic}' in the database."}

#         relevant_chunks = [metadata[i]["text"][:500] + "..." if len(metadata[i]["text"]) > 500 else metadata[i]["text"] for i in I[0]]

#         prompt = f"""
#         Siz {grade}-sinf o'quvchisining '{topic}' mavzusidagi uy vazifasini baholaydigan o'qituvchisiz.
#         O'quvchining topshirgan matni quyidagicha:
#         \"{homework_text}\"

#         Baholash mezonlari:
#         - Savolga to‘g‘ri, aniq va mavzuga mos javob berilganmi?
#         - Mazmuni darslikdagi ma’lumotlarga mos keladimi?
#         - Har bir javob haqida 2–4 jumladan iborat tushunarli fikr bildiring
#         - Har bir javobga alohida paragrafda baho (1-5) qo‘ying
#         - Har bir javobga to‘g‘ri / noto‘g‘ri / qisman to‘g‘ri deb belgilang
#         - Umumiy fikr va umumiy baho ham ajratilgan paragrafda yozilsin

#         MUHIM: O'quvchining yozgan javoblarini tuzatmang. Faqat qanday yozilgan bo‘lsa, shunga qarab baho bering.
#         Javobda aniqlik yetishmasa yoki noto‘g‘ri bo‘lsa, shuni aniqlab ko‘rsating, lekin o‘zingiz to‘g‘rilamang.

#         Javob formatini quyidagicha saqlang (paragraflar bo‘lsin):

#         Problem 1:
#         [Fikr-mulohaza 2-4 jumla. To‘g‘ri / noto‘g‘ri / qisman to‘g‘ri degan belgi bilan yakunlang.]
#         Grade 1: [2-5]

#         Problem 2:
#         [...]

#         Problem N:
#         [...]
        
#         Overall Feedback:
#         [Umumiy baho va xulosa, alohida paragrafda.]

#         Overall Grade:
#         [2-5]

#         Faqat shu formatda va {full_lang} tilida javob bering.

#         """

#         response = gemini_model.generate_content(prompt)
#         response_text = response.text.strip()

#         evaluations = []
#         overall_feedback = "No overall feedback provided by Gemini."
#         overall_grade = 1
#         problem_pattern = re.compile(
#             r'Problem (\d+):\s*(.*?)(?:\n|$)\s*Grade \1:\s*(\d)',
#             re.DOTALL
#         )
#         overall_pattern = re.compile(
#             r'Overall Feedback:\s*(.*?)(?:\n|$)\s*Overall Grade:\s*(\d)',
#             re.DOTALL
#         )

#         for match in problem_pattern.finditer(response_text):
#             problem_num = match.group(1)
#             feedback = match.group(2).strip()
#             grade = int(match.group(3))
#             grade = max(2, min(5, grade))
#             evaluations.append({
#                 "problem": problem_num,
#                 "feedback": feedback,
#                 "grade": grade
#             })

#         overall_match = overall_pattern.search(response_text)
#         if overall_match:
#             overall_feedback = overall_match.group(1).strip()
#             overall_grade = int(overall_match.group(2))
#             overall_grade = max(2, min(5, overall_grade))

#         if not evaluations:
#             return {
#                 "error": "No valid evaluations parsed from Gemini response.",
#                 "raw_response": response_text
#             }

#         return {
#             "evaluations": evaluations,
#             "overall_feedback": overall_feedback,
#             "overall_grade": overall_grade
#         }

#     except Exception as e:
#         print(f"Error checking homework: {e}")
#         return {
#             "error": f"Failed to process homework: {e}",
#             "raw_response": response_text if 'response_text' in locals() else ""
#         }



# import os
# import json
# import faiss
# import numpy as np
# import google.generativeai as genai
# from dotenv import load_dotenv
# import re
# import requests
# import time
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# load_dotenv()

# def load_resources():
#     """Load FAISS index and metadata."""
#     try:
#         index = faiss.read_index("embeddings/index.faiss")
#         with open("embeddings/database.json", "r", encoding="utf-8") as f:
#             metadata = json.load(f)
#         return index, metadata
#     except Exception as e:
#         print(f"Error loading resources: {e}")
#         return None, None

# def setup_gemini():
#     """Set up Gemini API."""
#     try:
#         api_key = os.getenv("GEMINI_API_KEY")
#         model_name = os.getenv("GEMINI_MODEL")
#         if not api_key:
#             raise ValueError("GEMINI_API_KEY environment variable not set.")
#         genai.configure(api_key=api_key)
#         model = genai.GenerativeModel(model_name)
#         return model
#     except Exception as e:
#         print(f"Error setting up Gemini: {e}")
#         return None

# def setup_jina_session():
#     """Set up Jina AI API session with retries."""
#     session = requests.Session()
#     retries = Retry(total=3, backoff_factor=5, status_forcelist=[429, 500, 502, 503, 524])
#     session.mount('https://', HTTPAdapter(max_retries=retries))
#     return session

# def get_jina_embedding(text, session):
#     url = 'https://api.jina.ai/v1/embeddings'
#     headers = {
#         'Content-Type': 'application/json',
#         'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}'
#     }
#     data = {
#         "model": "jina-embeddings-v2-base-en",
#         "input": [{"text": text}]
#     }
#     try:
#         response = session.post(url, headers=headers, json=data, timeout=60)
#         response.raise_for_status()
#         embeddings = response.json().get('data', [])
#         if not embeddings:
#             raise ValueError("No embeddings returned by Jina AI API.")
#         embedding = np.array(embeddings[0]['embedding'], dtype=np.float32)
#         print(f"✅ Embedding generated with dimension {embedding.shape[0]}")
#         return embedding
#     except requests.RequestException as e:
#         error_msg = str(e)
#         if hasattr(e, 'response') and e.response is not None:
#             error_msg += f" - Response: {e.response.text}"
#         print(f"Failed to get Jina embedding: {error_msg}")
#         return None
#     except ValueError as e:
#         print(f"Failed to process Jina response: {str(e)}")
#         return None

# def check_homework(homework_text=None, topic=None, grade=10, lang="uz", question=None, student_answer=None):
#     """
#     Evaluate homework based on provided text OR a specific question-answer pair.
#     Returns evaluations and grades using Gemini.
#     """
#     gemini_model = setup_gemini()
#     if not gemini_model:
#         return {"error": "Failed to set up Gemini."}

#     lang_map = {"uz": "uzbek", "ru": "russian", "en": "english"}
#     if lang not in lang_map:
#         return {"error": f"Unsupported language. Choose 'uz', 'ru', or 'en'."}
#     full_lang = lang_map[lang]

#     try:
#         if question and student_answer:
#             session = setup_jina_session()
#             embedding = get_jina_embedding(question + " " + student_answer, session)

#             if embedding is None:
#                 return {"error": "Failed to get embedding from Jina API."}

#             index, metadata = load_resources()
#             if not index or not metadata:
#                 return {"error": "Failed to load FAISS resources."}

#             embedding = embedding / np.linalg.norm(embedding)
#             embedding = np.array([embedding]).astype('float32')

#             D, I = index.search(embedding, k=5)  # top 5 related textbook chunks

#             valid_indices = [i for i in I[0] if i >= 0 and i < len(metadata)]
#             if not valid_indices:
#                 return {"error": "No relevant context found in textbook database."}

#             relevant_chunks = [
#                 metadata[i]["text"][:500] + "..." if len(metadata[i]["text"]) > 500 else metadata[i]["text"]
#                 for i in valid_indices
#             ]

#             context_knowledge = "\n\n".join(relevant_chunks)

#             prompt = f"""
#                 Siz {grade}-sinf o'quvchisining quyidagi savolga bergan javobini faqat darslikdagi ma’lumotlarga asoslangan holda baholovchi o'qituvchisiz.

#                 📚 Darslikdan topilgan kontekst (faqat mana shu ma’lumotlar asosida baholang):
#                 \"\"\"
#                 {context_knowledge}
#                 \"\"\"

#                 Savol: {question}

#                 O‘quvchining javobi: {student_answer}

#                 Baholash mezonlari:
#                 - Javob savolga to‘g‘ri, aniq va mavzuga mos berilganmi?
#                 - Mazmuni darslikdagi ma’lumotlarga mos keladimi?
#                 - 2–4 jumladan iborat tushunarli fikr bildiring
#                 - Javobga to‘g‘ri / noto‘g‘ri / qisman to‘g‘ri degan belgi qo‘ying
#                 - Javobga 2 dan 5 gacha baho qo‘ying
#                 - Umumiy xulosa ham bering

#                 Javob formatini quyidagicha bo‘lsin:

#                 Problem 1:
#                 [Fikr-mulohaza 2-4 jumla. To‘g‘ri / noto‘g‘ri / qisman to‘g‘ri degan belgi bilan yakunlang.]
#                 Grade 1: [2-5]

#                 Overall Feedback:
#                 [Umumiy xulosa va tushuntirish]

#                 Overall Grade:
#                 [2-5]

#                 Faqat {full_lang} tilida va faqat darslik ma’lumotlariga asoslangan holda javob yozing.
#             """

#             response = gemini_model.generate_content(prompt)
#             response_text = response.text.strip()


#         elif homework_text:
#             # OCR HOMEWORK MODE (Use your existing logic here)
#             return {"error": "OCR-based homework grading not included in this snippet. Provide QA pair."}

#         else:
#             return {"error": "You must provide either homework_text or question and student_answer."}

#         # --- Parse model output ---
#         evaluations = []
#         overall_feedback = "No overall feedback provided."
#         overall_grade = 1

#         problem_pattern = re.compile(r'Problem (\d+):\s*(.*?)(?:\n|$)\s*Grade \1:\s*(\d)', re.DOTALL)
#         overall_pattern = re.compile(r'Overall Feedback:\s*(.*?)(?:\n|$)\s*Overall Grade:\s*(\d)', re.DOTALL)

#         for match in problem_pattern.finditer(response_text):
#             evaluations.append({
#                 "problem": match.group(1),
#                 "feedback": match.group(2).strip(),
#                 "grade": max(2, min(5, int(match.group(3))))
#             })

#         overall_match = overall_pattern.search(response_text)
#         if overall_match:
#             overall_feedback = overall_match.group(1).strip()
#             overall_grade = max(2, min(5, int(overall_match.group(2))))

#         return {
#             "evaluations": evaluations,
#             "overall_feedback": overall_feedback,
#             "overall_grade": overall_grade
#         }

#     except Exception as e:
#         return {
#             "error": f"Failed to evaluate: {str(e)}",
#             "raw_response": response_text if 'response_text' in locals() else ""
#         }

# if __name__ == "__main__":
#     question = "X topologik fazo bo‘lsin va A uning ichidagi qism to‘plam bo‘lsin. Isbotlang: A ning yopilishi, cl(A), A va A ning chegaraviy nuqtalari to‘plami birikmasiga teng."
#     answer = "A ning yopilishi, cl(A), A ni o‘z ichiga oluvchi eng kichik yopiq to‘plamdir. Bu A va A ning barcha chegaraviy nuqtalaridan iborat bo‘ladi."
#     result = check_homework(topic="Algebraik ifodalar", grade=8, lang="uz", question=question, student_answer=answer)

#     print("\n--- NATIJA ---")
#     print(json.dumps(result, ensure_ascii=False, indent=2))

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
JINA_API_KEY = "jina_b5d8ea61235c48ad9e5af70719254fce5XsUnWspjaU21021cx5XvH6Zqbll"
GEMINI_API_KEY = "AIzaSyAL2dFskGajHSf0N3LbYsClqs_9LvAc3tQ"
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

