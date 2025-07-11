# import json
# import faiss
# import numpy as np
# import requests
# import google.generativeai as genai

# # --- Constants ---
# JINA_API_KEY = "jina_baed60ee21474a95a290a9444bcbe3dc7449oaVkAZ5oE0WqcI2mp4lL0Itu"
# GEMINI_API_KEY = "AIzaSyAL2dFskGajHSf0N3LbYsClqs_9LvAc3tQ"
# JINA_URL = "https://api.jina.ai/v1/embeddings"
# MODEL_NAME = "jina-embeddings-v4"
# TOP_K = 5

# # --- Load FAISS index ---
# index = faiss.read_index("embeddings/index2.faiss")

# # --- Load database.json ---
# with open("embeddings/database2.json", "r", encoding="utf-8") as f:
#     database = json.load(f)  # assume it's a list of dicts

# # --- Configure Gemini ---
# genai.configure(api_key=GEMINI_API_KEY)
# gemini = genai.GenerativeModel("gemini-2.5-flash")

# # --- Function: get embedding from jina ---
# def get_embedding(text):
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {JINA_API_KEY}"
#     }

#     payload = {
#         "model": MODEL_NAME,
#         "input": [{"text": text}]
#     }

#     response = requests.post(JINA_URL, headers=headers, json=payload)

#     if response.status_code != 200:
#         print("❌ Jina error response:", response.text)
#         response.raise_for_status()

#     embedding = response.json()["data"][0]["embedding"]
#     return np.array(embedding, dtype="float32")


# # --- Function: search index ---
# def search_faiss(query_text):
#     query_vector = get_embedding(query_text).reshape(1, -1)
#     distances, indices = index.search(query_vector, TOP_K)
#     return [database[i]["text"] for i in indices[0] if i < len(database)]

# # --- Function: ask a question ---
# def ask_question(question):
#     chunks = search_faiss(question)
#     context = "\n\n".join(chunks)

#     prompt = f"""Quyidagi matnga asoslanib savolga javob bering:

# {context}

# Savol: {question}
# Javob:"""

#     response = gemini.generate_content(prompt)
#     return response.text.strip()

# # --- MAIN: test it ---
# if __name__ == "__main__":
#     question = input("Savolingizni yozing: ")
#     answer = ask_question(question)
#     print("\n📘 Javob:\n", answer)
    


# import fitz  # PyMuPDF
# import json
# import faiss
# import re
# import numpy as np
# import requests
# import google.generativeai as genai

# # Setup Gemini
# genai.configure(api_key="AIzaSyAL2dFskGajHSf0N3LbYsClqs_9LvAc3tQ")
# model = genai.GenerativeModel("gemini-2.5-flash")

# # Load FAISS and database
# index = faiss.read_index("embeddings/index2.faiss")
# with open("embeddings/database2.json", "r", encoding="utf-8") as f:
#     db_texts = json.load(f)

# # Embedding
# def get_jina_embedding(text):
#     response = requests.post(
#         "https://api.jina.ai/v1/embeddings",
#         headers={"Authorization": "Bearer jina_baed60ee21474a95a290a9444bcbe3dc7449oaVkAZ5oE0WqcI2mp4lL0Itu"},
#         json={"input": text, "model": "jina-embeddings-v4"}
#     )
#     return np.array(response.json()["data"][0]["embedding"], dtype=np.float32)

# # Gemini check
# def check_answer_with_gemini(question, answer, context):
#     prompt = f"""
# Quyida savol, berilgan javob va kontekst (to‘g‘ri javob mavjud bo‘lishi mumkin bo‘lgan matn) keltirilgan.

# Savol:
# {question}

# Javob:
# {answer}

# Kontekst:
# {context}

# Iltimos, quyidagi formatda javob bering:

# 1. Verdict: "To‘g‘ri" yoki "Noto‘g‘ri"
# 2. To‘g‘ri javob: Agar javob noto‘g‘ri bo‘lsa, to‘g‘ri javobni yozing. Agar to‘g‘ri bo‘lsa, shunchaki "Javob to‘g‘ri" deb yozing.
# """

#     response = model.generate_content(prompt)
#     text = response.text.strip()

#     # Robustly extract verdict and correct answer
#     verdict_match = re.search(r"1\.\s*Verdict:\s*(.*)", text)
#     correct_answer_match = re.search(r"2\.\s*To‘g‘ri javob:\s*(.*)", text)

#     verdict = verdict_match.group(1).strip() if verdict_match else "Noma'lum"
#     correct_answer = correct_answer_match.group(1).strip() if correct_answer_match else "Noma'lum"

#     return verdict, correct_answer



# # Extract Q&A from PDF
# def extract_qa_from_pdf(path):
#     doc = fitz.open(path)
#     print(f"📄 PDF has {len(doc)} pages")
#     text = ""
#     for page in doc:
#         page_text = page.get_text()
#         print(f"--- Page text ---\n{page_text}\n")
#         text += page_text + "\n"
#         text += page.get_text() + "\n"

#     # Find all question blocks using regex
#     qa_raw = re.findall(r"(\d+\s+savol:.*?(?=(\d+\s+savol:|$)))", text, flags=re.DOTALL)

#     qa_pairs = []
#     for qa, _ in qa_raw:
#         lines = qa.strip().split('\n')
#         if len(lines) >= 2:
#             question = lines[0].replace("savol:", "").strip()
#             answer_lines = [line.replace("Javob:", "").strip() for line in lines[1:]]
#             answer = ' '.join(answer_lines).strip()
#             qa_pairs.append((question, answer))
#     return qa_pairs


# # Main Pipeline
# def process_pdf(pdf_path):
#     qa_pairs = extract_qa_from_pdf(pdf_path)
#     results = []

#     for i, (question, given_answer) in enumerate(qa_pairs, start=1):
#         query_vector = get_jina_embedding(question)
#         D, I = index.search(np.array([query_vector]), k=1)
#         matched_context = db_texts[I[0][0]]

#         verdict, correct_answer = check_answer_with_gemini(question, given_answer, matched_context)

#         print(f"Q{i}: {question}")
#         print(f"A: {given_answer}")
#         print(f"✅ Verdict: {verdict}")
#         if verdict.lower().startswith("noto"):
#             print(f"To‘g‘ri javob (Gemini): {correct_answer}")
#         print("---\n")

#         results.append({
#             "question": question,
#             "answer": given_answer,
#             "verdict": verdict,
#             "correct_answer": correct_answer
#         })

#     return results

# if __name__ == "__main__":
#     results = process_pdf("images/JT.pdf")
#     for r in results:
#         print(f"Q: {r['question']}\nA: {r['answer']}\nVerdict: {r['verdict']}\nCorrect answer: {r['correct_answer']}---")

# import json
# import faiss
# import numpy as np
# import requests
# import google.generativeai as genai
# import re

# # --- Constants ---
# JINA_API_KEY="jina_b5d8ea61235c48ad9e5af70719254fce5XsUnWspjaU21021cx5XvH6Zqbll"
# GEMINI_API_KEY = "AIzaSyAL2dFskGajHSf0N3LbYsClqs_9LvAc3tQ"
# JINA_URL = "https://api.jina.ai/v1/embeddings"
# MODEL_NAME = "jina-embeddings-v4"
# TOP_K = 3  
# SIMILARITY_THRESHOLD = 0.9

# # --- Load FAISS index and database ---
# index = faiss.read_index("embeddings/index2.faiss")
# with open("embeddings/database2.json", "r", encoding="utf-8") as f:
#     database = json.load(f)

# # --- Configure Gemini ---
# genai.configure(api_key=GEMINI_API_KEY)
# gemini = genai.GenerativeModel("gemini-2.5-flash")

# # --- Get embedding from Jina ---
# def get_embedding(text):
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {JINA_API_KEY}"
#     }
#     payload = {
#         "model": MODEL_NAME,
#         "input": [{"text": text}]
#     }
#     response = requests.post(JINA_URL, headers=headers, json=payload)
#     if response.status_code != 200:
#         print("❌ Jina error response:", response.text)
#         response.raise_for_status()
#     embedding = response.json()["data"][0]["embedding"]
#     return np.array(embedding, dtype="float32")

# # --- FAISS Search ---
# def get_context_for_question(question):
#     query_vector = get_embedding(question).reshape(1, -1)
#     distances, indices = index.search(query_vector, TOP_K)
    
#     matched_texts = [database[i]["text"] for i in indices[0] if i >= 0]
#     all_text = "\n\n".join(matched_texts)
#     return all_text, distances[0][0]

# # --- Use Gemini to check answer ---
# def validate_answer_with_gemini(question, user_answer, correct_context):
#     prompt = f"""
# Quyida savol, foydalanuvchi javobi va kontekst (to‘g‘ri javob mavjud bo‘lishi mumkin bo‘lgan matn) keltirilgan.

# Savol:
# {question}

# Foydalanuvchi javobi:
# {user_answer}

# To‘g‘ri kontekst:
# {correct_context}

# Iltimos, quyidagi formatda javob bering:

# 1. Verdict: "To‘g‘ri" yoki "Noto‘g‘ri"
# 2. To‘g‘ri javob: Agar foydalanuvchi javobi noto‘g‘ri bo‘lsa, to‘g‘ri javobni yozing. Agar to‘g‘ri bo‘lsa, shunchaki "Javob to‘g‘ri" deb yozing.
# """

#     response = gemini.generate_content(prompt)
#     text = response.text.strip()

#     # Extract verdict and correct answer
#     verdict_match = re.search(r"1\.\s*Verdict:\s*(.*)", text)
#     correct_answer_match = re.search(r"2\.\s*To‘g‘ri javob:\s*(.*?)(?:\n\d\.|\Z)", text, re.DOTALL)

#     verdict = verdict_match.group(1).strip() if verdict_match else "Noma'lum"
#     correct_answer = correct_answer_match.group(1).strip() if correct_answer_match else "Noma'lum"

#     return verdict, correct_answer

# # --- Main Logic ---
# if __name__ == "__main__":
#     question = "Gerxard Shryoder hukumati iqtisodda va ijtimoiy sohada qanday chuqur islohotlar o‘tkazdi?"
#     print("Savol:", question)
#     user_answer = input("Sizning javobingiz: ")

#     correct_context, distance = get_context_for_question(question)

#     if distance > SIMILARITY_THRESHOLD:
#         print("❌ Savol bazada topilmadi yoki yetarlicha o‘xshash emas.")
#         exit()

#     verdict, correct_answer = validate_answer_with_gemini(question, user_answer, correct_context)

#     print("\nNatija:")
#     print("Foydalanuvchi javobi:", user_answer)
#     print("Verdict:", verdict)
#     print("To‘g‘ri javob:", correct_answer)


import json
import faiss
import numpy as np
import requests
import google.generativeai as genai
import re

# --- Constants ---
JINA_API_KEY = "jina_b5d8ea61235c48ad9e5af70719254fce5XsUnWspjaU21021cx5XvH6Zqbll"
GEMINI_API_KEY = "AIzaSyAL2dFskGajHSf0N3LbYsClqs_9LvAc3tQ"
JINA_URL = "https://api.jina.ai/v1/embeddings"
MODEL_NAME = "jina-embeddings-v4"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.7

# --- Load FAISS index and database ---
index = faiss.read_index("embeddings/index2.faiss")
with open("embeddings/database2.json", "r", encoding="utf-8") as f:
    database = json.load(f)

# --- Configure Gemini ---
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

# --- Get embedding from Jina ---
def get_embedding(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    payload = {
        "model": MODEL_NAME,
        "input": [{"text": text}]
    }
    response = requests.post(JINA_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print("❌ Jina error response:", response.text)
        response.raise_for_status()
    embedding = response.json()["data"][0]["embedding"]
    return np.array(embedding, dtype="float32")

# --- FAISS Search ---
def get_context_for_question(question):
    query_vector = get_embedding(question).reshape(1, -1)
    distances, indices = index.search(query_vector, TOP_K)
    
    # Collect matched chunks with their details
    matched_chunks = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and idx < len(database):
            matched_chunks.append({
                "index": int(idx),
                "text": database[idx]["text"],
                "distance": float(distances[0][i])
            })
    
    # Concatenate all matched texts for context
    all_text = "\n\n".join(chunk["text"] for chunk in matched_chunks)
    min_distance = float(distances[0][0]) if len(distances[0]) > 0 else float('inf')
    
    return all_text, min_distance, matched_chunks

# --- Use Gemini to check answer ---
def validate_answer_with_gemini(question, user_answer, correct_context):
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

    response = gemini.generate_content(prompt)
    text = response.text.strip()

    # Extract verdict and correct answer
    verdict_match = re.search(r"1\.\s*Verdict:\s*(.*)", text)
    correct_answer_match = re.search(r"2\.\s*To‘g‘ri javob:\s*(.*?)(?:\n\d\.|\Z)", text, re.DOTALL)

    verdict = verdict_match.group(1).strip() if verdict_match else "Noma'lum"
    correct_answer = correct_answer_match.group(1).strip() if correct_answer_match else "Noma'lum"

    return verdict, correct_answer

# --- Main Logic ---
if __name__ == "__main__":
    question = "Gerxard Shryoder hukumati iqtisodda va ijtimoiy sohada qanday chuqur islohotlar o‘tkazdi?"
    print("Savol:", question)
    user_answer = input("Sizning javobingiz: ")

    correct_context, distance, matched_chunks = get_context_for_question(question)

    # Print matched chunks
    print("\nFAISS tomonidan topilgan o‘xshash chunklar:")
    for i, chunk in enumerate(matched_chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Indeks: {chunk['index']}")
        print(f"  Masofa: {chunk['distance']:.4f}")
        print(f"  Matn: {chunk['text'][:200]}..." if len(chunk['text']) > 200 else f"  Matn: {chunk['text']}")
    
    if distance > SIMILARITY_THRESHOLD:
        print("❌ Savol bazada topilmadi yoki yetarlicha o‘xshash emas.")
        exit()

    verdict, correct_answer = validate_answer_with_gemini(question, user_answer, correct_context)

    print("\nNatija:")
    print("Foydalanuvchi javobi:", user_answer)
    print("Verdict:", verdict)
    print("To‘g‘ri javob:", correct_answer)