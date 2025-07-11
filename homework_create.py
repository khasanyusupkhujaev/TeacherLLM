import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def load_resources():
    """Load FAISS index, metadata, and SentenceTransformer model."""
    try:
        index = faiss.read_index("embeddings/teacher_rag_index.faiss")
        with open("embeddings/chunks_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        print(f"Index loaded: ntotal={index.ntotal}, is_trained={index.is_trained}")
        return index, metadata, model
    except Exception as e:
        print(f"Error loading resources: {e}")
        return None, None, None

def configure_gemini():
    """Configure Gemini API with environment variables."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_MODEL")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        if not model_name:
            raise ValueError("GEMINI_MODEL not found in environment variables.")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return None

def create_homework(topic, grade, lang="uz", number_of_questions=3):
    """Create a homework assignment using Gemini based on topic, grade, language, and number of questions."""
    index, metadata, model = load_resources()
    gemini_model = configure_gemini()
    
    if not index or not metadata or not model:
        return "Failed to load resources."
    if not gemini_model:
        return "Failed to configure Gemini."

    lang_map = {"uz": "uzbek", "ru": "russian"}
    if lang not in lang_map:
        return "Unsupported language. Choose 'uz' for Uzbek or 'ru' for Russian."
    full_lang = lang_map[lang]

    try:
        topic_embedding = model.encode(topic)
        topic_embedding = np.array([topic_embedding]).astype('float32')

        D, I = index.search(topic_embedding, k=number_of_questions) 
        relevant_chunks = [metadata[i]["text"] for i in I[0]]

        print(f"Query Topic: {topic}")
        print(f"Distances: {D[0]}")
        print(f"Indices: {I[0]}")
        print(f"Relevant Chunks: {relevant_chunks}")

        distance_threshold = 3.0  
        print(f"Distance Threshold: {distance_threshold}")
        if all(d > distance_threshold for d in D[0]):
            return f"No such topic '{topic}' in the database."

        prompt = f"""
        You are an expert educator creating a homework assignment for Grade {grade} students on the topic '{topic}'.
        The assignment must be written in {full_lang.capitalize()}.
        Below is relevant content extracted from educational materials:
        
        """
        for i, chunk in enumerate(relevant_chunks, 1):
            prompt += f"Content {i}: {chunk[:500] + '...' if len(chunk) > 500 else chunk}\n"

        prompt += f"""
        Based on this content, create a homework assignment with {number_of_questions} questions tailored to the grade level.
        Each question should include:
        - A clear task or question related to the topic.
        - Instructions for what the student should do (e.g., solve a problem, write an explanation).
        - Ensure the questions are engaging and appropriate for Grade {grade} students.
        
        Format the output as follows:
        {'Uy ishi {grade}-sinf uchun - Mavzu: {topic}' if full_lang == 'uzbek' else 'Домашнее задание для {grade}-го класса - Тема: {topic}'}
        """
        for q in range(1, number_of_questions + 1):
            prompt += f"{'Savol {q}: [Question and instructions]' if full_lang == 'uzbek' else 'Вопрос {q}: [Question and instructions]'}\n"
        prompt += "Ensure each question is on a separate line."

        response = gemini_model.generate_content(prompt)
        homework_content = response.text.strip()

        return homework_content
    except Exception as e:
        return f"Error creating homework: {e}"