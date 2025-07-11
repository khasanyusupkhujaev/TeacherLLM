import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

def load_resources():
    """Load FAISS index, metadata, and SentenceTransformer model."""
    try:
        index = faiss.read_index("embeddings/teacher_rag_index.faiss")
        with open("embeddings/chunks_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        return index, metadata, model
    except Exception as e:
        print(f"Error loading resources: {e}")
        return None, None, None

def configure_gemini():
    """Configure Gemini API."""
    try:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found.")
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return None

def preprocess_text(text):
    """Preprocess text for embedding."""
    text = text.lower().strip()
    text = text.replace("–", "-").replace("\n", " ").replace("  ", " ")
    return text

def generate_content(topic, grade, lang="uz"):
    """Generate lesson plan based on topic, grade, and language, and return matched chunks."""
    index, metadata, model = load_resources()
    gemini_model = configure_gemini()

    if not index or not metadata or not model:
        return "Failed to load resources.", []
    if not gemini_model:
        return "Failed to configure Gemini.", []

    lang_map = {"uz": "uzbek", "ru": "russian", "en": "english"}
    if lang not in lang_map:
        return "Unsupported language. Choose 'uz', 'ru', or 'en'.", []
    full_lang = lang_map[lang]

    try:
        topic = preprocess_text(topic)
        topic_embedding = model.encode(topic)
        topic_embedding = topic_embedding / np.linalg.norm(topic_embedding)
        topic_embedding = np.array([topic_embedding]).astype('float32')

        D, I = index.search(topic_embedding, k=5) 
        print(f"Topic: {topic}, Distances: {D[0]}, Indices: {I[0]}")

        distance_threshold = 1.0
        if all(d > distance_threshold for d in D[0]):
            print("\nSearching metadata for exact topic match...")
            for i, meta in enumerate(metadata):
                if "POPULATSIYA – TURNING TUZILISH" in meta["text"]:
                    print(f"Found in metadata[{i}]: {meta['text'][:100]}...")
            return f"No relevant content found for topic '{topic}' in FAISS search.", []

        relevant_chunks = []
        matched_chunks_full = []
        for i in I[0]:
            full_chunk = metadata[i]["text"]
            matched_chunks_full.append(full_chunk)
            chunk = full_chunk[:500] + '...' if len(full_chunk) > 500 else full_chunk
            relevant_chunks.append(chunk)
            print(f"Chunk (Index {i}, Distance {D[0][list(I[0]).index(i)]:.6f}): {chunk[:100]}...")

        print("\nMatched Chunks (Full Text):")
        for i, chunk in enumerate(matched_chunks_full, 1):
            print(f"Chunk {i} (Index {I[0][i-1]}):\n{chunk}\n{'-'*50}")

        prompt = f"""
        You are an expert educator creating a lesson plan for a Grade {grade} class on the topic '{topic}'.
        The content must be written in {full_lang.capitalize()} and based solely on the provided educational material below.
        Do not use external knowledge or generate content for topics not covered in the provided chunks.
        If the chunks are insufficient or irrelevant, return: "Cannot generate content: Insufficient relevant material."

        Relevant content from educational materials:
        """
        for i, chunk in enumerate(relevant_chunks, 1):
            prompt += f"Content {i}: {chunk}\n"

        prompt += f"""
            Create a lesson plan for Grade {grade} students, including:
            - Introduction: Explain the topic's importance and relevance.
            - Key Concepts: List at least 3 concepts, clearly summarized.
            - Example Problems: Provide 2 problems with solutions.
            - Classroom Activities: Suggest 2 engaging activities (e.g., discussions, hands-on tasks).
            - Homework: Assign 3 problems or tasks related to the topic.
            - Ensure content is age-appropriate and engaging.

            Format the output in Markdown:
            # {'Dars mazmuni {grade}-sinf uchun - Mavzu: {topic}' if full_lang == 'uzbek' else 'Содержание урока для {grade}-го класса - Тема: {topic}' if full_lang == 'russian' else 'Lesson Plan for Grade {grade} - Topic: {topic}'}
            ## {'Kirish' if full_lang == 'uzbek' else 'Введение' if full_lang == 'russian' else 'Introduction'}
            [Introduction text]
            ## {'Asosiy tushunchalar' if full_lang == 'uzbek' else 'Ключевые понятия' if full_lang == 'russian' else 'Key Concepts'}
            1. [Concept 1]
            2. [Concept 2]
            3. [Concept 3]
            ## {'Misol masalalar' if full_lang == 'uzbek' else 'Примеры задач' if full_lang == 'russian' else 'Example Problems'}
            1. [Problem 1 with solution]
            2. [Problem 2 with solution]
            ## {'Sinfdagi faoliyat' if full_lang == 'uzbek' else 'Классные активности' if full_lang == 'russian' else 'Classroom Activities'}
            1. [Activity 1]
            2. [Activity 2]
            ## {'Uy vazifasi' if full_lang == 'uzbek' else 'Домашнее задание' if full_lang == 'russian' else 'Homework Assignment'}
            1. [Task 1]
            2. [Task 2]
            3. [Task 3]

            If insufficient material, return only: "Cannot generate content: Insufficient relevant material."
            """
        response = gemini_model.generate_content(prompt)
        content = response.text.strip()

        if content == "Cannot generate content: Insufficient relevant material":
            return f"No relevant content found for topic '{topic}'.", matched_chunks_full

        return content, matched_chunks_full
    except Exception as e:
        return f"Error generating content: {e}", []

if __name__ == '__main__':
    test_topic = "POPULATSIYA – TURNING TUZILISH"
    test_grade = 10
    test_lang = "uz"
    print(f"\nGenerating content for topic: {test_topic}, Grade: {test_grade}, Language: {test_lang}")
    content, matched_chunks = generate_content(test_topic, test_grade, test_lang)
    print("\nGenerated Content:")
    print(content)