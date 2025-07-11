import nltk
import os
from nltk.tokenize import sent_tokenize

def chunk_text(text, max_chunk_size=500):
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_size = [], "", 0
    for sentence in sentences:
        sentence_size = len(sentence.split())
        if current_size + sentence_size > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk, current_size = sentence, sentence_size
        else:
            current_chunk += " " + sentence
            current_size += sentence_size
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

output_dir = "books/11/" 

for txt_file in os.listdir(output_dir):
    if txt_file.endswith(".txt"):
        with open(os.path.join(output_dir, txt_file), "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        with open(os.path.join(output_dir, txt_file.replace(".txt", "_chunks.txt")), "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"Chunk {i}:\n{chunk}\n\n")