# import os
# import faiss
# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
# chunks_with_metadata = []
# embeddings = []
# grades = range(1, 12)

# for grade in grades:
#     output_dir = f"books/{grade}/"
#     if not os.path.exists(output_dir):
#         print(f"Directory does not exist: {output_dir}")
#         continue
#     for chunk_file in os.listdir(output_dir):
#         if chunk_file.endswith("_chunks.txt"):
#             chunk_path = os.path.join(output_dir, chunk_file)
#             try:
#                 with open(chunk_path, "r", encoding="utf-8") as f:
#                     content = f.read()
#                 chunks = content.split("\n\n")
#                 for i, chunk in enumerate(chunks):
#                     if chunk.strip():
#                         chunk_text = chunk.split("\n", 1)[1] if "\n" in chunk and "Chunk" in chunk else chunk
#                         try:
#                             embedding = model.encode(chunk_text)
#                             embeddings.append(embedding)
#                             chunks_with_metadata.append({
#                                 "id": f"{chunk_file}_{i}",
#                                 "text": chunk_text,
#                                 "source": chunk_file
#                             })
#                         except Exception as e:
#                             print(f"Error encoding chunk {i} in {chunk_file}: {e}")
#                 print(f"Processed chunks from: {chunk_file} in grade {grade}")
#             except Exception as e:
#                 print(f"Error processing file {chunk_file}: {e}")

# if embeddings:
#     embeddings = np.array(embeddings).astype('float32')
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)
#     faiss.write_index(index, "embeddings/teacher_rag_index.faiss")
#     with open("embeddings/chunks_metadata.json", "w", encoding="utf-8") as f:
#         json.dump(chunks_with_metadata, f, ensure_ascii=False)
#     print(f"Stored {len(embeddings)} embeddings in FAISS")
# else:
#     print("No embeddings were generated.")


import os
import json
import numpy as np
import faiss
import requests
import time
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Jina API details
url = 'https://api.jina.ai/v1/embeddings'
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer jina_b5d8ea61235c48ad9e5af70719254fce5XsUnWspjaU21021cx5XvH6Zqbll' 
}

# Load previous database if exists
db_path = 'embeddings/database2.json'
if os.path.exists(db_path):
    with open(db_path, 'r', encoding='utf-8') as f:
        database = json.load(f)
    logger.info(f"Loaded existing database with {len(database)} entries.")
else:
    database = []

# Track existing files for skipping
existing_keys = set((entry['directory'], entry['file']) for entry in database)

# List of embeddings for new data
embeddings_list = []

# Set up requests session with retries
session = requests.Session()
retries = Retry(total=3, backoff_factor=5, status_forcelist=[429, 500, 502, 503, 524])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Create embeddings directory
os.makedirs('embeddings/chunked', exist_ok=True)

def clean_chunk_text(chunk):
    if not chunk.strip():
        return None
    lines = chunk.split("\n")
    if lines and lines[0].startswith("Chunk"):
        return "\n".join(lines[1:]).strip()
    return chunk.strip()

for grade in range(7,8):
    dir_path = f'books/{grade}'
    if not os.path.exists(dir_path):
        logger.warning(f"Directory {dir_path} does not exist")
        continue

    for filename in os.listdir(dir_path):
        if not filename.endswith('_chunks.txt'):
            logger.info(f"Skipping non-chunk file: {filename}")
            continue

        if (dir_path, filename) in existing_keys:
            logger.info(f"Skipping already processed file: {dir_path}/{filename}")
            continue

        logger.info(f"Processing file: {dir_path}/{filename}")
        output_dir = 'embeddings/chunked'

        try:
            with open(f'{dir_path}/{filename}', 'r', encoding='utf-8') as f:
                content = f.read()
            chunks = content.split("\n\n")
            input_data = [{"text": clean_chunk_text(chunk)} for chunk in chunks if clean_chunk_text(chunk)]
        except Exception as e:
            logger.error(f"Failed to read {filename}: {str(e)}")
            continue

        if not input_data:
            logger.warning(f"No valid content found in {filename}")
            continue

        batch_size = 50
        for i in range(0, len(input_data), batch_size):
            batch = input_data[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} chunks")

            data = {
                "model": "jina-embeddings-v4",
                "task": "text-matching",
                "input": batch
            }

            try:
                response = session.post(url, headers=headers, json=data, timeout=60)
                response.raise_for_status()
                embeddings = response.json().get('data', [])

                if len(embeddings) != len(batch):
                    logger.warning(f"Mismatch: {len(embeddings)} embeddings for {len(batch)} chunks")

                for j, embedding in enumerate(embeddings):
                    embeddings_list.append(embedding['embedding'])
                    database.append({
                        'file': filename,
                        'directory': dir_path,
                        'text': batch[j]['text'],
                        'embedding': embedding['embedding'],
                        'index': len(database),  # ensure consistent index
                        'is_chunked': True,
                        'faiss_index': len(embeddings_list) - 1
                    })

                logger.info(f"Successfully processed batch {i//batch_size + 1}")
            except requests.RequestException as e:
                logger.error(f"Failed to process batch {i//batch_size + 1} of {filename}: {str(e)}")
                continue

            time.sleep(5)

        # Save intermediate embedding file per file
        embedding_filename = f'{output_dir}/{filename.replace(".txt", "_embeddings.json")}'
        try:
            with open(embedding_filename, 'w', encoding='utf-8') as f:
                json.dump(embeddings, f, ensure_ascii=False)
            logger.info(f"Saved embeddings to {embedding_filename}")
        except Exception as e:
            logger.error(f"Failed to save embeddings for {filename}: {str(e)}")

# Save updated database.json
try:
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved updated database with {len(database)} entries.")
except Exception as e:
    logger.error(f"Failed to save metadata: {str(e)}")

# Build and save FAISS index
if embeddings_list:
    try:
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        if os.path.exists('embeddings/index2.faiss'):
            # Load and append to existing index
            index = faiss.read_index('embeddings/index2.faiss')
            index.add(embeddings_array)
        else:
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)

        faiss.write_index(index, 'embeddings/index2.faiss')
        logger.info("FAISS index saved to embeddings/index2.faiss")
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {str(e)}")
else:
    logger.warning("No new embeddings to index")
