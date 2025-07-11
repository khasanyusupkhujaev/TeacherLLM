import re
import os
import json

output_dir = "books/11/"
metadata_dir = "books/11/metadata/"
os.makedirs(metadata_dir, exist_ok=True)

def clean_text(text):
    # Keep original logic
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r"@ELEKTRON_DARSLIKBOT\s+dan\s+yuklab\s+olindi", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[;*$>I]+|\b[ft]\b", "", text)  # Remove stray symbols
    
    # Extract publication metadata
    publication = {}
    
    # Title (match patterns like "Informatika va axborot texnologiyalari 1-sinf")
    title_match = re.search(r"(Informatika\s+va\s+axborot\s+texnologiyalari|Matematika|Adabiyot)\s+[\w\s\-\d]+sinf", text, re.IGNORECASE)
    if title_match:
        publication["title"] = title_match.group(0).strip()
    
    # Authors (match individual names and remove duplicates)
    author_matches = re.findall(r"[A-Z]\.\s*[A-Z]\.\s*\w+", text)
    unique_authors = list(dict.fromkeys(author_matches))  # Remove duplicates while preserving order
    if unique_authors:
        publication["author"] = ", ".join(unique_authors)
    
    # ISBN (match format like "978-9943-7209-1-6")
    isbn_match = re.search(r"ISBN\s*(\d{3}-\d{4}-\d{4}-\d{1})", text)
    if isbn_match:
        publication["isbn"] = isbn_match.group(1)
    
    # Publisher (match known publishers)
    publisher_match = re.search(r"(Novda\s+Edutainment|Ozbekiston\s+milliy\s+ensiklopediyasi)", text, re.IGNORECASE)
    if publisher_match:
        publication["publisher"] = publisher_match.group(0)
    
    # Year (match 4-digit years like 2023)
    year_match = re.search(r"\b(20\d{2})\b", text)
    if year_match:
        publication["year"] = int(year_match.group(1))
    
    # Remove extracted metadata from text to avoid redundancy
    if title_match:
        text = re.sub(re.escape(title_match.group(0)), "", text)
    for author in unique_authors:
        text = re.sub(re.escape(author), "", text)
    if isbn_match:
        text = re.sub(re.escape(isbn_match.group(0)), "", text)
    if publisher_match:
        text = re.sub(re.escape(publisher_match.group(0)), "", text)
    if year_match:
        text = re.sub(re.escape(year_match.group(0)), "", text)
    
    # Remove additional metadata (e.g., UOK, KBK, editors, artists)
    text = re.sub(r"UO['‘]K\s*\d+.*?\b|KBK\s*\d+.*?\b|Muharrirlar:.*?(\n|$)|Badiiy\s+muharrir.*?(\n|$)|Rassomlar:.*?(\n|$)|Kompyuterda\s+sahifalovchi.*?(\n|$)|Nashriyot\s+litsenziyasi.*?(\n|$)", "", text)
    
    # Remove redundant publication details (e.g., "Toshkent-2023", "80 b")
    text = re.sub(r"Toshkent-20\d{2}|-\s*\d+\s*b\.?", "", text)
    
    return text.strip(), publication

for txt_file in os.listdir(output_dir):
    if txt_file.endswith(".txt"):
        file_path = os.path.join(output_dir, txt_file)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        cleaned_text, publication = clean_text(text)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        # Save publication metadata
        with open(os.path.join(metadata_dir, txt_file.replace(".txt", "_meta.json")), "w", encoding="utf-8") as f:
            json.dump(publication, f, ensure_ascii=False, indent=2)