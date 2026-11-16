# Reads the PDFs in data/, chunks them, and saves the embeddings in ChromaDB.
import os
import re

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION = "pregnancy_docs"
CHUNK_SIZE = 500  # words
OVERLAP = 50  # words


def clean_text(text):
    text = re.sub(r"-\s*\n\s*", "", text)  # join words split across lines
    text = re.sub(r"\.{3,}", " ", text)  # dotted lines in tables of contents
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_pdf(path):
    # keep a page number for every word so each chunk knows where it came from
    words, word_pages = [], []
    for page_num, page in enumerate(PdfReader(path).pages, start=1):
        page_words = clean_text(page.extract_text() or "").split()
        words += page_words
        word_pages += [page_num] * len(page_words)

    chunks = []
    for start in range(0, len(words) - OVERLAP, CHUNK_SIZE - OVERLAP):
        end = min(start + CHUNK_SIZE, len(words))
        first, last = word_pages[start], word_pages[end - 1]
        page = str(first) if first == last else f"{first}-{last}"
        chunks.append({"text": " ".join(words[start:end]), "page": page})
    return chunks


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_DIR)

    # start fresh so running this twice doesn't create duplicates
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for filename in sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")):
        chunks = chunk_pdf(os.path.join(DATA_DIR, filename))
        print(f"{filename}: {len(chunks)} chunks")
        if not chunks:
            continue

        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        collection.add(
            ids=[f"{filename}-{i}" for i in range(len(chunks))],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[{"source": filename, "page": c["page"]} for c in chunks],
        )

    print(f"Done. {collection.count()} chunks saved in {DB_DIR}/")


if __name__ == "__main__":
    main()
