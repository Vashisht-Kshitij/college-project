# Reads the PDFs in data/, chunks them, and saves the embeddings in ChromaDB.
import os
import re

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION = "pregnancy_docs"
CHUNK_WORDS = 150  # small enough for MiniLM, which only reads the first 256 tokens
OVERLAP_WORDS = 30


def clean_text(text):
    text = re.sub(r"-\s*\n\s*", "", text)  # join words split across lines
    text = re.sub(r"\.{3,}", " ", text)  # dotted lines in tables of contents
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text):
    # split after . ! ? and at bullet points (the guidelines use a lot of bullets)
    parts = re.split(r"(?<=[.!?])\s+|\s*[•∙●►]\s*", text)
    return [p.strip() for p in parts if p.strip()]


def word_count(sentences):
    return sum(len(s.split()) for s, _ in sentences)


def make_chunk(sentences):
    first, last = sentences[0][1], sentences[-1][1]
    page = str(first) if first == last else f"{first}-{last}"
    return {"text": " ".join(s for s, _ in sentences), "page": page}


def chunk_pdf(path):
    # list of (sentence, page number) so each chunk knows where it came from
    sentences = []
    for page_num, page in enumerate(PdfReader(path).pages, start=1):
        for sentence in split_sentences(clean_text(page.extract_text() or "")):
            words = sentence.split()
            for i in range(0, len(words), CHUNK_WORDS):  # cut very long "sentences" like tables
                sentences.append((" ".join(words[i:i + CHUNK_WORDS]), page_num))

    # add whole sentences until the chunk is full, then start the next chunk with the last ~30 words
    chunks, current, new_words = [], [], 0
    for sentence, page in sentences:
        if new_words and word_count(current) + len(sentence.split()) > CHUNK_WORDS:
            chunks.append(make_chunk(current))
            overlap = []
            while current and word_count(overlap) + len(current[-1][0].split()) <= OVERLAP_WORDS:
                overlap.insert(0, current.pop())
            current, new_words = overlap, 0
        current.append((sentence, page))
        new_words += len(sentence.split())
    if new_words:
        chunks.append(make_chunk(current))
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
