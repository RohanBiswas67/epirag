"""
EpiRAG -- ingest.py

Two modes:

  LOCAL:
    python ingest.py
    Reads PDFs from ./papers/, saves persistent ChromaDB to ./chroma_db/

  CLOUD (HF Spaces):
    from ingest import build_collection_in_memory
    collection, embedder = build_collection_in_memory()
    Downloads PDFs from HF dataset at startup, builds ChromaDB in RAM.
    No papers/ folder needed in the repo.
"""

import os
import re
import fitz
import chromadb
from sentence_transformers import SentenceTransformer

# Config
PAPERS_DIR      = "./papers"
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "epirag"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 100
EMBED_MODEL     = "all-MiniLM-L6-v2"
CHROMA_BATCH    = 5000
HF_DATASET_ID   = "RohanB67/papers"


def extract_text(pdf_path: str) -> tuple[str, str]:
    doc  = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]


def _embed_and_add(collection, embedder, docs, ids, metas):
    total, all_embeddings = len(docs), []
    for i in range(0, total, 64):
        batch = docs[i:i + 64]
        all_embeddings.extend(embedder.encode(batch, show_progress_bar=False).tolist())
        print(f"  Embedded {min(i + 64, total)}/{total}", flush=True)
    for i in range(0, total, CHROMA_BATCH):
        j = min(i + CHROMA_BATCH, total)
        collection.add(
            documents=docs[i:j],
            embeddings=all_embeddings[i:j],
            ids=ids[i:j],
            metadatas=metas[i:j]
        )
        print(f"  Stored {j}/{total}", flush=True)


def _load_pdfs(papers_dir: str):
    pdf_files = sorted(f for f in os.listdir(papers_dir) if f.endswith(".pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {papers_dir}/")

    docs, ids, metas, chunk_index = [], [], [], 0
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}", flush=True)
        chunks = chunk_text(extract_text(os.path.join(papers_dir, pdf_file)))
        print(f"  -> {len(chunks)} chunks", flush=True)

        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            ids.append(f"{pdf_file}_chunk_{chunk_index}")
            metas.append({
                "source":      pdf_file,
                "chunk_index": i,
                "paper_name":  pdf_file.replace(".pdf", "").replace("_", " ")
            })
            chunk_index += 1

    return docs, ids, metas, len(pdf_files)


def _download_papers_from_hf(dest_dir: str = PAPERS_DIR):
    """
    Pull all PDF files from HF dataset RohanB67/papers into dest_dir.
    Uses huggingface_hub already available in HF Spaces environment.
    """
    from huggingface_hub import list_repo_files, hf_hub_download
    os.makedirs(dest_dir, exist_ok=True)
    pdf_files = [
        f for f in list_repo_files(HF_DATASET_ID, repo_type="dataset")
        if f.endswith(".pdf")
    ]
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in HF dataset {HF_DATASET_ID}")

    print(f"Downloading {len(pdf_files)} papers from {HF_DATASET_ID}...", flush=True)
    for fname in pdf_files:
        local_path = os.path.join(dest_dir, os.path.basename(fname))
        if os.path.exists(local_path):
            print(f"  Cached: {fname}", flush=True)
            continue
        hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename=fname,
            repo_type="dataset",
            local_dir=dest_dir,
            local_dir_use_symlinks=False
        )
        print(f"  Downloaded: {fname}", flush=True)
    print(f"All papers ready in {dest_dir}", flush=True)


# -- In-memory build (HF Spaces) ----------------------------------------------
def build_collection_in_memory(papers_dir: str = PAPERS_DIR):
    print("=== EpiRAG: building in-memory corpus ===", flush=True)
    _download_papers_from_hf(papers_dir)
    embedder = SentenceTransformer(EMBED_MODEL)
    client   = chromadb.EphemeralClient()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    docs, ids, metas, n_pdfs = _load_pdfs(papers_dir)
    print(f"\nEmbedding {len(docs)} chunks from {n_pdfs} papers...", flush=True)
    _embed_and_add(collection, embedder, docs, ids, metas)
    print(f"In-memory corpus ready: {len(docs)} chunks / {n_pdfs} papers", flush=True)
    return collection, embedder


# -- Persistent build (local dev) ---------------------------------------------
def ingest_papers(papers_dir: str = PAPERS_DIR, chroma_dir: str = CHROMA_DIR):
    os.makedirs(papers_dir, exist_ok=True)
    os.makedirs(chroma_dir, exist_ok=True)
    print(f"Loading embedding model: {EMBED_MODEL}", flush=True)
    embedder = SentenceTransformer(EMBED_MODEL)
    client   = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Cleared existing collection.", flush=True)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    docs, ids, metas, n_pdfs = _load_pdfs(papers_dir)
    print(f"\nEmbedding {len(docs)} chunks...", flush=True)
    _embed_and_add(collection, embedder, docs, ids, metas)
    print(f"\nDone. {len(docs)} chunks from {n_pdfs} papers saved to {chroma_dir}", flush=True)


if __name__ == "__main__":
    ingest_papers()