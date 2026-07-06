# rag.py – PubMed RAG without LangChain
import os
import streamlit as st
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import NotFoundError   # ✅ added
from groq import Groq
from sentence_transformers import SentenceTransformer

from pubmed import PubMedRetriever

# ----------------------------------------------------------------------
# 🔐 API key: Streamlit secrets (cloud) or .env (local)
# ----------------------------------------------------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in secrets or .env")

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "medical_articles"

# ----------------------------------------------------------------------
# Initialise components
# ----------------------------------------------------------------------
client = None          # Groq client
vector_store = None    # Chroma collection
embedder = None        # SentenceTransformer


def initialize_components():
    global client, vector_store, embedder
    if client is None:
        client = Groq(api_key=GROQ_API_KEY)
    if embedder is None:
        embedder = SentenceTransformer(EMBEDDING_MODEL)
    if vector_store is None:
        chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        class SentenceTransformerEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __init__(self, model):
                self.model = model
            def __call__(self, texts):
                return self.model.encode(texts, convert_to_numpy=True).tolist()
        embed_fn = SentenceTransformerEmbeddingFunction(embedder)
        try:
            collection = chroma_client.get_collection(COLLECTION_NAME)
        except NotFoundError:   # ✅ fixed exception
            collection = chroma_client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=embed_fn
            )
        vector_store = collection


# ----------------------------------------------------------------------
# Text splitter (recursive character split)
# ----------------------------------------------------------------------
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in ['. ', '? ', '! ', '\n\n', '\n', '.', '?', '!']:
                last_sep = text.rfind(sep, start, end)
                if last_sep != -1:
                    end = last_sep + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end - overlap if end < len(text) else len(text)
    return chunks


# ----------------------------------------------------------------------
# Process PubMed articles
# ----------------------------------------------------------------------
def process_urls(search_term: str, max_results: int = 20):
    yield "Initializing Components"
    initialize_components()

    yield "Resetting vector store...✅"
    try:
        all_ids = vector_store.get()["ids"]
        if all_ids:
            vector_store.delete(ids=all_ids)
    except Exception:
        pass

    yield f"Searching PubMed for '{search_term}' (max {max_results} results)..."
    pmid_list = PubMedRetriever.search_pubmed_articles(search_term, max_results=max_results)
    if not pmid_list:
        yield "No PubMed IDs found. Exiting."
        return
    yield f"Found {len(pmid_list)} articles. Fetching abstracts..."

    articles = PubMedRetriever.fetch_pubmed_abstracts(pmid_list)
    yield f"Fetched {len(articles)} articles. Building documents..."

    ids = []
    documents = []
    metadatas = []

    for art in articles:
        abstract_text = "\n".join(
            f"{label}: {text}" for label, text in art["abstract"].items()
        )
        content = f"Title: {art['title']}\n{abstract_text}"
        source_str = f"PMID: {art['pmid']} – {art['title']}"

        chunks = split_text(content)
        for chunk in chunks:
            doc_id = str(uuid4())
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "pmid": art["pmid"],
                "journal": art["journal"],
                "authors": art["authors"],
                "publication_date": art["publication_date"],
                "source": source_str
            })

    yield f"Splitting into {len(documents)} chunks. Adding to vector database..."
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        vector_store.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    yield "Done adding documents to vector database. ✅"


# ----------------------------------------------------------------------
# Answer a query
# ----------------------------------------------------------------------
def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized. Call process_urls first.")

    results = vector_store.query(query_texts=[query], n_results=5)
    if not results["documents"]:
        return "I couldn't find any relevant information.", "No sources available."

    context_parts = []
    sources_set = set()
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(doc)
        sources_set.add(meta.get("source", "Unknown source"))

    context = "\n\n".join(context_parts)
    sources = "; ".join(sources_set) if sources_set else "No sources available."

    prompt = f"""You are a medical expert. Use the following context to answer the question.
If you don't know the answer, say you don't know. Cite the sources using PMID numbers.

Context:
{context}

Question: {query}

Answer:"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=500,
    )
    answer = completion.choices[0].message.content

    return answer, sources


# ----------------------------------------------------------------------
# Local test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    search_term = "hypertension treatment guidelines 2024"
    for msg in process_urls(search_term, max_results=10):
        print(msg)
    answer, sources = generate_answer("What are the latest treatment guidelines for hypertension?")
    print(f"\nAnswer: {answer}")
    print(f"Sources: {sources}")
