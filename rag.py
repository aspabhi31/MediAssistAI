# rag.py – PubMed Q&A with sources (imports work on any LangChain version)
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv

# Core LangChain imports (always available via langchain-core)
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# ------------------------------------------------------------------
# Resilient imports for text splitter, embeddings, vector store, and QA chain
# ------------------------------------------------------------------

# 1. Text Splitter – try dedicated package, fallback to community, then core
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_community.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. Embeddings – try huggingface package, fallback to community
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# 3. Vector store – try chroma package, fallback to community
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# 4. Retrieval QA – try chains, fallback to community, then legacy import
try:
    from langchain.chains import RetrievalQA
except ImportError:
    try:
        from langchain_community.chains import RetrievalQA
    except ImportError:
        try:
            from langchain.chains.retrieval_qa.base import RetrievalQA
        except ImportError:
            from langchain import RetrievalQA   # very old versions

# ------------------------------------------------------------------

from langchain_groq import ChatGroq
from pubmed import PubMedRetriever

load_dotenv()

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "medical_articles"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store
    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)
    if vector_store is None:
        ef = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"trust_remote_code": True})
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_urls(search_term: str, max_results: int = 20):
    yield "Initializing Components"
    initialize_components()
    yield "Resetting vector store...✅"
    vector_store.reset_collection()

    yield f"Searching PubMed for '{search_term}' (max {max_results} results)..."
    pmid_list = PubMedRetriever.search_pubmed_articles(search_term, max_results=max_results)
    if not pmid_list:
        yield "No PubMed IDs found. Exiting."
        return
    yield f"Found {len(pmid_list)} articles. Fetching abstracts..."

    articles = PubMedRetriever.fetch_pubmed_abstracts(pmid_list)
    yield f"Fetched {len(articles)} articles. Building documents..."

    documents = []
    for art in articles:
        abstract_text = "\n".join(f"{label}: {text}" for label, text in art["abstract"].items())
        content = f"Title: {art['title']}\n{abstract_text}"
        source_str = f"PMID: {art['pmid']} – {art['title']}"
        metadata = {
            "pmid": art["pmid"],
            "journal": art["journal"],
            "authors": art["authors"],
            "publication_date": art["publication_date"],
            "source": source_str
        }
        documents.append(Document(page_content=content, metadata=metadata))

    yield f"Splitting {len(documents)} documents into chunks..."
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "], chunk_size=CHUNK_SIZE)
    docs = text_splitter.split_documents(documents)

    yield f"Adding {len(docs)} chunks to vector database..."
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding documents to vector database. ✅"


def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized. Call process_urls first.")

    prompt_template = """You are a medical expert. Use the following context to answer the question.
    If you don't know the answer, say you don't know. Cite the sources using PMID numbers.

    Context: {context}

    Question: {question}

    Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = chain.invoke({"query": query})
    answer = result["result"]
    source_docs = result.get("source_documents", [])

    sources_set = set()
    for doc in source_docs:
        src = doc.metadata.get("source")
        if src:
            sources_set.add(src)

    sources = "; ".join(sources_set) if sources_set else "No sources available."
    return answer, sources


if __name__ == "__main__":
    search_term = "hypertension treatment guidelines 2024"
    for msg in process_urls(search_term, max_results=10):
        print(msg)

    answer, sources = generate_answer("What are the latest treatment guidelines for hypertension?")
    print(f"\nAnswer: {answer}")
    print(f"Sources: {sources}")