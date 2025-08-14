# rag_core.py (only the FAISS section shown)
import os
import platform
from typing import List, Dict, Any
from uuid import uuid4

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

DEFAULT_BACKEND = "faiss"  # <- force FAISS universally for stability
USE_FAISS = os.getenv("VECTOR_BACKEND", DEFAULT_BACKEND).lower() == "faiss"

# -----------------------------
# Embeddings & LLM
# -----------------------------
def get_embeddings():
    return OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))

def get_llm():
    return ChatOpenAI(
        model=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
    )

# -----------------------------
# FAISS backend (default)
# -----------------------------
if USE_FAISS:
    try:
        from langchain_community.vectorstores import FAISS
    except Exception:
        from langchain.vectorstores import FAISS  # legacy fallback

    _faiss_store = None
    _faiss_dir = os.getenv("FAISS_DIR", "./faiss_db")

    def _load_faiss_if_exists():
        """Load FAISS index from disk if present."""
        from pathlib import Path
        global _faiss_store
        if _faiss_store is not None:
            return
        if Path(_faiss_dir).exists():
            try:
                _faiss_store = FAISS.load_local(
                    _faiss_dir,
                    get_embeddings(),
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                _faiss_store = None  # fall through; will build on first add

    def init_chroma(_persist_dir: str = "./chroma_db"):
        """Signature compatibility; FAISS is in-memory + persisted via save_local."""
        _load_faiss_if_exists()
        return None, None

    def add_to_vectorstore(_coll, _embeddings, chunks: List[Dict[str, Any]]):
        """Build or extend FAISS index, then persist to disk."""
        global _faiss_store
        texts = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        embs = get_embeddings()
        if _faiss_store is None:
            _faiss_store = FAISS.from_texts(texts=texts, embedding=embs, metadatas=metas)
        else:
            _faiss_store.add_texts(texts=texts, metadatas=metas, embedding=embs)
        # persist
        os.makedirs(_faiss_dir, exist_ok=True)
        _faiss_store.save_local(_faiss_dir)

    def _bm25_rerank(query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not docs:
            return docs
        tokens = [d["text"].split() for d in docs]
        bm25 = BM25Okapi(tokens)
        scores = bm25.get_scores(query.split())
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [docs[i] for i in order]

    def retrieve(_coll, _embeddings, query: str, k: int = 6) -> List[Dict[str, Any]]:
        _load_faiss_if_exists()
        if _faiss_store is None:
            return []
        lc_docs = _faiss_store.similarity_search(query, k=k * 2)  # over-fetch
        docs = [{"text": d.page_content, "metadata": d.metadata} for d in lc_docs]
        return _bm25_rerank(query, docs, top_k=k)

# -----------------------------
# Chroma backend (Windows/macOS default)
# -----------------------------

else:
    # Import Chroma lazily to avoid Cloud sqlite issues when FAISS is used
    import chromadb
    from chromadb.utils import embedding_functions

    def init_chroma(persist_dir: str = "./chroma_db"):
        try:
            client = chromadb.PersistentClient(path=persist_dir)

            # 🔑 Force Chroma to use OpenAI embeddings (no ONNX download)
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            )

            coll = client.get_or_create_collection(
                "big_bill_collection",
                metadata={"hnsw:space": "cosine"},
                embedding_function=ef,   # <- IMPORTANT
            )
            return client, coll
        except Exception as e:
            raise RuntimeError(
                "Vector store initialization failed (Chroma). "
                "If on Streamlit Cloud, prefer FAISS. "
                f"Original error: {e}"
            )
        
# else:
#     # Import Chroma lazily to avoid Cloud sqlite issues when FAISS is used
#     import chromadb

#     def init_chroma(persist_dir: str = "./chroma_db"):
#         """
#         Initialize Chroma across versions:
#           - 0.5.x → PersistentClient(path=...)
#           - 0.4.x → Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=...))

#         On failure, fall back to an ephemeral client so the app can still answer (no persistence).
#         """
#         ver = _pv.parse(getattr(chromadb, "__version__", "0.5.0"))
#         try:
#             if ver >= _pv.parse("0.5.0"):
#                 # 0.5.x API
#                 client = chromadb.PersistentClient(path=persist_dir)
#                 coll = client.get_or_create_collection(
#                     "big_bill_collection",
#                     metadata={"hnsw:space": "cosine"},
#                 )
#             else:
#                 # 0.4.x API
#                 from chromadb.config import Settings
#                 client = chromadb.Client(
#                     Settings(
#                         chroma_db_impl="duckdb+parquet",
#                         persist_directory=persist_dir,
#                     )
#                 )
#                 coll = client.get_or_create_collection(
#                     "big_bill_collection",
#                     metadata={"hnsw:space": "cosine"},
#                 )
#             return client, coll

#         except Exception as e:
#             # Final fallback: ephemeral client (no persistence)
#             try:
#                 if ver >= _pv.parse("0.5.0"):
#                     client = chromadb.EphemeralClient()
#                 else:
#                     client = chromadb.Client()  # 0.4.x ephemeral
#                 coll = client.get_or_create_collection(
#                     "big_bill_collection",
#                     metadata={"hnsw:space": "cosine"},
#                 )
#                 return client, coll
#             except Exception:
#                 raise RuntimeError(
#                     "Vector store initialization failed (Chroma). "
#                     "If on Streamlit Cloud, prefer FAISS. "
#                     f"Original error: {e}"
#                 )

    def add_to_vectorstore(coll, _embeddings, chunks: List[Dict[str, Any]]):
        """
        IMPORTANT: pass embeddings in explicitly so Chroma never tries to download ONNX models.
        """
        texts = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        ids = [c["id"] for c in chunks]

        # Compute embeddings with OpenAI
        emb = get_embeddings()
        vectors = emb.embed_documents(texts)

        coll.add(ids=ids, documents=texts, metadatas=metas, embeddings=vectors)

    def _bm25_rerank(query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not docs:
            return docs
        tokens = [d["text"].split() for d in docs]
        bm25 = BM25Okapi(tokens)
        scores = bm25.get_scores(query.split())
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [docs[i] for i in order]

    def retrieve(coll, _embeddings, query: str, k: int = 6) -> List[Dict[str, Any]]:
        """
        Query using precomputed embeddings to avoid Chroma's internal embedding pipeline.
        """
        emb = get_embeddings()
        qvec = emb.embed_query(query)

        res = coll.query(
            query_embeddings=[qvec],
            n_results=k * 8,  # over-fetch, then BM25 re-rank
            include=["metadatas", "documents"],
        )
        docs: List[Dict[str, Any]] = []
        if res and res.get("documents"):
            for text, meta in zip(res["documents"][0], res["metadatas"][0]):
                docs.append({"text": text, "metadata": meta})
        return _bm25_rerank(query, docs, top_k=k)


# -----------------------------
# Chunking & formatting
# -----------------------------
def chunk_text(text: str, doc_title: str, source_path: str) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(chunks):
        out.append({
            "id": f"{doc_title}-{i}-{uuid4().hex}",
            "text": c,
            "metadata": {"doc_title": doc_title, "source_path": source_path},
        })
    return out


def format_context(docs: List[Dict[str, Any]]) -> str:
    blocks = []
    for d in docs:
        meta = d.get("metadata", {})
        page = meta.get("page", "")
        sec = meta.get("section", "")
        head = f"{meta.get('doc_title','')}"
        if page or sec:
            head += f" (p.{page} {sec})"
        blocks.append(f"[{head}]\n{d['text']}")
    return "\n\n---\n\n".join(blocks)

# --- Diagnostics ---
def vector_count() -> int:
    """Return number of docs loaded in FAISS (or 0 if none)."""
    try:
        if USE_FAISS:
            # trigger load if needed
            try:
                _ = init_chroma()
            except Exception:
                return 0
            global _faiss_store
            if _faiss_store is None:
                return 0
            # FAISS doesn't expose a direct length; use index docstore length
            try:
                return len(_faiss_store.docstore._dict)  # type: ignore[attr-defined]
            except Exception:
                return 0
        else:
            # For Chroma (if ever used again), return 0 here
            return 0
    except Exception:
        return 0
