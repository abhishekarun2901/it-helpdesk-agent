import logging
import threading
from contextlib import suppress
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_client_instance: MilvusClient | None = None
_client_lock = threading.Lock()

embeddings = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

def get_milvus_client() -> MilvusClient:
    """Thread-safe singleton getter for MilvusClient."""
    global _client_instance
    with _client_lock:
        if _client_instance is None:
            db_path = Path(settings.MILVUS_PATH).resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _client_instance = MilvusClient(str(db_path))
        return _client_instance

def close_milvus_client():
    """Explicitly closes the Milvus-Lite embedded server and client."""
    global _client_instance
    with _client_lock:
        if _client_instance is not None:
            with suppress(Exception):
                _client_instance.close()
            _client_instance = None

def load_and_chunk_documents() -> list[Document]:
    docs_dir = settings.DOCS_DIR
    raw_documents = []

    if not docs_dir.exists():
        return []

    for file_path in docs_dir.glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = file_path.name
        raw_documents.extend(loaded_docs)

    if not raw_documents:
        return []

    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )
    return text_splitter.split_documents(raw_documents)

def init_vectorstore(force_reload: bool = False):
    """Initializes the collection, loads documents if needed, and loads the collection into memory."""
    client = get_milvus_client()
    collection_name = settings.COLLECTION_NAME

    if client.has_collection(collection_name):
        if not force_reload:
            client.load_collection(collection_name)
            return client
        client.drop_collection(collection_name)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="HNSW",
        params={"M": 8, "efConstruction": 64},
    )

    client.create_collection(
        collection_name=collection_name,
        dimension=384,
        metric_type="COSINE",
        index_params=index_params,
        auto_id=True,
    )

    chunks = load_and_chunk_documents()
    if chunks:
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        vectors = embeddings.embed_documents(texts)

        data = [
            {"vector": vec, "text": txt, "source": meta.get("source", "unknown")}
            for vec, txt, meta in zip(vectors, texts, metadatas)
        ]
        client.insert(collection_name=collection_name, data=data)

    client.load_collection(collection_name)
    return client

class MilvusLiteRetriever:
    """Retriever utilizing the shared singleton client."""

    def __init__(self, top_k: int = 2):
        self.top_k = top_k

    def invoke(self, query: str) -> list[Document]:
        client = get_milvus_client()
        collection_name = settings.COLLECTION_NAME

        try:
            client.load_collection(collection_name)
        except MilvusException as exc:
            logger.debug("Milvus collection already loaded or load skipped: %s", exc)

        query_vector = embeddings.embed_query(query)
        results = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=self.top_k,
            output_fields=["text", "source"],
        )

        documents = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get("entity", {})
                documents.append(
                    Document(
                        page_content=entity.get("text", ""),
                        metadata={"source": entity.get("source", "unknown")},
                    )
                )
        return documents

def get_retriever(k: int = 2) -> MilvusLiteRetriever:
    return MilvusLiteRetriever(top_k=k)