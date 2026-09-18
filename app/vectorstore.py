import logging
import threading
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from milvus_model.sparse import BM25EmbeddingFunction
from pymilvus import (
    AnnSearchRequest,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    MilvusException,
    RRFRanker,
)
from scipy.sparse import csr_matrix, issparse

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_client_instance: MilvusClient | None = None
_client_lock = threading.Lock()

# Dense model for semantic representation
embeddings = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Sparse BM25 model
bm25_ef = BM25EmbeddingFunction()


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
            try:
                _client_instance.close()
            except MilvusException as exc:
                logger.warning("Error during Milvus client shutdown: %s", exc)
            _client_instance = None


def _sparse_to_dict(matrix: csr_matrix) -> list[dict[int, float]]:
    """Converts a SciPy CSR sparse matrix into a list of index-value dicts for Milvus."""
    dict_list = []
    for row in range(matrix.shape[0]):
        start = matrix.indptr[row]
        end = matrix.indptr[row + 1]
        indices = matrix.indices[start:end]
        values = matrix.data[start:end]
        dict_list.append({int(i): float(v) for i, v in zip(indices, values)})
    return dict_list


def extract_metadata_for_file(file_name: str) -> dict[str, str]:
    """Enriches chunks with domain-specific metadata for scalar filtering."""
    file_lower = file_name.lower()
    if "vpn" in file_lower:
        return {
            "source": file_name,
            "category": "network",
            "security_level": "medium",
            "doc_type": "policy",
            "last_reviewed": "2026-01-15",
        }
    if "password" in file_lower:
        return {
            "source": file_name,
            "category": "identity_access",
            "security_level": "high",
            "doc_type": "policy",
            "last_reviewed": "2026-02-10",
        }
    if "software" in file_lower:
        return {
            "source": file_name,
            "category": "procurement",
            "security_level": "low",
            "doc_type": "guideline",
            "last_reviewed": "2026-03-01",
        }
    if "faq" in file_lower or "troubleshoot" in file_lower:
        return {
            "source": file_name,
            "category": "support_troubleshooting",
            "security_level": "low",
            "doc_type": "faq",
            "last_reviewed": "2026-04-12",
        }
    return {
        "source": file_name,
        "category": "general",
        "security_level": "low",
        "doc_type": "general_doc",
        "last_reviewed": "2026-01-01",
    }


def load_and_chunk_documents() -> list[Document]:
    """Loads text documents and splits them semantically with rich metadata."""
    docs_dir = settings.DOCS_DIR
    raw_documents = []

    if not docs_dir.exists():
        return []

    for file_path in docs_dir.glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        loaded_docs = loader.load()
        metadata = extract_metadata_for_file(file_path.name)
        for doc in loaded_docs:
            doc.metadata.update(metadata)
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
    """Initializes the collection with HNSW dense index and sparse inverted index."""
    client = get_milvus_client()
    collection_name = settings.COLLECTION_NAME

    if client.has_collection(collection_name):
        if not force_reload:
            client.load_collection(collection_name)
            return client
        client.drop_collection(collection_name)

    # 1. Define schema with Dense and Sparse vector fields
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="security_level", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="last_reviewed", dtype=DataType.VARCHAR, max_length=32),
        ],
        description="IT Helpdesk Knowledge Base with Hybrid Dense-Sparse Indexing",
        enable_dynamic_field=True,
    )

    # 2. Configure Index parameters (HNSW for Dense, SPARSE_INVERTED_INDEX for Sparse)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_hnsw_idx",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 128},
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_inverted_idx",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"drop_ratio_build": 0.2},
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )

    # 3. Ingest documents and compute vectors
    chunks = load_and_chunk_documents()
    if not chunks:
        client.load_collection(collection_name)
        return client

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # Dense embeddings
    dense_vectors = embeddings.embed_documents(texts)

    # Fit and generate BM25 sparse vectors
    bm25_ef.fit(texts)
    sparse_matrix = bm25_ef.encode_documents(texts)
    sparse_dicts = _sparse_to_dict(sparse_matrix) if issparse(sparse_matrix) else sparse_matrix

    data = []
    for i in range(len(chunks)):
        meta = metadatas[i]
        data.append({
            "dense_vector": dense_vectors[i],
            "sparse_vector": sparse_dicts[i],
            "text": texts[i],
            "source": meta.get("source", "unknown"),
            "category": meta.get("category", "general"),
            "security_level": meta.get("security_level", "low"),
            "doc_type": meta.get("doc_type", "general_doc"),
            "last_reviewed": meta.get("last_reviewed", "2026-01-01"),
        })

    client.insert(collection_name=collection_name, data=data)
    client.load_collection(collection_name)
    return client


class MilvusHybridRetriever:
    """Executes multi-vector hybrid search (HNSW dense + BM25 sparse) with RRF fusion."""

    def __init__(self, top_k: int = 2, filter_expr: str | None = None):
        self.top_k = top_k
        self.filter_expr = filter_expr

    def invoke(self, query: str, filter_expr: str | None = None) -> list[Document]:
        client = get_milvus_client()
        collection_name = settings.COLLECTION_NAME

        try:
            client.load_collection(collection_name)
        except MilvusException as exc:
            logger.debug("Collection already loaded: %s", exc)

        active_filter = filter_expr or self.filter_expr

        # 1. Dense query representation
        query_dense = embeddings.embed_query(query)
        dense_req = AnnSearchRequest(
            data=[query_dense],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=self.top_k * 2,
            expr=active_filter,
        )

        # 2. Sparse query representation (BM25 -> Python dict format)
        query_sparse = bm25_ef.encode_queries([query])
        sparse_data = _sparse_to_dict(query_sparse) if issparse(query_sparse) else query_sparse

        sparse_req = AnnSearchRequest(
            data=sparse_data,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=self.top_k * 2,
            expr=active_filter,
        )

        # 3. Hybrid search with Reciprocal Rank Fusion
        results = client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=self.top_k,
            output_fields=[
                "text",
                "source",
                "category",
                "security_level",
                "doc_type",
                "last_reviewed",
            ],
        )

        documents = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get("entity", {})
                documents.append(
                    Document(
                        page_content=entity.get("text", ""),
                        metadata={
                            "source": entity.get("source", "unknown"),
                            "category": entity.get("category", "general"),
                            "security_level": entity.get("security_level", "low"),
                            "doc_type": entity.get("doc_type", "general_doc"),
                            "last_reviewed": entity.get("last_reviewed", ""),
                        },
                    )
                )
        return documents


def get_retriever(k: int = 2, filter_expr: str | None = None) -> MilvusHybridRetriever:
    return MilvusHybridRetriever(top_k=k, filter_expr=filter_expr)