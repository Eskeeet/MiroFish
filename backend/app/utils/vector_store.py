"""
ChromaDB向量存储管理
单一集合，通过 graph_id 元数据字段区分不同图谱
"""

import threading
from typing import Any, Dict, List, Optional

from .logger import get_logger
from ..config import Config

logger = get_logger('mirofish.vector_store')

_client = None
_client_lock = threading.Lock()

COLLECTION_NAME = "mirofish_facts"


def get_client():
    """获取或创建ChromaDB持久化客户端（单例）"""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                import os
                import chromadb
                os.makedirs(Config.CHROMA_PERSIST_DIR, exist_ok=True)
                _client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIR)
                logger.info(f"ChromaDB客户端已初始化: {Config.CHROMA_PERSIST_DIR}")
    return _client


def get_collection():
    """获取或创建事实集合"""
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def upsert_facts(facts: List[Dict[str, Any]]):
    """
    批量 upsert 事实到向量存储

    Args:
        facts: 每项包含 {id, text, embedding, metadata}
    """
    if not facts:
        return
    collection = get_collection()
    collection.upsert(
        ids=[f["id"] for f in facts],
        documents=[f["text"] for f in facts],
        embeddings=[f["embedding"] for f in facts],
        metadatas=[f["metadata"] for f in facts],
    )
    logger.debug(f"ChromaDB upsert {len(facts)} 条事实")


def update_fact_metadata(fact_id: str, updates: Dict[str, Any]):
    """Merge temporal/provenance fields into an existing fact's metadata.

    Neo4j is the source of truth, but keeping the Chroma copy synchronized
    prevents invalidated facts from being presented as current during vector
    retrieval.
    """
    if not fact_id or not updates:
        return
    collection = get_collection()
    try:
        current = collection.get(ids=[fact_id], include=["metadatas"])
        if not current or not current.get("ids"):
            return
        metadata = dict((current.get("metadatas") or [{}])[0] or {})
        for key, value in updates.items():
            # Chroma metadata values must be scalar and cannot be None.
            metadata[key] = "" if value is None else value
        collection.update(ids=[fact_id], metadatas=[metadata])
    except Exception as e:
        logger.warning(f"更新事实 {fact_id} 的 ChromaDB 元数据失败: {e}")


def semantic_search(
    query_embedding: List[float],
    graph_id: str,
    n_results: int = 10,
    where: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """
    语义搜索

    Returns:
        结果列表，每项含 {id, document, metadata, score}
    """
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    filter_condition: Dict[str, Any] = {"graph_id": graph_id}
    if where:
        # Merge with $and if both filters present
        filter_condition = {"$and": [{"graph_id": graph_id}, where]}

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            where=filter_condition,
            include=["documents", "metadatas", "distances"],
        )
        items = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                items.append({
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "score": max(0.0, 1.0 - results["distances"][0][i]),
                })
        return items
    except Exception as e:
        logger.warning(f"ChromaDB搜索失败: {e}")
        return []


def delete_graph_facts(graph_id: str):
    """删除指定图谱的所有向量"""
    try:
        collection = get_collection()
        collection.delete(where={"graph_id": graph_id})
        logger.info(f"已删除图谱 {graph_id} 的所有向量")
    except Exception as e:
        logger.warning(f"删除图谱向量失败: {e}")
