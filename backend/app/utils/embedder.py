"""
Sentence-Transformers 嵌入模型管理
支持中英文混合文本
"""

import threading
from typing import List

from .logger import get_logger

logger = get_logger('mirofish.embedder')

_model = None
_model_lock = threading.Lock()

# 支持中英文混合的多语言模型
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def get_model():
    """获取或加载嵌入模型（单例，延迟加载）"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                logger.info(f"加载嵌入模型: {DEFAULT_MODEL}...")
                _model = SentenceTransformer(DEFAULT_MODEL)
                logger.info("嵌入模型加载完成")
    return _model


def embed(text: str) -> List[float]:
    """生成单个文本的嵌入向量"""
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_batch(texts: List[str]) -> List[List[float]]:
    """批量生成文本嵌入向量"""
    if not texts:
        return []
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=32)
    return [e.tolist() for e in embeddings]


def warmup():
    """预热模型（应用启动时调用，避免首次请求延迟）"""
    try:
        get_model()
        logger.info("嵌入模型预热完成")
    except Exception as e:
        logger.error(f"嵌入模型预热失败: {e}")
