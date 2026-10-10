"""
rag/__init__.py —— RAG（检索增强生成）包
================================================================================
RAG = Retrieval-Augmented Generation，检索增强生成。
    一句话解释：模型自己不知道的东西，先去资料库里查，把查到的内容
    塞进 prompt 里再让模型回答。

为什么需要 RAG？
    1. 模型的知识有截止日期，且不知道你的私有资料
    2. 直接问模型容易"编"（幻觉），有资料约束就可靠得多
    3. 把知识外置，更新资料不用重新训练模型

本包三个文件的职责（就是 RAG 的完整链路）：
    loader.py    : 原始文档 -> 切成小块（为什么要切？见文件内注释）
    embedder.py  : 文本块 -> 向量（把"语义"变成"可计算的数字"）
    store.py     : 向量入库 + 相似度检索（找出最相关的几块）

【关键理解：RAG 的质量瓶颈通常不在最后一步（生成），而在第一步（切块）】
    切得太碎 -> 语义不完整，检索到也没用
    切得太大 -> 一个块里混了太多主题，向量表示被"平均"掉，检索不精准
    这就是为什么 chunk_size / chunk_overlap 这两个参数那么重要。
================================================================================
"""

from __future__ import annotations

# 导入 core 会触发控制台编码修复（见 core/__init__.py 的说明）。
# 【为什么要在这里 import？】因为 rag 的 loader 会打印 emoji，
# 而使用方可能只 import rag 而不 import core。在这里保证初始化一定发生。
import core  # noqa: F401

from rag.loader import Chunk, load_documents, chunk_text
from rag.embedder import Embedder, HashingEmbedder, build_embedder
from rag.store import VectorStore, SearchHit

__all__ = [
    "Chunk",
    "load_documents",
    "chunk_text",
    "Embedder",
    "HashingEmbedder",
    "build_embedder",
    "VectorStore",
    "SearchHit",
]
