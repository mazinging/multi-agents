"""
eval_retrieval.py —— RAG 检索质量评估脚本
================================================================================
【这个脚本为什么存在？】
    在开发 RAG 的过程中，我（作者）多次遇到"改了参数，感觉好像好一点，
    但说不清到底好没好"的情况。这是非常危险的 ——
    凭感觉调参最终一定会调歪。

    所以必须建立一个【可量化】的评估机制。这个脚本做的事很简单：
        1. 准备一批"问题 -> 正确答案所在文档"的标注
        2. 对每个问题跑检索，看正确文档有没有出现在 top-k 里
        3. 统计 Recall@k 和 MRR

【两个核心指标】

    Recall@k（召回率）
        正确文档出现在前 k 个结果里的比例。
        "有没有找到" —— 这是检索最重要的问题。
        如果 Recall@5 只有 0.6，意味着 40% 的问题根本找不到资料，
        那后面 LLM 再强也没用（无米之炊）。

    MRR（Mean Reciprocal Rank，平均倒数排名）
        正确文档排名的倒数的平均值。
        第 1 名得 1.0，第 2 名得 0.5，第 3 名得 0.33……
        "找得够不够靠前" —— 因为 top-k 的名额有限，
        排第 8 名的正确文档在实际使用中可能已经超出了 k。

【为什么要写这个脚本而不是手动看几个例子？】
    因为手动看例子有严重的确认偏误：你会不自觉地把注意力放在
    你期望成功的那几个 case 上。而固定的评估集能给你一个
    不受情绪影响的数字。这个数字才是你做技术决策的依据。

    这也是本脚本想教给你的最重要的一课：
    【没有评估就没有优化】

运行方式:
    python eval_retrieval.py
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import core  # noqa: F401  触发控制台编码修复
from rag import VectorStore


# ==============================================================================
# 评估集：问题 -> 应该被召回的文档 ID
# ==============================================================================
# 【怎么构造评估集？】
#   这里我手工标注了 20 个问题，覆盖两类：
#       a) 明确型：问题里的词和文档标题/内容高度重合（容易）
#       b) 口语型：用户用日常说法提问，和文档措辞不同（困难，更有价值）
#   真实项目里，评估集应该来自真实的用户提问日志，
#   并由人工标注正确答案。这里的 20 条是演示用的最小规模。
#
# 【为什么要包含"口语型"问题？】
#   因为只测"明确型"会给出过分乐观的结论。用户不会照着你的文档标题提问，
#   他们会说"贵不贵""怎么弄""哪个好"。这些才是真实分布。
# ==============================================================================

EVAL_SET: list[tuple[str, str]] = [
    # ---- 明确型（术语与文档一致）----
    ("向量数据库有哪些主流方案", "01_vector_db_overview"),
    ("HNSW 索引的原理是什么", "01_vector_db_overview"),
    ("向量数据库的内存成本怎么算", "02_vector_db_cost"),
    ("Pgvector 和 Milvus 的区别", "02_vector_db_cost"),
    ("中文 embedding 模型有哪些", "03_embedding_models"),
    ("BGE 系列模型的维度是多少", "03_embedding_models"),
    ("文档切块大小怎么设置", "04_chunking_strategies"),
    ("chunk overlap 重叠比例取多少", "04_chunking_strategies"),
    ("RAG 的完整流程包含哪些阶段", "05_rag_architecture"),
    ("怎么防止 RAG 产生幻觉", "05_rag_architecture"),
    ("ReAct 循环的结构是什么", "06_react_pattern"),
    ("ReAct 实现中如何避免死循环", "06_react_pattern"),
    ("多智能体有哪些协作架构模式", "07_multi_agent_arch"),
    ("Orchestrator 协调者模式的优缺点", "07_multi_agent_arch"),
    ("企业知识库问答系统怎么落地", "08_enterprise_rag_plan"),
    ("企业 RAG 如何做权限隔离", "08_enterprise_rag_plan"),

    # ---- 口语型（用词与文档不同，考察检索的鲁棒性）----
    ("我们团队小，没运维，选哪个向量库省事", "02_vector_db_cost"),
    ("怎么把我的文档切好一点，检索总是不准", "04_chunking_strategies"),
    ("模型老是自己编答案怎么办", "05_rag_architecture"),
    ("该不该用 LangChain 还是自己写", "09_framework_selection"),
    ("上线之后要监控哪些指标", "10_deployment_ops"),
    ("线上服务怎么部署和扩容", "10_deployment_ops"),
]


def evaluate(k: int = 5, verbose: bool = True) -> dict[str, float]:
    """
    跑完整评估，返回指标字典。

    【关键实现细节：为什么要按 doc_id 判断而不是 chunk_id？】
        因为标注"正确答案在哪个块里"工作量大且不稳定（切块策略一变，
        chunk_id 全变）。标注到"文档级"更稳定，也足够反映检索能力 ——
        只要召回了正确的文档，后面精读环节就能找到具体内容。
        这是评估设计里"标注成本 vs 评估精度"的务实折中。
    """
    print("=" * 78)
    print(f"构建索引并评估检索质量（top_k={k}）")
    print("=" * 78)

    store = VectorStore()
    store.build_from_directory(verbose=False)
    print(f"\n{store.stats()}\n")

    hits_at_k = 0
    reciprocal_ranks: list[float] = []
    failures: list[tuple[str, str, list[str]]] = []

    for question, expected_doc in EVAL_SET:
        results = store.search(question, top_k=k)
        found_docs = [h.chunk.doc_id for h in results]

        if expected_doc in found_docs:
            hits_at_k += 1
            rank = found_docs.index(expected_doc) + 1   # 1-based
            reciprocal_ranks.append(1.0 / rank)
            marker = f"✓ 第{rank}名"
        else:
            reciprocal_ranks.append(0.0)
            marker = "✗ 未命中"
            failures.append((question, expected_doc, found_docs))

        if verbose:
            got = found_docs[0] if found_docs else "(空)"
            print(f"  {marker:<9} 期望 {expected_doc:<26} 实际首位 {got:<26} | {question}")

    n = len(EVAL_SET)
    recall = hits_at_k / n
    mrr = sum(reciprocal_ranks) / n

    print()
    print("=" * 78)
    print(f"  Recall@{k} = {recall:.3f}  ({hits_at_k}/{n})   —— 找得到吗")
    print(f"  MRR       = {mrr:.3f}              —— 排得够靠前吗")
    print("=" * 78)

    if failures:
        print(f"\n未命中的 {len(failures)} 个案例（这些是最值得分析的）:")
        for q, exp, got in failures:
            print(f"  ✗ 「{q}」")
            print(f"      期望: {exp}")
            print(f"      实得: {got if got else '(检索返回空)'}")

    return {"recall": recall, "mrr": mrr, "k": k}


if __name__ == "__main__":
    evaluate(k=5)
