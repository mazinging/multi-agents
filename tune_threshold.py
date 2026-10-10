"""
tune_threshold.py —— 阈值标定脚本（演示"用数据定参数"的方法）
================================================================================
【为什么要有这个脚本？】
    在开发检索功能时，我一开始是"凭感觉"设阈值的：先设 0.34，
    发现召回崩了；改成 0.20，发现噪音多了；再改成 0.45……
    这种来回摇摆是低效且不科学的。

    正确的做法是：把候选阈值都跑一遍，画出"召回率 vs 噪音率"的曲线，
    然后根据业务需要选择工作点。

【关键概念：召回与噪音的权衡（trade-off）】
    这两个指标天然对立：
        阈值调高 -> 更严格 -> 噪音少，但真资料也可能被拦掉（召回降）
        阈值调低 -> 更宽松 -> 召回全，但垃圾也进来了（噪音升）
    不存在"两个都最好"的参数，只有"最适合你业务"的平衡点。

    怎么选？
        - 医疗、法律、金融场景：宁可说"不知道"，也不能答错。
          选高阈值，接受召回率下降。
        - 内部知识助手、搜索建议：宁可多给一些让人自己筛。
          选低阈值，接受一些噪音。

【评估方法说明】
    召回率：用 eval_retrieval.py 里标注的 22 个相关问题测量。
    噪音率：用一组"肯定不该有结果"的查询测量，
           统计其中有多少个仍然返回了结果（假阳性率）。

运行方式:
    python tune_threshold.py
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import core  # noqa: F401
from rag import VectorStore
from eval_retrieval import EVAL_SET


# 肯定不该检索到任何东西的查询。用来测量"假阳性"。
# 【要求】这些查询要尽量贴近真实场景中的"无关提问"，
# 而不是随便写几个不存在的名词 —— 因为后者太容易被过滤，
# 会给出过于乐观的噪音率。
NOISE_QUERIES = [
    "完全不相关的外星人问题",
    "今天晚饭吃什么",
    "股票行情预测",
    "如何养一只猫",
    "2024年世界杯冠军是谁",
    "请帮我写一首诗",
    "人生意义是什么",
    "推荐一部好看的电影",
    "明天会下雨吗",
    "怎么学吉他",
]


def run() -> None:
    store = VectorStore()
    store.build_from_directory(verbose=False)

    print(f"知识库: {store.stats()}")
    print(f"评估集: {len(EVAL_SET)} 个相关问题, {len(NOISE_QUERIES)} 个无关查询\n")

    print("=" * 78)
    print(f"{'BM25阈值':>10} {'Recall@5':>10} {'MRR':>8} {'噪利率':>9}  说明")
    print("=" * 78)

    results = []
    for threshold in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        store._MIN_BM25_ABS = threshold

        # ---- 测召回 ----
        hits = 0
        rr_sum = 0.0
        for question, expected_doc in EVAL_SET:
            got = [h.chunk.doc_id for h in store.search(question, top_k=5)]
            if expected_doc in got:
                hits += 1
                rr_sum += 1.0 / (got.index(expected_doc) + 1)
        recall = hits / len(EVAL_SET)
        mrr = rr_sum / len(EVAL_SET)

        # ---- 测噪音 ----
        noisy = sum(1 for q in NOISE_QUERIES if store.search(q, top_k=5))
        noise_rate = noisy / len(NOISE_QUERIES)

        results.append((threshold, recall, mrr, noise_rate))
        print(f"{threshold:>10.1f} {recall:>10.3f} {mrr:>8.3f} {noise_rate:>8.0%}")

    print("=" * 78)
    print()
    print("【如何读这张表】")
    print("  看 Recall@5 和 噪利率 两个列，找一个平衡点。")
    print("  如果某个阈值能同时保持高 Recall 和低噪利率，那它就是这个语料下的最优解。")
    print()
    print("【本项目实测结论】")
    print("  阈值 3.0 附近是一个合理的平衡点：Recall 保持在高位，")
    print("  同时噪音被显著抑制。这也是 store.py 里 _MIN_BM25_ABS 取 3.0 的依据。")
    print()
    print("【重要提醒】")
    print("  这个最优值是【针对本项目的语料和 embedder】的。")
    print("  换语料、换 embedding 模型后必须重新标定 —— 这就是为什么要有这个脚本。")
    print("  真实项目里，这个流程应该自动化，作为 CI 的一部分。")
    print()
    print("  如果某个阈值怎么调都无法兼顾，说明【特征本身的区分度不够】，")
    print("  这时不该继续调参，而应该换特征（比如换更好的 embedding 模型）。")
    print("  这是本项目想传达的最重要的工程直觉。")


if __name__ == "__main__":
    run()
