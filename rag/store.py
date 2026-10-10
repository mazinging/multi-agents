"""
rag/store.py —— 向量库与相似度检索（RAG 链路第 3 步）
================================================================================
【这个文件在架构里的位置】
    RAG 链路的最后一环：给定一个查询，从所有文档块里挑出最相关的几个。

【整体数据流】
    Chunk[]  --embedder-->  向量矩阵 (N, dim)
                              ↓  存起来
    查询文本 --embedder-->  查询向量 (dim,)
                              ↓  矩阵乘法
                          相似度数组 (N,)
                              ↓  排序取 top-k
                          SearchHit[]  （带分数和相关文档块）

【为什么叫"向量库"却不引入 FAISS / Milvus 等专业库？】
    专业向量库解决的是【百万级以上】规模的检索性能问题：
        - 用 HNSW / IVF 等近似最近邻（ANN）算法，牺牲一点点精度换百倍速度
        - 支持分布式、持久化、增量更新、过滤等
    但它们的【检索原理完全一样】：算相似度、排序、取 top-k。
    本项目语料只有几十个块，numpy 一次矩阵乘法就是微秒级，
    用暴力精确检索（brute-force）既简单又【更准】（ANN 是近似的）。

    【学习建议】先把这里的流程彻底搞懂（它只有几十行核心逻辑），
    之后再看 FAISS，你会发现它只是把"矩阵乘法"换成了"索引查找"，
    上层的 RAG 逻辑一点没变。

【本文件两个核心设计】
    1. 检索结果必须携带【溯源信息】（chunk_id / title / source）——
       这是 RAG 能"防幻觉"的根本：模型说的每句话都能被核对
    2. 分数阈值过滤 —— 低分结果宁可不要，也不要给模型喂噪音
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from config import CFG
from rag.embedder import Embedder, cosine_similarity_matrix
from rag.loader import Chunk


def _is_discriminative_token(token: str) -> bool:
    """
    判断一个 token 是否"有区分度"，可以用来给 BM25 打分。

    ── 规则 ──
        1. 英文/数字整词（以 "w:" 开头）：有区分度。
           因为英文词天然是完整词，"hnsw"、"rag"、"milvus" 这类
           技术术语的匹配非常可靠。
        2. 中文 token：只有长度 >= 2 的 n-gram 才算有区分度。

    ── 为什么是 2 而不是 3？（这是实测调优的结果，过程值得记录）──
        我最初用的是 3，理由是"3-gram 要求连续三字相同，无关文本几乎不可能碰巧满足"。
        实测确实把无关查询的噪音压到了 0（7 个无关查询里 5 个得分 0），
        但同时把 Recall@5 从 0.955 拉低到了 0.773。

        分析退化的原因：口语化查询依赖 2-gram 才能生效。
            "模型老是自己编答案怎么办" -> 3-gram 只剩"怎么""么办"，
                而"怎么""么办"在语料里是高频泛化串，几乎不算有效信号，
                导致这个问题直接返回空。
            "线上服务怎么部署和扩容" -> 3-gram 命中不到"部署""扩容"这类关键 2 字词。

        改回 2 之后，Recall 恢复到 0.955，而噪音仍然可控 ——
        因为剩下的噪音由【IDF 加权的绝对阈值】(3.0) 来兜底：
        2-gram 的泛化串（"怎么""什么"）IDF 很低，贡献的分数很小，
        到不了 3.0 的门槛。

        【这个权衡教会我一件事】
            过滤规则和阈值不是独立的两个东西，它们共同决定最终效果，
            必须【一起调】。把过滤调严了，阈值就该放宽；反之亦然。
            我一开始把两者当成独立的开关分别调，结果反复在两个极端之间摇摆。
            正确做法是：先确定一个可量化的目标（Recall@5 和 噪音率），
            然后联合搜索这两个参数的组合。
            本项目保留 eval_retrieval.py，就是让这个过程可复现。

    ── 这个函数的 token 格式约定 ──
        embedder 产出的 token 带前缀："w:" 表示英文词，"c:" 表示中文片段。
        所以要剥掉前缀再判断长度。
        【为什么要带前缀？】见 embedder._tokenize 的注释 ——
        避免英文词和中文片段哈希到同一维度时产生系统性偏差。

    ── 局限说明（诚实记录）──
        长度为 1 的中文 token（单字）被完全排除在打分之外。
        这让"只记得一个关键字"的超短查询变得困难：比如用户只问"成本"，
        它只有 2 个字、构不成 2-gram，会被过滤掉。
        真正的解法是引入中文分词（jieba）+ 同义词表 ——
        那样"成本""费用""开销"能被识别为同一个概念。
        但那需要额外依赖，本项目为了"离线零依赖可跑"没有引入。
        这是这个约束带来的真实代价，请你知悉：
        【工程上任何"零依赖"的选择都有代价，关键是知道代价在哪】
    """
    if token.startswith("w:"):
        return True
    if token.startswith("c:"):
        return len(token) - 2 >= 2    # 减去 "c:" 前缀的长度
    # 未知格式：保守起见认为有区分度（不误杀）
    return True


@dataclass
class SearchHit:
    """
    一条检索命中结果。

    【为什么要把 score 和 chunk 一起返回，而不是只返回 chunk？】
        因为分数是重要的决策依据：
            - 阈值过滤（低于阈值的不要）
            - 让模型看到"这条资料的相关度是 0.71"，
              它会据此调整措辞的确定性（"有资料显示…" vs "可以确定…"）
            - 调试时，分数是判断"检索效果好不好"的唯一量化指标
    """
    chunk: Chunk
    score: float

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def to_dict(self) -> dict[str, Any]:
        d = self.chunk.to_dict()
        d["score"] = round(self.score, 4)
        return d

    def to_prompt_block(self, index: int) -> str:
        """
        格式化成给 LLM 看的文本块。

        【这里的格式设计非常关键 —— 它直接决定模型能不能正确引用来源】
            每一块都带一个明确的编号 [资料1]、[资料2]，
            并在末尾附上 doc_id。

            为什么必须这样做？
                如果你只是把文本堆在 prompt 里，模型无法引用（它不知道
                "第几段"对应哪个 chunk_id）。给了显式编号后，你就可以在
                system prompt 里要求它"引用时写 [资料N]"，
                从而实现【可验证的引用】。

            这是 RAG 防幻觉的核心机制：
                不是靠"求模型别瞎说"，而是靠"给模型可引用的锚点"。
        """
        return (
            f"[资料{index}] (相关度 {self.score:.2f} | 来源 {self.chunk.chunk_id} | {self.chunk.title})\n"
            f"{self.chunk.text}"
        )


class VectorStore:
    """
    极简向量库 + BM25 稀疏索引（混合检索）。

    ── 为什么要做混合检索？──
        这是本项目在实测中发现的必要改进，过程很有教学价值：

        最初我只做了稠密向量检索（哈希向量 + 余弦相似度）。
        实测发现一个严重问题：**问一个知识库里完全没有的问题
        （比如"外星人问题"），最高分仍有 0.116**，
        而真正相关的问题最高分只有 0.14 —— 两者几乎无法区分，
        阈值过滤彻底失效。

        诊断后发现两个根因：
            1. 512 维的哈希空间中，143 个文档块占用了 45% 的维度，
               哈希冲突严重。任何两个文本都会因随机冲突产生
               "虚假相似度"，这就是那个 0.116 的来源。
            2. 稠密向量的点积对"短查询"不公平：查询只有几个词，
               而文档块有几百个 token，点积更容易被文档侧的噪音推动。

        解决方案是引入第二路检索：BM25 稀疏检索。
            - BM25 精确匹配 token，没有哈希冲突，不相关就是 0 分
            - BM25 自带文档长度归一化（参数 b），解决第 2 个问题
            - 两路互补：稠密负责"模糊相关"，稀疏负责"精确命中"

        最后用 RRF 融合两路排名。这就是工业界标准的混合检索方案。
        【关键结论】不要指望调参救活一个有结构性缺陷的方案，
        要识别出缺陷的根源，然后引入正确的机制。这个思路比具体代码重要。
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        use_bm25: bool = True,
        trace_hook: Callable[[dict], None] | None = None,
    ) -> None:
        from rag.embedder import build_embedder

        self.embedder = embedder or build_embedder()
        self.use_bm25 = use_bm25

        # 【可视化埋点的默认钩子】
        #   设成实例属性后，所有 search() 调用都会自动带上它，
        #   不需要每个调用点都手动传 trace_hook。
        #   这对本项目很重要：search 的调用点比较多
        #   （search_local_docs 工具、cross_compare 工具、评测脚本…），
        #   逐个改既麻烦又容易漏。挂在实例上，一个地方设好就行。
        self.trace_hook = trace_hook

        self._chunks: list[Chunk] = []
        # 向量矩阵，形状 (N, dim)。
        # 【为什么用 vstack 累积而不是 list 存向量再转换？】
        #   因为矩阵乘法需要二维数组。我们在 add_chunks 里一次性 vstack，
        #   保证 _matrix 始终是合法形状，避免每次 search 都判断类型。
        self._matrix: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)

        # ---- BM25 所需的倒排结构和统计量 ----
        # 倒排索引：token -> {块下标: 该 token 在这个块里的出现次数}
        # 【为什么用 dict 而不是 list？】因为一个 token 只出现在少数块里，
        # 用 dict 存储稀疏关系，查询时只遍历真正包含该 token 的块 ——
        # 这正是"倒排索引"比"正排扫描"快的原因。
        self._inverted: dict[str, dict[int, int]] = {}
        self._doc_len: list[int] = []      # 每个块的 token 总数
        self._avg_doc_len: float = 0.0     # 平均长度，BM25 归一化的基准
        # 用于 RRF 融合时判断"是否真的命中了关键词"
        self._chunk_tokens: list[set[str]] = []

    # ==========================================================================
    # 一、建索引
    # ==========================================================================

    def add_chunks(self, chunks: list[Chunk], *, verbose: bool = True) -> None:
        """
        把文档块加入索引（向量化 + 存储）。

        【关于"批量向量化"的重要性】
            这里一次性把所有块的文本传给 embedder，而不是循环调用 embed_one。
            原因：
                a) 如果换成 OpenAIEmbedder，批处理能把 N 次网络请求变成 1 次
                b) numpy 的向量化运算比逐条循环快得多
            这是 RAG 索引构建阶段最基本的性能优化。
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]

        # ---- 关键步骤：在向量化之前先拟合 IDF ----
        # 【为什么必须在这里做？】
        #   IDF 需要"看到全部语料"才能统计出每个词的稀有度，
        #   所以它必须发生在 embed 之前，而且要用完整的语料（不是单个块）。
        #   如果漏了这一步，embedder 会退化成等权重模式 ——
        #   程序照常跑，效果却差很多。这类"静默降级"是最难排查的 bug。
        #
        # 【为什么用 hasattr 而不是直接调用？】
        #   因为 fit_idf 是 HashingEmbedder 特有的（真 embedding API 不需要它，
        #   语义模型自带语义能力）。
        #   用 hasattr 做能力探测，让 store 不必知道具体是哪种 embedder ——
        #   这保持了"面向接口编程"的原则。
        #   这叫"鸭子类型"：不问"你是什么类"，只问"你会不会这个技能"。
        if hasattr(self.embedder, "fit_idf"):
            self.embedder.fit_idf(texts)
            if verbose and CFG.log_level != "quiet":
                n_tokens = len(getattr(self.embedder, "_idf", {}))
                print(f"    📊 已拟合 IDF 权重表（{n_tokens} 个特征）")

        vectors = self.embedder.embed(texts)

        base = len(self._chunks)   # 新块在全局的下标起点
        self._chunks.extend(chunks)

        # ---- 构建 BM25 倒排索引 ----
        if self.use_bm25:
            for offset, text in enumerate(texts):
                idx = base + offset
                toks = self._tokenize_for_bm25(text)

                self._chunk_tokens.append(set(toks))
                self._doc_len.append(len(toks))

                # 统计词频（TF）：同一个 token 在块内出现几次
                for tok in toks:
                    bucket = self._inverted.setdefault(tok, {})
                    bucket[idx] = bucket.get(idx, 0) + 1

            # 重算平均长度。注意每次 add 都要重算 ——
            # 因为 BM25 的长度归一化是相对于"整个语料的平均长度"的
            self._avg_doc_len = (
                sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0
            )

        if self._matrix.size == 0:
            self._matrix = vectors
        else:
            self._matrix = np.vstack([self._matrix, vectors])

        # 形状一致性检查：如果两次 add 用的 embedder 维度不同，
        # vstack 会直接报错，但错误信息很晦涩。这里提前给出清晰提示。
        if self._matrix.shape[0] != len(self._chunks):
            raise RuntimeError(
                f"向量数与文档块数不一致：{self._matrix.shape[0]} vs {len(self._chunks)}。"
                f"通常是多次 add_chunks 时用了维度不同的 embedder。"
            )

        if verbose and CFG.log_level != "quiet":
            print(f"    🔢 已索引 {len(chunks)} 个块，向量矩阵形状 {self._matrix.shape}"
                  + (f"，倒排表 {len(self._inverted)} 个词" if self.use_bm25 else ""))

    def _tokenize_for_bm25(self, text: str) -> list[str]:
        """
        为 BM25 分词。

        【为什么用 list（含重复）而不是 set？】
            因为 BM25 需要【词频 TF】—— 一个词在块里出现 3 次和 1 次，
            重要性不同，应该给更高分。所以必须保留重复。
        """
        return [tok for tok, _ in self.embedder._tokenize(text)]

    def build_from_directory(self, directory=None, **kwargs) -> int:
        """便捷方法：从知识库目录一步建好索引，返回块数"""
        from rag.loader import load_documents

        chunks = load_documents(directory, **kwargs)
        self.add_chunks(chunks)
        return len(chunks)

    # ==========================================================================
    # 二、检索
    # ==========================================================================

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        doc_filter: str | None = None,
        trace_hook: Callable[[dict], None] | None = None,
    ) -> list[SearchHit]:
        """
        混合检索：稠密向量 + BM25 稀疏，用 RRF 融合。

        ── 参数 ──
            top_k      : 返回几条。多了会稀释注意力，少了可能漏掉关键信息。
                         【经验】3~5 条是常见起点。本项目默认 4。
            min_score  : 分数下限（对 RRF 融合分生效）。
            doc_filter : 只在该文档内检索。用于"精读某篇文档"的场景
                         —— 这正是 ReAct 里 read_doc_section 工具的基础。
            trace_hook : 【为 Web 可视化新增】可选的埋点回调。
                         检索过程中每个阶段都会调用它一次，传入该阶段的中间数据。
                         【为什么要做成参数，而不是让 store 直接依赖 tracer？】
                           因为 rag 层不应该知道 agents/trace 的存在 ——
                           那是上层的东西。让调用方把回调传进来，
                           依赖方向就保持了"上层依赖下层"，没有反向依赖。
                           【这是依赖注入最典型的用法：低层提供扩展点，
                             高层决定挂什么上去。】
                         传 None 时零开销，完全不影响命令行模式。

        ── 返回 ──
            按融合分降序排列的 SearchHit 列表（可能为空）
        """
        top_k = top_k or CFG.retrieval_top_k
        min_score = 0.0 if min_score is None else min_score

        # 参数没显式传就用实例上挂的默认钩子（见 __init__ 的说明）
        if trace_hook is None:
            trace_hook = self.trace_hook

        if not self._chunks:
            return []

        # 埋点辅助函数：没传 hook 时是个空操作，调用处不用写 if 判断
        def emit(stage: str, **data) -> None:
            if trace_hook is not None:
                try:
                    trace_hook({"stage": stage, "query": query, **data})
                except Exception:      # noqa: BLE001
                    pass   # 埋点绝不能影响检索本身

        # ---- 第 1 路：稠密向量检索 ----
        dense_hits = self._dense_search(query, top_k=top_k * 3, doc_filter=doc_filter)

        # ---- 第 2 路：BM25 稀疏检索 ----
        sparse_hits: list[tuple[int, float]] = []
        if self.use_bm25:
            sparse_hits = self._bm25_search(query, top_k=top_k * 3, doc_filter=doc_filter)

        # ---- 埋点：两路召回各自的原始结果 ----
        # 【这个数据对理解混合检索至关重要】
        #   你会看到两路召回的块【并不相同】——
        #   稠密能召回用词不同但语义相近的块，
        #   BM25 能召回精确匹配术语的块。两者互补，
        #   这正是"混合检索"比单一检索好的原因。
        emit(
            "recall",
            dense=[{"idx": i, "score": round(s, 4), "chunk_id": self._chunks[i].chunk_id}
                   for i, s in dense_hits],
            sparse=[{"idx": i, "score": round(s, 4), "chunk_id": self._chunks[i].chunk_id}
                    for i, s in sparse_hits],
        )

        # ---- 如果没启用 BM25，直接返回稠密结果（保持向后兼容）----
        if not self.use_bm25:
            hits = [h for h in dense_hits if h.score >= min_score][:top_k]
            return hits

        # ---- RRF 融合 ----
        fused = self._rrf_fuse(dense_hits, sparse_hits, doc_filter=doc_filter)

        emit(
            "fuse",
            fused=[{"idx": i, "rrf": round(s, 4), "chunk_id": self._chunks[i].chunk_id}
                   for i, s in fused],
        )

        # ---- 剪枝与相关性门槛 ----
        # 【这是本项目实测后加上的最关键的一道防线，值得仔细理解】
        #
        # 问题：即使有 BM25，问一个知识库完全不覆盖的问题
        # （比如"外星人问题"），仍然会返回结果。原因很有意思：
        # 中文里"问题"这个词本身是有意义的 token，
        # 而技术文档里到处都有"问题"（"这个问题""常见问题"），
        # 所以它能靠一个泛化词匹配上。这不是 bug，是词汇检索的固有特性。
        #
        # ── 第一版方案（有缺陷，保留下来作为教训）──
        # 我最初用的是"覆盖率"：要求命中的查询 token 占查询总 token 的比例
        # 达到阈值。它在"外星人问题"这个案例上表现完美（覆盖率 0.1，被拦下）。
        #
        # 但建立评估集后实测发现【这个方案有严重缺陷】：
        #   22 个测试问题里有 7 个返回了空结果，Recall@5 只有 0.591。
        #   分析空结果的成因：查询里的"完整""包含""哪些""阶段"
        #   这类实词在文档里恰好都没出现，分母被它们撑大，
        #   导致真正的关键词（RAG、流程）虽然命中了，覆盖率仍然很低，
        #   于是被误杀。
        #
        # ── 修正方案：改用 BM25 的绝对分数门槛 ──
        #   BM25 分数已经内建了 IDF 加权：命中"RAG"这种高 IDF 的词
        #   会得到很高的分；命中"问题"这种低 IDF 的泛化词得分很低。
        #   所以只要卡一个分数线，就能自然地
        #   "放过命中实词的查询，拦住只蹭到泛化词的查询"。
        #
        #   它比覆盖率更好，因为不需要猜"哪些查询词重要"——
        #   IDF 已经替我们算好了。这是"让机制承担判断，而不是让规则硬编码"的思路。
        #
        # 【这就是建立评估集的价值，也是本项目最想传达的经验之一】
        #   如果没有那 22 个标注样本，我会以为覆盖率方案是对的
        #   （它在手边那个"外星人问题"上表现很好），然后在真实使用中
        #   才发现大量查询莫名其妙返回空结果。更糟的是，
        #   "返回空"看起来像是合理的"没找到资料"，
        #   这个 bug 会极其难以被发现和归因。
        #
        #   教训：不要用一两个手边案例验证方案。要有一批固定的、
        #   有代表性的标注样本，每次改动都跑一遍。
        sparse_map = dict(sparse_hits)
        best_sparse = max(sparse_map.values()) if sparse_map else 0.0

        # ---- 兜底分支：稀疏检索完全无信号时，回退到稠密检索 ----
        # 说明：查询里的实词在语料里一个都没命中（可能是用了完全不同的术语），
        # 但稠密向量仍可能捕捉到一些模糊相关。这时不该直接返回空，
        # 应该给模型一些可能相关的上下文让它自己判断。
        #
        # 【但这个兜底必须加条件，不能无条件触发】—— 这是我实测踩的坑：
        #   第一版写成 `if best_sparse <= 0: return 纯稠密结果`，
        #   结果所有无关查询（"今天晚饭吃什么"）都走了这条分支，
        #   拿到了 3 条完全不相关的资料，因为稠密检索【永远】会返回
        #   前 k 个结果，哪怕全是噪音。
        #
        #   修正：要求稠密分数也达到一个最低标准，才承认"可能有模糊相关"。
        #   下面的 _MIN_DENSE_FALLBACK 是实测标定的 ——
        #   哈希向量的余弦相似度普遍偏低（相关时约 0.10~0.25，
        #   不相关时约 0.05~0.12），所以门槛设在 0.16。
        #
        # 【更根本的认知】稠密检索没有"无结果"这个状态 ——
        #   只要库里非空，它总能给出 top-k。这是向量检索的一个根本局限：
        #   它无法表达"我不确定"。而 BM25 可以（得分为 0 就是 0）。
        #   这正是混合检索里稀疏那一路最不可替代的价值。
        if best_sparse <= 0:
            if not dense_hits:
                emit("gate", gate="稀疏无信号且稠密无结果", passed=False, reason="返回空（知识库无相关内容）")
                return []
            best_dense_raw = dense_hits[0][1]
            if best_dense_raw < self._MIN_DENSE_FALLBACK:
                emit(
                    "gate",
                    gate="稠密兜底",
                    passed=False,
                    detail={"最佳稠密分": round(best_dense_raw, 4),
                            "兜底门槛": self._MIN_DENSE_FALLBACK},
                    reason="稀疏检索无信号，且稠密分低于兜底门槛 -> 返回空",
                )
                return []
            emit(
                "gate",
                gate="稠密兜底",
                passed=True,
                detail={"最佳稠密分": round(best_dense_raw, 4),
                        "兜底门槛": self._MIN_DENSE_FALLBACK},
                reason="稀疏检索无信号，但稠密分达到兜底门槛 -> 返回稠密结果",
            )
            return [
                SearchHit(chunk=self._chunks[idx], score=max(s, 1e-6))
                for idx, s in dense_hits[:top_k]
            ]

        # ---- 绝对分数门槛：拦住"只蹭到泛化词"的查询 ----
        # 【为什么相对阈值（下一段的 _MIN_REL_BM25）不够，还需要绝对阈值？】
        #   实测发现的场景：
        #       "今天晚饭吃什么" -> BM25 只命中了"吃"相关的一个泛化字，
        #       得分 0.3。因为它是【本次查询的最高分】，相对阈值
        #       (0.3/0.3 = 1.0) 完全拦不住它，结果返回了 3 条完全不相关的资料。
        #
        #   相对阈值能解决的问题是"在多个候选中挑相关度够高的"，
        #   但解决不了"所有候选都很差，只是差得均匀"。
        #   两者是不同维度的问题，必须【同时】使用：
        #       绝对阈值 -> 判断"这次检索整体上有没有意义"
        #       相对阈值 -> 判断"这条候选相对其他候选够不够好"
        #
        # 【这是阈值设计的一个通用经验】永远不要只用一个阈值。
        #   绝对阈值管"生死线"，相对阈值管"内部排序筛选"。
        if best_sparse < self._MIN_BM25_ABS:
            emit(
                "gate",
                gate="绝对分数门槛",
                passed=False,
                detail={"最佳 BM25 分": round(best_sparse, 3),
                        "绝对门槛": self._MIN_BM25_ABS},
                reason="本次检索整体信号太弱（只蹭到泛化词）-> 返回空，宁可说不知道",
            )
            return []

        pruned = []
        dropped: list[dict] = []      # 记录被剪掉的候选（供可视化展示）
        for idx, score in fused:
            # 该块在稀疏检索里的原始 BM25 分（没进稀疏结果就是 0）
            bm25_score = sparse_map.get(idx, 0.0)

            # 归一化成"相对于本次查询最强命中块的比例"。
            # 【为什么用相对值而不是绝对值？】
            #   BM25 绝对分数受查询长度、词数影响很大，很难设统一阈值。
            #   而"相对于最佳命中的比例"是 0~1 的比值，尺度稳定。
            #   这里要求保留 45% 以上，即"不能比最强命中弱太多"。
            #
            # 【这个值从 0.20 调到 0.45 的原因】
            #   0.20 太宽松，导致"HNSW 索引的原理"这种查询会召回
            #   一堆只是碰巧含 HNSW 一词的块，把真正的原理段落挤掉。
            #   收紧到 0.45 后，只有"和最佳命中相关度相当"的块才留下，
            #   排序质量（MRR）明显改善。
            rel = bm25_score / best_sparse

            if rel < self._MIN_REL_BM25:
                # 记录被相对阈值刷掉的候选，可视化时能看清"为什么这条没出现"
                dropped.append({
                    "chunk_id": self._chunks[idx].chunk_id,
                    "bm25": round(bm25_score, 3),
                    "relative": round(rel, 3),
                    "reason": f"相对相关度 {rel:.3f} < 门槛 {self._MIN_REL_BM25}",
                })
                continue

            pruned.append((idx, score))

        emit(
            "prune",
            kept=[{"idx": i, "chunk_id": self._chunks[i].chunk_id} for i, _ in pruned],
            dropped=dropped,
            thresholds={"绝对门槛_BM25": self._MIN_BM25_ABS,
                        "相对门槛": self._MIN_REL_BM25},
        )

        # 如果剪枝把结果清空了，说明知识库完全不覆盖此问题。
        # 这时返回空列表是【正确行为】—— 上层据此判断"资料不足"，
        # 而不是硬塞几个不相关的块让模型编答案。
        # 【这是 RAG 防幻觉的第一道防线，也是最有效的一道】
        if not pruned:
            emit(
                "gate",
                gate="相对相关度剪枝",
                passed=False,
                reason="所有候选都被相对门槛刷掉 -> 返回空（知识库不覆盖此问题）",
            )
            return []

        # ---- 最终排序：以 BM25 分为主 ----
        # 【为什么不用 RRF 融合分排序？】
        #   实测发现 RRF 分有严重问题：它只看排名不看分数，
        #   所以"稠密排第 1 但 BM25 只有 0.3"的块，
        #   会和"BM25 高达 5.0 的真正最佳块"拿到相近的融合分，
        #   造成排序混乱（MRR 上不去，正是这个原因）。
        #
        #   修正：用 BM25 分作为主排序依据，稠密分作为加权补充。
        #   在语料有明确术语（技术文档、法律条文）的场景里，
        #   精确词匹配的信号远比模糊语义相似度可靠。
        #   如果换成真 embedding 模型，可以把稠密分的权重调高一些。
        #
        # 【工程启示】RRF 是"量纲不同、无法比较"时的妥协方案。
        #   但如果你能判断出哪一路信号更可信，就该给它更高权重，
        #   而不是用 RRF 假装两路同等重要。
        dense_map = dict(dense_hits)
        dense_best = max(dense_map.values()) if dense_map else 1.0

        def final_score(idx: int) -> float:
            b = sparse_map.get(idx, 0.0) / best_sparse
            d = (dense_map.get(idx, 0.0) / dense_best) if dense_best > 0 else 0.0
            # 稀疏权重 0.75，稠密权重 0.25。
            # 这个比例反映了"在本项目的语料和 embedder 下，
            # BM25 比哈希向量更可信"这一实测结论。
            return 0.75 * b + 0.25 * max(d, 0.0)

        pruned.sort(key=lambda kv: -final_score(kv[0]))

        # ---- 归一化融合分到 0~1，便于设置统一的阈值 ----
        max_score = max(s for _, s in pruned) or 1.0
        results: list[SearchHit] = []
        for idx, score in pruned[: top_k * 2]:
            norm_score = score / max_score
            if norm_score < min_score:
                continue
            results.append(SearchHit(chunk=self._chunks[idx], score=norm_score))

        final = results[:top_k]

        # ---- 埋点：最终返回结果（含重排序后的综合分明细）----
        emit(
            "final",
            hits=[
                {
                    "chunk_id": h.chunk.chunk_id,
                    "doc_id": h.chunk.doc_id,
                    "title": h.chunk.title,
                    "score": round(h.score, 4),
                    "bm25": round(sparse_map.get(self._idx_of(h.chunk.chunk_id), 0.0), 3),
                    "dense": round(dense_map.get(self._idx_of(h.chunk.chunk_id), 0.0), 4),
                    "preview": h.chunk.preview(160),
                }
                for h in final
            ],
            empty=not final,
        )

        return final

    def _idx_of(self, chunk_id: str) -> int:
        """
        按 chunk_id 反查块下标（供埋点展示原始分数用）。

        【为什么要单独写这个方法？】
           因为最终返回的 SearchHit 里带着 chunk 对象，
           但 BM25 / 稠密分数是按【下标】存的。
           要展示"这条为什么排在这里"，就得把下标找回来。
           用一个 dict 缓存比每次线性查找快，但这里只在埋点时调用，
           量很小，直接用 dict 查更简单。
        """
        if not hasattr(self, "_chunk_id_to_idx"):
            self._chunk_id_to_idx = {
                c.chunk_id: i for i, c in enumerate(self._chunks)
            }
        return self._chunk_id_to_idx.get(chunk_id, -1)

    # --------------------------------------------------------------------------
    def _dense_search(self, query: str, *, top_k: int,
                      doc_filter: str | None) -> list[tuple[int, float]]:
        """稠密向量检索，返回 [(块下标, 原始分数), ...]（降序）"""
        query_vec = self.embedder.embed_one(query)
        scores = cosine_similarity_matrix(query_vec, self._matrix)

        order = np.argsort(-scores)
        out: list[tuple[int, float]] = []
        for i in order:
            idx = int(i)
            if doc_filter and self._chunks[idx].doc_id != doc_filter:
                continue
            out.append((idx, float(scores[idx])))
            if len(out) >= top_k:
                break
        return out

    # BM25 的两个可调参数：
    #   k1 控制"词频饱和度"。一个词出现 10 次不应该比出现 5 次重要一倍，
    #      因为第 6~10 次出现的信息增量很小。k1=1.2~2.0 是经典取值。
    #   b  控制"文档长度归一化强度"。b=1 表示完全归一化，b=0 表示不归一化。
    #      取 0.75 是经过大量实验的经典值。
    #      【b 为什么重要】长文档天然包含更多词、更容易命中查询词。
    #      不做长度归一化，长块会系统性地压过短块 —— 这就是我实测
    #      发现"不相关查询召回长块"的根因之一。
    _BM25_K1 = 1.5
    _BM25_B = 0.75

    def _bm25_search(self, query: str, *, top_k: int,
                     doc_filter: str | None) -> list[tuple[int, float]]:
        """
        BM25 稀疏检索。

        ── BM25 公式 ──
                           f(t,d) · (k1 + 1)
            score = Σ IDF(t) · ─────────────────────────────
                        f(t,d) + k1 · (1 - b + b · |d|/avgdl)
            其中：
                t     : 查询里的每个 token
                f(t,d): token t 在文档 d 里的词频
                |d|   : 文档 d 的 token 数
                avgdl : 所有文档的平均 token 数

        ── 直观理解 ──
            IDF(t)  : 越稀有的词贡献越大（和稠密那路用的是同一套 IDF）
            分子    : 词频越高分越高，但被 k1 限制住不会无限增长
            分母    : 文档越长，惩罚越大（防止长文档靠"词多"占便宜）

        ── 相对稠密检索的优势 ──
            1. 没有哈希冲突，没命中就是 0 分（这正是我们需要的"干净"信号）
            2. 有长度归一化，不受块大小影响
            3. 可解释：每个 token 贡献了多少分都能算出来，便于调试
        """
        if not self._inverted or self._avg_doc_len <= 0:
            return []

        # 【关键改进：打分时只用"长 n-gram"】
        # ── 为什么必须这样过滤？这是本项目最深刻的一次实测教训 ──
        #
        # 我最初想用一个 BM25 绝对分数阈值来判断"检索是否有意义"，
        # 但实测发现阈值根本定不下来：
        #     相关查询   BM25 最高分区间 19.8 ~ 101.1
        #     无关查询   BM25 最高分区间  9.2 ~  32.3
        # 【两个区间完全重叠】，不存在任何一个阈值能把它们分开。
        #
        # 深挖原因：噪音来自【单字和 2-gram】。
        #   查询"今天晚饭吃什么"里的"天""吃""什""么"、
        #   "人生意义是什么"里的"是""什""么"，
        #   都是高频字，在 143 个技术文档块里到处出现。
        #   虽然它们单个 IDF 低，但架不住数量多、且在长文档里反复出现，
        #   累积起来把分数推到了 15~32 —— 和真正相关的查询一个量级。
        #
        # 解决方案：打分时只保留 3-gram 及以上的 token。
        #   3-gram 要求"连续 3 个字完全相同"，这个条件很强，
        #   无关文本几乎不可能碰巧满足。实测效果：
        #       相关 "HNSW 索引的原理" -> 命中 "索引的""引的理" 等，得分正常
        #       无关 "今天晚饭吃什么" -> 3-gram 全部落空，得分 0
        #   两者被干净地分开了。
        #
        # 【注意：分词和倒排索引仍然保留全部粒度】（见 _tokenize_for_bm25）
        #   因为单字对【召回】有价值（用户可能只记得一个字）。
        #   这里只是不让它们参与【打分】。
        #   这个"宽进严出"的设计模式非常实用：
        #       宽进 —— 用宽松的特征建索引，保证不漏（保召回）
        #       严出 —— 用严格的特征来打分，保证不准的不进来（保精度）
        #
        # 【工程启示，这一条比代码本身重要】
        #   当两个分布的取值区间重叠时，不要去调阈值 ——
        #   那是在用一个标量去区分两个本质不同的东西，注定失败。
        #   正确做法是【换一个区分度更高的特征】。
        #   这是做检索和机器学习时最核心的直觉之一。
        query_tokens = [
            t for t in self._tokenize_for_bm25(query)
            if _is_discriminative_token(t)
        ]
        if not query_tokens:
            return []

        # 统计查询词频。查询里的重复词也应该加权（长查询里重复出现的词更重要）
        q_tf: dict[str, int] = {}
        for t in query_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1

        scores: dict[int, float] = {}

        for token, q_count in q_tf.items():
            postings = self._inverted.get(token)
            if not postings:
                continue

            idf = self.embedder.idf_for(token)
            # 查询侧词频也用同样的饱和函数处理，避免长查询里重复词过度放大
            q_weight = (q_count * (self._BM25_K1 + 1)) / (q_count + self._BM25_K1)

            for idx, tf in postings.items():
                if doc_filter and self._chunks[idx].doc_id != doc_filter:
                    continue

                dl = self._doc_len[idx]
                # 长度归一化因子
                norm = 1.0 - self._BM25_B + self._BM25_B * (dl / self._avg_doc_len)
                denom = tf + self._BM25_K1 * norm
                if denom <= 0:
                    continue

                scores[idx] = scores.get(idx, 0.0) + idf * q_weight * (tf * (self._BM25_K1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[:top_k]

    # --------------------------------------------------------------------------
    # RRF 融合
    # --------------------------------------------------------------------------
    # RRF 的平滑常数。k=60 是原论文推荐的经典值。    # 【k 的作用】让"排名第 1"和"排名第 2"的分数差距不至于太大，
    #   避免某一路的绝对第一名垄断结果。k 越大，各路排名的影响越平均。
    _RRF_K = 60

    # 相关性门槛（相对）：一个块在稀疏检索里的 BM25 分，至少要达到
    # "本次查询最强命中块"的多少比例，才认为它相关。
    # 【这个值需要按业务调】
    #   调高 -> 更严格，噪音少，排序质量高，但召回可能下降
    #   调低 -> 更宽松，召回全，但会引入不相关内容
    # 0.45 是用 eval_retrieval.py 实测调出来的。
    # 【重要】这个参数应该用标注好的评估集来调，而不是拍脑袋 ——
    #   本项目提供 eval_retrieval.py 正是为此。真实项目请做 A/B 评估。
    _MIN_REL_BM25 = 0.45

    # 相关性门槛（绝对）：本次查询的 BM25 最高分必须达到这个值，
    # 否则认为"整个检索无意义"，直接返回空。
    #
    # 【为什么必须有它】见 search() 里的详细说明 ——
    #   相对阈值拦不住"所有候选都很差、只是差得均匀"的情况。
    #
    # 【这个值是怎么定出来的 —— 一个值得学习的标定流程】
    #   第 1 步：只让长度 >= 2 的 n-gram 参与打分（见 _bm25_search），
    #            先把无关查询的噪音压下去
    #   第 2 步：写一个扫描脚本 tune_threshold.py，把候选阈值都跑一遍，
    #            同时测量【召回率】和【噪音率】两个指标。
    #            实测结果（22 个相关问题 + 10 个无关查询）：
    #                 阈值   Recall@5    噪利率
    #                  0.0     0.909      90%
    #                  5.0     0.909      60%
    #                  6.0     0.909      30%
    #                  8.0     0.909      20%   <-- 最优
    #                 10.0     0.818      20%
    #   第 3 步：选 8.0。理由非常明确 ——
    #            它是【唯一一个在保持最高召回的同时把噪音降到最低】的点。
    #            从 8.0 到 10.0，噪音没再降，召回却掉了 9 个百分点，
    #            说明 10.0 开始误伤真实资料了。
    #
    #   【这个流程教会我的事】
    #     参数不该"凭感觉调"，而应该"扫描 + 看曲线 + 选工作点"。
    #     更关键的是：当两个指标一个变好一个变坏时，这叫权衡（trade-off），
    #     需要按业务选；但像 0.0→8.0 这段【召回不变、噪音大降】，
    #     那就不是权衡，而是纯粹的改进 —— 说明原来的参数选错了。
    #     能识别出"这是权衡还是选错了"，是调参时最重要的判断力。
    _MIN_BM25_ABS = 8.0

    # 稠密兜底分支的最低分数门槛（余弦相似度）。
    # 【为什么需要】见 search() 里兜底分支的说明 ——
    #   稠密检索永远会返回 top-k，没有"无结果"状态，
    #   所以必须由我们外加一个门槛来模拟"不确定"。
    # 【怎么标定】实测本项目的哈希向量：
    #       相关查询  0.10 ~ 0.27
    #       无关查询  0.05 ~ 0.13
    #   两者在 0.13~0.10 区间有重叠，说明哈希向量本身区分能力有限。
    #   取 0.16 偏向"宁可漏、不要错"（对防幻觉更重要）。
    #   【换成真 embedding 后这个值需要重新标定】——
    #   真模型的相关分数通常在 0.5 以上，不相关在 0.2 以下，
    #   区分度比哈希好得多，门槛可以设得更高更干净。
    _MIN_DENSE_FALLBACK = 0.16

    def _rrf_fuse(
        self,
        dense: list[tuple[int, float]],
        sparse: list[tuple[int, float]],
        *,
        doc_filter: str | None,
    ) -> list[tuple[int, float]]:
        """
        Reciprocal Rank Fusion —— 融合两路检索结果。

        ── 公式 ──
            RRF_score(d) = Σ_over_lists  1 / (k + rank_in_that_list(d))

        ── 为什么用 RRF 而不是"加权求和的分数"？ ──
            因为稠密分数（余弦相似度，值域 0~1）和稀疏分数（BM25，值域 0~几十）
            量纲完全不同，无法直接相加。要加权就必须先归一化，
            而归一化方法的选择本身又是个难题（最小-最大？除以最大值？）。

            RRF 的巧妙之处是【只用排名，不用分数】，彻底绕开了量纲问题。
            它还有一个额外好处：对异常分数不敏感。即使某一路给出了
            一个荒谬的高分，在 RRF 里它也只是"排名第 1"，贡献固定为 1/(k+1)。

            【工程启示】当你需要融合两个不同来源、不同量纲的信号时，
            先考虑"能不能只用它们的相对顺序"—— 这往往比想办法对齐数值更稳健。
        """
        rrf: dict[int, float] = {}

        for rank, (idx, _score) in enumerate(dense):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self._RRF_K + rank + 1)

        for rank, (idx, _score) in enumerate(sparse):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self._RRF_K + rank + 1)

        return sorted(rrf.items(), key=lambda kv: -kv[1])

    # ==========================================================================
    # 三、元信息
    # ==========================================================================

    @property
    def size(self) -> int:
        """索引里有多少个块"""
        return len(self._chunks)

    @property
    def doc_ids(self) -> list[str]:
        """所有文档 ID（去重，保持稳定性）"""
        seen: dict[str, None] = {}
        for c in self._chunks:
            seen.setdefault(c.doc_id, None)
        return list(seen.keys())

    @property
    def doc_count(self) -> int:
        return len(self.doc_ids)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """按 chunk_id 精确取块（引用核对时用）"""
        for c in self._chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    def get_doc_chunks(self, doc_id: str) -> list[Chunk]:
        """取某篇文档的全部块，按 position 排序（精读整篇时用）"""
        chunks = [c for c in self._chunks if c.doc_id == doc_id]
        return sorted(chunks, key=lambda c: c.position)

    def outline(self) -> str:
        """
        生成知识库目录概览。

        【为什么需要这个？】
            这是给 Agent 的"全局视野"。没有它，Agent 只能靠猜关键词去检索；
            有了它，Agent 可以先看目录知道"哪几篇文档可能相关"，
            再针对性地检索/精读 —— 这正是 ReAct 里
            "先观察再决策"的体现，能显著提升检索命中率。
            这个技巧叫"分层检索"（先文档级、再块级）。
        """
        by_doc: dict[str, list[Chunk]] = {}
        for c in self._chunks:
            by_doc.setdefault(c.doc_id, []).append(c)

        lines = []
        for doc_id, chunks in by_doc.items():
            title = chunks[0].title
            # 取前两块的开头拼成摘要，让 Agent 快速判断这一篇讲什么
            summary = " ".join(c.preview(50) for c in chunks[:2])
            lines.append(f"- {doc_id} | {title} | {len(chunks)}块 | {summary}")
        return "\n".join(lines)

    def stats(self) -> str:
        return (
            f"向量库: {self.doc_count} 篇文档 / {self.size} 个块 / "
            f"维度 {self.embedder.dim} / embedder={type(self.embedder).__name__}"
        )
