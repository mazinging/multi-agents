"""
rag/embedder.py —— 文本向量化（RAG 链路第 2 步）
================================================================================
【这个文件在架构里的位置】
    把"文本"翻译成"向量"，让计算机能计算语义相似度。

【第一性原理：为什么向量能表示语义？】
    核心思想：语义相近的文本，在向量空间里的距离也相近。

    想象一个二维平面：
        x 轴 = "技术深度"，y 轴 = "成本高低"
        "便宜的方案"      -> (0.1, 0.9)
        "低成本的轻量级工具" -> (0.2, 0.85)    <- 离上一点很近 ✓
        "企业级重型架构"   -> (0.9, 0.2)      <- 离上面两点都很远 ✓

    真实 embedding 模型有几百到几千维，每一维都代表某种"潜在语义特征"，
    但这些维度是人类难以命名的（所以叫"隐空间"）。
    模型通过海量文本训练，学会了"把经常出现在相似语境里的词放到相近位置"。

【本实现的两种 Embedder】
    1. HashingEmbedder  —— 手写、离线、零依赖（默认）
    2. OpenAIEmbedder   —— 调真实 embedding API

【必须诚实说明 HashingEmbedder 的局限】
    它是"词袋 + 特征哈希"，本质只做了【词汇重叠度】匹配：
        "向量数据库" 和 "向量检索库"  -> 有共同字"向量"，能匹配上  ✓
        "数据库"     和 "DB"         -> 没有共同字，匹配不上       ✗
    也就是说，它抓不住"同义词"和"深层语义"。

    那为什么还要用它？
        a) 完全离线，不需要 API Key，任何人都能跑起来
        b) 结构和真 embedding 完全一致（同样的接口、同样的检索流程），
           理解了流程后换成真模型，代码只需改一行配置
        c) 对技术文档这种"术语重复率高"的语料，它的召回效果其实够用

    【学习建议】先跑通流程理解每一步，再切真 embedding 对比"语义召回"的差异，
    你会非常直观地体会到"为什么 RAG 的效果主要取决于 embedding 质量"。
================================================================================
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

import numpy as np

from config import CFG


# ==============================================================================
# 一、抽象接口
# ==============================================================================

class Embedder(ABC):
    """
    向量化器统一接口。

    约定：
        embed(texts) -> np.ndarray，形状 (len(texts), dim)
        - 返回的是【已归一化】的向量（L2 范数为 1）

    【为什么要约定"已归一化"？】
        归一化后，余弦相似度 = 向量点积。
        点积计算比完整的余弦公式（要先算两个模长）快得多，
        而且向量库可以一次性用矩阵乘法算完所有相似度。
        把归一化放在"入库时做一次"，而不是"每次查询时做 N 次"，
        这是很典型的"预处理换运行速度"优化。
    """

    dim: int

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """子类实现：把一批文本转成 (n, dim) 的归一化向量矩阵"""
        raise NotImplementedError

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        对外统一入口，负责形状校验。

        【为什么要有这层包装？】
            如果一个 batch 里混进了空字符串或 None，子类的实现可能返回
            畸形的数组（比如 0 行），后面矩阵运算就会报出很难懂的错误。
            在这里统一拦住，错误信息会清晰得多。
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        cleaned = [t if isinstance(t, str) else str(t) for t in texts]
        vectors = self._embed_batch(cleaned)

        # 形状断言：把"静默的数据错位"变成"立即的明确报错"
        if vectors.shape != (len(cleaned), self.dim):
            raise ValueError(
                f"Embedder 返回形状异常: 期望 {(len(cleaned), self.dim)}，"
                f"实际 {vectors.shape}"
            )
        return vectors

    def embed_one(self, text: str) -> np.ndarray:
        """便捷方法：单个文本 -> 单个向量 (dim,)"""
        return self.embed([text])[0]


# ==============================================================================
# 二、HashingEmbedder —— 手写离线实现
# ==============================================================================
#
# 【给这份实现的诚实评价（实测后补充）】
#
#   我最初只用了 3-gram，实测检索效果很差：相关文档的相似度只有 0.19 左右，
#   而且经常把不相关的块排到前面。原因是 3-gram 对中文匹配【过于严格】。
#
#   中文表达同一件事时措辞变化很大：文档写"成本测算"，用户问"贵不贵"，
#   两者的 3-gram 重叠为零，完全匹配不上；甚至连"切块大小怎么定"这种
#   和文档标题几乎同义的问题，都排不到第一名。
#
#   改进方案是引入【多粒度特征】（见下面 _tokenize 的实现）：
#       单字    -> 负责"召回"，宽松，只要有一个字搭上就有基础分
#       2-gram  -> 中等区分度
#       3-gram  -> 负责"区分"，严格，词序一致才得高分
#   并给不同粒度不同的权重。这是中文检索的常用技巧，
#   短 n-gram 保召回、长 n-gram 保精度，效果比单一粒度好得多。
#
#   【但必须清楚：这仍然是"词汇匹配"，不是"语义匹配"】
#   它永远不会知道"向量数据库"和"Vector DB"是同一个东西，
#   也不知道"贵不贵"和"成本"是同一个意思 —— 它只是恰好共享了"成""本"两个字。
#   要真正的语义能力，必须换真的 embedding 模型。
#   这个局限正是本文件想教给你的最重要一课。
# ==============================================================================

# 【中文分词策略】不引入 jieba 等分词库（为了零依赖离线），
# 改用【多粒度字符 n-gram】。下面这个元组定义了要用哪几种粒度。
#
# 为什么需要多粒度？举个真实例子：
#     文档："向量数据库成本测算"      用户查询："向量库贵不贵"
#     只用 3-gram：前者有"向量数/量数据/数据库/据库成/库成本/成本测/本测算"
#                  后者有"向量库/量库贵/库贵不/贵不贵"
#                  —— 重叠为 0，完全检索不到（这就是我实测遇到的问题）
#     加上单字后：两者都含"向""量""库""成""本" —— 至少有基础分
#
# 权衡：单字粒度会引入噪音（"量"字在"测量""质量""向量"里都出现），
# 所以【不能只用单字】。多粒度组合起来，短 n-gram 保召回、长 n-gram 保精度，
# 是实践中很有效的折中方案。
_NGRAM_SIZES = (1, 2, 3)

# 英文/数字单词的正则
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]*|\d+")

# 中文（含扩展区）字符正则 —— 用于把连续中文片段抠出来
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")

# 中文常见虚词/标点，作为极简停用字过滤。
# 【为什么要过滤？】"的""了""是"这类字在每个块里都出现，
# 它们会给所有文档都加上相同的分数，相当于给所有人都加 1 分 ——
# 不改变排序，却抬高了"不相关文档"的绝对分数，让阈值过滤失效。
# 生产环境应该用完整的停用词表；这里只放最高频的几个，够用即可。
_STOP_CHARS = set("的了是在和与及或而也就都很更最一二三四五六七八九十这那有个我你他它们之其所以因为但是")

# 单个 token 的权重：越长的 n-gram 越有区分度，权重越高。
# 【这个权重设计是让检索质量变好的关键】
#   单字权重 1.0：只用于兜底召回
#   2-gram 权重 1.5：有一定区分度
#   3-gram 权重 2.0：区分度最高，词序完全一致才得分
# 这样"词序和用词都接近"的文档会显著高于"只有零散单字重合"的文档。
_NGRAM_WEIGHT = {1: 1.0, 2: 1.5, 3: 2.0}


class HashingEmbedder(Embedder):
    """
    基于【特征哈希 + 多粒度 n-gram + IDF 加权】的离线向量化器。

    ── 完整流程 ──
        1. 抽取特征（token），带粒度权重：
             英文/数字 -> 整词（权重 2.0）
             中文     -> 1/2/3-gram（权重 1.0 / 1.5 / 2.0）
        2. 每个 token 通过哈希映射到 [0, dim) 里的某一维，
           按 (符号 × 粒度权重 × IDF权重) 累加到该维
        3. 所有 token 处理完后，对向量做 L2 归一化

    ── 为什么第 2 步要有 +1/-1 的随机符号？ ──
        这叫"符号哈希技巧"（signed hashing）。
        如果所有 token 都只做 +1，那么两个完全不同的文本也可能因为
        哈希冲突（不同 token 落到同一维）而累积出正相关，
        产生虚假相似度。
        给每个 token 一个确定性的 ±1，冲突时正负可能相互抵消，
        期望上让"无关文本的相似度 ≈ 0"。
        【这是哈希技巧能逼近真实词表的关键，很值得记住】

    ── 为什么用哈希而不是真实的词表？ ──
        真实词表需要先扫一遍全语料建立 vocabulary，然后保存成文件。
        哈希方案不需要构建词表、不需要保存、支持无限新词
        （新词直接哈希到某一维就行，只是可能有冲突）。
        代价就是冲突带来的精度损失。

    ── 哈希函数的选择 ──
        用 hashlib.md5 而不是 Python 内置的 hash()：
        内置 hash() 对字符串有随机盐（PYTHONHASHSEED），
        同一个词在不同进程里哈希值不同 —— 这会导致你存的索引
        在下次运行时无法复用（查询向量和文档向量对不上）。
        md5 是确定性的，跨进程、跨机器都一致。
        【这是手写向量化时最容易踩的坑之一】
    """

    def __init__(self, dim: int | None = None, *, use_idf: bool = True) -> None:
        self.dim = dim or CFG.embedding_dim

        # 是否启用 IDF 加权（见 fit_idf 的说明）。
        # 默认开启，因为实测它对检索质量提升非常明显。
        self.use_idf = use_idf

        # token -> IDF 权重。初始为空，需要调用 fit_idf 后才生效。
        # 【这里有个重要的工程权衡】IDF 必须"见过语料"才能算出来，
        # 所以 embedder 变成了有状态的 —— 必须先 fit 再 embed。
        # 如果没 fit 就直接用，会退化成"所有 token 权重相同"，
        # 效果变差但不会报错（优雅降级）。
        self._idf: dict[str, float] = {}
        self._default_idf = 1.0

    # --------------------------------------------------------------------------
    # IDF 拟合：本文件对检索质量影响最大的一个设计
    # --------------------------------------------------------------------------
    def fit_idf(self, corpus: list[str]) -> None:
        """
        在整个语料上统计 IDF 权重。

        ── 什么是 IDF，为什么它能大幅改善检索质量？ ──

        IDF = Inverse Document Frequency（逆文档频率）。
        核心思想：一个词如果在所有文档里都出现，它就没有区分能力。

        具体到本项目的实测问题：
            "的""在""是"这些字，以及"文档""使用""可以"这类高频词，
            几乎每个块里都有。它们会给所有块都加上相同的分数 ——
            相当于给所有候选人加同样的分，不改变排序，
            却把"完全不相关文档"的绝对分数抬高到 0.1 以上。
            这导致：
                a) 真正的相关文档（0.19）和噪音文档（0.11）差距被压缩
                b) 无法用阈值过滤掉噪音（设 0.15 会把相关的也滤掉）

            加上 IDF 后，这些高频词的权重被压到接近 0，
            而"成本""切块""HNSW"这类有区分度的词权重被放大。
            相关文档的分数会明显上升，噪音文档的分数会明显下降，
            两者被拉开 —— 检索质量就上来了。

        ── 计算公式 ──
            IDF(t) = ln((N + 1) / (df(t) + 1)) + 1

            其中 N = 文档总数，df(t) = 包含 t 的文档数。
            这个变体（平滑 IDF）保证了 IDF 恒为正数，
            避免"某个词在所有文档都出现时 IDF = 0"导致该特征被完全抹掉。
            加 1 是标准的平滑手段。

        ── 为什么用 ln 而不是直接 1/df？ ──
            因为要压缩动态范围。df 从 1 变到 2（权重减半）的影响，
            不应该和 df 从 1000 变到 2000 一样大。
            对数让"稀有词"的优势平滑增长，而不是极端放大。

        ── 这是"训练"吗？ ──
            严格说这是"统计拟合"，不涉及梯度下降。
            但它和机器学习里的"在训练集上估计参数"本质相同：
            都是先从数据里学出统计规律，再用这个规律处理新数据。
            【重要】IDF 统计的是语料，不是查询。查询进来时才用已拟合的 IDF。

        ── 局限 ──
            这是一个纯统计方法，不知道怎么处理"同义词"。
            而且 IDF 表需要和索引一起持久化 ——
            换了语料必须重新 fit，否则新词的 IDF 会缺失，
            只能用默认值。本实现里默认值取语料平均 IDF，是一种折中。
        """
        if not corpus:
            return

        n_docs = len(corpus)

        # 统计每个 token 出现在多少个【文档】里（不是出现多少次）
        # 【为什么要用 set？】因为 IDF 关心的是"出现在几篇文档里"，
        # 同一篇文档里出现 100 次和 1 次，对 df 的贡献都是 1。
        # 这是 IDF 和 TF 的核心区别。
        df: dict[str, int] = {}
        for text in corpus:
            unique_tokens = {tok for tok, _ in self._tokenize(text)}
            for tok in unique_tokens:
                df[tok] = df.get(tok, 0) + 1

        # 计算 IDF
        self._idf = {
            tok: math.log((n_docs + 1) / (d + 1)) + 1.0
            for tok, d in df.items()
        }

        # 默认 IDF：语料里没出现过的新 token 用什么权重。
        # 取"只出现在 1 篇文档里的词"的 IDF，即最稀有的情况 ——
        # 因为新词大概率是罕见的专有名词，给高权重比给平均权重更合理。
        self._default_idf = math.log((n_docs + 1) / 2) + 1.0

    def _idf_of(self, token: str) -> float:
        """查一个 token 的 IDF；没拟合过或没见过则用默认值"""
        if not self.use_idf or not self._idf:
            return 1.0        # 未启用 or 未拟合 -> 退化为等权重
        return self._idf.get(token, self._default_idf)

    # --------------------------------------------------------------------------
    def _tokenize(self, text: str) -> list[tuple[str, float]]:
        """
        抽取特征 token，返回 [(token, 权重), ...]。

        ── 三路特征 ──
            1. 英文/数字整词  —— 权重 2.0（和 3-gram 同级，因为整词信息量高）
            2. 中文 n-gram   —— n ∈ {1,2,3}，权重随 n 递增
            3. （可选扩展）你可以在这里加同义词表展开，比如把"贵"映射成"成本"，
               这是不换 embedding 模型也能提升召回的手段，值得自己动手试试

        【为什么要 lower()】
            "Vector" 和 "vector" 应视为同一特征，否则词汇重叠度被低估。

        【为什么返回 (token, 权重) 而不是纯 token 列表】
            因为不同粒度的 token 区分能力不同，必须让它们在向量里占不同分量。
            如果一视同仁，单字（数量多、噪音大）会淹没 3-gram 的信号，
            检索质量反而会下降 —— 这是我在实测中确认过的。
        """
        text = text.lower()
        tokens: list[tuple[str, float]] = []

        # ---- 第 1 路：英文/数字整词 ----
        # 加一个前缀 "w:" 是为了避免英文词和中文 n-gram 哈希到同一维度时
        # 产生语义串扰（虽然哈希本来就会冲突，但至少让"w:abc"和"abc 的中文片段"
        # 走完全不同的哈希输入，减少系统性偏差）
        for w in _WORD_RE.findall(text):
            tokens.append((f"w:{w}", 2.0))

        # ---- 第 2 路：中文多粒度 n-gram ----
        for run in _CJK_RUN_RE.findall(text):
            for n in _NGRAM_SIZES:
                if len(run) < n:
                    continue
                weight = _NGRAM_WEIGHT.get(n, 1.0)
                for i in range(len(run) - n + 1):
                    gram = run[i:i + n]

                    # 单字粒度时过滤停用字。
                    # 【为什么只过滤单字，不过滤 2/3-gram？】
                    #   因为"的"单独出现没有信息量，但"目的""的确"里的"的"
                    #   是有信息量的（属于更大词组的一部分）。
                    #   只在单字层面过滤，既去掉了噪音又不会误伤。
                    if n == 1 and gram in _STOP_CHARS:
                        continue

                    # 用 "c:" 前缀标记中文片段，理由同上面的 "w:"
                    tokens.append((f"c:{gram}", weight))

        return tokens

    # --------------------------------------------------------------------------
    # 稀疏检索（BM25）—— 本文件的第二个关键能力
    # --------------------------------------------------------------------------
    def token_keys(self, text: str) -> set[str]:
        """
        返回文本去重后的 token 集合，供稀疏检索使用。

        【为什么要把这个方法暴露出来（而不是只内部用）？】
            因为稀疏检索（BM25）需要"倒排索引"：token -> 包含它的文档列表。
            倒排索引的构建由 store.py 负责，它需要拿到 token 集合。
            把分词逻辑留在 embedder 里（而不是让 store 自己分词），
            保证稠密和稀疏两路用的是【完全相同的分词规则】——
            否则两路结果的尺度无法对齐。这是一个容易出错的地方。
        """
        return {tok for tok, _ in self._tokenize(text)}

    def idf_for(self, token: str) -> float:
        """对外暴露 IDF 查询（BM25 需要用它算词权重）"""
        return self._idf_of(token)

    @staticmethod
    def _hash_token(token: str, dim: int) -> tuple[int, float]:
        """
        把 token 映射到 (维度索引, 符号)。

        用 md5 前 8 个字节：
            前 4 字节 -> 维度索引
            后 4 字节的最低位 -> 符号（正/负）

        为什么用两个不同的哈希派生值而不是同一个？
            如果用同一个哈希的低位同时决定位置和符号，会引入相关性偏差。
            用哈希的不同部分能保证两者近似独立。
        """
        h = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if (h[4] & 1) == 0 else -1.0
        return idx, sign

    # --------------------------------------------------------------------------
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = self._tokenize(text)
            if not tokens:
                # 空文本或纯符号文本：留一个零向量。
                # 【注意】零向量和任何向量的余弦相似度都是 0（因为分母为 0），
                # store.py 里要专门处理这个边界情况。
                continue

            # 把 token 的权重累加到对应维度上。
            # 最终权重 = 粒度权重(n-gram长度) × IDF权重(区分度)
            # 这两个权重是【正交】的，各管一件事：
            #     粒度权重：这个词本身有多具体（3-gram 比单字具体）
            #     IDF权重 ：这个词在语料里有多稀有（稀有 = 有区分度）
            # 两者相乘是标准做法，效果优于只用其中一个。
            #
            # 注意这里用 += 而不是 = ：不同 token 哈希到同一维时要累加，
            # 这正是"词袋模型"的本质 —— 一个维度代表"某类特征出现了多少次"。
            for tok, weight in tokens:
                idx, sign = self._hash_token(tok, self.dim)
                out[row, idx] += sign * weight * self._idf_of(tok)

            # ---- L2 归一化 ----
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
            # norm == 0 说明正负符号恰好完全抵消（极罕见）。
            # 此时保留零向量，交给检索层处理。

        return out


# ==============================================================================
# 三、OpenAIEmbedder —— 真实 embedding API
# ==============================================================================

class OpenAIEmbedder(Embedder):
    """
    调 OpenAI 兼容的 /v1/embeddings 接口。

    接口形状和 HashingEmbedder 完全一致，所以上层代码（store、tools）
    一行都不用改 —— 这就是面向接口编程的价值。

    【批处理的重要工程考虑】
        API 通常限制单次请求的文本条数（比如 2048 条）和总 token 数。
        所以要分批发送。这里的 batch_size 是保守值。
    """

    def __init__(self, dim: int | None = None, batch_size: int = 64) -> None:
        self.dim = dim or CFG.embedding_dim
        self.batch_size = batch_size
        if not CFG.llm_api_key:
            raise ValueError("embedder_mode='openai' 但未设置 LLM_API_KEY")

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        import json
        import urllib.request

        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            payload = {"model": CFG.embedding_model, "input": batch}
            req = urllib.request.Request(
                url=f"{CFG.llm_base_url.rstrip('/')}/embeddings",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {CFG.llm_api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=CFG.llm_timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            # 按 index 排序，因为 API 不保证返回顺序
            items = sorted(body["data"], key=lambda d: d["index"])
            all_vectors.extend(item["embedding"] for item in items)

        arr = np.asarray(all_vectors, dtype=np.float32)

        # 真实 API 的实际维度可能与配置不符（比如配置写的 512，实际返回 1536）。
        # 这里以实际返回为准并更新 self.dim，避免下游形状断言失败。
        if arr.size:
            self.dim = arr.shape[1]

        # 同样做归一化，保证下游逻辑统一
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


# ==============================================================================
# 四、工厂函数
# ==============================================================================

def build_embedder(mode: str | None = None) -> Embedder:
    """根据配置返回 embedder 实例"""
    mode = mode or CFG.embedder_mode
    if mode == "hashing":
        return HashingEmbedder()
    if mode == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"未知的 embedder_mode: {mode}（可选 'hashing' / 'openai'）")


# ==============================================================================
# 五、相似度计算（独立成函数，方便单测和复用）
# ==============================================================================

def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    计算一个查询向量与一批向量的余弦相似度。

    【为什么可以简化成矩阵乘法？】
        余弦相似度 = (a·b) / (|a| * |b|)
        因为入库时已经把 matrix 里每一行归一化成 |b| = 1，
        所以分母里的 |b| 全是 1，只剩：
            余弦相似度 = a·b / |a|
        再对 query 也归一化，就变成了纯粹的 a·b —— 一次矩阵乘法搞定。

        如果每次查询都要现算所有文档向量的模长，那是 N 倍的无谓开销。
        这就是"入库时归一化"这个设计的回报。

    返回形状 (N,) 的相似度数组，值域约 [-1, 1]（HashingEmbedder 下通常是 [0, 1]）。
    """
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)

    q = np.asarray(query, dtype=np.float32).reshape(-1)
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0:
        # 查询向量是零向量（比如查询词全是停用词/未知符号）：
        # 无法定义相似度，返回全 0，让上层自然得到"没有结果"
        return np.zeros((matrix.shape[0],), dtype=np.float32)

    return matrix @ (q / q_norm)
