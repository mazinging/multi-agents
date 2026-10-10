"""
tools/rag_tools.py —— 基于 RAG 的具体工具实现
================================================================================
【这个文件在架构里的位置】
    把 rag/ 里的检索能力，包装成 Agent 可以调用的"工具"。

【为什么需要这一层包装？不直接让 Agent 调 store.search()？】
    三个理由：
        1. Agent 是通过文本 Action 来调工具的，它只会说
           "Action: search_local_docs / Action Input: 向量数据库"，
           程序需要有人把这个文本翻译成函数调用 —— 就是这一层
        2. 工具层的返回值是【为模型优化的呈现】，而不是原始对象。
           store.search() 返回 SearchHit 对象列表，
           工具层把它格式化成带编号、带来源的可读文本
        3. 工具层承担参数校验和错误处理，让 Agent 能自我纠正

【本项目的五个工具，构成一个完整的能力阶梯】
    list_knowledge_base  : 看目录     —— 建立全局视野（不知道有什么，就先看目录）
    search_local_docs    : 粗检索     —— 找到候选文档
    read_doc_section     : 精读       —— 读具体内容（对应 ReAct 的"深入观察"）
    extract_key_points   : 提炼       —— 从长文本里抽要点
    cross_compare        : 对比       —— 跨文档做横向对比（分析型任务的核心）

    【这个阶梯的设计思路】从"广"到"深"再到"综合"。
    它对应了人类做调研的自然过程：先扫一眼目录，找到几篇相关的，
    再逐篇细看，然后提炼要点，最后横向对比得出结论。
    【为 Agent 设计工具时，模仿人类专家的实际工作流程是个很好的方法。】
================================================================================
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from rag.store import VectorStore


# ==============================================================================
# 工具 1：查看知识库目录
# ==============================================================================

class ListKnowledgeBaseTool(BaseTool):
    """
    让 Agent 先看到"有什么"，再决定"查什么"。

    【为什么这个工具很重要？】
        没有它，Agent 只能盲猜关键词去检索。而有了目录，
        Agent 可以先判断"哪几篇文档可能相关"，然后针对性地检索或精读。
        这个策略叫"分层检索"（先文档级、再块级），
        实测能显著提升命中率，尤其是当用户的措辞和文档标题不一致时。

    【类比】就像你去图书馆查资料，先看分类目录比直接冲进书库乱翻高效得多。
    """

    name = "list_knowledge_base"
    description = (
        "列出知识库中所有文档的目录（文档ID、标题、内容摘要）。"
        "当你不知道知识库里有哪些资料、或需要决定该查哪些文档时，"
        "应该先调用这个工具。参数随便填一个占位符即可。"
    )
    parameters = "无需参数，填任意字符（如 'all'）"

    def __init__(self, store: "VectorStore") -> None:
        self.store = store

    def run(self, arg: str) -> ToolResult:
        outline = self.store.outline()
        content = (
            f"知识库共 {self.store.doc_count} 篇文档、{self.store.size} 个内容块。\n"
            f"目录如下（格式：文档ID | 标题 | 块数 | 摘要）：\n\n{outline}"
        )
        return ToolResult.success(
            content,
            data={"doc_ids": self.store.doc_ids},
            doc_count=self.store.doc_count,
        )


# ==============================================================================
# 工具 2：检索文档块
# ==============================================================================

class SearchDocsTool(BaseTool):
    """
    RAG 的核心检索工具。

    【返回值设计的关键点：给每个结果编号 [资料1]、[资料2]】
        这不是为了好看。编号是【引用溯源的基础】：
        Agent 后续在报告里写"根据 [资料1]"，
        我们就能程序化地反查 [资料1] 对应哪个 chunk_id，
        从而校验这个引用是否真实存在、内容是否支持结论。

        没有编号，模型只能含糊地说"根据资料显示"，
        你就无法核对它到底有没有依据 —— 幻觉就无从发现。

        【这就是 RAG 防幻觉的核心机制】不是靠请求模型"别说谎"，
        而是靠提供可验证的引用锚点。
    """

    name = "search_local_docs"
    description = (
        "在本地知识库中检索与查询最相关的内容块（混合检索：语义向量 + 关键词）。"
        "返回结果带编号和来源ID，可以在后续回答中引用。"
        "如果返回结果为 0 条，说明知识库没有覆盖该话题，"
        "此时应该换关键词重试，或直接说明资料不足。"
    )
    parameters = "检索关键词或问题描述（自然语言即可）"

    def __init__(self, store: "VectorStore", top_k: int | None = None) -> None:
        self.store = store
        self.top_k = top_k

    def run(self, arg: str) -> ToolResult:
        query = arg.strip()

        # ---- 参数校验要给出可操作的提示 ----
        if not query or query.lower() in ("none", "无", "null"):
            return ToolResult.failure(
                "search_local_docs 需要一个检索关键词，但你传的是空值。"
                "请填写你想查询的具体内容，例如「向量数据库选型」。"
            )

        hits = self.store.search(query, top_k=self.top_k)

        # ---- 检索为空时的返回，是最需要精心设计的地方 ----
        # 【为什么？】因为这是"知识库没覆盖"的信号，
        # 而 Agent 接下来怎么反应，直接决定了会不会产生幻觉。
        # 所以这里必须【明确告诉 Agent 该怎么做】，
        # 而不是只丢一个"没找到"就完事。
        if not hits:
            return ToolResult.failure(
                f"未检索到与「{query}」相关的内容。这表示知识库可能不覆盖此话题。\n"
                f"你可以：\n"
                f"  1. 换用更具体或更常见的技术术语重试一次\n"
                f"  2. 调用 list_knowledge_base 查看知识库实际覆盖了哪些主题\n"
                f"  3. 如果确认知识库没有相关内容，请【如实说明资料不足】，"
                f"不要凭常识推测作答。",
                query=query,
            )

        blocks = [h.to_prompt_block(i + 1) for i, h in enumerate(hits)]
        content = (
            f"检索「{query}」得到 {len(hits)} 条结果：\n\n"
            + "\n\n---\n\n".join(blocks)
        )

        return ToolResult.success(
            content,
            # data 里放结构化结果，供程序做引用校验（不进 prompt）
            data=[h.to_dict() for h in hits],
            query=query,
            hit_count=len(hits),
            top_score=round(hits[0].score, 4),
        )


# ==============================================================================
# 工具 3：精读指定文档
# ==============================================================================

class ReadDocSectionTool(BaseTool):
    """
    读取指定文档的完整内容（或其中一段）。

    【为什么有了检索还需要精读？】
        检索是把文档【切碎】后按相似度取几块，好处是精准，
        坏处是丢失了上下文。一个块可能写着"成本可以降低 40%"，
        但"相对于什么降低"在上一块里 —— 单看这一块会误解。

        精读则把整篇文档按顺序读完，保留了完整的论证链条。
        ReAct 的价值正体现在这里：模型可以先粗检索定位到文档，
        再决定精读哪一篇来获取完整信息。这是"根据观察调整策略"。

    【参数格式】支持两种写法：
        - "doc_03"           -> 读整篇
        - "doc_03#2"         -> 只读第 3 块
    这种"一个参数两种语义"的设计是为了在"参数少"和"功能全"之间折中。
    """

    name = "read_doc_section"
    description = (
        "精读指定文档的完整内容，或其中某一个内容块。"
        "当你已经从检索结果中确定了相关文档、需要获取完整上下文和准确细节时使用。"
        "返回结果同样带来源ID，可被引用。"
    )
    parameters = "文档ID（如 doc_03）读取整篇；或 块ID（如 doc_03#2）读取单块"

    def __init__(self, store: "VectorStore") -> None:
        self.store = store

    def run(self, arg: str) -> ToolResult:
        ref = arg.strip().strip("\"'")

        if not ref:
            return ToolResult.failure(
                "read_doc_section 需要一个文档ID或块ID。"
                "请先用 search_local_docs 查出 doc_id，再传进来。"
            )

        # 容错：模型有时会传 "doc_id=doc_03" 或 "文档 doc_03" 这种带装饰的形式
        ref = re.sub(r"^(?:doc_id|文档|id)\s*[=:：]?\s*", "", ref, flags=re.IGNORECASE)
        ref = ref.split()[0] if ref.split() else ref

        # ---- 情况 A：指定了具体块（形如 doc_03#2）----
        if "#" in ref:
            chunk = self.store.get_chunk(ref)
            if chunk is None:
                return self._not_found(ref)
            content = (
                f"文档 {chunk.doc_id} 的第 {chunk.position + 1} 块（{chunk.chunk_id}）:\n"
                f"标题: {chunk.title}\n\n{chunk.text}"
            )
            return ToolResult.success(
                content,
                data=chunk.to_dict(),
                chunk_id=chunk.chunk_id,
            )

        # ---- 情况 B：指定了整篇文档 ----
        chunks = self.store.get_doc_chunks(ref)
        if not chunks:
            return self._not_found(ref)

        parts = [f"# {chunks[0].title}\n（文档ID: {ref}，共 {len(chunks)} 块）\n"]
        for c in chunks:
            parts.append(f"--- 第 {c.position + 1} 块 [{c.chunk_id}] ---\n{c.text}")

        content = "\n\n".join(parts)

        # 【重要的工程细节：超长内容要截断】
        # 一篇完整文档可能有上万字，全塞进 prompt 会：
        #   1. 超出上下文窗口
        #   2. 成本剧增
        #   3. 稀释注意力，模型反而抓不住重点
        # 但截断时必须【明确告知模型被截断了】，
        # 否则它会以为自己看到的就是全部，可能做出错误判断。
        MAX_LEN = 4000
        truncated = False
        if len(content) > MAX_LEN:
            content = content[:MAX_LEN] + (
                f"\n\n…（此文档共 {len(content)} 字符，以上为前 {MAX_LEN} 字符，"
                f"内容已截断。如需后续内容，请用 {ref}#N 的形式读取指定块）"
            )
            truncated = True

        return ToolResult.success(
            content,
            data=[c.to_dict() for c in chunks],
            doc_id=ref,
            chunk_count=len(chunks),
            truncated=truncated,
        )

    def _not_found(self, ref: str) -> ToolResult:
        """
        找不到文档时的错误返回。

        【这是"错误信息引导自我纠正"的典型例子】
            直接返回"未找到"没有用，模型不知道下一步干什么。
            返回可用文档列表，模型下一轮就能选对。
            这类错误信息的设计，能显著降低 Agent 的无效循环次数。
        """
        available = ", ".join(self.store.doc_ids)
        return ToolResult.failure(
            f"未找到文档或块「{ref}」。\n"
            f"知识库中可用的文档ID有：{available}\n"
            f"请从上面这些ID中重新选择，或先用 search_local_docs 检索确认。",
            requested=ref,
        )


# ==============================================================================
# 工具 4：提炼要点
# ==============================================================================

class ExtractKeyPointsTool(BaseTool):
    """
    从检索结果中提炼要点。

    【为什么用"程序化提炼"而不是"让 LLM 提炼"？】
        这是一个刻意的设计选择，值得说明：
            用 LLM 提炼：效果好，但增加一次调用（成本+延迟），
                        而且可能引入幻觉（LLM 会"补充"原文没有的内容）
            用规则提炼：快、免费、绝对忠实于原文（不会幻觉）

        对于"提炼要点"这个任务，规则方法的关键在于
        【如何判断一句话是"要点"】。本实现用两个信号：
            1. 句子长度适中（太短是标题碎片，太长是叙述而非要点）
            2. 包含结论性/数值性标记词（"建议""应该""必须""可以"，
               或包含数字 —— 技术文档里的数字往往是关键数据）

        这个启发式不完美，但它有一个巨大优势：
        【它绝不会编造】。在防幻觉优先的场景下，这个权衡是值得的。

        【更重要的学习点】不是所有环节都该用 LLM。
        能用确定性规则做好的事，就别用不确定的模型。
        判断标准是：这个环节需要"理解"还是只需要"筛选"？
    """

    name = "extract_key_points"
    description = (
        "从一段文本（或指定文档）中提炼关键要点，返回按重要性筛选出的句子列表。"
        "当你需要快速把握一篇长文档的核心结论、或在写报告前整理素材时使用。"
    )
    parameters = "文档ID（如 doc_03）或一段直接的文本"

    # 结论性标记词。出现在句子里，说明这句在表达观点/建议/规范。
    _SIGNAL_WORDS = (
        "建议", "应该", "必须", "需要", "可以", "推荐", "结论",
        "关键", "核心", "重要", "注意", "原则", "最好", "通常",
        "优于", "适合", "不适合", "避免", "原因", "因为", "所以",
    )

    def __init__(self, store: "VectorStore") -> None:
        self.store = store

    def run(self, arg: str) -> ToolResult:
        raw = arg.strip()
        if not raw:
            return ToolResult.failure(
                "extract_key_points 需要文档ID或文本内容。请传入要提炼的对象。"
            )

        # 判断参数是"文档ID"还是"直接文本"
        source = ""
        ref = ""
        first_word = raw.split()[0] if raw.split() else ""
        bare = re.sub(r"#\d+$", "", first_word)
        if bare in self.store.doc_ids:
            chunks = self.store.get_doc_chunks(bare)
            source = "\n".join(c.text for c in chunks)
            ref = bare
        else:
            source = raw

        # ---- 按句子切分 ----
        sentences = re.split(r"(?<=[。！？；!?;])\s*|\n+", source)
        sentences = [s.strip() for s in sentences if s and len(s.strip()) > 10]

        if not sentences:
            return ToolResult.failure(
                f"从「{raw[:30]}…」中未提取到有效句子，内容可能太短或格式特殊。"
            )

        # ---- 打分：有标记词 + 有数字的句子得分更高 ----
        scored: list[tuple[float, str]] = []
        for s in sentences:
            score = 0.0
            score += sum(1.2 for w in self._SIGNAL_WORDS if w in s)
            # 数字往往是技术文档里的关键数据（成本、维度、阈值、百分比）
            score += 0.8 * len(re.findall(r"\d+(?:\.\d+)?\s*(?:%|GB|MB|维|倍|毫秒|条|个)", s))
            score += 0.5 * len(re.findall(r"\d+", s))
            # 长度适中的句子优先（过短信息量少，过长不是"要点"）
            if 25 <= len(s) <= 120:
                score += 1.0
            scored.append((score, s))

        scored.sort(key=lambda kv: -kv[0])
        top = scored[:6]

        lines = [f"{i+1}. {s}" for i, (_sc, s) in enumerate(top)]
        content = (
            f"从 {'文档 ' + ref if ref else '提供的内容'} 中提炼出 {len(top)} 条要点"
            f"（按重要性排序）：\n\n" + "\n".join(lines)
        )

        return ToolResult.success(
            content,
            data=[{"point": s, "score": round(sc, 2)} for sc, s in top],
            source=ref or "(直接文本)",
            point_count=len(top),
        )


# ==============================================================================
# 工具 5：跨文档对比
# ==============================================================================

class CrossCompareTool(BaseTool):
    """
    对多个文档/方案做横向对比。

    【这个工具是"分析型任务"的核心】
        Researcher 负责"把资料找全"，而横向对比是 Analyst 的工作。
        两者需要不同的工具，这正是"角色专精"的体现 ——
        不是同一个 Agent 换个人设 prompt，而是给它不同的能力。

    【参数格式：逗号分隔的多个检索词】
        例如: "Pgvector, Milvus, Qdrant"
        工具会为每个词分别检索，然后并列展示结果。

        【为什么并列展示就能支持"对比"？】
            因为对比的本质是"让差异可被观察"。
            把三个方案的资料并排放在一起，模型（或人）
            一眼就能看出哪个文档提到成本、哪个提到运维。
            这个"结构化呈现"本身就是对比能力的基础。
            如果只是把三份资料顺序堆叠，很难看出对应关系。
    """

    name = "cross_compare"
    description = (
        "对多个对象（方案、产品、技术）做横向对比。"
        "输入用逗号分隔的多个查询词，工具会分别检索并并列展示结果，"
        "便于发现各对象在不同维度上的差异。"
        "当你需要比较几个方案的优劣时使用。"
    )
    parameters = "逗号分隔的多个对比对象，例如: Pgvector, Milvus, Qdrant"

    def __init__(self, store: "VectorStore", per_item_k: int = 2) -> None:
        self.store = store
        self.per_item_k = per_item_k

    def run(self, arg: str) -> ToolResult:
        raw = arg.strip()
        if not raw:
            return ToolResult.failure(
                "cross_compare 需要至少一个对比对象。"
                "请用逗号分隔，例如「Pgvector, Milvus」。"
            )

        # 支持中英文逗号、顿号分隔 —— 模型可能用任何一种
        items = [x.strip() for x in re.split(r"[,，、;；]", raw) if x.strip()]

        if len(items) < 2:
            return ToolResult.failure(
                f"cross_compare 需要至少 2 个对象才能对比，但你只提供了 1 个：「{items[0]}」。"
                f"请用逗号分隔多个对象，例如「{items[0]}, Milvus」。"
            )

        # 限制数量上限，防止模型传进来 20 个对象导致：
        #   1. 检索次数爆炸（每个对象一次检索）
        #   2. 返回内容过长，撑爆 prompt
        MAX_ITEMS = 5
        truncated_items = False
        if len(items) > MAX_ITEMS:
            items = items[:MAX_ITEMS]
            truncated_items = True

        sections = []
        comparison_data = {}
        total_hits = 0

        for item in items:
            hits = self.store.search(item, top_k=self.per_item_k)
            comparison_data[item] = [h.to_dict() for h in hits]
            total_hits += len(hits)

            if not hits:
                # 【注意这里不返回 failure】
                # 因为"某一个对象没查到"不该让整个对比失败 ——
                # 其他对象的信息仍然有价值。这是"部分失败"的处理方式：
                # 降级而不是中止，并把缺失情况明确告知模型。
                sections.append(f"【{item}】\n（知识库中未找到关于此项的资料）")
                continue

            body = "\n".join(
                f"  - [{h.chunk.chunk_id}] {h.chunk.preview(110)}" for h in hits
            )
            sections.append(f"【{item}】\n{body}")

        content = (
            f"对 {len(items)} 个对象做横向对比"
            f"（每个对象取最相关的 {self.per_item_k} 条资料）：\n\n"
            + "\n\n".join(sections)
        )
        if truncated_items:
            content += f"\n\n（注意：你提供了超过 {MAX_ITEMS} 个对象，仅对比了前 {MAX_ITEMS} 个）"
        if total_hits == 0:
            content += (
                "\n\n【重要】所有对象都未检索到资料，说明知识库不覆盖这些话题。"
                "请如实说明资料不足，不要凭常识编造对比结论。"
            )

        return ToolResult.success(
            content,
            data=comparison_data,
            items=items,
            total_hits=total_hits,
        )


# ==============================================================================
# 工厂函数
# ==============================================================================

def build_default_registry(store: "VectorStore") -> "ToolRegistry":
    """
    构建包含全部工具的标准注册表。

    【注册顺序是有讲究的】
        按"使用频率"和"推荐优先级"排序：
            list_knowledge_base 放第一（鼓励 Agent 先建立全局视野）
            search_local_docs   放第二（最常用）
            其余按需
        实测中，prompt 里靠前的工具更容易被模型选中，
        所以顺序可以起到"引导 Agent 行为"的作用。
    """
    from tools.base import ToolRegistry

    registry = ToolRegistry()
    registry.register(ListKnowledgeBaseTool(store))
    registry.register(SearchDocsTool(store))
    registry.register(ReadDocSectionTool(store))
    registry.register(ExtractKeyPointsTool(store))
    registry.register(CrossCompareTool(store))
    return registry
