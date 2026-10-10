"""
rag/loader.py —— 文档加载与切块（RAG 链路第 1 步）
================================================================================
【这个文件在架构里的位置】
    RAG 流水线的入口：把磁盘上的原始文档，变成一个个"可被检索的最小单元"。

【整体数据流】
    knowledge/*.md  ->  Document  ->  Chunk[]  ->  (交给 embedder 向量化)

【核心问题一：为什么要切块（chunking）？】
    不能把整篇文档作为一个检索单元，原因：
        1. 太长塞不进 LLM 的上下文窗口
        2. 一个长文档里往往有多个主题，向量化后整个文档的向量是"平均语义"，
           问任何一个具体问题都会匹配到它，但匹配度都不高 —— 检索精度崩了
        3. 召回 5 篇长文 = 塞进去几万字，又贵又慢，还容易让模型抓不住重点
    所以要把文档切成小块，让"一个块 ≈ 一个相对独立的语义单元"。

【核心问题二：切块的两个关键参数】
    chunk_size（块大小）
        太小：语义不完整。比如把"推荐使用方案 A，因为它的成本更低"
              切成两块，单独看任一块都答不了"为什么推荐 A"
        太大：语义被稀释。一块里混了"成本""性能""部署"三个话题，
              它的向量是三者混合，检索"成本"时会匹配到它但分数不高，
              同时也会匹配"性能""部署"，造成噪音
        经验值：中文 250~500 字符是比较稳的起点

    chunk_overlap（重叠）
        如果一句话正好落在切分点上，前半句在块 A、后半句在块 B，
        那么无论召回 A 还是 B，这句话都是残缺的。
        让相邻块重叠一段，保证任何一句话至少完整出现在某一个块里。
        【生活类比】像用剪刀裁报纸，与其一刀切断一行字，
        不如让两段重叠一点，保证没有字被切成两半。

【核心问题三：为什么按段落优先，而不是无脑按长度切？】
    因为文档的作者已经用段落表达了他的语义边界。
    "在段落边界切"比"在第 300 个字切"更可能保持语义完整。
    所以本实现采用【两级策略】：
        第一级：先按自然段落（空行）分组
        第二级：段落太长时，才在句号/问号等句子边界做二次切分
    这叫"尊重原文结构"的切块策略，效果明显优于暴力定长切分。

    这个思路可以推广到所有文档处理场景：
        Markdown 按标题切、代码按函数切、HTML 按标签切 ——
        永远优先利用原文已有的结构信息。
================================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import CFG


@dataclass
class Chunk:
    """
    一个文本块 —— RAG 的"最小检索单元"。

    【字段设计说明】
        doc_id     : 文档级 ID。多个块共享同一个 doc_id，
                     这是实现"引用溯源"的基础 —— 模型说"根据 [doc_03]"，
                     你能立刻找到是哪篇文档
        chunk_id   : 块级唯一 ID（形如 doc_03#2），用于精确引用到某一段
        text       : 块的正文内容
        title      : 文档标题，方便人阅读引用来源
        source     : 原始文件路径，最彻底的溯源
        position   : 这是文档里的第几块（从 0 开始），便于定位上下文
        metadata   : 预留的扩展字段（作者、日期、标签等）

    【为什么 doc_id 和 chunk_id 都要有？】
        doc_id 回答"来自哪篇文档"，chunk_id 回答"来自这篇文档的哪一段"。
        引用溯源需要精确到段，否则模型可能引用了文档 A，
        但结论其实来自文档 B —— 你无法核对。
    """
    chunk_id: str
    doc_id: str
    text: str
    title: str
    source: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_len(self) -> int:
        return len(self.text)

    def preview(self, n: int = 80) -> str:
        """生成短预览，用于日志显示"""
        flat = " ".join(self.text.split())
        return flat[:n] + ("…" if len(flat) > n else "")

    def citation(self) -> str:
        """
        生成人类可读的引用标签，比如：
            [doc_03#2 · 向量数据库选型对比]
        报告里带上这个，读者就能顺着找到原文。
        """
        return f"[{self.chunk_id} · {self.title}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "title": self.title,
            "source": self.source,
            "position": self.position,
            "char_len": self.char_len,
        }


# ==============================================================================
# 一、文本切分
# ==============================================================================

# 句子结束符。中文用 。！？；，英文用 .!?;
# 【注意】把分号和换行也算进来，是因为技术文档里常有列表项和分号分隔的长句，
# 在这些位置断开通常也是安全的
_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])\s*|\n+")


def _split_sentences(text: str) -> list[str]:
    """
    把文本切成句子。

    【为什么用 re.split 而不是简单的 text.split("。")】
        因为要保留分隔符。`(?<=[。！？])` 是"后行断言"：
        匹配位置的前面必须是这些结束符之一，但结束符本身不被消耗掉，
        所以切分后句号会留在前一句的末尾。
        如果不这样做，切出来的句子会丢掉标点，重新拼接时可读性很差。

    【注意：这是一个"够用就好"的实现】
        真正的句子切分还要处理引号、省略号、英文缩写（如 "e.g." "Mr."）等。
        生产环境可以用专门的库。学习阶段保持简单，重点是理解"按句子边界切"
        这个策略本身。
    """
    parts = [p.strip() for p in _SENTENCE_END.split(text) if p and p.strip()]
    return parts


def chunk_text(
    text: str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    把一段长文本切分成若干块。

    ── 算法流程 ──
        第 1 步：按空行拆成自然段落
        第 2 步：逐段累积到当前块，直到超过 chunk_size，就封箱
        第 3 步：如果单个段落本身就超过 chunk_size，先按句子切碎它
        第 4 步：封箱时，从上一块的尾部取 overlap 个字符作为下一块的开头

    ── 为什么要做第 3 步？ ──
        如果不处理超长段落，一个 5000 字的段落会变成一个 5000 字的块，
        chunk_size 就形同虚设了。所以必须有"段落内部再切分"的降级逻辑。

    ── 为什么第 4 步的 overlap 用"字符"而不是"整句"？ ──
        简单可靠。用整句做 overlap 更优雅，但实现复杂、容易出现死循环
        （比如某句话比 chunk_size 还长）。学习项目里字符重叠足够说明问题。

    ── 【重要修正】overlap 必须对齐到边界，不能硬切 ──
        最初的实现是直接 `current[-chunk_overlap:]`，无脑按字符数截取。
        这在网页可视化里暴露出了一个严重问题：
        看知识库原文时，很多块的开头是【半个词】：
            01_vector_db_overview#7  ->  "lvus 是独立部署的分布式系统…"   （Mi 被切掉了）
            01_vector_db_overview#2  ->  "hbor）算法。它不保证找到…"      （Nearest Neigh 被切掉）
            01_vector_db_overview#1  ->  "个向量"，这是一个高维空间…"     （前引号被切掉）

        【为什么这是个真问题，而不只是"不好看"？】
            1. 检索质量：切碎的词会产出无意义的 n-gram（"lvus"、"hbor"），
               污染倒排索引，还可能和别的词产生虚假匹配
            2. 阅读体验：溯源时看到"lvus"会让人怀疑数据是不是坏了
            3. 引用可读性：报告里引用这种块，读者无法判断上下文

        【修正思路：从"截取 N 个字符"改成"截取 N 个字符、然后向右对齐到最近的边界"】
            边界包括：段落分隔（\n\n）、句末标点、以及中英文的词边界。
            具体做法是先多截一点，再从左往右找第一个边界位置。

        【为什么是"向右对齐"而不是"向左对齐"？】
            向左对齐会让 overlap 变短甚至变 0，削弱重叠的作用。
            向右对齐只让 overlap 略微变长，功能不打折。
            宁可多留几个字，也不要破坏语义完整性。
    """
    chunk_size = chunk_size or CFG.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else CFG.chunk_overlap

    # 参数校验：overlap 必须小于 size，否则会无限循环
    # 【这类防御性检查很值得养成习惯】它把"参数配错"这种低级错误
    # 从"运行时诡异卡死"变成"启动时明确报错"
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})，"
            f"否则切分会陷入死循环"
        )

    text = text.strip()
    if not text:
        return []

    # ---- 第 1 步：按空行拆段落 ----
    # 用 \n\s*\n 匹配"一个或多个空行"，兼容 Windows 的 \r\n
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs: list[str] = []
    for p in raw_paragraphs:
        p = p.strip()
        if not p:
            continue
        # ---- 第 3 步：超长段落降级处理 ----
        if len(p) <= chunk_size:
            paragraphs.append(p)
        else:
            paragraphs.extend(_pack_sentences(_split_sentences(p), chunk_size))

    # ---- 第 2 步 + 第 4 步：累积封箱，带重叠 ----
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        # 超过上限：先把当前累积的内容封箱
        if current:
            chunks.append(current)
            # 从上一块尾部截取 overlap，作为新块的开头（对齐到词/句边界）
            tail = _boundary_tail(current, chunk_overlap)
            current = (tail + "\n\n" + para).strip() if tail else para
        else:
            # current 为空说明这个段落自己就超了（理论上不该发生，
            # 因为上面已经用 _pack_sentences 拆过了）——但防御性兜底仍然必要，
            # 避免极端输入（比如一整段没有任何标点的长文本）导致内容丢失
            chunks.append(para[:chunk_size])
            current = para[chunk_size:]

        # 重叠后可能又超了，再封一次
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            # 这里同样要做边界对齐 —— 否则同样会产生"半个词"的问题
            tail = _boundary_tail(current[:chunk_size], chunk_overlap)
            current = tail

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c.strip()]


def _boundary_tail(text: str, overlap: int) -> str:
    """
    从 text 的尾部取出约 overlap 个字符，并【对齐到语义边界】。

    【要解决的问题】
        直接 `text[-overlap:]` 会把词切成两半：
            "Milvus 是..."  ->  "lvus 是..."     （Mi 被切掉）
            "Nearest Neighbor"  ->  "hbor"        （前半截没了）
        这种碎片会污染倒排索引，也让溯源时看到的原文难以阅读。

    【算法】
        1. 先多取一些（overlap 的 1.6 倍），保证够长
        2. 在这个窗口里找【最靠左】的边界位置
        3. 从那个位置切到末尾

        第 2 步的"最靠左"是关键：它让取出的片段尽量接近 overlap 长度，
        而不是随便找个边界就切（那样可能只剩几个字，重叠就失去意义了）。

    【边界怎么定义？按优先级找】
        段落分隔 > 句末标点 > 空白 > 中英文交界

    【为什么英文词要特殊处理？】
        英文单词之间靠空格分隔，但中文没有空格。
        如果不区分，"Nearest Neighbor" 会被当成一个连续的"词"，
        边界检测就找不到该切的地方。
        所以我们把"拉丁字母/数字"和"其他字符"的交界也算作边界 ——
        这样 "Mi|lvus" 不会被识别为边界（都是字母），
        但 "的|Milvus" 会（中文和拉丁字母交界）。
        【注意】这意味着 "Milvus" 整个词是安全的，
        但如果我们恰好从 "Milvus" 中间开始切，仍然会切碎它。
        这是字符级 overlap 的固有局限，彻底解决需要分词。
    """
    if overlap <= 0 or not text:
        return ""

    # 窗口取大一点，给边界检测留出腾挪空间
    window_start = max(0, len(text) - int(overlap * 1.6))
    window = text[window_start:]

    # 找最靠左的边界。
    # 注意：从下标 1 开始找，避免切出空片段。
    for i in range(1, len(window)):
        # 段落分隔
        if window[i:i + 2] == "\n\n":
            return window[i:].strip()
        ch, prev = window[i], window[i - 1]

        # 句末标点后
        if prev in "。！？；.!?;\n":
            return window[i:].strip()

        # 空白后
        if prev in " \t":
            return window[i:].strip()

        # 中英文交界（拉丁字母/数字 vs 其它）
        prev_is_latin = prev.isascii() and (prev.isalnum())
        cur_is_latin = ch.isascii() and (ch.isalnum())
        if prev_is_latin != cur_is_latin:
            return window[i:].strip()

    # 找不到任何边界（比如整段是无标点的长串），退回按字符切。
    # 【必须有这个兜底】否则可能返回空串，导致 overlap 完全失效。
    return text[-overlap:].strip()


def _pack_sentences(sentences: list[str], chunk_size: int) -> list[str]:
    """
    把句子列表打包成不超过 chunk_size 的段落。

    这是 chunk_text 的辅助函数，逻辑和主循环类似但更简单 ——
    因为这里不需要 overlap（overlap 在外层统一处理，避免叠加两层重叠
    导致同一段内容被重复太多次）。
    """
    out: list[str] = []
    buf = ""
    for s in sentences:
        # 单句就超长（比如一整段没有标点）：硬切
        if len(s) > chunk_size:
            if buf:
                out.append(buf)
                buf = ""
            for i in range(0, len(s), chunk_size):
                out.append(s[i:i + chunk_size])
            continue

        candidate = (buf + s) if buf else s
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            out.append(buf)
            buf = s

    if buf:
        out.append(buf)
    return out


# ==============================================================================
# 二、文档加载
# ==============================================================================

# 从 Markdown 里提取一级/二级标题作为文档标题
_TITLE_RE = re.compile(r"^#{1,2}\s+(.+)$", re.MULTILINE)


def load_documents(
    directory: Path | None = None,
    *,
    glob_pattern: str = "*.md",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    verbose: bool = True,
) -> list[Chunk]:
    """
    加载目录下所有文档，切块，返回 Chunk 列表。

    【为什么返回扁平化的 Chunk 列表，而不是"文档 -> 块列表"的嵌套结构？】
        因为下游（embedder / store）只关心"块"，不关心它来自哪篇文档 ——
        文档归属信息已经编码在 chunk.doc_id 里了。
        扁平结构让向量库的实现更简单：一个块就是一行。
        这是"数据结构服从消费方需求"的一个例子。

    【doc_id 的生成方式】
        用文件名（不含扩展名），比如 "vector_db_comparison.md" -> "vector_db_comparison"。
        为什么不用自增数字（doc_1, doc_2）？
            因为数字 ID 在文档增删后会变化，导致引用失效。
            基于文件名的 ID 是稳定的 —— 重新构建索引后引用依然有效。
            【工程原则】ID 应该稳定且可读，不要用"当前顺序"当 ID。
    """
    directory = directory or CFG.knowledge_dir

    if not directory.exists():
        raise FileNotFoundError(
            f"知识库目录不存在: {directory}\n"
            f"请先创建该目录并放入 .md 文档。"
        )

    files = sorted(directory.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"知识库目录 {directory} 下没有匹配 {glob_pattern} 的文件")

    all_chunks: list[Chunk] = []

    for path in files:
        raw = path.read_text(encoding="utf-8", errors="replace")

        # ---- 提取标题 ----
        m = _TITLE_RE.search(raw)
        title = m.group(1).strip() if m else path.stem

        # ---- 切块 ----
        text_chunks = chunk_text(raw, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        doc_id = path.stem
        for i, tc in enumerate(text_chunks):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{i}",
                    doc_id=doc_id,
                    text=tc,
                    title=title,
                    source=str(path.relative_to(directory.parent)),
                    position=i,
                    metadata={"file": path.name},
                )
            )

        if verbose and CFG.log_level != "quiet":
            print(f"    📄 {path.name:<38} -> {len(text_chunks):>2} 块  ({len(raw):>5} 字符)")

    return all_chunks
