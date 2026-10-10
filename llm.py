"""
llm.py —— 大模型调用层（双模式：离线 Mock / 真实 API）
================================================================================
【这个文件在架构里的位置】
    Agent 想"思考"，就必须调用 LLM。所有 Agent 都通过这里拿模型输出。
    这是整个系统唯一与"模型"耦合的地方 —— 换成任何模型，只改这个文件。

【为什么要有 MockLLM】
    调试多智能体系统时，难的不是"模型聪不聪明"，而是：
        - ReAct 循环有没有正确解析出 Action / Action Input
        - 工具参数有没有传错
        - Critic 打回后，Orchestrator 有没有真的重新派活
        - 黑板上的数据有没有被下游 Agent 正确读到
    这些全是【工程问题】，和模型无关。用真 API 调试的代价：
        花钱、慢、输出不稳定（这次能跑下次崩，分不清是代码错还是模型抽风）、
        没 Key 的人根本跑不起来。
    MockLLM 用规则代码模拟模型输出：固定输入 -> 固定输出，零成本、毫秒级、不需要网络。
    代价：它不会真推理，所以只能验证"流程正确性"，不能验证"报告质量"。

    学习路径建议：
        第 1 步  用 MockLLM 把全流程跑通，看日志、打断点，彻底搞懂数据怎么流
        第 2 步  切 llm_mode="openai"，观察同样的代码在真模型下行为有何不同

【本文件三个关键设计（都是实战教训）】
    1. LLMClient 抽象基类 —— 让上层不关心底下是 mock 还是真模型
    2. parse_react_output 的容错 —— 真模型经常不按格式输出，必须能兜住
    3. call_with_retry 的统一重试 —— 网络抖动、格式错误都能自动补救
================================================================================
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from config import CFG


# ==============================================================================
# 第 1 部分：抽象基类
# ==============================================================================

class LLMClient(ABC):
    """
    所有 LLM 实现的统一接口。

    为什么用抽象基类（ABC）而不是鸭子类型？
        ABC 能在实例化时就把"忘记实现某个方法"报错，而不是等到运行到那一行才炸。
        整个项目只有这一个接口，Agent 只认这个接口，不认具体实现。
        —— 这就是"依赖倒置"：高层模块（Agent）不依赖低层模块（LLM），两者都依赖抽象。
    """

    # 记录累计调用次数 / token 估算，方便观察成本
    def __init__(self) -> None:
        self.call_count = 0
        self.prompt_chars = 0
        self.completion_chars = 0

    @abstractmethod
    def _generate(self, prompt: str, system: str = "") -> str:
        """子类实现真正的生成逻辑"""
        raise NotImplementedError

    def generate(self, prompt: str, system: str = "") -> str:
        """
        对外唯一入口。所有的计数、日志、耗时统计都在这里统一做，
        子类只管生成内容，不用重复写这些样板代码。
        """
        self.call_count += 1
        self.prompt_chars += len(prompt) + len(system)

        t0 = time.time()
        text = self._generate(prompt, system)
        elapsed = time.time() - t0

        self.completion_chars += len(text)

        if CFG.log_level == "verbose":
            print("\n" + "=" * 78)
            print(f"[LLM PROMPT] ({len(prompt)} chars)")
            print(prompt)
            print("-" * 78)
            print(f"[LLM RESPONSE] ({elapsed:.2f}s)")
            print(text)
            print("=" * 78 + "\n")

        return text

    def stats(self) -> str:
        return (
            f"LLM 调用 {self.call_count} 次 | "
            f"输入约 {self.prompt_chars} 字符 | 输出约 {self.completion_chars} 字符"
        )


# ==============================================================================
# 第 2 部分：MockLLM —— 离线假模型
# ==============================================================================

class MockLLM(LLMClient):
    """
    规则驱动的假模型。

    【工作原理】
        它不"理解"任何东西，只是对 prompt 做关键词匹配，然后返回一段
        预先写好的、格式正确的文本。为了让 Agent 的多轮 ReAct 循环真的能
        走完，它内部维护了一点"记忆"（比如某个角色已经搜过哪些词），
        这样第二轮才会给出不同的输出，而不是傻傻地重复第一步。

    【关键点】
        它的输出格式必须和我们对真模型要求的格式【完全一致】，
        否则"mock 能跑通、真模型跑不通"就失去了参考价值。
    """

    def __init__(self) -> None:
        super().__init__()
        # 简单的会话内记忆：记录每个角色已经执行过哪些动作，
        # 用来决定"下一步该干嘛"，让多轮循环能自然收敛到 Final Answer
        self._memory: dict[str, list[str]] = {}

    # --------------------------------------------------------------------------
    # 工具方法
    # --------------------------------------------------------------------------
    def _recall(self, key: str, item: str) -> bool:
        """检查 key（通常用角色名）下是否已经做过 item"""
        return item in self._memory.get(key, [])

    def _remember(self, key: str, item: str) -> None:
        self._memory.setdefault(key, []).append(item)

    @staticmethod
    def _role_of(prompt: str) -> str:
        """
        从 prompt 里判断"现在是谁在说话"。

        真实场景当然不需要这么做（真模型能读懂 system prompt），
        但 Mock 必须知道自己扮演哪个角色，才能返回符合该角色预期的输出。

        【这里踩过坑，教训值得记录】
            第一版实现是扫描 prompt 里有没有 "角色：researcher" 这样的标记，
            以及"调研员""分析师"这类中文词。
            结果判定全部失败，所有角色都被误判成 unknown，返回了同一段兜底文本。

            失败原因有两个，都很典型：
              1. 我假设上层会写 "角色：xxx"，但 base.py 实际不生成这个字段
                 —— 【凭想象假设上游格式】是写解析代码时的头号错误
              2. 中文词判断依赖 system_prompt 被拼进 prompt，
                 但实际上 system 是【单独的参数】传给 LLM 的，
                 并不在 prompt 里！（见 LLMClient.generate 的签名：
                 generate(prompt, system=...)）

            这个第 2 点是很关键的理解：prompt 和 system 是两路输入，
            MockLLM._generate 只收到 prompt，收不到 system。
            所以任何要识别角色的逻辑，都必须只看 prompt。

            【修正后的判据：用角色【独有的工具集】来判断】
                每个角色的工具子集是独特的（见各 Agent 的 __init__）：
                    researcher -> 有 search_local_docs + list_knowledge_base
                    analyst    -> 有 cross_compare，没有 search_local_docs
                    writer/critic -> 完全没有工具列表
                这个特征稳定、唯一，且确实是 prompt 的一部分（工具说明会写进 prompt）。
                用"结构性特征"而不是"人设描述词"来识别，可靠得多 ——
                这和 _generate 里选任务类型判据的原则是一致的。
        """
        # ---- 有工具的 Agent：用工具集签名识别 ----
        has_search = "### 工具名: search_local_docs" in prompt
        has_outline = "### 工具名: list_knowledge_base" in prompt
        has_compare = "### 工具名: cross_compare" in prompt
        has_tools = "### 工具名:" in prompt

        # 调研员是唯一同时拥有 list_knowledge_base 和 search_local_docs 的角色
        if has_search and has_outline:
            return "researcher"
        # 分析师有 cross_compare 但没有 search_local_docs（刻意的权限隔离）
        if has_compare and not has_search:
            return "analyst"
        # 其他有工具的情况：按最先出现的工具保守判断
        if has_search:
            return "researcher"
        if has_compare:
            return "analyst"

        # ---- 无工具的 Agent：靠 prompt 骨架区分 ----
        # Critic 的 build_task 会加"程序化校验结果"这段
        if "程序化校验结果" in prompt:
            return "critic"
        # Writer 的 prompt 以"直接输出你的最终结果"结尾
        if "直接输出你的最终结果" in prompt:
            return "writer"

        # ---- 兜底：用角色名字符串（兼容显式写法的上层）----
        for role in ("orchestrator", "researcher", "analyst", "writer", "critic"):
            if f"角色：{role}" in prompt or f"你是 {role}" in prompt:
                return role

        return "unknown"

    @staticmethod
    def _extract_topic(prompt: str) -> str:
        """
        从 prompt 里抽取"用户原始问题"作为检索关键词。
        Agent 的 prompt 里会带上任务描述，用正则捞出来。
        """
        patterns = [
            r"用户(?:原始)?问题[:：]\s*(.+)",
            r"原始任务[:：]\s*(.+)",
            r"任务[:：]\s*(.+)",
        ]
        for p in patterns:
            m = re.search(p, prompt)
            if m:
                return m.group(1).strip().split("\n")[0][:60]
        return "向量数据库选型"

    # --------------------------------------------------------------------------
    # 主入口
    # --------------------------------------------------------------------------
    def _generate(self, prompt: str, system: str = "") -> str:
        """
        根据 prompt 的特征，分派到不同的"剧本"。
        注意分派顺序：越具体的判断要越靠前，否则会被宽泛条件先吃掉。
        """
        role = self._role_of(prompt)

        # ======================================================================
        # 判断任务类型
        # ======================================================================
        # 【这段分派逻辑我改了三版，踩的坑很值得记录 ——
        #   它是本项目里"想当然的判据会失效"的最佳案例】
        #
        # 第一版：让上层在 prompt 里插入 [TASK:PLAN] 这样的标记。
        #   问题：上层有十几个地方拼 prompt，任一处忘了加标记就走错分支，
        #   而且症状隐蔽（不报错，只是行为奇怪）。
        #
        # 第二版：改用关键词检测，用"sub_tasks"判断"这是规划任务"。
        #   结果【所有角色的 prompt 都被误判成规划任务】，
        #   每个 Agent 都返回同一段 JSON，整个流程形同虚设。
        #   原因：Orchestrator 把任务规划（含 "sub_tasks" 字段）
        #   通过 context 传给了下游 Agent ——
        #   规划 JSON 的原文出现在了研究员/分析师的 prompt 里！
        #
        # 第三版：改用"coordinator 拆解成若干子任务"这个短语，还是失败 ——
        #   因为 prompt 里同时出现了"工具调用格式"（说明是 ReAct 角色）
        #   和规划内容（说明 context 里有计划），两个信号冲突。
        #
        # 【最终的、正确的判据：看 prompt 的"结构骨架"，而不是"内容里有什么"】
        #   关键洞察：base.py 的 _build_prompt 生成的 prompt 有固定的分段标题，
        #   而不同角色的 prompt 骨架是不同的：
        #       有工具的 Agent : 必含 "## 工具调用格式（必须严格遵守）"
        #       无工具的 Agent : 必含 "## 现在请直接输出你的最终结果"
        #       Critic        : 必含 "## 程序化校验结果"
        #       Orchestrator  : 必含 "## 输出格式（严格遵守，这是机器要解析的）"
        #
        #   这些标题是【框架生成的骨架】，不会被 context 里的业务内容污染。
        #   而 "sub_tasks" 是【业务内容】，会被 context 带进来。
        #
        #   【这个教训非常重要，可以推广到所有分类场景：
        #     判据要选"由生产者严格控制的、结构化产生的"特征，
        #     而不是"可能出现在任何地方的自由文本"。
        #     前者稳定可靠，后者充满意外。
        #     这和机器学习里"用元数据特征 vs 用文本内容特征"的权衡是一样的。】
        # ======================================================================

        # ---- 1. Critic：审校任务 ----
        # "程序化校验结果" 是 critic.py 的 build_task 独有的段落标题
        if "程序化校验结果" in prompt:
            return self._critique(prompt)

        # ---- 2. Orchestrator：任务拆解 ----
        # "机器要解析" 这个短语只出现在 orchestrator.PLAN_PROMPT 里
        if "这是机器要解析的" in prompt:
            return self._plan(prompt)

        # ---- 3. 有工具的 Agent：走 ReAct 剧本 ----
        # 这是 base.py 的 _build_prompt 在 include_tools=True 时必加的骨架标题
        # 【放在 plan 判断之后】因为 plan 的 prompt 也要求结构化输出，
        # 但它没有工具，所以不会含这个标题，两者不会冲突。
        if "## 工具调用格式（必须严格遵守）" in prompt:
            return self._react(prompt, role)

        # ---- 4. 无工具的生成型 Agent（Writer）：总结/撰写剧本 ----
        if "## 现在请直接输出你的最终结果" in prompt:
            return self._summarize(prompt, role)

        # ---- 5. 显式标记（保留作为上层的可选手段）----
        if "[TASK:SUMMARIZE]" in prompt:
            return self._summarize(prompt, role)
        if "[TASK:REACT]" in prompt:
            return self._react(prompt, role)

        # ---- 兜底：绝不返回空串 ----
        # （返回空串会让上层解析崩溃，且难以定位问题。
        #   返回一段中性但格式正确的内容，至少流程能走下去。）
        return (
            f"Thought: 我已获得足够信息。\n"
            f"Final Answer: 根据现有资料，就「{self._extract_topic(prompt)}」"
            f"这一问题，建议综合评估后做出决定。"
        )

    # --------------------------------------------------------------------------
    # 剧本 1：任务规划（Orchestrator 用）
    # --------------------------------------------------------------------------
    def _plan(self, prompt: str) -> str:
        topic = self._extract_topic(prompt)
        # 要求真模型输出 JSON，是因为规划结果需要被程序【结构化消费】：
        # Orchestrator 要拿 sub_tasks 去逐个派活，纯文本没法可靠解析。
        # 【重要】凡是需要程序消费的输出，一律要求 JSON，并且必须写容错解析。
        return json.dumps(
            {
                "topic": topic,
                "sub_tasks": [
                    {
                        "id": "T1",
                        "assignee": "researcher",
                        "goal": f"检索关于「{topic}」的技术资料，产出带引用来源的证据清单",
                    },
                    {
                        "id": "T2",
                        "assignee": "analyst",
                        "goal": "基于证据清单做多维度对比分析，输出对比表格与推荐结论",
                    },
                    {
                        "id": "T3",
                        "assignee": "writer",
                        "goal": "把证据与分析整合成一份结构化的选型报告",
                    },
                ],
                "acceptance_criteria": [
                    "每个关键结论都有可追溯的引用来源",
                    "有明确的对比维度（至少 3 个）",
                    "给出可执行的最终推荐",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------------------------
    # 剧本 2：ReAct 循环（所有工具型 Agent 用）
    # --------------------------------------------------------------------------
    def _react(self, prompt: str, role: str) -> str:
        """
        ReAct 的核心：Thought -> Action -> Observation 循环。

        Mock 的策略是"按剧本推进"：
            第 1 轮：先检索（Action）
            第 2 轮：再精读（Action）
            第 3 轮：信息够了，给 Final Answer
        这样能保证循环必然收敛，不会死循环。

        真实模型下这个决策由模型自己做，可能会出现"反复搜同一个词"的问题，
        所以 base.py 里必须有 max_react_steps 兜底。

        【重要：脚本必须适配"当前角色手里实际有哪些工具"】
            这是我实测踩的一个典型坑：
            Mock 脚本写死了"先 search_local_docs、再 read_doc_section"，
            但 Analyst 的工具集里【没有】 search_local_docs
            （这是刻意的角色隔离，见 analyst.py 的说明）。
            结果 Analyst 每轮都在调用一个不存在的工具，拿到错误，
            整个演示看起来像"Agent 完全不会工作"。

            【这个 bug 很有教育意义】它暴露了一个真实问题：
            写 mock / 测试替身时，必须让它遵守与真实实现相同的约束，
            否则 mock 会掩盖问题，或者（像这里）制造出不存在的问题。

            修法：脚本先看 prompt 里列出的工具清单，从中选择合适的工具。
            这样无论角色有什么工具，Mock 都能做出合理的选择。
        """
        topic = self._extract_topic(prompt)
        mem_key = f"{role}:{topic}"

        # ---- 从 prompt 里解析出当前可用的工具清单 ----
        # prompt 里工具说明的格式是 "### 工具名: xxx"，正好可以正则提取
        available = re.findall(r"###\s*工具名[:：]\s*([a-zA-Z_][a-zA-Z0-9_]*)", prompt)

        # 按"优先级"选择要用的工具：
        #   有检索能力就先检索（这是最符合 ReAct 精神的第一步）
        #   没有检索就用对比（Analyst 的情况）
        #   都没有就用精读
        def pick(*preferred: str) -> str | None:
            for name in preferred:
                if name in available:
                    return name
            return None

        # ---- 阶段 1：先做一次"广撒网"式的动作 ----
        if not self._recall(mem_key, "phase1"):
            self._remember(mem_key, "phase1")

            tool = pick("search_local_docs", "list_knowledge_base", "cross_compare")
            if tool is None:
                # 连广撒网的工具都没有（比如只有 extract_key_points）：
                # 直接跳到收尾，避免产生无效的工具调用
                self._remember(mem_key, "phase2")
            elif tool == "cross_compare":
                # cross_compare 需要至少 2 个对象，用逗号分隔
                return (
                    f"Thought: 我需要对比多个方案在不同维度上的表现，"
                    f"应该用 cross_compare 把它们并列查询。\n"
                    f"Action: {tool}\n"
                    f"Action Input: 轻量方案, 重型方案"
                )
            elif tool == "list_knowledge_base":
                return (
                    f"Thought: 我还不清楚知识库覆盖了哪些主题，"
                    f"应该先查看目录以确定检索方向。\n"
                    f"Action: {tool}\n"
                    f"Action Input: all"
                )
            else:
                return (
                    f"Thought: 我需要先了解「{topic}」有哪些相关资料，"
                    f"应该先在本地知识库中检索。\n"
                    f"Action: {tool}\n"
                    f"Action Input: {topic}"
                )

        # ---- 阶段 2：深入一步（精读或提炼）----
        if not self._recall(mem_key, "phase2"):
            self._remember(mem_key, "phase2")

            tool = pick("read_doc_section", "extract_key_points")
            if tool is not None:
                # 从上一轮的 Observation 里找出一个真实存在的 doc_id
                doc_id = self._pick_doc_id(prompt)
                if tool == "read_doc_section":
                    return (
                        f"Thought: 检索返回了几个候选文档。为了拿到准确的细节和数据，"
                        f"我需要精读其中最相关的一篇。\n"
                        f"Action: {tool}\n"
                        f"Action Input: {doc_id}"
                    )
                return (
                    f"Thought: 我已经找到了相关文档，现在需要从中提炼出关键要点，"
                    f"以便形成结构化的结论。\n"
                    f"Action: {tool}\n"
                    f"Action Input: {doc_id}"
                )

        # ---- 阶段 3：资料够了，输出结论 ----
        answer = self._final_answer_for(role, topic)

        return f"Thought: 我已经获得了足够的信息，可以给出最终结论了。\nFinal Answer: {answer}"

    @staticmethod
    def _final_answer_for(role: str, topic: str) -> str:
        """
        按角色生成不同的 Final Answer。

        【为什么要分角色写不同的答案？】
            如果所有角色都返回同一段文本，你就无法从输出上
            分辨"这个结论是调研员给的还是分析师给的" ——
            多智能体的分工就变得不可观察了。

            Mock 的价值在于让流程【可观察】。
            所以每个角色的产出应该有明显的差异化特征：
                researcher -> 给"带来源ID的证据清单"格式
                analyst    -> 给"带对比维度的分析"格式
                other      -> 通用说明
            这样你在日志里一眼就能看出信息是怎么一步步演化的。
        """
        if role == "researcher":
            # 模拟"带来源ID的证据清单"格式，这也是 system_prompt 要求的形式。
            # 【注意】这里的 doc_id 是刻意编的示例，用于演示格式。
            # 真实模型会从检索到的 Observation 里引用真实的 ID ——
            # 这也正是 Critic 的 programmatic_check 要校验的东西。
            return (
                f"围绕「{topic}」的调研结果如下：\n\n"
                f"1. [01_vector_db_overview#0] 向量数据库的核心职责是存储和检索高维向量，"
                f"在 RAG 中负责找出与查询语义最相近的文档块。\n"
                f"2. [01_vector_db_overview#5] 数据量在一万条以下时，暴力检索（Flat）"
                f"的性能已经足够，不必引入复杂索引。\n"
                f"3. [02_vector_db_cost#2] 一千万条 768 维向量的 HNSW 索引实际需要约 65 GB 内存，"
                f"内存是主要成本项。\n"
                f"4. [02_vector_db_cost#6] Pgvector 在 500 万条以内的查询延迟通常在 10~30 毫秒。\n"
                f"5. [07_multi_agent_arch#0] 多智能体系统把复杂任务拆给多个各具专长的 Agent 协作完成。"
            )
        if role == "analyst":
            return (
                f"基于调研证据，对「{topic}」的分析如下：\n\n"
                f"## 对比维度一：成本\n"
                f"轻量方案（Pgvector/Chroma）几乎无额外成本，可复用现有基础设施；"
                f"重型方案（Milvus）需要大内存机器，千万级向量需约 65 GB 内存（依据 [02_vector_db_cost#2]）。\n\n"
                f"## 对比维度二：运维复杂度\n"
                f"Pgvector 作为 PostgreSQL 扩展，可复用现有备份监控体系；"
                f"Milvus 需维护 etcd、MinIO、Pulsar 等组件，需要专职人员（依据 [01_vector_db_overview#7]）。\n\n"
                f"## 对比维度三：中文语义效果\n"
                f"效果不取决于向量数据库本身，而取决于 embedding 模型的选择（依据 [03_embedding_models#0]）。\n\n"
                f"## 推荐结论\n"
                f"建议采用轻量方案起步。对于 5 人左右、无专职运维的团队，"
                f"Pgvector 在成本与运维复杂度两个维度上都有明显优势，"
                f"且其检索层接口标准化，未来数据量增长后易于迁移（依据 [02_vector_db_cost#6]）。\n\n"
                f"## 落地建议（第一步）\n"
                f"先用 Pgvector + bge-base-zh 搭建最小可用版本，"
                f"并用 100~200 篇代表性文档测出 Recall@5 基线（依据 [08_enterprise_rag_plan] 的分阶段路线）。"
            )
        return (
            f"我已基于证据与分析完成「{topic}」相关章节的撰写，"
            f"所有结论均标注了引用来源。"
        )

    @staticmethod
    def _pick_doc_id(prompt: str) -> str:
        """
        从 prompt 中已经出现的 Observation 里，解析出可用的 doc_id。

        【为什么要这么做？这是 MockLLM 能否"表现得更像真模型"的关键】
            真模型的行为是：看着工具返回的内容，决定下一步读哪篇文档。
            它提取的是【真实存在的 doc_id】，因为工具返回里就有。

            MockLLM 要模拟这个行为，就必须从 prompt 里把 doc_id 抠出来。
            如果抠不出来（比如返回一个不存在的 "auto"），
            工具会报"未找到文档"，虽然这能验证错误路径，
            但会让整个演示看起来像是"Agent 在工作，只是一直失败" ——
            这不是我们想展示的正常流程。

        【这里踩过的坑】
            我最初只匹配 "doc_id=xxx" 这种形式，但实际工具返回的格式是
                [资料1] (相关度 1.00 | 来源 01_vector_db_overview#0 | 标题)
            doc_id 出现在"来源 "后面，而且是 "doc_id#块号" 的形式。
            匹配不上，于是 MockLLM 一直返回 "auto"，流程就卡在错误路径上了。

            【教训】写解析逻辑时，不要凭想象假设上游的格式，
              要去实际打印出来看。这个 bug 我如果一开始就打印 Observation
              确认格式，就能立刻发现，而不是等到跑完整流程才暴露。
              "先看真实数据，再写解析代码" —— 这个顺序能省很多时间。
        """
        # 匹配工具返回里的 "来源 xxx#N" 形式（search_local_docs 的格式）
        m = re.search(r"来源\s+([A-Za-z0-9_\-]+)#\d+", prompt)
        if m:
            return m.group(1)

        # 匹配 "[xxx#N]" 形式（cross_compare 和 read_doc_section 的格式）
        # 【为什么要加这一条？】
        #   Analyst 没有 search_local_docs，它用的是 cross_compare，
        #   而 cross_compare 的输出格式是 "  - [10_deployment_ops#8] ..."，
        #   没有"来源 "前缀。只匹配第一种格式会导致 Analyst 拿不到 doc_id，
        #   只能返回 "auto" 并撞上错误路径。
        #   【教训：工具多了以后，每个工具的返回格式都不同，
        #     解析逻辑必须覆盖所有实际会遇到的格式。
        #     这也是"返回格式应该统一"的一个论据 ——
        #     如果所有工具都返回统一格式，解析就会简单得多。】
        m = re.search(r"\[([A-Za-z0-9_\-]+)#\d+\]", prompt)
        if m:
            return m.group(1)

        # 兼容 "doc_id=xxx" 这种显式写法
        m = re.search(r"doc_id[=:]\s*([A-Za-z0-9_\-]+)", prompt)
        if m:
            return m.group(1)

        # 兼容工具报错时列出的可用文档（取第一个）
        m = re.search(r"可用的文档ID有：([A-Za-z0-9_\-]+)", prompt)
        if m:
            return m.group(1)

        # 实在找不到：返回一个明显无效的值，让流程走错误路径。
        # 【这是有意的设计】宁可让错误显式暴露，也不要静默地返回一个
        # 看似合理但错误的值 —— 后者会让你误以为流程是正常的。
        return "auto"

    # --------------------------------------------------------------------------
    # 剧本 3：总结（各 Agent 收尾用）
    # --------------------------------------------------------------------------
    def _summarize(self, prompt: str, role: str) -> str:
        """
        无工具角色的生成剧本（Writer 用）。

        【为什么 Writer 的剧本要单独写，不能和 ReAct 共用？】
            因为 Writer 不走 ReAct 循环（见 writer.py 的说明），
            它的 prompt 里没有工具说明，是"直接输出最终结果"的形式。
            如果给它 ReAct 格式的输出（Thought + Action），
            上层会把它当成"要调用工具"来处理，流程就乱了。

            【这体现了 mock 的一个核心要求：
              mock 的输出格式必须和"真实模型在该场景下应该输出的格式"一致。
              否则 mock 能跑通、真实模型跑不通，mock 就失去了参考价值。】
        """
        topic = self._extract_topic(prompt)

        if role == "writer":
            # 报告用 Markdown，方便直接落盘成人能读的文件。
            # 【注意这里的引用格式 [xxx#N] 是刻意写的】
            #   为了让 Critic 的 programmatic_check 有东西可校验 ——
            #   这些 ID 如果在知识库里存在，校验会通过；
            #   如果不存在，会被标记为"编造引用"。
            #   这样你就能观察到程序化防幻觉机制实际工作的样子。
            return (
                f"# {topic} —— 技术选型报告\n\n"
                f"## 一、背景与目标\n\n"
                f"针对「{topic}」这一需求，本文基于本地知识库检索到的资料进行分析，"
                f"目标是在成本、运维复杂度与中文语义效果之间找到适合当前团队规模的平衡点。\n\n"
                f"## 二、候选方案概览\n\n"
                f"根据调研证据，主要候选方案分为两类：\n"
                f"- **轻量方案**：Pgvector、Chroma，部署简单、运维成本低\n"
                f"- **重型方案**：Milvus、Qdrant，功能完整、扩展性强\n\n"
                f"## 三、多维度对比\n\n"
                f"| 对比维度 | 轻量方案 | 重型方案 |\n"
                f"| --- | --- | --- |\n"
                f"| 成本 | 低，可复用现有基础设施 | 高，千万级需约 65 GB 内存 [02_vector_db_cost#2] |\n"
                f"| 运维复杂度 | 低，Pgvector 复用数据库运维体系 | 高，需维护多个组件 [01_vector_db_overview#7] |\n"
                f"| 中文语义效果 | 取决于 embedding 模型选择 [03_embedding_models#0] | 同左 |\n"
                f"| 扩展性 | 百万级以内够用 [02_vector_db_cost#6] | 可扩展到十亿级 |\n\n"
                f"## 四、推荐结论\n\n"
                f"建议采用**轻量方案（Pgvector）起步**，理由如下：\n\n"
                f"1. 在成本维度上，Pgvector 无需额外硬件投入，可复用现有的 PostgreSQL 运维体系 [02_vector_db_cost#6]\n"
                f"2. 在运维复杂度维度上，无需专职人员维护，符合当前团队规模的实际情况 [01_vector_db_overview#7]\n"
                f"3. 向量检索层接口标准化，未来数据量增长后可平滑迁移到其他方案，迁移成本可控\n\n"
                f"## 五、落地建议\n\n"
                f"**第一步**：用 Pgvector 配合 bge-base-zh 模型搭建最小可用版本，"
                f"收集 100~200 篇有代表性的文档建立索引 [08_enterprise_rag_plan]\n\n"
                f"**第二步**：构造 50~100 个真实问题作为评估集，测出 Recall@5 基线，"
                f"并据此优化切块策略 [04_chunking_strategies]\n\n"
                f"**第三步**：加入 rerank 模型提升精度，再根据实际使用反馈决定是否需要引入混合检索\n\n"
                f"## 六、引用来源\n\n"
                f"01_vector_db_overview, 02_vector_db_cost, 03_embedding_models, "
                f"04_chunking_strategies, 08_enterprise_rag_plan\n"
            )

        return f"关于「{topic}」的要点总结完成。"

    # --------------------------------------------------------------------------
    # 剧本 4：审校（Critic 用）
    # --------------------------------------------------------------------------
    def _critique(self, prompt: str) -> str:
        """
        Critic 需要输出结构化评审结果，因为 Orchestrator 要用
        verdict 字段决定"是继续还是打回重做"。

        这里 Mock 的策略：第一轮故意挑刺（不通过），第二轮才通过。
        为什么这么设计？—— 为了让你能【看到打回重做的完整流程】。
        如果 Mock 永远说"通过"，那多 Agent 最有价值的那条路径你根本看不到。
        """
        round_no = 1
        m = re.search(r"第\s*(\d+)\s*轮", prompt)
        if m:
            round_no = int(m.group(1))

        # 第一轮：判不通过，逼出返工流程
        if round_no == 1 and not self._recall("critic", "rejected_once"):
            self._remember("critic", "rejected_once")
            return json.dumps(
                {
                    "verdict": "revise",
                    "score": 58,
                    "issues": [
                        "对比维度只有 2 个，不足以支撑选型结论，建议补充「扩展性」维度",
                        "推荐结论缺少可执行的落地步骤",
                    ],
                    "suggestions": [
                        "补充至少 3 个对比维度",
                        "在结论后增加「第一步做什么」的落地建议",
                    ],
                    "reviewer_notes": "方向正确，但论据密度不足，需补充后再提交。",
                },
                ensure_ascii=False,
                indent=2,
            )

        # 第二轮：通过
        return json.dumps(
            {
                "verdict": "pass",
                "score": 82,
                "issues": [],
                "suggestions": ["后续可补充真实压测数据进一步增强说服力"],
                "reviewer_notes": "对比维度完整，结论有据可依，落地建议可执行，予以通过。",
            },
            ensure_ascii=False,
            indent=2,
        )


# ==============================================================================
# 第 3 部分：真实 LLM（OpenAI 兼容协议）
# ==============================================================================

class OpenAICompatLLM(LLMClient):
    """
    任何提供 /v1/chat/completions 的服务都能用：
        DeepSeek、Moonshot、智谱、本地 Ollama、vLLM……
    只要改 config 里的 base_url 和 model 即可。

    【为什么不用 openai 官方 SDK？】
        为了让你看清"一次 HTTP 请求到底发生了什么"。
        生产环境用 SDK 没问题，但学习阶段手写一次 HTTP 调用，
        你会明白 SDK 帮你隐藏了哪些东西（鉴权头、重试、流式解析……）。
    """

    def __init__(self) -> None:
        super().__init__()
        if not CFG.llm_api_key:
            raise ValueError(
                "llm_mode='openai' 但未设置 API Key。\n"
                "请设置环境变量 LLM_API_KEY，或把 config.py 里的 llm_mode 改回 'mock'。"
            )
        # 延迟导入 urllib，避免 mock 模式下产生无谓的依赖负担
        import urllib.request  # noqa: F401  (仅确认可用)

    def _generate(self, prompt: str, system: str = "") -> str:
        import urllib.error
        import urllib.request

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": CFG.llm_model,
            "messages": messages,
            "temperature": CFG.llm_temperature,
        }

        req = urllib.request.Request(
            url=f"{CFG.llm_base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {CFG.llm_api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=CFG.llm_timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 把响应体读出来，否则只看到 "400 Bad Request" 根本不知道哪错了
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:500]
            except Exception:
                pass
            raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"LLM 网络不可达: {e.reason}") from e

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"LLM 返回结构异常: {json.dumps(body)[:500]}") from e


# ==============================================================================
# 第 4 部分：工厂函数
# ==============================================================================

def build_llm(mode: str | None = None) -> LLMClient:
    """
    工厂：根据配置返回对应的 LLM 实例。
    上层代码只写 `llm = build_llm()`，永远不知道也不关心拿到的是哪个实现。
    """
    mode = mode or CFG.llm_mode
    if mode == "mock":
        return MockLLM()
    if mode == "openai":
        return OpenAICompatLLM()
    raise ValueError(f"未知的 llm_mode: {mode}（可选 'mock' / 'openai'）")


# ==============================================================================
# 第 5 部分：ReAct 输出解析（本文件最需要细看的部分）
# ==============================================================================

@dataclass
class ReactStep:
    """
    一次 ReAct 步骤的解析结果。

    raw          : 模型原始输出，调试时最有用
    thought      : 思考内容
    action       : 要调用的工具名（None 表示已经给出最终答案）
    action_input : 工具参数
    final_answer : 最终答案（不为 None 表示循环应该结束）
    parse_error  : 解析失败的原因（不为 None 说明这轮输出格式有问题）
    """
    raw: str = ""
    thought: str = ""
    action: str | None = None
    action_input: str = ""
    final_answer: str | None = None
    parse_error: str | None = None

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


# 允许"Action:"后面跟工具名的各种写法。
# 【为什么要写这么多别名？】真模型不会严格照你的格式来，
# 它可能写 "Action: search" / "动作：search" / "Action： search"。
# 一个健壮的解析器必须容忍这些变体，否则线上会频繁失败。
#
# 【注意这里必须包含 Thought —— 这是我踩过的一个 bug】
#   我最初漏掉了 Thought 的别名，导致所有输出的 thought 字段都是空的。
#   Action 和 Final Answer 解析都正常，流程看起来完全跑得通，
#   只是日志里看不到任何"思考过程"。
#   这个 bug 的隐蔽之处在于：它【不影响功能，只丢信息】。
#   而 Thought 恰恰是学习 ReAct 时最想看的东西。
#   教训：解析器要覆盖【协议里定义的每一个字段】，不能只覆盖"必需"的字段。
_THOUGHT_ALIASES = {
    "thought": "thought",
    "thinking": "thought",
    "思考": "thought",
    "想法": "thought",
    "推理": "thought",
}
_ACTION_ALIASES = {
    "action": "action",
    "动作": "action",
    "tool": "action",
    "工具": "action",
}
_INPUT_ALIASES = {
    "action input": "action_input",
    "action_input": "action_input",
    "actioninput": "action_input",
    "输入": "action_input",
    "参数": "action_input",
    "动作输入": "action_input",
}
_ANSWER_ALIASES = {
    "final answer": "final_answer",
    "final_answer": "final_answer",
    "finalanswer": "final_answer",
    "最终答案": "final_answer",
    "答案": "final_answer",
}


def _match_label(line: str, aliases: dict[str, str]) -> tuple[str | None, str]:
    """
    判断一行是不是 "标签: 内容" 的形式。

    返回 (标准字段名, 内容)；不是标签则返回 (None, "")。

    实现要点：
        - 用「中英文冒号」都要支持
        - 标签本身可能带 Markdown 装饰，比如 "**Action**: xxx"，
          所以要先剥掉 * ` # 这些符号
        - 标签匹配不区分大小写、忽略空格和下划线
    """
    stripped = line.strip()
    if not stripped:
        return None, ""

    # 剥掉 markdown 强调符号，并在冒号处切一次
    cleaned = stripped.lstrip("*#->` ").strip()
    # 同时支持 : 和 ：
    m = re.match(r"^([^:：]{1,20})[:：]\s*(.*)$", cleaned)
    if not m:
        return None, ""

    label_raw, content = m.group(1), m.group(2)

    # 归一化标签：去掉装饰符、空格、下划线，转小写
    label = label_raw.strip().strip("*`").lower()
    label = label.replace("_", " ").replace("  ", " ").strip()
    label_nospace = label.replace(" ", "")

    for table in (_THOUGHT_ALIASES, _ACTION_ALIASES, _INPUT_ALIASES, _ANSWER_ALIASES):
        if label in table:
            return table[label], content.strip()
        if label_nospace in table:
            return table[label_nospace], content.strip()

    return None, ""


def parse_react_output(text: str) -> ReactStep:
    """
    把模型的一段自由文本，解析成结构化的 ReActStep。

    【这是整个 Agent 最脆弱的环节，必须重点理解】

    理想情况下模型会乖乖输出：
        Thought: 我需要先查资料
        Action: search_local_docs
        Action Input: 向量数据库

    但现实里它会输出各种花样：
        1. 把好几步写在一行
        2. 用中文标签"动作："、"最终答案："
        3. 加粗：**Action**: xxx
        4. 唠叨一大堆解释，格式标签藏在中间
        5. 该给 JSON 的时候给 ```json ... ``` 代码块
        6. 干脆忘了写 Action，只有一坨散文

    解析器的职责：尽可能把能救的都救回来；实在救不回来，
    就返回带 parse_error 的结果，让上层决定"重试"还是"降级处理"。
    【千万不要抛异常】—— 一次格式错误不该让整个流程崩溃。

    ── 解析策略（优先级从高到低）──
        1. 逐行扫描标签，这是最可靠的方式
        2. Action Input 可能跨行（比如是段 JSON），要能吃下后续行
        3. 找不到任何标签时，把整段当作 final_answer 兜底
    """
    step = ReactStep(raw=text)
    if not text or not text.strip():
        step.parse_error = "模型返回了空内容"
        return step

    lines = text.splitlines()

    current_field: str | None = None   # 正在收集哪个字段（用于处理跨行内容）
    buffer: list[str] = []

    def flush() -> None:
        """把 buffer 里累积的内容写回 step 对应字段"""
        nonlocal current_field, buffer
        if current_field and buffer:
            content = "\n".join(buffer).strip()
            if content:
                if current_field == "thought":
                    step.thought = (step.thought + "\n" + content).strip()
                elif current_field == "action":
                    step.action = content.split()[0] if content.split() else None
                elif current_field == "action_input":
                    step.action_input = (step.action_input + "\n" + content).strip()
                elif current_field == "final_answer":
                    # 【注意这里必须处理 None 的情况】
                    #   final_answer 的初始值是 None（因为它同时承担"是否已结束"的标志作用）。
                    #   第一次累积时若直接写 (None + "\n" + content) 就会 TypeError。
                    #   这是我实测踩到的 bug。
                    #   【教训】区分"未设置"（None）和"空字符串"（""）是有价值的，
                    #     但每次拼接前都得检查 —— 这是用 None 做哨兵值的代价。
                    #     这里用 `or ""` 一次性兜住。
                    step.final_answer = ((step.final_answer or "") + "\n" + content).strip()
        buffer = []

    for line in lines:
        field, content = _match_label(line, {})

        if field is not None:
            # 遇到新的标签，先把上一段内容落盘
            flush()
            current_field = field
            buffer = [content] if content else []
        else:
            # 不是标签行 —— 归到当前字段里（实现跨行收集）
            if current_field is not None:
                buffer.append(line)
            # 如果还没遇到任何标签，先忽略（通常是模型的寒暄/前言）

    flush()   # 处理最后一段

    # --------------------------------------------------------------------------
    # 兜底逻辑：一个有效标签都没解析到
    # --------------------------------------------------------------------------
    if step.action is None and step.final_answer is None:
        # 有时候模型会把 Final Answer 写成"最终答案是：xxx"这种自然语言，
        # 再给一次机会用宽松正则捞一下
        loose = re.search(
            r"(?:final\s*answer|最终答案|答案)\s*[:：]?\s*(.+)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if loose:
            step.final_answer = loose.group(1).strip()
            if not step.thought:
                step.thought = text[: loose.start()].strip()
            return step

        # 实在不行，整段当答案 —— 这叫"优雅降级"。
        # 与其报错中断，不如给出一个可能不太合格但可用的结果，
        # 并把问题记录在 parse_error 里。
        step.final_answer = text.strip()
        step.parse_error = "未识别到标准 ReAct 标签，已将整段输出降级为最终答案"
        return step

    # --------------------------------------------------------------------------
    # 一致性检查：给了 Action 却没有 Action Input
    # --------------------------------------------------------------------------
    if step.action and not step.action_input:
        step.parse_error = f"解析到 Action='{step.action}' 但缺少 Action Input"

    # --------------------------------------------------------------------------
    # 同时出现 Action 和 Final Answer 时，以 Final Answer 为准。
    # 这是常见现象：模型一边说"我要结束"，一边又写了个动作。
    # 判断规则：既然它已经给出答案，就认为任务完成，避免多做一次无用工具调用。
    # --------------------------------------------------------------------------
    if step.action and step.final_answer:
        step.action = None
        step.action_input = ""

    return step


# ==============================================================================
# 第 6 部分：带重试的调用工具
# ==============================================================================

def call_with_retry(
    fn: Callable[[], Any],
    *,
    validator: Callable[[Any], str | None] | None = None,
    max_retries: int | None = None,
    desc: str = "LLM 调用",
) -> Any:
    """
    统一的重试包装器。

    【为什么需要它】
        两类失败在真实环境里非常常见：
            1. 网络类：超时、5xx —— 重试通常就好了
            2. 格式类：模型输出不符合预期 —— 重试一次往往也能好
               （温度 > 0 时，模型每次输出都略有不同）

    【参数】
        fn        : 实际要执行的操作（无参函数，用 lambda 或 partial 包装）
        validator : 校验函数。返回 None 表示通过；返回字符串表示失败原因。
                    这是"把校验逻辑和重试逻辑解耦"的经典做法 ——
                    校验规则由调用方决定，重试机制由这里统一提供。
        max_retries: 最大重试次数

    返回 fn 的结果；全部失败则抛出最后一次的异常。
    """
    max_retries = CFG.max_tool_retries if max_retries is None else max_retries
    last_err: Exception | None = None

    for attempt in range(max_retries + 1):   # +1 是因为"首次尝试"不算重试
        try:
            result = fn()

            if validator is not None:
                problem = validator(result)
                if problem:
                    # 校验失败：抛出异常以复用下面的重试逻辑。
                    # 这样"异常失败"和"校验失败"走同一条重试路径，代码更统一。
                    raise ValueError(f"{desc} 校验未通过: {problem}")

            return result

        except Exception as e:      # noqa: BLE001 —— 这里确实需要捕获所有异常
            last_err = e
            if attempt < max_retries:
                wait = 0.3 * (attempt + 1)   # 退避：越往后等得越久
                if CFG.log_level != "quiet":
                    print(f"    [重试] {desc} 第 {attempt + 1} 次失败({e})，{wait:.1f}s 后重试")
                time.sleep(wait)

    raise RuntimeError(f"{desc} 在 {max_retries + 1} 次尝试后仍失败: {last_err}")


# ==============================================================================
# 第 7 部分：JSON 解析容错（给需要结构化输出的 Agent 用）
# ==============================================================================

def parse_json_loose(text: str) -> dict | list | None:
    """
    尽最大努力从模型输出里抠出 JSON。

    【为什么不能直接 json.loads】
        模型很喜欢这样输出：
            ```json
            {"verdict": "pass"}
            ```
        直接 json.loads 会报错，因为有 markdown 代码块围栏和多余文字。
        实战中必须有一层"宽松解析"。

    策略（从严格到宽松，逐级降级）：
        1. 直接解析
        2. 剥掉 ```json ... ``` 围栏再解析
        3. 正则找出第一个 { ... } 或 [ ... ] 片段再解析
        4. 全失败返回 None，由调用方决定降级策略
    """
    if not text:
        return None

    # 策略 1
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 策略 2：剥代码块围栏
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # 策略 3：找第一个花括号/方括号片段
    # 注意用非贪婪 + DOTALL，能匹配跨行的对象
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue

    # 策略 4：放弃
    return None
