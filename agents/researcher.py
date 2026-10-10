"""
agents/researcher.py —— 调研员 Agent
================================================================================
【角色定位：发散、探索、求证】

    认知模式是"把相关信息尽可能找全，并标注来源"。
    它是整个流程的第一步，产出的【证据清单】是所有后续工作的基础。

【为什么这个角色值得独立成一个 Agent？】

    因为"找资料"和"用资料"需要完全不同的思维模式：
        找资料时：要发散，宁可多找一些，担心遗漏
        分析时：  要收敛，砍掉无关信息，聚焦结论
        写报告时：要面向读者，注重表达和组织

    让一个 Agent 同时具备"发散"和"收敛"两种倾向，效果一定打折 ——
    它要么找资料时就开始下结论（漏掉反例），
    要么写报告时还在堆砌资料（重点不明）。

    【这就是角色划分的核心依据：认知模式冲突，才需要拆分】。
    如果两个角色的思维模式一样，拆开只是增加成本。

【本角色的关键设计：强制引用溯源】
    Researcher 产出的每条证据都必须带 chunk_id。
    这不是"建议"，而是 system_prompt 里的硬性要求。
    为什么这么强调？
        因为后面 Critic 要靠这些 ID 做程序化校验 ——
        如果 Researcher 不记来源，整条防幻觉链条就断了。
    设计多智能体系统时，【上游的输出格式必须满足下游的校验需求】，
    这需要提前规划，而不是事后打补丁。
================================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.base import BaseReActAgent, extract_citations
from core.blackboard import (
    SLOT_EVIDENCE,
    SLOT_PLAN,
    SLOT_USER_QUERY,
)
from tools import build_default_registry

if TYPE_CHECKING:
    from core.blackboard import Blackboard
    from core.trace import TraceRecorder
    from llm import LLMClient
    from rag.store import VectorStore


class ResearcherAgent(BaseReActAgent):
    """
    调研员：检索资料并产出带引用的证据清单。
    """

    role = "researcher"

    # ==========================================================================
    # 角色提示词 —— 这是"人格"的核心
    # ==========================================================================
    # 【写角色提示词的几条实战经验，都在这里体现】
    #
    #   1. 说清"你是谁"和"你的产出是什么"
    #      模糊的人设（"你是一个有用的助手"）没有约束力。
    #      明确"你的产出是证据清单，每条必须有来源"才有约束力。
    #
    #   2. 明确"不该做什么"
    #      这比"该做什么"更重要。调研员最危险的倾向是
    #      "资料不够就自己补充常识"—— 那会直接导致幻觉。
    #      所以要明确禁止，并给出替代行为（如实说明资料不足）。
    #
    #   3. 给出可检验的输出格式
    #      "每条证据一行，格式：[来源ID] 内容摘要"
    #      比"请给出结构化的证据"有效得多，因为前者可检验。
    #
    #   4. 解释"为什么"，而不只是下命令
    #      现代 LLM 对"为什么"的理解能力很强。
    #      告诉它"引用来源是为了让结论可被验证"，
    #      比单纯命令"必须带来源"效果更好 —— 它会在边界情况下
    #      自己做出符合意图的判断。
    # ==========================================================================
    system_prompt = """你是一名严谨的技术调研员（researcher），负责为团队收集决策所需的资料。

## 你的唯一职责
从知识库中检索并整理【有来源依据】的事实性信息，形成证据清单。

## 你的产出要求（非常重要）
每一条证据都必须包含来源ID，格式如下：
  - [来源ID] 具体内容
例如：
  - [02_vector_db_cost#4] HNSW 索引的内存开销约为向量本体的 1.2 倍

## 你必须遵守的原则
1. 只报告知识库中真实存在的内容。绝对不要凭常识、经验或推测补充信息。
2. 如果知识库没有覆盖某个方面，必须如实说明"知识库未涉及此方面"，
   而不是用你的先验知识填补。这一点关系到整个团队的结论可信度。
3. 优先收集【具体数据】（数字、维度、成本、延迟、百分比），
   因为具体的数字比笼统的描述更有决策价值。
4. 如果发现不同文档的说法有出入，要如实指出这种矛盾，而不是自行选择其一。

## 工作方法建议
- 先用 list_knowledge_base 了解知识库覆盖范围，再针对性地检索
- 检索到候选文档后，用 read_doc_section 精读关键文档获取完整上下文
- 检索无结果时，换用更常见的技术术语重试，而不是重复同样的查询

## 什么时候结束
当你能对任务要求的各个方面都给出有来源支撑的信息时，输出 Final Answer。
Final Answer 应当是一份整理好的证据清单，而不是检索过程的复述。"""

    def __init__(
        self,
        llm: "LLMClient",
        store: "VectorStore",
        *,
        blackboard: "Blackboard | None" = None,
        tracer: "TraceRecorder | None" = None,
    ) -> None:
        # ---- 工具子集：调研员只需要"找资料"类工具 ----
        # 【为什么不给它全部工具？】
        #   1. prompt 更短，模型更聚焦（实测能提升工具选择的准确率）
        #   2. 从机制上防止它越界去做分析或写报告 —— 那会破坏分工，
        #      导致多个 Agent 重复劳动、或者分析做得比 Analyst 还粗糙
        #   3. cross_compare 明确不给：横向对比是 Analyst 的职责。
        #      如果调研员自己对比了，Analyst 就没事可做，架构就退化了
        #
        # 【这是"用工具权限实现角色隔离"的具体实践】
        registry = build_default_registry(store).subset([
            "list_knowledge_base",
            "search_local_docs",
            "read_doc_section",
            "extract_key_points",
        ])

        super().__init__(
            llm, registry, blackboard=blackboard, tracer=tracer,
            # 调研任务需要多轮检索，给它比其他角色更多的步数
            max_steps=7,
        )
        self.store = store

    # ==========================================================================
    def build_task(self, **kwargs) -> tuple[str, str]:
        """
        构造调研任务。

        【从黑板读什么？】
            SLOT_USER_QUERY : 用户的原始问题 —— 这是调研的根本目标
            SLOT_PLAN       : Orchestrator 的任务规划 —— 包含验收标准

        【为什么必须读 PLAN？】
            因为"验收标准"直接决定了调研的广度。
            如果标准要求"至少 3 个对比维度"，调研员就必须
            把成本、效果、运维三个方面的资料都找齐，
            而不是找到一方面就收工。
            【上游把验收标准传给下游，是保证最终产出合格的关键机制。】
            很多多智能体系统失败的原因是：下游不知道"什么算做好了"。
        """
        query = ""
        plan_text = ""
        if self.blackboard:
            query = self.blackboard.read(SLOT_USER_QUERY, "")
            plan = self.blackboard.read(SLOT_PLAN, None)
            if plan:
                # 把规划转成文本。它可能是 dict，也可能是字符串
                if isinstance(plan, dict):
                    import json
                    plan_text = json.dumps(plan, ensure_ascii=False, indent=2)
                else:
                    plan_text = str(plan)

        task = (
            f"请围绕以下问题开展资料调研：\n\n「{query}」\n\n"
            f"你的目标是产出一份【带来源ID的证据清单】，"
            f"覆盖回答该问题所需的各个方面。"
        )

        context_parts = []
        if plan_text:
            context_parts.append(
                f"## 任务规划与验收标准（由协调者制定，你必须满足这些标准）\n{plan_text}"
            )

        context = "\n\n".join(context_parts)
        return task, context

    # ==========================================================================
    def postprocess(self, answer: str, steps) -> str:
        """
        后处理：把证据清单写入黑板。

        【为什么要在 postprocess 里写黑板，而不是在 run() 之后由外部写？】
            因为"什么算这个 Agent 的产出"只有 Agent 自己最清楚。
            让 Agent 负责把自己的成果放到该放的地方，
            调用方就不需要知道"Researcher 的产出该存到哪个槽位" ——
            这是"信息隐藏"原则的体现，降低了 Orchestrator 的耦合度。

        【同时收集引用 ID】这些 ID 会随证据一起存进黑板，
        供 Critic 后续做引用有效性校验。
        """
        # 从答案里提取所有 [xxx#N] 形式的引用。
        # 【注意】这里用 base.py 里的公共函数，而不是自己写正则 ——
        #   见 base.py 中 CITATION_PATTERN 的注释：
        #   我自己写的正则版本漏匹配了含中文的 ID，
        #   而且同样的 bug 在四个文件里都存在，修起来容易漏。
        #   统一实现是唯一可靠的做法。
        cited = extract_citations(answer)

        if self.blackboard:
            self.blackboard.write(
                SLOT_EVIDENCE,
                {
                    "content": answer,
                    "citations": cited,
                },
                writer=self.role,
                note=f"收集到 {len(cited)} 条带来源的证据",
            )

        return answer
