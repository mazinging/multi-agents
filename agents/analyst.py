"""
agents/analyst.py —— 分析师 Agent
================================================================================
【角色定位：收敛、对比、归纳】

    认知模式是"从杂乱的证据中提炼出结构化的判断"。
    它消费 Researcher 的证据清单，产出对比分析和推荐结论。

【角色划分的对比：为什么 Analyst 不能和 Researcher 合并？】

    Researcher 的倾向是"发散"：
        看到什么相关就收集什么，宁可多不可少。
        它的失败模式是"遗漏"。

    Analyst 的倾向是"收敛"：
        必须做减法，砍掉不相关的维度，聚焦到能支撑决策的少数几点。
        它的失败模式是"抓不住重点"。

    这两个倾向是【对立】的。如果合成一个 Agent：
        - 要么它一直保持发散，产出一堆罗列但没有结论
        - 要么它过早收敛，资料没找全就开始下结论
    两种都是常见且难以修复的问题。

    【把对立倾向的认知模式拆到不同 Agent，是多智能体最实在的价值之一。
      不是为了"看起来更专业"，而是因为一个 prompt 装不下相互冲突的要求。】

【本角色的核心产出要求：必须有"对比维度"】
    这是从 Critic 的验收标准倒推出来的。
    分析的价值在于"让差异可被观察" —— 而差异必须在明确的维度上才有意义。
    "A 比 B 好"是无意义的判断；"在运维成本这一维度上，A 优于 B"才有决策价值。
================================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.base import BaseReActAgent, extract_citations
from core.blackboard import SLOT_ANALYSIS, SLOT_EVIDENCE, SLOT_PLAN, SLOT_USER_QUERY
from tools import build_default_registry

if TYPE_CHECKING:
    from core.blackboard import Blackboard
    from core.trace import TraceRecorder
    from llm import LLMClient
    from rag.store import VectorStore


class AnalystAgent(BaseReActAgent):
    """
    分析师：基于证据做多维度对比分析，给出推荐结论。
    """

    role = "analyst"

    system_prompt = """你是一名技术分析师（analyst），负责把调研证据转化为可支撑决策的分析结论。

## 你的职责
基于【调研员提供的证据】做横向对比分析，输出结构化的判断和推荐。

## 你的产出要求（非常重要）
你的分析必须包含：
1. **对比维度**：明确列出至少 3 个对比维度（如成本、效果、运维复杂度、扩展性等）
2. **每个维度的逐项对比**：说明各候选方案在该维度上的表现差异
3. **推荐结论**：给出明确推荐，并说明理由
4. **落地建议**：给出可执行的第一步行动（不要只说"建议评估"这种空话）

## 你必须遵守的原则
1. 【只能基于已有证据分析】。你的已知信息里包含了调研员收集的证据。
   如果某个维度缺乏证据支持，必须明确标注"该维度缺乏资料"，
   而不要用你的先验知识补充。这是团队结论可信度的基础。
2. 【不要重复罗列证据】。调研员已经列过原始资料了，
   你的价值在于【归纳和对比】，而不是复制粘贴。
3. 判断要有依据。不要写"A 更好"，要写"在运维复杂度这一维度上，
   A 优于 B，因为……（依据来源）"。
4. 如果证据之间存在矛盾，要指出矛盾并说明这如何影响结论的可靠性。

## 什么是好的分析（正例 vs 反例）
反例（不合格）：
  "Milvus 功能强大，Pgvector 比较简单，建议根据需求选择。"
  —— 没有维度、没有依据、没有明确推荐，等于没说。

正例（合格）：
  "从【运维复杂度】维度看：Pgvector 复用现有 PostgreSQL，
   无需额外组件（依据 [02_vector_db_cost#6]）；Milvus 需要维护
   etcd、MinIO、Pulsar 等多个组件（依据 [01_vector_db_overview#7]）。
   因此对于无专职运维的小团队，Pgvector 明显更优。"

## 工作方法
- 你可以用 extract_key_points 从证据里提炼要点
- 用 cross_compare 对多个候选方案做并列检索，发现维度差异
- 先确定对比维度，再逐维度分析，最后综合推荐

## 什么时候结束
当你能给出包含【3个以上维度 + 明确推荐 + 落地建议】的完整分析时，
输出 Final Answer。"""

    def __init__(
        self,
        llm: "LLMClient",
        store: "VectorStore",
        *,
        blackboard: "Blackboard | None" = None,
        tracer: "TraceRecorder | None" = None,
    ) -> None:
        # ---- 工具子集：分析师需要"对比"和"补充查询"能力 ----
        # 【关键设计：给它 cross_compare 但不给它 search_local_docs】
        #
        #   为什么要给 cross_compare？
        #       这是分析型任务的核心工具，"并列展示"是发现差异的基础操作。
        #
        #   为什么【不给】search_local_docs？
        #       这是一个刻意的限制。如果给分析师自由检索的能力，
        #       它会倾向于"自己再查一遍"来补全缺的维度 ——
        #       这看起来合理，实际上破坏了架构：
        #           a) 重复劳动（调研员已经查过了）
        #           b) 分析师的检索往往不如调研员充分（它步数更少）
        #           c) 最关键：如果分析师自己找资料，
        #              "证据"和"分析"就混在一起了，Critic 无法区分
        #              "这条结论有调研证据支持"还是"分析师自己找的"
        #              引用校验链条就断了
        #
        #       正确做法是：分析师发现证据不足时，应该【如实指出】，
        #       由 Orchestrator 决定是否让调研员补充调研。
        #       这个"发现问题的人不负责解决问题"的原则，
        #       保证了各环节职责清晰、可追溯。
        registry = build_default_registry(store).subset([
            "cross_compare",
            "extract_key_points",
            "read_doc_section",
        ])

        super().__init__(
            llm, registry, blackboard=blackboard, tracer=tracer,
            max_steps=5,
        )
        self.store = store

    # ==========================================================================
    def build_task(self, **kwargs) -> tuple[str, str]:
        """
        构造分析任务。

        【为什么要把完整证据传给它，而不是传摘要？】
            这一点值得特别说明，因为它违反了"最小必要上下文"原则。

            通常我们会做上下文裁剪以避免 prompt 膨胀。但这里【不能裁】：
            分析师的职责就是"基于证据做判断"，
            如果证据被摘要过，它就可能遗漏关键维度 ——
            而遗漏恰恰是分析任务最严重的失败模式。
            摘要过程本身也会丢失信息（谁来保证摘要没漏掉重要的点？）。

            【判断标准】什么时候该给完整上下文？
                当接收方的任务是"必须看到全部细节才能做对"时，给完整的。
                当接收方的任务是"基于结论做决策"时，给摘要。
                分析师属于前者，Writer 属于两者之间。

            【这是一个很实际的架构判断，没有统一答案，
              要问"如果裁剪了，接收方还能完成它的职责吗？"】
        """
        query = ""
        evidence_text = ""
        plan_note = ""

        if self.blackboard:
            query = self.blackboard.read(SLOT_USER_QUERY, "")

            evidence = self.blackboard.read(SLOT_EVIDENCE, None)
            if isinstance(evidence, dict):
                evidence_text = evidence.get("content", "")
            elif evidence:
                evidence_text = str(evidence)

            plan = self.blackboard.read(SLOT_PLAN, None)
            if isinstance(plan, dict):
                criteria = plan.get("acceptance_criteria", [])
                if criteria:
                    plan_note = "验收标准（你的分析必须满足）：\n" + "\n".join(
                        f"  - {c}" for c in criteria
                    )

        task = (
            f"请基于调研证据，对以下问题做多维度对比分析：\n\n「{query}」\n\n"
            f"要求：至少 3 个对比维度 + 明确推荐结论 + 可执行的落地建议。"
        )

        context_parts = []
        if plan_note:
            context_parts.append(plan_note)
        if evidence_text:
            context_parts.append(f"## 调研员收集的证据（你的分析只能基于这些）\n{evidence_text}")
        else:
            # 【重要：证据缺失时的处理】
            # 不静默继续，而是明确告知。否则分析师会开始"自己编"。
            # 这类"上游数据缺失"的情况在多智能体里很常见
            # （比如上游失败了、或者流程被中断了），
            # 必须有明确的降级处理，而不是让它带着空上下文硬跑。
            context_parts.append(
                "## ⚠ 警告：没有收到调研证据\n"
                "上游调研员未能提供证据。你【不应该】凭常识编造分析。\n"
                "请在最终结论中明确说明：由于缺乏调研资料，无法给出有依据的分析。"
            )

        context = "\n\n".join(context_parts)
        return task, context

    # ==========================================================================
    def postprocess(self, answer: str, steps) -> str:
        """把分析结论写入黑板（附上引用，供 Critic 校验）"""
        # 用 base.py 的公共引用提取函数（见其中关于正则 bug 的说明）
        cited = extract_citations(answer)

        if self.blackboard:
            self.blackboard.write(
                SLOT_ANALYSIS,
                {
                    "content": answer,
                    "citations": cited,
                },
                writer=self.role,
                note=f"分析结论，引用 {len(cited)} 条来源",
            )

        return answer
