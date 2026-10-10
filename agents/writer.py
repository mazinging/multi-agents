"""
agents/writer.py —— 撰写者 Agent
================================================================================
【角色定位：组织、表达、面向读者】

    认知模式是"把结构化的结论变成易读的文章"。
    它消费证据 + 分析，产出最终报告。

【这个角色的特殊之处：它不需要工具，不走 ReAct 循环】

    这是本项目里唯一一个"纯生成型" Agent。
    为什么？因为它的工作本质是【加工已有信息】，而不是【获取新信息】：
        输入：证据清单 + 分析结论（都已经在黑板上了）
        输出：组织好的报告

    让它去检索反而是错的 —— 那会破坏分工，也会让它偏离
    "忠实呈现证据和结论"这个职责（自己检索来的东西会掺入新判断，
    而它不该做判断，那是 Analyst 的活）。

    【工程价值】不走 ReAct 循环意味着：
        少 1~5 次 LLM 调用 -> 更快、更便宜
    不要为了"架构统一"让所有 Agent 都走复杂路径。
    【按任务本质选择最简实现，是很重要的工程判断力】。
    这是 BaseReActAgent 里那条 `if not self.registry.names` 分支的用途。

【本角色的核心约束：不许引入新事实】
    这是防幻觉链条上容易被忽略的一环。
    Writer 的诱惑是"为了让文章流畅，自己补充一些过渡性的常识"。
    比如证据只说"Pgvector 运维简单"，它可能写成
    "Pgvector 运维简单，因为它基于 PostgreSQL，而团队通常已经熟悉 SQL"
    —— 后半句是它自己加的，证据里没有。
    这种"善意的补充"正是最隐蔽的幻觉来源，因为它读起来非常合理。

    所以 prompt 里必须明确禁止，并要求它标注引用。
================================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.base import BaseReActAgent, extract_citations
from core.blackboard import (
    SLOT_ANALYSIS,
    SLOT_DRAFT,
    SLOT_EVIDENCE,
    SLOT_PLAN,
    SLOT_USER_QUERY,
)

if TYPE_CHECKING:
    from core.blackboard import Blackboard
    from core.trace import TraceRecorder
    from llm import LLMClient


class WriterAgent(BaseReActAgent):
    """
    撰写者：把证据与分析组织成结构化报告。

    【注意构造函数：没有 store 参数】
        因为它不需要检索，也就不需要知识库。
        这个签名本身就说明了它的职责边界 ——
        【接口设计可以传达架构约束】。
        如果有人想给 Writer 加检索能力，看到构造函数没有 store
        就会意识到"这个类在设计上就不该碰知识库"。
    """

    role = "writer"

    system_prompt = """你是一名技术文档撰写者（writer），负责把团队的调研与分析整理成报告。

## 你的职责
将【调研证据】和【分析结论】组织成一份结构清晰、可直接交付的技术报告。

## 你必须遵守的原则（最重要的一条）
【绝对不要在报告中引入任何新的技术事实】。
你只能使用团队已经收集到的证据和分析结论。
即使你觉得补充某句话能让文章更流畅、更完整，也不要补 ——
因为那句话没有来源支撑，会污染整个报告的可信度。
你可以在已有事实之间建立逻辑连接（"因此""相比之下"），
但不能添加新的事实性陈述。

## 报告结构要求
1. **背景与目标**：简述要解决的问题
2. **候选方案概览**：列出涉及的方案/技术
3. **多维度对比**：这是核心章节，必须保留分析师给出的对比维度
   （建议用表格形式呈现，便于阅读）
4. **推荐结论**：明确给出推荐，并说明依据
5. **落地建议**：给出可执行的第一步
6. **引用来源**：列出报告中引用的所有来源ID

## 格式要求
- 使用 Markdown 格式
- 关键结论后面必须标注来源，格式如 [02_vector_db_cost#4]
- 表格用 Markdown 表格语法
- 语言简洁专业，避免空洞的套话

## 特别注意
如果分析结论中标注了"某维度缺乏资料"，你必须在报告中
如实保留这个说明，不能为了让报告看起来完整而隐去它。
诚实地呈现信息边界，比制造完整的假象更有价值。"""

    def __init__(
        self,
        llm: "LLMClient",
        *,
        blackboard: "Blackboard | None" = None,
        tracer: "TraceRecorder | None" = None,
    ) -> None:
        # registry 传空 -> BaseReActAgent 会自动走"无工具"路径
        super().__init__(llm, None, blackboard=blackboard, tracer=tracer)

    # ==========================================================================
    def build_task(self, **kwargs) -> tuple[str, str]:
        """
        构造撰写任务。

        【注意这里传的是"证据 + 分析"两者，而不是只有分析】
            为什么两者都要？
                Analyst 出于篇幅考虑，会把证据压缩成结论。
                但报告里有时需要引用原始数据（比如"6.5 GB 内存"这个数字），
                而分析结论里可能只写了"内存开销较大"。
                给 Writer 原始证据，它才能写出有说服力的具体内容。

            【这又回到了"给完整还是给摘要"的判断】
                Writer 需要具体数据来支撑论述，所以证据要给。
                但给的方式可以是"证据 + 分析"并列，
                让它自己决定哪里用数据、哪里用结论。

        【反面案例：只给分析结论会怎样？】
            报告会变成全是"某某方案较好"这种空泛的判断，
            没有任何可验证的具体内容 —— 读起来像 AI 生成的套话。
            这正是很多 RAG 报告质量差的原因：中间层把信息压得太狠了。
        """
        query = ""
        evidence_text = ""
        analysis_text = ""
        criteria_text = ""

        if self.blackboard:
            query = self.blackboard.read(SLOT_USER_QUERY, "")

            evidence = self.blackboard.read(SLOT_EVIDENCE, None)
            if isinstance(evidence, dict):
                evidence_text = evidence.get("content", "")
            elif evidence:
                evidence_text = str(evidence)

            analysis = self.blackboard.read(SLOT_ANALYSIS, None)
            if isinstance(analysis, dict):
                analysis_text = analysis.get("content", "")
            elif analysis:
                analysis_text = str(analysis)

            plan = self.blackboard.read(SLOT_PLAN, None)
            if isinstance(plan, dict):
                crit = plan.get("acceptance_criteria", [])
                if crit:
                    criteria_text = "\n".join(f"  - {c}" for c in crit)

        task = (
            f"请撰写一份关于以下问题的技术选型报告：\n\n「{query}」\n\n"
            f"报告需要结构完整、结论明确、所有关键结论都标注来源。"
        )

        parts = []
        if criteria_text:
            parts.append(f"## 验收标准（报告必须满足）\n{criteria_text}")
        if evidence_text:
            parts.append(f"## 调研证据（可引用的原始资料）\n{evidence_text}")
        if analysis_text:
            parts.append(f"## 分析结论（报告的核心内容骨架）\n{analysis_text}")

        if not evidence_text and not analysis_text:
            parts.append(
                "## ⚠ 警告：没有收到任何证据和分析\n"
                "你无法撰写有依据的报告。请如实说明资料不足，不要编造内容。"
            )

        return task, "\n\n".join(parts)

    # ==========================================================================
    def postprocess(self, answer: str, steps) -> str:
        """把草稿写入黑板"""
        # 用 base.py 的公共引用提取函数（见其中关于正则 bug 的说明）
        cited = extract_citations(answer)

        if self.blackboard:
            self.blackboard.write(
                SLOT_DRAFT,
                {
                    "content": answer,
                    "citations": cited,
                },
                writer=self.role,
                note=f"报告草稿 {len(answer)} 字符，{len(cited)} 条引用",
            )

        return answer
