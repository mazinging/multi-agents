"""
agents/critic.py —— 审校者 Agent
================================================================================
【角色定位：怀疑、挑错、找漏洞】

    认知模式是"主动寻找问题"，与 Writer 的"组织表达"完全对立。

【这个角色是多智能体架构里最有价值的部分，值得重点理解】

    为什么？因为它是【单智能体架构无法替代】的一环。

    单 Agent 写完就结束了，没有任何检查环节。它的失败是静默的 ——
    报告看起来完整、读起来通顺，但可能有编造的引用、
    可能漏掉了关键维度、可能前后矛盾。用户很难发现。

    多 Agent 架构里可以插入一个独立的 Critic，带来两个好处：
        1. 【独立视角】它没有"自己写的东西总是对的"这种偏见。
           模型很难批判自己的输出 —— 这是被反复验证的现象。
           换一个 Agent、换一个 prompt，批判能力就恢复了。
        2. 【结构化判定】它可以输出机器可读的结论，
           让程序据此决定流程分支（通过/打回重做）。

【本实现的 Critic 有两层校验，这是一个关键设计】

    第一层：程序化校验（确定性，不会出错）
        - 引用的 chunk_id 是否真实存在？
        - 引用的 ID 是否在 Researcher/Writer 实际看到过的资料范围内？
        这一层用集合运算完成，100% 可靠。任何一个无效引用都是硬伤。

    第二层：LLM 校验（主观判断，会出错）
        - 对比维度是否足够？
        - 结论是否有依据？
        - 落地建议是否可执行？
        这一层交给 LLM，因为它需要"理解"。

    【为什么必须两层都有？】
        只用 LLM：它可能忽略掉明显的引用错误
                  （LLM 对 ID 这种精确字符串的核对能力很弱）
        只用程序：无法判断"维度够不够"这类需要理解的问题

        【这是多智能体质量保证的核心思路：
          能用确定性代码做的校验，绝不交给模型；
          需要理解的判断，才交给模型。
          让程序和模型各做自己擅长的事。】

【为什么 MockLLM 下第一轮故意判不通过？】
    为了让学习者能看到"打回重做"的完整流程。
    如果 Mock 永远说通过，多智能体最有价值的那条路径就永远看不到。
    这是教学上的刻意设计 —— 详见 llm.py 的 MockLLM._critique。
================================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.base import CITATION_PATTERN, BaseReActAgent
from core.blackboard import (
    SLOT_DRAFT,
    SLOT_ANALYSIS,
    SLOT_CRITIQUE,
    SLOT_EVIDENCE,
    SLOT_PLAN,
    SLOT_USER_QUERY,
)
from llm import call_with_retry, parse_json_loose

if TYPE_CHECKING:
    from core.blackboard import Blackboard
    from core.trace import TraceRecorder
    from llm import LLMClient
    from rag.store import VectorStore


class CriticAgent(BaseReActAgent):
    """
    审校者：检查报告质量，输出结构化评审结论。

    【注意：这个类覆盖了 postprocess，并且重度使用 programmatic_check】
        它的产出不是一个"答案"，而是一个【判定】：
            verdict = "pass"   -> 报告合格，流程结束
            verdict = "revise" -> 报告有问题，带着修改意见打回重做
    """

    role = "critic"

    system_prompt = """你是一名严格的技术审校者（critic），负责检查报告质量。

## 你的职责
审查报告草稿，找出其中的问题，给出明确的通过/打回判定。

## 你的审查维度
1. **引用有效性**：每个关键结论是否都有来源标注？来源是否支持该结论？
2. **覆盖完整性**：是否满足验收标准？对比维度是否达到要求的数量？
3. **逻辑自洽性**：前后是否有矛盾？推荐结论是否与对比分析一致？
4. **可执行性**：落地建议是否具体可执行，而不是"建议进一步评估"这类空话？
5. **是否存在无依据的陈述**：有没有报告里写了、但证据中找不到的说法？

## 你的判定标准
- verdict = "pass"：报告质量达到可交付水平
- verdict = "revise"：存在需要修正的问题

【重要】你要严格但公正。
不要为了显示严格而挑无关紧要的毛病（比如措辞偏好）；
也不要因为报告读起来通顺就放过实质问题。
真正值得打回的问题是：编造的引用、缺失的关键维度、无依据的结论。

## 你必须输出的格式（严格遵守，这是机器要解析的）
{
  "verdict": "pass" 或 "revise",
  "score": 0-100 的整数,
  "issues": ["问题1", "问题2"],
  "suggestions": ["可执行的修改建议1", "修改建议2"],
  "reviewer_notes": "总体评语"
}

## 关于修改建议的要求
建议必须【具体可执行】。
反例："内容不够充分" —— 收到这条建议的人不知道该改什么。
正例："补充「扩展性」维度，说明数据量增长到千万级时各方案的表现" ——
      明确指出了缺什么、补什么。
你的建议质量直接决定了打回重做的效果。"""

    def __init__(
        self,
        llm: "LLMClient",
        store: "VectorStore",
        *,
        blackboard: "Blackboard | None" = None,
        tracer: "TraceRecorder | None" = None,
    ) -> None:
        # Critic 不需要工具：它的工作是完全基于已有材料做判断
        super().__init__(llm, None, blackboard=blackboard, tracer=tracer)
        self.store = store

    # ==========================================================================
    def build_task(self, **kwargs) -> tuple[str, str]:
        """构造审校任务，带上轮次信息（MockLLM 靠轮次决定判 pass 还是 revise）"""
        round_no = kwargs.get("round_no", 1)

        query = ""
        draft = ""
        evidence_citations: list[str] = []
        criteria: list[str] = []

        if self.blackboard:
            query = self.blackboard.read(SLOT_USER_QUERY, "")

            d = self.blackboard.read(SLOT_DRAFT, None)
            if isinstance(d, dict):
                draft = d.get("content", "")
            elif d:
                draft = str(d)

            ev = self.blackboard.read(SLOT_EVIDENCE, None)
            if isinstance(ev, dict):
                evidence_citations = ev.get("citations", [])

            plan = self.blackboard.read(SLOT_PLAN, None)
            if isinstance(plan, dict):
                criteria = plan.get("acceptance_criteria", [])

        # ---- 先跑程序化校验，把确定性结论直接给 LLM 看 ----
        # 【这个设计很关键】
        #   与其让 LLM 自己去找引用错误（它不擅长核对精确 ID），
        #   不如我们用代码查好，把结果作为"已知事实"告诉它。
        #   这样 LLM 只需要判断"这些已知问题有多严重"，
        #   而不是做它不擅长的精确匹配。
        #   【让每个组件做它擅长的事】—— 这是分层校验的精髓。
        prog = self.programmatic_check(draft, evidence_citations)

        task = (
            f"请审校以下技术报告（第 {round_no} 轮审校）。\n\n"
            f"原始问题：「{query}」\n\n"
            f"## 待审校的报告草稿\n{draft}\n"
        )

        parts = []
        if criteria:
            parts.append(
                "## 验收标准\n" + "\n".join(f"  - {c}" for c in criteria)
            )

        parts.append(
            "## 程序化校验结果（已由代码确认，请作为已知事实采信）\n"
            f"  - 报告中出现的引用总数: {prog['total_citations']}\n"
            f"  - 其中【无效引用】(知识库中不存在): {prog['invalid'] or '无'}\n"
            f"  - 其中【越界引用】(不在调研证据范围内): {prog['out_of_scope'] or '无'}\n"
            f"  - 调研阶段可用的来源共 {len(prog['available'])} 条\n"
            f"  - 报告是否包含 Markdown 表格: {'是' if prog['has_table'] else '否'}\n"
            f"  - 报告字数: {prog['length']}\n"
        )

        if prog["invalid"]:
            parts.append(
                "## ⚠ 严重问题：存在无效引用\n"
                f"以下引用 ID 在知识库中根本不存在，属于【编造引用】：\n"
                f"  {', '.join(prog['invalid'])}\n"
                "这是幻觉的直接证据，必须判定为 revise。"
            )

        if prog["out_of_scope"]:
            parts.append(
                "## ⚠ 问题：存在越界引用\n"
                f"以下引用 ID 虽然存在于知识库，但不在本次调研证据范围内：\n"
                f"  {', '.join(prog['out_of_scope'])}\n"
                "说明报告引入了未经调研的资料，请判定为 revise 或至少扣分。"
            )

        return task, "\n\n".join(parts)

    # ==========================================================================
    # 程序化校验 —— 本文件最重要的方法
    # ==========================================================================

    def programmatic_check(self, draft: str, evidence_citations: list[str]) -> dict:
        """
        用确定性代码检查引用有效性。

        【这个方法是"程序化防幻觉"的核心实现，请仔细理解】

        为什么不能只靠 LLM 审查引用？
            因为引用 ID 是形如 "02_vector_db_cost#4" 的精确字符串。
            让 LLM 在几百行报告里逐字核对每个 ID 是否真实存在，
            它的准确率很低 —— 这是 LLM 的固有弱点（精确字符串匹配）。
            实测中，LLM 经常会漏掉明显的编造引用。

            而用代码做这件事是 100% 可靠的：把所有出现的 ID 抓出来，
            和知识库里的 ID 集合做差集运算，一眼就能看出哪些是编造的。

        【两级校验的区别，非常重要】
            第一级：ID 是否存在于【知识库】中？
                    -> 不存在 = 凭空编造（最严重）
            第二级：ID 是否在【本次调研证据范围】内？
                    -> 存在但越界 = 报告引入了没调研过的资料
                       这也很严重，因为 Critic 无法确认这个资料
                       是否支持报告里的结论（调研员没看过它）

            第二级比第一级更隐蔽，也更容易被忽略。
            实现它的前提是：Researcher 必须记录它收集到的所有引用 ID。
            这体现了前面说的"上游输出格式必须满足下游校验需求"。

        【返回什么？】
            返回一个结构化的校验报告，会被拼进 Critic 的 prompt，
            作为"已知事实"提供给 LLM。
            这样 LLM 就不用自己去找引用错误了，只需要判断严重程度。
        """
        import re

        # 抓出报告里所有形如 [xxx#N] 的引用。
        #
        # 【这里的正则我踩过一个危险的 bug，务必理解】
        #   我最初写的是 r"\[([A-Za-z0-9_\-]+#\d+)\]"，
        #   只允许英文数字下划线连字符。
        #   结果：编造的引用 [99_不存在文档#7] 完全没被匹配到，
        #   程序化校验报告"invalid=[]"，编造引用被顺利放过。
        #
        #   这个 bug 极其危险，因为它是"安全检查静默失效"——
        #   校验器不报错，只是什么都没查到，看起来像是"一切正常"。
        #   如果没有 test_hallucination_guard.py 这个回归测试，
        #   它会一直潜伏到线上，而且永远不会有人发现。
        #
        #   修正后的字符类 [^\[\]#\s]+ 表示"除方括号、井号、空白外的任意字符"。
        #   这种"反向排除"的写法比"正向枚举允许字符"更稳健 ——
        #   出现新的字符类型时不会漏匹配。
        #   【原则：对安全检查类逻辑，宁可匹配得宽一些，
        #     让不该通过的东西被抓住并报错，也不要静默漏过。】
        #
        #   现在这个正则统一放在 base.CITATION_PATTERN，
        #   避免四个文件各写一遍、修一处漏三处。
        found = CITATION_PATTERN.findall(draft)
        total = len(found)
        unique_found = sorted(set(found))

        # ---- 第一级：ID 是否真实存在于知识库 ----
        # store.get_chunk() 是 O(n) 查找，但报告里的引用数量很少（几十条），
        # 而且只在 Critic 阶段调用一次，性能完全可以接受。
        # 【不要过早优化】如果这里要处理上万条引用，才需要考虑建索引。
        invalid = [cid for cid in unique_found if self.store.get_chunk(cid) is None]

        # ---- 第二级：是否越出本次调研范围 ----
        available = set(evidence_citations)
        # 只对"存在但不在证据范围内"的引用报越界
        out_of_scope = [
            cid for cid in unique_found
            if cid not in available and cid not in invalid
        ]

        return {
            "total_citations": total,
            "unique_citations": unique_found,
            "invalid": invalid,
            "out_of_scope": out_of_scope,
            "available": sorted(available),
            "has_table": "|" in draft and "---" in draft,
            "length": len(draft),
        }

    # ==========================================================================
    def postprocess(self, answer: str, steps) -> str:
        """
        把审校结论解析成结构化 dict 并写入黑板。

        【为什么必须结构化？】
            因为 Orchestrator 要根据 verdict 字段决定流程分支：
                pass   -> 结束流程，输出最终报告
                revise -> 带着 suggestions 打回给 Writer 重做

            如果 Critic 只输出一段自然语言评语，
            Orchestrator 就无法可靠地判断"到底通过没有"——
            它得去猜"这个评语看起来像是通过了吗"，这非常脆弱。

            【这是多智能体设计里一个反复出现的主题：
              凡是需要被程序消费的输出，都必须结构化，
              并且必须有解析容错。】

        【解析失败的降级策略】
            如果 JSON 解析失败（模型没按格式输出），不能直接崩。
            本实现的降级策略是：
                - 用关键词猜判定（"通过"/"pass" 出现在文本里 -> pass）
                - 猜不出来时【保守地判为 revise】
            为什么保守判 revise？
                因为"误判为通过"会让不合格的报告流出去，
                而"误判为 revise"只是多花一轮成本。
                在质量把关的环节，宁可严一点。
                【这是防幻觉场景下的通用原则：失败时倒向保守的一侧。】
        """
        parsed = parse_json_loose(answer)

        if isinstance(parsed, dict):
            verdict = str(parsed.get("verdict", "")).lower().strip()
            # 归一化各种可能的写法（模型可能写 "PASS"、"Pass"、"通过"）
            if verdict in ("pass", "通过", "ok", "approved"):
                verdict = "pass"
            elif verdict in ("revise", "revise_needed", "打回", "修改", "fail", "不通过"):
                verdict = "revise"
            else:
                # verdict 字段值不认识，用关键词兜底
                verdict = self._guess_verdict(answer)

            result = {
                "verdict": verdict,
                "score": self._safe_int(parsed.get("score"), default=60),
                "issues": self._as_list(parsed.get("issues")),
                "suggestions": self._as_list(parsed.get("suggestions")),
                "reviewer_notes": str(parsed.get("reviewer_notes", "")),
                "parsed_ok": True,
            }
        else:
            # 降级：JSON 解析失败
            result = {
                "verdict": self._guess_verdict(answer),
                "score": 50,
                "issues": [],
                "suggestions": [],
                "reviewer_notes": answer[:600],
                "parsed_ok": False,
            }

        # ---- 追加程序化校验结果 ----
        # 【为什么要把校验结果放进最终结论？】
        #   因为它比 LLM 的主观判断更可靠。
        #   如果代码发现了无效引用（编造引用），
        #   那么无论 LLM 说 pass 还是 revise，都必须是 revise。
        #   【程序化校验拥有否决权】—— 这是分层校验的关键规则：
        #   确定性的证据优先于模型的判断。
        if self.blackboard:
            draft = self.blackboard.read(SLOT_DRAFT, {})
            draft_text = draft.get("content", "") if isinstance(draft, dict) else str(draft)
            ev = self.blackboard.read(SLOT_EVIDENCE, {})
            ev_cites = ev.get("citations", []) if isinstance(ev, dict) else []

            check = self.programmatic_check(draft_text, ev_cites)
            result["programmatic"] = {
                "invalid_citations": check["invalid"],
                "out_of_scope_citations": check["out_of_scope"],
                "total_citations": check["total_citations"],
            }

            if check["invalid"]:
                result["verdict"] = "revise"
                result["issues"] = list(result["issues"]) + [
                    f"存在编造的引用（知识库中不存在）: {', '.join(check['invalid'])}"
                ]
                result["reviewer_notes"] = (
                    str(result["reviewer_notes"])
                    + " [程序化校验否决：检测到无效引用，强制判定为 revise]"
                )

            self.blackboard.write(
                SLOT_CRITIQUE,
                result,
                writer=self.role,
                note=f"判定 {result['verdict']}，得分 {result['score']}",
            )

        return answer

    # ==========================================================================
    # 解析辅助
    # ==========================================================================

    @staticmethod
    def _guess_verdict(text: str) -> str:
        """
        从自由文本里猜判定结果（JSON 解析失败时的降级手段）。

        【保守原则】猜不出来时返回 "revise"。
            宁可多审一轮，也不要放过有问题的报告。
            在质量把关环节，假阴性（漏过问题）比假阳性（多审一轮）代价大得多。
        """
        low = text.lower()
        if '"revise"' in low or '"pass"' in low or "不通过" in text or "需要修改" in text:
            # 有明确信号：优先看 revise（因为 revise 是更常见的明确表述）
            if "revise" in low or "不通过" in text or "需要修改" in text:
                return "revise"
            if "pass" in low or "通过" in text:
                return "pass"
        # 无法判断 -> 保守判 revise
        return "revise"

    @staticmethod
    def _safe_int(value, default: int = 60) -> int:
        """
        安全地转成整数。

        【为什么需要这个？】
            模型可能把 score 输出成 "82分"、"82/100"、82.5、"高" 等。
            直接 int() 会抛异常。这里做容错：先尝试直接转，
            失败则用正则抠出第一个数字。
            【凡是解析模型输出的地方，都要假设格式可能不对。】
        """
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            import re
            m = re.search(r"\d+", value)
            if m:
                return int(m.group(0))
        return default

    @staticmethod
    def _as_list(value) -> list[str]:
        """
        安全地转成字符串列表。

        【为什么需要？】模型有时把 issues 输出成字符串而不是数组：
            "issues": "维度不足"          <- 应该是 ["维度不足"]
            "issues": ["a", "b"]          <- 正常
            "issues": None                <- 可能
        三种都要能处理，否则下游遍历时会崩。
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, (list, tuple)):
            return [str(x) for x in value if str(x).strip()]
        return [str(value)]
