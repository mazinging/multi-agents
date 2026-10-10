"""
agents/orchestrator.py —— 协调者 Agent
================================================================================
【角色定位：拆解任务、派活、汇总、判定返工】

    它是整个多智能体系统的"大脑"，但不做具体工作。
    它决定"谁在什么时候做什么"，然后检查做得怎么样。

【这里有一个重要的架构选择，值得仔细说明】

    本项目【没有】让 Orchestrator 走 ReAct 循环（BaseReActAgent 的循环）。
    它是一个纯粹的"控制流"组件，用普通 Python 代码实现。

    为什么？
        因为它要做的事是确定性的流程控制：
            拆解 -> 派给调研员 -> 派给分析师 -> 派给撰写者 -> 审校
            -> 通过则结束，不通过则打回
        这个流程是固定的、可预先确定的。
        用一个 ReAct 循环（让 LLM 自己决定下一步调哪个 Agent）
        来做这件事，会带来三个问题：
            1. 不确定：同一个任务，LLM 可能这次先派给分析师，下次先派给调研员
            2. 无法保证完成：LLM 可能忘记调用某个必需的环节
            3. 成本更高：每一轮都要调 LLM 来决定"下一步干什么"

    【什么时候该用 LLM 做控制流，什么时候该用代码？】
        用代码：流程是【已知且固定】的。此时 LLM 只会引入不确定性。
        用 LLM：流程需要【根据中间结果动态调整】的。
                比如"如果调研发现资料严重不足，就跳过分析直接告知用户"
                —— 这种分支难以穷举，适合让 LLM 判断。

    本项目的混合做法：
        主流程用代码固定（保证可靠、可控、可调试）
        任务【拆解】用 LLM（这部分需要理解用户意图，是创造性的）
        【返工判定】用代码读 Critic 的结构化结论（确定性的分支）

    【这是我在本项目里最想传达的架构判断之一：
      不要因为"用了 LLM"就以为所有决策都该交给 LLM。
      把确定性的事情用代码写死，把不确定的事情交给模型，
      这样得到的系统既可靠又灵活。】

【协作模式：协调者 + 黑板】
    本项目采用混合模式（见 knowledge/07_multi_agent_arch.md）：
        - Orchestrator 控制【流程】（协调者模式的清晰性）
        - Agent 之间通过 Blackboard 交换【数据】（黑板模式的解耦性）

    为什么不让 Orchestrator 把数据也传来传去？
        因为那样它就得知道"Researcher 的产出格式是什么"、
        "Writer 需要什么格式的输入" —— 所有人改动都要改它，
        它会变成一个巨大的、脆弱的中心节点。
        数据走黑板，Orchestrator 只管"谁来执行"，职责就干净了。
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config import CFG
from core.blackboard import (
    SLOT_CRITIQUE,
    SLOT_DRAFT,
    SLOT_EVIDENCE,
    SLOT_FINAL,
    SLOT_PLAN,
    SLOT_REVISION_HISTORY,
    SLOT_USER_QUERY,
    Blackboard,
)
from core.message import Message, MsgKind
from llm import LLMClient, call_with_retry, parse_json_loose

if TYPE_CHECKING:
    from agents.analyst import AnalystAgent
    from agents.critic import CriticAgent
    from agents.researcher import ResearcherAgent
    from agents.writer import WriterAgent
    from core.trace import TraceRecorder


# ==============================================================================
# 流程结果
# ==============================================================================

@dataclass
class PipelineResult:
    """
    一次完整协作流程的结果。

    【为什么要记录 rounds 和 history？】
        因为"返工了几轮"是衡量系统健康度的重要指标：
            总是 1 轮就通过 -> Critic 可能太宽松，或者在走过场
            总是 3 轮才通过 -> 上游质量太差，或者 Critic 太苛刻
        这个数据在调试和调优时非常有价值。
        没有它，你只能看到"最后报告好不好"，
        看不到"这个过程是否健康"。
    """
    topic: str = ""
    final_report: str = ""
    rounds: int = 0
    passed: bool = False
    history: list[dict] = field(default_factory=list)
    messages: int = 0

    def summary(self) -> str:
        status = "通过" if self.passed else "未通过"
        return (
            f"流程结束 | {self.rounds} 轮 | 审校{status} | "
            f"黑板消息 {self.messages} 条 | 报告 {len(self.final_report)} 字符"
        )


# ==============================================================================
# 协调者
# ==============================================================================

class OrchestratorAgent:
    """
    协调者：编排整个多智能体协作流程。

    【注意它不继承 BaseReActAgent】
        这是一个刻意的设计，理由是上面说的：
        协调工作是确定性的控制流，不需要 ReAct 循环。
        【接口应该反映职责本质】—— 如果它继承 ReAct 基类，
        会给人"它也是一个会思考的 Agent"的错觉，
        而实际上它更像一个"工作流引擎"。
    """

    role = "orchestrator"

    # 任务拆解的 prompt。
    # 【为什么要求输出 JSON？】
    #   因为拆解结果要被程序消费（Orchestrator 要遍历 sub_tasks 决定派活顺序）。
    #   自然语言无法可靠解析。
    # 【凡是需要程序消费的输出，必须结构化 + 容错解析】—— 这是贯穿本项目的原则。
    PLAN_PROMPT = """你是一个多智能体系统的协调者（orchestrator）。

用户提出了以下问题：
「{query}」

请你把回答这个问题所需的工作拆解成若干子任务，分派给以下专业角色：

- researcher : 调研员，负责检索资料、收集带来源的证据
- analyst    : 分析师，负责基于证据做多维度对比分析
- writer     : 撰写者，负责把证据与分析组织成报告

## 要求
1. 子任务应该按【执行顺序】排列（后面的任务依赖前面的产出）
2. 每个子任务说明：负责角色、具体目标
3. 给出【验收标准】——什么样的产出算合格。这很重要，
   下游 Agent 会依据它判断自己是否做到位了。

## 输出格式（严格遵守，这是机器要解析的）
{{
  "topic": "一句话概括主题",
  "sub_tasks": [
    {{"id": "T1", "assignee": "researcher", "goal": "具体目标"}},
    {{"id": "T2", "assignee": "analyst", "goal": "具体目标"}},
    {{"id": "T3", "assignee": "writer", "goal": "具体目标"}}
  ],
  "acceptance_criteria": ["标准1", "标准2", "标准3"]
}}

只输出 JSON，不要有其他内容。"""

    def __init__(
        self,
        llm: LLMClient,
        *,
        researcher: "ResearcherAgent",
        analyst: "AnalystAgent",
        writer: "WriterAgent",
        critic: "CriticAgent",
        blackboard: Blackboard | None = None,
        tracer: "TraceRecorder | None" = None,
    ) -> None:
        self.llm = llm
        self.bb = blackboard or Blackboard()
        self.tracer = tracer

        # ---- 持有各个角色的引用 ----
        # 【为什么在这里注入，而不是在 Orchestrator 内部 new 出来？】
        #   这叫"依赖注入"（Dependency Injection）。
        #   好处：
        #       1. 可以替换实现（比如测试时传入 mock 的 Researcher）
        #       2. 各个 Agent 的构造在外部完成，Orchestrator 不需要
        #          知道它们需要什么参数（比如 store、工具集）
        #       3. 依赖关系显式化 —— 看构造函数就知道 Orchestrator
        #          依赖哪四个角色，一目了然
        #   【如果内部 new，Orchestrator 就和具体实现强耦合了，
        #     而且依赖关系被藏在代码里，很难看清架构。】
        self.researcher = researcher
        self.analyst = analyst
        self.writer = writer
        self.critic = critic

        # 把四个角色的黑板统一到 Orchestrator 的黑板上。
        # 【这一步很容易被忽略，但非常关键】
        #   如果各 Agent 持有不同的 Blackboard 实例，
        #   它们就看不到彼此的数据 —— 协作会完全失效，
        #   而且症状很隐蔽（每个 Agent 都正常执行，只是互相看不见）。
        #   在构造函数里统一，能避免这个陷阱。
        for agent in (self.researcher, self.analyst, self.writer, self.critic):
            agent.blackboard = self.bb

        # 让轨迹记录器订阅黑板消息（旁路记录，不影响业务）
        if self.tracer:
            self.bb.subscribe(self.tracer)

    # ==========================================================================
    # 主流程
    # ==========================================================================

    def run(self, query: str) -> PipelineResult:
        """
        执行完整的协作流程。

        ── 流程 ──
            0. 记录用户问题到黑板
            1. 拆解任务（LLM）
            2. 派发给 Researcher  -> 收集证据
            3. 派发给 Analyst     -> 对比分析
            4. 进入"写作-审校"循环：
                 a. 派发给 Writer -> 产出草稿
                 b. 交给 Critic 审校
                 c. 通过 -> 结束；不通过 -> 带着修改意见回到 a
        """
        result = PipelineResult(topic=query)

        # ---- 第 0 步：把用户问题写进黑板 ----
        # 【为什么放黑板而不是直接传给各 Agent？】
        #   因为每个 Agent 都需要知道"用户到底问的是什么"。
        #   如果靠参数传递，就要在每个调用点都传一遍，很容易漏。
        #   放黑板后，Agent 在自己的 build_task 里按需读取。
        #   这体现了黑板模式的核心价值：共享上下文不用重复传递。
        self.bb.write(SLOT_USER_QUERY, query, writer="user", note="用户原始问题")
        self.bb.post(Message(
            sender="user", recipient="orchestrator",
            kind=MsgKind.TASK, payload={"goal": query},
        ))

        if self.tracer:
            self.tracer.record_event("pipeline_start", query=query)

        # ---- 第 1 步：拆解任务 ----
        plan = self.plan(query)
        self.bb.write(SLOT_PLAN, plan, writer=self.role, note="任务规划与验收标准")

        # ---- 第 2 步：调研 ----
        # 【注意这里的调用方式：build_task() 返回 (task, context) 元组，
        #   而 run() 的 context 是 keyword-only 参数，所以必须显式解包传递。】
        #   这里不用 `run(*build_task())` 是因为那样会把 context 当成
        #   位置参数传进去。用解包+关键字参数最清晰：
        self._assign("researcher", plan, self.researcher)
        r_task, r_ctx = self.researcher.build_task()
        research_result = self.researcher.run(r_task, context=r_ctx)
        self._report_back("researcher", research_result, plan)

        # ---- 第 3 步：分析 ----
        self._assign("analyst", plan, self.analyst)
        a_task, a_ctx = self.analyst.build_task()
        analysis_result = self.analyst.run(a_task, context=a_ctx)
        self._report_back("analyst", analysis_result, plan)

        # ---- 第 4 步：写作-审校循环 ----
        round_no = 0
        passed = False

        while round_no < CFG.max_revision_rounds:
            round_no += 1

            if CFG.log_level != "quiet":
                print(f"\n{'═' * 72}")
                print(f"  第 {round_no} 轮：撰写 -> 审校")
                print(f"{'═' * 72}")

            # ---- 4a. 撰写（或基于修改意见返工）----
            self._assign("writer", plan, self.writer, round_no=round_no)

            task, context = self.writer.build_task()
            if round_no > 1:
                # 【返工的关键：把上一轮的问题和建议明确传给 Writer】
                #   如果不传，Writer 会产出和上一轮几乎一样的内容 ——
                #   它不知道自己哪里做得不好，只能重新猜一遍。
                #   实测中，"不传修改意见的返工"基本上等于白跑一轮。
                context = self._build_revision_context(context, round_no)

            writer_result = self.writer.run(task, context=context)
            self._report_back("writer", writer_result, plan, round_no=round_no)

            # ---- 4b. 审校 ----
            self._assign("critic", plan, self.critic, round_no=round_no)
            ctask, ccontext = self.critic.build_task(round_no=round_no)
            critic_result = self.critic.run(ctask, context=ccontext)
            self._report_back("critic", critic_result, plan, round_no=round_no)

            # ---- 4c. 读取判定结果 ----
            # 【注意这里：判定用代码读黑板，而不是解析 Critic 的自然语言输出】
            #   因为 Critic 的 postprocess 已经把结论结构化了，
            #   直接读字段是可靠的。如果去解析自然语言，
            #   就会重新引入"猜测"的脆弱性。
            #   【让上游负责结构化，下游就能简单可靠地消费。】
            critique = self.bb.read(SLOT_CRITIQUE, {})
            verdict = critique.get("verdict", "revise") if isinstance(critique, dict) else "revise"

            record = {
                "round": round_no,
                "verdict": verdict,
                "score": critique.get("score") if isinstance(critique, dict) else None,
                "issues": critique.get("issues", []) if isinstance(critique, dict) else [],
            }
            result.history.append(record)

            if CFG.log_level != "quiet":
                print(f"\n  📋 第 {round_no} 轮审校结果: {verdict} "
                      f"(得分 {record['score']})")
                for issue in record["issues"][:3]:
                    print(f"      - {issue}")

            # ---- 4d. 分支判定 ----
            if verdict == "pass":
                passed = True
                break

            if round_no >= CFG.max_revision_rounds:
                # 【达到返工上限怎么办？这是很重要的设计决策】
                #   选项 A：继续循环 -> 可能无限循环，成本失控
                #   选项 B：直接失败 -> 浪费了所有已完成的工作
                #   选项 C（本项目采用）：接受当前版本，但明确标记"未通过审校"
                #
                #   为什么选 C？
                #       因为部分可用的产出 > 什么都没有。
                #       而且明确标记能让使用者知道"这份报告有已知问题"，可以自行判断。
                #       这比悄悄交付一份不合格报告（瞒报）或直接丢弃（浪费）都好。
                #   【这就是"优雅降级"原则在流程层的应用。】
                if CFG.log_level != "quiet":
                    print(f"\n  ⚠ 已达最大返工轮次 {CFG.max_revision_rounds}，"
                          f"接受当前版本（标记为未通过审校）")
                break

        # ---- 第 5 步：输出最终结果 ----
        draft = self.bb.read(SLOT_DRAFT, {})
        final_text = draft.get("content", "") if isinstance(draft, dict) else str(draft)

        # 在报告头部加上流程元信息。
        # 【为什么要加？】让使用者知道这份报告的生成过程是否顺利。
        #   如果显示"未通过审校"，读者就会带着更多怀疑去读 —— 这是诚实的做法。
        header = (
            f"<!-- 生成信息: {round_no} 轮迭代 | "
            f"审校{'通过' if passed else '未通过'} | "
            f"黑板消息 {self.bb.message_count} 条 -->\n\n"
        )
        final_report = header + final_text

        self.bb.write(SLOT_FINAL, final_report, writer=self.role,
                      note=f"最终报告，审校{'通过' if passed else '未通过'}")

        result.final_report = final_report
        result.rounds = round_no
        result.passed = passed
        result.messages = self.bb.message_count

        if self.tracer:
            self.tracer.record_event(
                "pipeline_end", rounds=round_no, passed=passed,
                report_len=len(final_report), messages=result.messages,
            )

        # 把返工历史也存进黑板（供事后分析）
        self.bb.write(SLOT_REVISION_HISTORY, result.history,
                      writer=self.role, note="各轮审校记录")

        # ---- 最后才发 FINAL 消息 ----
        # 【为什么必须放在所有字段都赋值之后？】
        #   因为 Message 的 payload 里用了 result.summary()，
        #   而 summary() 会读取 result.rounds / messages 等字段。
        #   我最初把这条 post 放在了赋值之前，结果消息里显示的是
        #   "0 轮 | 黑板消息 0 条 | 报告 0 字符" —— 全是初始值。
        #   【这是一个很典型的"赋值顺序"bug：数据已经算好了，
        #     只是读取的时机比写入早。】
        #   修法就是把它移到末尾。这类 bug 不报错，只是数据不对，
        #   必须靠仔细检查"谁在什么时候读了什么"来发现。
        self.bb.post(Message(
            sender=self.role, recipient="user", kind=MsgKind.FINAL,
            payload={"summary": result.summary(), "passed": passed},
        ))

        return result

    # ==========================================================================
    # 任务拆解
    # ==========================================================================

    def plan(self, query: str) -> dict:
        """
        用 LLM 把用户问题拆解成结构化计划。

        【这是整个流程里唯一让 LLM 决定"做什么"的地方】
            其他环节都是确定的控制流。
            因为"如何拆解一个问题"需要理解用户意图，是创造性的工作，
            适合交给 LLM。而"拆完之后按什么顺序执行"是确定性的，
            适合用代码。
            【这个边界划分很关键：让 LLM 做创造性工作，
              让代码做确定性工作。】

        【解析失败怎么办？必须有一个兜底计划】
            如果 LLM 输出格式不对（JSON 解析失败），
            我们不能就此中止整个流程 —— 那太脆弱了。
            本实现的降级策略是：使用一个固定的默认计划。
            这个默认计划覆盖了标准的"调研-分析-撰写"三段式，
            对大多数技术调研任务都适用。

            【这是"默认值思维"：给每个可能失败的地方准备一个合理的默认值，
              系统的鲁棒性就会大幅提升。
              不要问"失败了怎么办"，而要问"失败时用什么合理的东西顶上"。】
        """
        if CFG.log_level != "quiet":
            print(f"\n{'═' * 72}")
            print("  第 0 步：任务拆解")
            print(f"{'═' * 72}")

        prompt = self.PLAN_PROMPT.format(query=query)

        try:
            raw = call_with_retry(
                lambda: self.llm.generate(prompt, system="你是一个任务规划专家。"),
                validator=lambda t: None if t and t.strip() else "返回为空",
                desc="任务拆解",
            )
        except RuntimeError as e:
            if CFG.log_level != "quiet":
                print(f"  ⚠ 任务拆解失败（{e}），使用默认计划")
            return self._default_plan(query)

        parsed = parse_json_loose(raw)

        # ---- 校验解析结果的结构 ----
        # 【为什么解析出 JSON 还不够，还要校验字段？】
        #   因为 LLM 可能返回 {"foo": "bar"} 这种"合法 JSON 但缺字段"的结果。
        #   如果不校验，后面访问 plan["sub_tasks"] 就会 KeyError。
        #   【解析成功 != 数据可用】，这是两件事。
        if not isinstance(parsed, dict) or "sub_tasks" not in parsed:
            if CFG.log_level != "quiet":
                print("  ⚠ 任务拆解返回结构不完整，使用默认计划")
            return self._default_plan(query)

        # 补全缺失的可选字段
        parsed.setdefault("topic", query[:60])
        parsed.setdefault("acceptance_criteria", [
            "每个关键结论都有可追溯的引用来源",
            "包含至少 3 个对比维度",
            "给出可执行的推荐结论",
        ])

        if CFG.log_level != "quiet":
            print(f"  ✓ 拆解出 {len(parsed['sub_tasks'])} 个子任务")
            for st in parsed["sub_tasks"]:
                print(f"      [{st.get('id','?')}] -> {st.get('assignee','?')}: "
                      f"{str(st.get('goal',''))[:70]}")
            print(f"  ✓ 验收标准 {len(parsed['acceptance_criteria'])} 条")

        return parsed

    @staticmethod
    def _default_plan(query: str) -> dict:
        """
        默认计划（LLM 拆解失败时的兜底）。

        【为什么这个兜底值得认真写，而不是随便给个空的？】
            因为它会在异常路径上被使用。
            如果它质量太差，那"LLM 失败"就会导致"整个流程结果很差"——
            异常路径变成了一条质量断崖。
            把兜底方案做得足够好，异常路径的质量就有保障。
            【衡量系统健壮性的标准，就是看它在异常路径上的表现。】
        """
        return {
            "topic": query[:60],
            "sub_tasks": [
                {"id": "T1", "assignee": "researcher",
                 "goal": f"检索关于「{query}」的技术资料，产出带引用来源的证据清单"},
                {"id": "T2", "assignee": "analyst",
                 "goal": "基于证据清单做多维度对比分析，输出对比表格与推荐结论"},
                {"id": "T3", "assignee": "writer",
                 "goal": "把证据与分析整合成一份结构化的选型报告"},
            ],
            "acceptance_criteria": [
                "每个关键结论都有可追溯的引用来源",
                "包含至少 3 个对比维度",
                "给出可执行的推荐结论与落地建议",
            ],
        }

    # ==========================================================================
    # 派发与回报（消息层面的协作）
    # ==========================================================================

    def _assign(self, assignee: str, plan: dict, agent, *, round_no: int = 0) -> None:
        """
        派发任务：在消息流水里留一条 TASK 记录。

        【这个方法的实际作用是什么？】
            从功能上说，它不改变任何业务数据 ——
            Agent 从黑板读数据，不依赖这条消息。
            但它有【可观测性】价值：
                它让"谁在什么时候被派了什么活"变得可见。
                跑完流程后，看消息流水就能重建整个协作过程。

            【这引出一个重要认知：多智能体系统里，
              "记录协作过程"本身就是一个功能，不是附加品。
              因为这类系统的调试难度远高于单体程序，
              没有完整的协作记录，出问题时你根本无从下手。】
        """
        goal = ""
        for st in plan.get("sub_tasks", []):
            if isinstance(st, dict) and st.get("assignee") == assignee:
                goal = st.get("goal", "")
                break
        if not goal:
            goal = f"执行 {assignee} 角色任务"

        self.bb.post(Message.task(
            sender=self.role, recipient=assignee, goal=goal, round_no=round_no,
        ))

    def _report_back(self, sender: str, run_result, plan: dict, *, round_no: int = 0) -> None:
        """
        Agent 回报结果：留一条 RESULT 消息。

        【注意这里包含了 stopped 状态】
            如果 Agent 是被强制中断的（max_steps / repeated / llm_error），
            这条信息会体现在消息里，让复盘时能看出"这一步其实没做好"。
            仅仅记录"完成了"是不够的 —— 必须记录"完成得怎么样"。
        """
        self.bb.post(Message.result(
            sender=sender,
            recipient=self.role,
            content=run_result.answer[:300],
            round_no=round_no,
            steps=run_result.steps,
            stopped=run_result.stopped,
            citations=len(run_result.citations),
        ))
    def _build_revision_context(self, base_context: str, round_no: int) -> str:
        """
        构造返工时的上下文：把上一轮的审校意见追加进去。

        【这是"打回重做"能真正生效的关键】
            对比两种情况：
                不传意见：Writer 重新生成一遍，结果和上次差不多
                          （它不知道自己哪里做错了，只能重新猜）
                传意见：  Writer 明确知道"要补充扩展性维度，
                          要把结论改成可执行的步骤"，针对性修改

            实测差别非常大。不传意见的返工，基本上是浪费一轮。

        【实现上的小细节：为什么只传最近一轮的意见？】
            如果传所有历史轮次的意见，会出现两个问题：
                a) prompt 越来越长
                b) 早期已经被解决的问题会被反复提及，让 Writer 困惑
                   （"我已经改了这一点，为什么还让我改？"）
            只传最近一轮，指向性最明确。
            【这也是上下文管理的一个通用原则：
              历史信息不是越多越好，越相关越好。】
        """
        critique = self.bb.read(SLOT_CRITIQUE, {})
        if not isinstance(critique, dict):
            return base_context

        issues = critique.get("issues", [])
        suggestions = critique.get("suggestions", [])
        notes = critique.get("reviewer_notes", "")

        revision_block = [
            f"## ⚠ 重要：这是第 {round_no} 轮修订",
            "上一轮的报告未通过审校。审校者提出了以下问题，你【必须】针对性修正：",
        ]

        if issues:
            revision_block.append("\n### 存在的问题")
            revision_block.extend(f"  {i+1}. {x}" for i, x in enumerate(issues))

        if suggestions:
            revision_block.append("\n### 具体的修改建议（请逐条落实）")
            revision_block.extend(f"  {i+1}. {x}" for i, x in enumerate(suggestions))

        if notes:
            revision_block.append(f"\n### 审校者总体评语\n{notes}")

        revision_block.append(
            "\n请在保留原有正确内容的基础上，针对上述问题做实质性修改。"
            "不要只是换个说法，而要真正解决问题。"
        )

        return base_context + "\n\n" + "\n".join(revision_block)

    # ==========================================================================
    def __repr__(self) -> str:
        return (
            f"<OrchestratorAgent roles=[researcher, analyst, writer, critic] "
            f"max_revision_rounds={CFG.max_revision_rounds}>"
        )
