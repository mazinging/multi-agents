"""
agents/base.py —— ReAct 循环引擎（本项目最核心的文件）
================================================================================
【这个文件在架构里的位置】
    它是所有 Agent 的"大脑"。其他 5 个 Agent 文件都只是它的子类，
    通过配置（角色提示词 + 工具集）来定义自己是谁。

【ReAct 循环到底在做什么？】

    一个 ReAct Agent 的工作方式是不断重复：
        ┌─────────────────────────────────────────┐
        │  1. 把"当前已知的一切"拼成 prompt        │
        │  2. 让 LLM 输出下一步：                 │
        │       Thought: 我在想什么                │
        │       Action: 我要调用什么工具           │
        │       Action Input: 参数是什么           │
        │     或者                               │
        │       Final Answer: 我得出的结论         │
        │  3. 如果是 Action -> 执行工具 -> 拿到 Observation
        │  4. 把 Observation 追加到历史里，回到第 1 步 │
        │  5. 如果是 Final Answer -> 结束，返回结果   │
        └─────────────────────────────────────────┘

    关键理解：LLM 本身是【无状态】的。它不记得上一轮说了什么。
    是【我们把历史拼进 prompt】，才让它看起来像是"记得"。
    这就是为什么下面 _build_prompt 要花那么多笔墨 ——
    Agent 的"记忆"完全由这一段的拼接方式决定。

【本实现处理的五个工程问题（都是实战必踩的坑）】
    1. 死循环         -> max_react_steps 硬上限 + 重复动作检测
    2. 格式解析失败   -> 复用 llm.parse_react_output 的容错 + 重试
    3. 工具参数错误   -> 交给工具层的友好错误信息，让模型自我纠正
    4. 上下文膨胀     -> 对历史 Observation 做截断
    5. 模型直接给答案 -> 允许第 0 轮就 Final Answer（有些任务不需要工具）
================================================================================
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from config import CFG
from llm import LLMClient, ReactStep, call_with_retry, parse_react_output
from tools.base import ToolRegistry, ToolResult

if TYPE_CHECKING:
    from core.blackboard import Blackboard
    from core.trace import TraceRecorder


# ==============================================================================
# 引用提取（全项目统一实现）
# ==============================================================================
#
# 【为什么要把这个正则提到 base.py 作为公共函数？】
#   因为它原本在 researcher / analyst / writer / critic 四个文件里各写了一遍，
#   而且我最初写的那版有 bug：
#       r"\[([A-Za-z0-9_\-]+#\d+)\]"      <- 只允许英文数字下划线
#   这个字符类匹配不到含中文的引用 ID，导致编造的引用
#   （如 [99_不存在文档#7]）完全不被识别，
#   防幻觉校验静默失效。
#
#   修复时我改了一处，然后意识到【同样的 bug 在另外三处也存在】。
#   这正是"重复代码"最典型的危害：
#       一个 bug 要改 N 次，而且很容易漏掉某一处。
#       更糟的是，漏掉的那一处会静默失效，你不会收到任何提示。
#
#   所以把它提取成公共函数。这样：
#       a) 修一次就全好了
#       b) 全项目的引用格式定义只有一处，不会出现"各处理解不一致"
#       c) 可以单独为它写测试
#
#   【这是"消除重复"最有力的论据：不是为了代码好看，
#     而是因为重复会让修复变得不可靠。】
CITATION_PATTERN = re.compile(r"\[([^\[\]#\s]+#\d+)\]")


def extract_citations(text: str) -> list[str]:
    """
    从文本里提取所有引用 ID（去重且排序）。

    引用格式约定为 [文档ID#块号]，例如 [02_vector_db_cost#4]。
    """
    if not text:
        return []
    return sorted(set(CITATION_PATTERN.findall(text)))


# ==============================================================================
# 运行结果
# ==============================================================================

@dataclass
class AgentRunResult:
    """
    Agent 一次运行的完整结果。

    【为什么 answer 之外还要返回这么多东西？】
        因为多智能体系统里，调用方（通常是 Orchestrator）需要的不只是答案：
            steps      : 跑了几步 -> 用于成本统计、判断是否"跑得很吃力"
            tool_calls : 调了哪些工具 -> 用于判断证据是否充分
            citations  : 引用了哪些来源 -> 用于 Critic 做引用校验
            trace      : 完整推理轨迹 -> 用于调试和可观测
            stopped    : 结束原因 -> 区分"正常完成"和"被上限截断"

        【尤其是 stopped 字段】它区分了三种结束方式：
            "final_answer" -> 模型自己给出了答案（正常）
            "max_steps"    -> 达到步数上限被迫停止（可能未完成任务）
            "repeated"     -> 检测到重复动作被强制中断（模型卡住了）
        上层据此可以做出不同反应：正常完成就接受，
        被截断则可能需要重试或降低期望。没有这个字段，
        你无法区分"Agent 说完成了"和"Agent 被强行打断了"。
    """
    agent: str
    answer: str
    steps: int = 0
    tool_calls: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    stopped: str = "final_answer"
    elapsed: float = 0.0
    raw_history: list[ReactStep] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.stopped == "final_answer"

    def summary(self) -> str:
        return (
            f"[{self.agent}] {self.steps}步 / {len(self.tool_calls)}次工具调用 / "
            f"{len(self.citations)}条引用 / {self.elapsed:.1f}s / 结束={self.stopped}"
        )


# ==============================================================================
# ReAct 引擎
# ==============================================================================

class BaseReActAgent:
    """
    所有 Agent 的基类。

    ── 子类需要提供什么 ──
        role           : 角色标识（用于日志、消息署名）
        system_prompt  : 角色人设 + 任务目标 + 输出要求（最重要的部分）
        registry       : 该角色可用的工具集
        build_task()   : 从黑板读取上游数据，构造本轮任务描述

    ── 子类可以覆盖什么 ──
        postprocess()  : 对最终答案做后处理（比如抽取引用、格式化 JSON）
        needs_tools    : 如果为 False，跳过 ReAct 循环直接调一次 LLM
                         （适合纯生成型角色，如 Writer）
    """

    role: str = "agent"
    system_prompt: str = ""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry | None = None,
        *,
        blackboard: "Blackboard | None" = None,
        tracer: "TraceRecorder | None" = None,
        max_steps: int | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry or ToolRegistry()
        self.blackboard = blackboard
        self.tracer = tracer
        self.max_steps = max_steps or CFG.max_react_steps

    # ==========================================================================
    # 对外入口
    # ==========================================================================

    def run(self, task: str, *, context: str = "") -> AgentRunResult:
        """
        执行一次完整的 Agent 任务。

        ── 参数 ──
            task    : 本次要完成的具体任务（由调用方或 build_task 生成）
            context : 额外的上下文（通常是从黑板读来的上游产物）

        【为什么 task 和 context 要分开？】
            这是"要做什么"和"已知什么"的分离，对应 prompt 工程里
            两个完全不同的问题：
                task    -> 影响"目标是否清晰"（模型知不知道要干嘛）
                context -> 影响"信息是否充分"（模型有没有足够素材）
            分开之后，调试时能快速定位问题出在哪一边：
                答非所问 -> task 没写清楚
                内容空洞 -> context 不够或没被利用
            如果混成一个字符串，你就很难判断该怎么改。
        """
        t0 = time.time()

        if CFG.log_level != "quiet":
            print(f"\n{'─' * 72}")
            print(f"▶ {self.role} 开始执行")
            print(f"  任务: {task[:110]}{'…' if len(task) > 110 else ''}")
            print(f"{'─' * 72}")

        if self.tracer:
            self.tracer.record_event(
                "agent_start", agent=self.role, task=task[:300], has_context=bool(context)
            )

        # ---- 分派：有些角色不需要工具循环 ----
        if not self.registry.names:
            result = self._run_without_tools(task, context)
        else:
            result = self._run_react_loop(task, context)

        result.elapsed = time.time() - t0

        if CFG.log_level != "quiet":
            print(f"\n◀ {self.role} 结束 | {result.summary()}")

        if self.tracer:
            self.tracer.record_event(
                "agent_end",
                agent=self.role,
                stopped=result.stopped,
                steps=result.steps,
                answer_preview=result.answer[:400],
            )

        return result

    # ==========================================================================
    # 路径 A：无工具，单次 LLM 调用
    # ==========================================================================

    def _run_without_tools(self, task: str, context: str) -> AgentRunResult:
        """
        纯生成型 Agent 的路径。

        【什么时候用这条路径？】
            当一个角色的工作本质是"加工信息"而不是"获取信息"时 ——
            最典型的是 Writer：它的输入是已经齐备的证据和分析，
            要做的是组织成文章，不需要（也不该）自己去检索。
            让它去检索反而会破坏分工。

        【这条路径的价值】少一轮 ReAct 循环 = 少一次 LLM 调用 = 更快更便宜。
        不要为了"架构统一"而让所有 Agent 都走 ReAct 循环。
        【按任务本质选择最简路径，是很好的工程判断】
        """
        prompt = self._build_prompt(task, context, include_tools=False)
        raw = call_with_retry(
            lambda: self.llm.generate(prompt, system=self.system_prompt),
            validator=lambda t: None if t and t.strip() else "返回内容为空",
            desc=f"{self.role} 生成",
        )

        if self.tracer:
            self.tracer.record_react(
                agent=self.role, step=0, phase="final", content=raw[:2000]
            )

        # ----------------------------------------------------------------------
        # 【关键：必须调用 postprocess —— 这是我实测踩到的一个严重 bug】
        #
        # 我最初写这条路径时忘了调 postprocess，后果是：
        #   Writer 不会把草稿写进黑板 -> Orchestrator 读不到草稿 -> 最终报告是空的
        #   Critic 不会把审校结论写进黑板 -> Orchestrator 读到空值 -> 永远判定为 revise
        #
        # 症状非常隐蔽：整个流程跑完不报任何错，只是"最终报告 0 字符"
        # 和"审校永远不通过"。因为每个 Agent 都"成功执行"了，
        # 只是它们的成果没有落地到共享状态里。
        #
        # 【这个 bug 的教学价值】它说明了"两条执行路径必须保持行为一致"的重要性。
        #    走 ReAct 循环的路径调了 postprocess，走无工具路径的没调 ——
        #    同一个基类的两条分支行为不一致，是一个典型的设计疏漏。
        #    修法就是这里补上调用，让两条路径的"收尾行为"统一。
        #
        #    更进一步的教训：当你给一个类加分支时，要把所有"公共收尾动作"
        #    抽出来，确保每条分支都执行。否则很容易漏掉其中一条。
        # ----------------------------------------------------------------------
        answer = self.postprocess(raw.strip(), [])

        return AgentRunResult(
            agent=self.role,
            answer=answer,
            steps=1,
            stopped="final_answer",
        )

    # ==========================================================================
    # 路径 B：完整 ReAct 循环
    # ==========================================================================

    def _run_react_loop(self, task: str, context: str) -> AgentRunResult:
        """
        ReAct 主循环。

        【循环的退出条件有三个，缺一不可】
            1. 模型输出 Final Answer      -> 正常完成
            2. 达到 max_steps             -> 硬上限，防止无限循环
            3. 检测到重复动作             -> 模型卡住了，强制中断
        只有第 1 种是"正常完成"。第 2、3 种都要在返回值里标记出来，
        让上层知道"这个结果可能不完整"。
        """
        # history 是本循环的【短期记忆】。
        # 它按时间顺序累积每一步的 Thought / Action / Observation，
        # 每轮都会被拼进 prompt 送给模型 —— 这就是 ReAct 的"记忆机制"。
        history: list[str] = []
        tool_calls: list[str] = []
        citations: list[str] = []
        raw_steps: list[ReactStep] = []

        # 用于重复动作检测：记录上一轮的 (action, input)
        last_action: tuple[str, str] | None = None
        repeat_count = 0

        step = 0
        while step < self.max_steps:
            step += 1

            # ---- 1. 拼 prompt 并调用 LLM ----
            prompt = self._build_prompt(task, context, history=history)

            try:
                raw = call_with_retry(
                    lambda: self.llm.generate(prompt, system=self.system_prompt),
                    validator=lambda t: None if t and t.strip() else "返回内容为空",
                    desc=f"{self.role} 第{step}步推理",
                )
            except RuntimeError as e:
                # LLM 连续失败（网络问题等）。
                # 【关键决策】不抛异常中止整个流程，而是降级返回一个说明。
                # 因为多智能体流程很长，一处 LLM 抖动不该让全部工作白做。
                # 上层（Orchestrator）会看到 stopped="llm_error" 并决定怎么处理。
                if CFG.log_level != "quiet":
                    print(f"    ⚠ LLM 调用失败: {e}")
                return AgentRunResult(
                    agent=self.role,
                    answer=f"（推理中断：LLM 调用失败 {e}）",
                    steps=step,
                    tool_calls=tool_calls,
                    citations=citations,
                    stopped="llm_error",
                    raw_history=raw_steps,
                )

            # ---- 2. 解析输出 ----
            parsed = parse_react_output(raw)
            raw_steps.append(parsed)

            if parsed.thought and CFG.log_level != "quiet":
                thought_display = " ".join(parsed.thought.split())
                print(f"    💭 Thought: {thought_display[:150]}")

            if self.tracer:
                self.tracer.record_react(
                    agent=self.role, step=step, phase="thought",
                    content=parsed.thought, extra={"parse_error": parsed.parse_error},
                )

            # 解析有问题的提示（不中断，只是提示 —— 因为解析器已尽力降级）
            if parsed.parse_error and CFG.log_level == "verbose":
                print(f"    ⚠ 解析提示: {parsed.parse_error}")

            # ---- 3. 判断是否结束 ----
            if parsed.is_final:
                if CFG.log_level != "quiet":
                    preview = " ".join(parsed.final_answer.split())[:200]
                    print(f"    ✅ Final Answer: {preview}")

                if self.tracer:
                    self.tracer.record_react(
                        agent=self.role, step=step, phase="final",
                        content=parsed.final_answer or "",
                    )

                answer = self.postprocess(parsed.final_answer or "", raw_steps)
                return AgentRunResult(
                    agent=self.role,
                    answer=answer,
                    steps=step,
                    tool_calls=tool_calls,
                    citations=citations,
                    stopped="final_answer",
                    raw_history=raw_steps,
                )

            # ---- 4. 执行工具 ----
            action = parsed.action or ""
            action_input = parsed.action_input or ""

            # 重复动作检测 —— 防止模型"卡住"
            # 【为什么需要这个？除了步数上限之外的额外保护】
            #   步数上限是"钝"的：它允许模型重复 5 次同样的动作才停。
            #   而重复检测是"锐"的：第 2 次重复就中断，节省时间和成本。
            #   实测中，模型陷入重复是很常见的失败模式
            #   （尤其是在工具返回空结果时，它会不甘心地再试一次同样的查询）。
            current = (action, action_input)
            if current == last_action:
                repeat_count += 1
                if CFG.log_level != "quiet":
                    print(f"    ⚠ 检测到重复动作 ({action}) 第 {repeat_count} 次")
                if repeat_count >= 2:
                    return self._forced_stop(
                        "repeated", history, tool_calls, citations, raw_steps, step,
                        reason=f"连续重复调用 {action}({action_input}) 且无新信息",
                    )
            else:
                repeat_count = 0
            last_action = current

            if CFG.log_level != "quiet":
                print(f"    🔧 Action: {action}({action_input[:80]})")

            if self.tracer:
                self.tracer.record_react(
                    agent=self.role, step=step, phase="action",
                    content=f"{action}({action_input})",
                )

            result: ToolResult = self.registry.execute(action, action_input)
            tool_calls.append(f"{action}({action_input})")

            # 从工具的结构化返回里收集引用（用于后续的引用校验）
            citations.extend(self._collect_citations(result))

            observation = result.content
            if CFG.log_level != "quiet":
                obs_preview = " ".join(observation.split())[:170]
                status = "✓" if result.ok else "✗"
                print(f"    {status} Observation: {obs_preview}")

            if self.tracer:
                self.tracer.record_react(
                    agent=self.role, step=step, phase="observation",
                    content=observation[:3000],
                    extra={"ok": result.ok, "meta": result.meta},
                )

            # ---- 5. 把这一步追加到历史 ----
            history.append(self._format_step(parsed, observation))

        # ---- 循环结束：达到步数上限 ----
        return self._forced_stop(
            "max_steps", history, tool_calls, citations, raw_steps, step,
            reason=f"达到最大步数上限 {self.max_steps}",
        )

    # ==========================================================================
    # prompt 构建 —— Agent 的"记忆"和"人格"都在这里形成
    # ==========================================================================

    def _build_prompt(
        self,
        task: str,
        context: str,
        history: list[str] | None = None,
        include_tools: bool = True,
    ) -> str:
        """
        拼装送给 LLM 的完整 prompt。

        ── 结构（顺序很重要）──
            1. 任务目标        —— 让模型知道要做什么
            2. 可用工具（含格式说明）—— 让模型知道能做什么
            3. 上游上下文      —— 让模型知道已有什么（Evidence/Analysis 等）
            4. 历史轨迹        —— 让模型知道已经做过什么（ReAct 的记忆）
            5. 格式要求 + 收尾指令 —— 引导模型输出正确格式

        【为什么顺序是这样？】
            这是"从宏观到微观"、"从不变到变化"的组织方式：
                任务目标是最稳定的（整个 run 不变）
                历史轨迹变化最快（每步都在增长）
            把变化的放后面，有两个好处：
                a) KV 缓存友好（前缀不变，部分推理框架能复用缓存）
                b) 即使历史被截断，前面的任务目标和工具说明仍然完整
            这个顺序在长上下文场景下很重要 —— 如果任务目标被截断掉了，
            模型就完全跑偏了。

        【第 5 部分的"收尾指令"为什么必须存在？】
            因为模型有很强的"聊天惯性" —— 它会倾向于输出一段
            漂亮的总结然后停下，而不是输出 Action。
            明确要求"如果你还需要信息，必须输出 Action"能显著
            提高它继续调用工具的概率。
            这是 prompt 工程里很实际的一条经验：
            【不要指望模型推断你的意图，要明确写出你期望的行为】
        """
        parts: list[str] = []

        # ---- 1. 任务 ----
        parts.append(f"## 你的任务\n{task}")

        # ---- 2. 工具说明 ----
        if include_tools and len(self.registry) > 0:
            parts.append(
                "## 你可以使用的工具\n"
                + self.registry.catalog()
                + "\n\n## 工具调用格式（必须严格遵守）\n"
                "当你需要调用工具时，输出：\n"
                "Thought: <你的思考，说明为什么要调用这个工具>\n"
                "Action: <工具名，必须完全等于上面列出的名字>\n"
                "Action Input: <参数，只写参数本身，不要加引号或额外说明>\n\n"
                "当你已经收集足够信息、可以给出结论时，输出：\n"
                "Thought: <说明为什么信息已经足够>\n"
                "Final Answer: <你的完整结论>"
            )

        # ---- 3. 上游上下文 ----
        if context and context.strip():
            parts.append(f"## 已知信息（来自其他 Agent 的产物）\n{context}")

        # ---- 4. 历史轨迹（ReAct 的记忆）----
        if history:
            parts.append("## 你已经执行过的步骤\n" + "\n\n".join(history))

        # ---- 5. 收尾引导 ----
        if include_tools and len(self.registry) > 0:
            parts.append(
                "## 现在请继续\n"
                "如果你还需要更多信息才能完成任务，请输出 Thought + Action + Action Input。\n"
                "如果你已经掌握足够信息，请输出 Thought + Final Answer。\n"
                "注意：不要重复调用之前已经做过的相同工具和相同参数。"
            )
        else:
            parts.append("## 现在请直接输出你的最终结果（不需要调用工具）。")

        return "\n\n".join(parts)

    # ==========================================================================
    # 辅助方法
    # ==========================================================================

    def _format_step(self, parsed: ReactStep, observation: str) -> str:
        """
        把一轮 Thought/Action/Observation 格式化成历史条目。

        【这个格式必须和 prompt 里教模型的格式一致】
            因为模型看到历史里的 "Thought: ... Action: ... Observation: ..."，
            会模仿这个模式继续输出。格式一致本身就是一种强力的格式约束 ——
            模型倾向于复制它在上下文里看到的模式（这叫 in-context learning）。
            所以"历史格式规范"间接提升了模型输出的格式正确率。
            这是一个很实用但常被忽略的技巧。

        【为什么要截断 observation？】
            见下面 OBS_MAX_CHARS 的说明。
        """
        return (
            f"Thought: {parsed.thought}\n"
            f"Action: {parsed.action}\n"
            f"Action Input: {parsed.action_input}\n"
            f"Observation: {self._truncate_observation(observation)}"
        )

    # 单次观察结果最多保留多少字符。
    # 【为什么必须截断？】这是 ReAct 实现里最重要的成本控制点。
    #   每轮循环都会把全部历史拼进 prompt。如果每次工具返回 4000 字符，
    #   跑 6 轮就是 24000 字符的历史 —— prompt 会迅速膨胀到几万 token，
    #   导致成本剧增、延迟上升，而且早期信息会淹没最新信息。
    #   实测中，不截断的实现跑几轮后 prompt 就能超过 10 万字符。
    #
    #   截断到 1200 字符的权衡：
    #       保留了足够的信息供模型判断"这轮查到了什么"
    #       又不至于让历史失控
    #   【注意】使用这个工具的 Agent 如果真需要完整内容，
    #   应该把结果存到黑板（比如证据清单），而不是依赖 prompt 里的历史。
    #   这就回到了"共享状态 vs 消息传递"的架构讨论 ——
    #   重数据放黑板，prompt 里只放摘要和引用。
    OBS_MAX_CHARS = 1200

    def _truncate_observation(self, text: str) -> str:
        if len(text) <= self.OBS_MAX_CHARS:
            return text
        return (
            text[: self.OBS_MAX_CHARS]
            + f"\n…（观察结果过长已截断，原长 {len(text)} 字符）"
        )

    @staticmethod
    def _collect_citations(result: ToolResult) -> list[str]:
        """
        从工具返回的结构化数据里收集引用 ID。

        【为什么要做这件事？】
            这是引用校验（防幻觉）的数据基础。

            在 RAG 场景里，一个关键问题是："模型说的结论，
            真的有资料支持吗？" 要回答这个问题，需要知道
            "模型这次看到了哪些资料"。

            工具返回的 data 字段里有 chunk_id（比如 "03_embedding_models#5"），
            收集起来就得到"本次看到的资料集合"。
            之后 Critic 检查报告里的引用时，就可以核对：
                报告引用的 ID 是否在这个集合里？
            不在 -> 说明模型凭空捏造了一个引用 -> 判定为幻觉。

            【这是程序化防幻觉的核心思路】
                不要把这件事交给"另一个 LLM 去判断"（它会出错），
                而是用集合运算做确定性校验。
                能用确定性方法做好的校验，就不要用模型。
        """
        cites: list[str] = []
        data = result.data

        # data 可能是单个 dict、dict 列表、或者嵌套结构，都要能处理
        def harvest(obj: Any) -> None:
            if isinstance(obj, dict):
                cid = obj.get("chunk_id")
                if isinstance(cid, str) and cid:
                    cites.append(cid)
                for v in obj.values():
                    harvest(v)
            elif isinstance(obj, list):
                for item in obj:
                    harvest(item)

        harvest(data)
        return cites

    def _forced_stop(
        self,
        reason: str,
        history: list[str],
        tool_calls: list[str],
        citations: list[str],
        raw_steps: list[ReactStep],
        step: int,
        *,
        stop_reason: str = "",
    ) -> AgentRunResult:
        """
        非正常结束时的处理：不返回空，而是返回"部分成果 + 明确说明"。

        【为什么不让它返回空字符串？】
            因为 Agent 已经做了很多工作（可能查了很多资料），
            直接丢弃太浪费。更好的做法是：
                1. 保留已有的轨迹信息（让下游知道做了什么）
                2. 给出一个基于已有信息的粗略答案
                3. 明确标记 stopped 状态，让上层知道这是不完整的结果

            【这就是"优雅降级"】在工程系统里，
            一个部分可用的结果 + 明确的警告，
            远比一个干净的错误或一个空白要好。
            上层（比如 Critic）可以据此决定是否接受、
            或者给这个 Agent 更多步数重试。
        """
        if CFG.log_level != "quiet":
            print(f"    ⚠ 强制中断: {stop_reason or reason}")

        if self.tracer:
            self.tracer.record_event(
                "agent_forced_stop", agent=self.role,
                reason=stop_reason or reason, steps=step,
            )

        # 用已有的历史拼一个"尽力而为"的答案说明
        attempted = "；".join(tool_calls[-4:]) if tool_calls else "（未调用任何工具）"
        answer = (
            f"（{self.role} 未能给出完整结论：{stop_reason or reason}）\n"
            f"已尝试的操作：{attempted}\n"
            f"请上层 Agent 根据现有信息判断，或补充资料后重试。"
        )

        return AgentRunResult(
            agent=self.role,
            answer=answer,
            steps=step,
            tool_calls=tool_calls,
            citations=citations,
            stopped=reason,
            raw_history=raw_steps,
        )

    # ==========================================================================
    # 子类扩展点
    # ==========================================================================

    def build_task(self, **kwargs) -> tuple[str, str]:
        """
        构造 (task, context)。子类必须实现。

        【为什么用返回值而不是让子类直接读黑板？】
            因为返回 (task, context) 是个明确的契约，
            强制子类回答两个问题："要做什么"和"已知什么"。
            如果让子类随便读黑板，很容易写成"把所有东西都塞进 context"，
            导致 prompt 臃肿、重点不明。
            用契约约束子类的实现，是一种"用接口引导设计"的手法。
        """
        raise NotImplementedError

    def postprocess(self, answer: str, steps: list[ReactStep]) -> str:
        """
        对最终答案做后处理。默认什么都不做，子类可覆盖。

        【典型用途】
            - 从答案里抽取 JSON 并结构化（Critic 需要）
            - 清理模型加的多余客套话
            - 规范化引用格式
        """
        return answer

    # ==========================================================================
    def __repr__(self) -> str:
        return f"<{type(self).__name__} role={self.role} tools={self.registry.names}>"
