"""
core/message.py —— Agent 之间的消息协议
================================================================================
【这个文件在架构里的位置】
    通信层。多 Agent 系统里，"Agent 之间怎么传话"是个必须先定死的契约。
    定义了契约，各个 Agent 才能独立开发、独立测试。

【核心设计决策：黑板模式 vs 直接调用】

    方案 A（直接调用）：Orchestrator 直接 `researcher.run()`，拿返回值传给 analyst
        + 简单，代码少
        - Agent 之间强耦合：Researcher 改了返回类型，Analyst 就得跟着改
        - 无法并行：必须严格串行
        - 难以追踪：信息在函数调用栈里流动，事后无法复盘"谁给了谁什么"
        - 无法插入新角色：想加个 FactChecker，得改所有调用点

    方案 B（共享黑板 + 消息）：所有 Agent 读写同一个 Blackboard，通过 Message 通信
        + 解耦：Agent 只依赖"消息格式"，不依赖具体是谁生产的
        + 可追踪：每条消息都留痕，能完整复盘信息流
        + 可扩展：加新 Agent 只需订阅/发布消息，不动老代码
        + 可并行：无依赖的 Agent 能同时跑
        - 复杂度略高，需要想清楚"谁该读谁写"

    本项目选 B —— 因为学习多智能体的重点恰恰是"协作机制"本身。

【Message 的设计要点】
    一条消息 = 谁(who) + 给谁(to) + 什么类型(kind) + 内容(payload) + 元数据
    "kind" 字段是关键：它让接收方可以用统一方式判断"这条消息是不是给我的、
    我该怎么处理"，而不用解析 payload 的内部结构。
================================================================================
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class MsgKind(str, Enum):
    """
    消息类型枚举。

    为什么继承 str？
        这样 `msg.kind == "task"` 和 `msg.kind == MsgKind.TASK` 都能成立，
        JSON 序列化时也会自动变成字符串。这是 Python 枚举的一个实用技巧。

    为什么用枚举而不是裸字符串？
        裸字符串写错一个字母（"tsak"）只会在运行时静默出错，
        枚举写错会立刻 AttributeError —— 把错误提前到开发阶段。
    """
    TASK = "task"              # 派发任务
    RESULT = "result"          # 提交结果
    CRITIQUE = "critique"      # 审校意见
    REVISION = "revision"      # 返工要求
    EVIDENCE = "evidence"      # 证据（供下游使用）
    ANALYSIS = "analysis"      # 分析结论
    DRAFT = "draft"            # 报告草稿
    FINAL = "final"            # 最终产出
    LOG = "log"                # 过程日志（不参与业务逻辑，仅供追踪）


@dataclass
class Message:
    """
    一条 Agent 间消息。

    【字段设计说明】
        sender / recipient : 通信双方。recipient 用 "broadcast" 表示广播。
        kind               : 消息类型，接收方靠它决定处理方式
        payload            : 真正的业务内容（通常是 dict）
        round_no           : 第几轮（Critic 打回后轮次 +1，便于追踪返工历史）
        task_id            : 关联的任务 ID（把散落的消息串成一条任务线）
        msg_id / ts        : 唯一 ID 和时间戳，用于排序与去重

    【为什么 payload 用 dict 而不是强类型】
        学习项目里 dict 更灵活、更容易打印查看。
        生产环境建议用 Pydantic 定义每个 kind 对应的 schema，
        这样能在消息产生时就校验格式，而不是等消费方报错。
    """
    sender: str
    recipient: str
    kind: MsgKind
    payload: dict[str, Any] = field(default_factory=dict)
    round_no: int = 0
    task_id: str = ""
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    ts: float = field(default_factory=time.time)

    # --------------------------------------------------------------------------
    # 便捷构造：用类方法提供常见消息的快捷创建方式
    # 好处：调用处 `Message.task(...)` 比 `Message(sender=..., recipient=...,
    # kind=MsgKind.TASK, ...)` 短得多，也不容易漏参数
    # --------------------------------------------------------------------------
    @classmethod
    def task(cls, sender: str, recipient: str, goal: str, task_id: str = "",
             round_no: int = 0, **extra) -> "Message":
        """
        创建任务消息。

        【注意 round_no 和 task_id 是显式参数，而不是塞进 **extra】
            这是一个容易犯的错误：如果把 round_no 放在 **extra 里，
            它就会跑进 payload 字典，而 Message 自身的 round_no 字段
            仍然是默认值 0。
            结果：轨迹里所有消息的 round_no 都是 0，
            你无法从消息流水上看出"这条是第几轮返工的" ——
            而返工轮次恰恰是观察多智能体协作最有价值的信息之一。

            【教训】当一个字段既存在于"结构"里又可能出现在"载荷"里时，
              必须明确它到底该放哪一层。这里 round_no 是消息的
              路由/分类元数据，应该放结构层（Message 自己的字段），
              这样 trace 和日志才能统一按它排序和分组。
        """
        return cls(
            sender=sender,
            recipient=recipient,
            kind=MsgKind.TASK,
            payload={"goal": goal, **extra},
            task_id=task_id,
            round_no=round_no,
        )

    @classmethod
    def result(cls, sender: str, recipient: str, content: Any, task_id: str = "",
               round_no: int = 0, **extra) -> "Message":
        return cls(
            sender=sender,
            recipient=recipient,
            kind=MsgKind.RESULT,
            payload={"content": content, **extra},
            task_id=task_id,
            round_no=round_no,
        )

    # --------------------------------------------------------------------------
    def brief(self, max_len: int = 70) -> str:
        """
        生成一行摘要，用于控制台日志。
        调试时你需要的是"一眼看清发生了什么"，而不是刷屏的完整 JSON。
        """
        text = ""
        for key in ("goal", "content", "summary", "notes", "reason"):
            if key in self.payload:
                v = self.payload[key]
                text = v if isinstance(v, str) else str(v)
                break
        if not text:
            text = str(self.payload)

        text = " ".join(text.split())     # 把换行压成空格，保证单行显示
        if len(text) > max_len:
            text = text[:max_len] + "…"
        return f"[{self.kind.value}] {self.sender} → {self.recipient}: {text}"

    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的 dict（Enum 要手动转成字符串）"""
        d = asdict(self)
        d["kind"] = self.kind.value
        return d
