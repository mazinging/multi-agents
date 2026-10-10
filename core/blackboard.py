"""
core/blackboard.py —— 共享黑板（多 Agent 的"公共工作台"）
================================================================================
【这个文件在架构里的位置】
    多 Agent 协作的核心。所有 Agent 通过它交换信息，而不是互相直接调用。

【什么是黑板模式（Blackboard Pattern）】
    想象一间会议室里有一块大白板：
        - 每个人都能看到白板上已有的内容
        - 每个人根据自己的专长，把新内容写上去
        - 没人需要知道"上一条信息是谁写的"，只需要看内容本身
    这就是黑板模式。它源自 AI 领域的经典架构，特别适合"多个专家协作解决
    一个复杂问题"的场景 —— 正好就是多智能体。

【本实现包含两个东西】
    1. slots    —— 结构化状态槽。按 key 存中间产物（证据、分析、草稿……）
                   这是 Agent 之间传递【业务数据】的主通道
    2. messages —— 消息流水。按时间顺序记录所有通信
                   这是复盘【协作过程】的依据

    两者关系：slots 是"当前状态快照"，messages 是"变更日志"。
    就像数据库和数据 binlog 的关系。

【为什么不用全局变量】
    全局变量也能实现共享，但没有：
        - 类型化的读写接口（get_set 带默认值和校验）
        - 变更历史（谁什么时候改的）
        - 版本控制（Critic 打回后要恢复/对比旧版本）
        - 多任务隔离（一个黑板服务一次会话）
    Blackboard 本质上是"带审计能力的全局状态"。

【学习提示：注意 read/write 权限的隐含约定】
    代码层面没有强制谁能写哪个槽，但在设计层面有约定（见 SLOT_OWNERS）。
    真实系统里可以考虑加权限校验 —— 这也是个很好的练习。
================================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from config import CFG
from core.message import Message, MsgKind


# ==============================================================================
# 槽位命名常量
# ==============================================================================
# 【为什么要定义成常量而不是直接写字符串？】
#   写错 "evidnece" 这种拼写错误，用字符串时只会静默地存进去、然后下游读不到，
#   你会花很久才找到问题。用常量的话，IDE 能自动补全，写错立刻报错。
# ==============================================================================

SLOT_USER_QUERY = "user_query"          # 用户原始问题
SLOT_PLAN = "plan"                      # Orchestrator 的任务规划
SLOT_EVIDENCE = "evidence"              # Researcher 的证据清单（含 doc_id 溯源）
SLOT_ANALYSIS = "analysis"              # Analyst 的分析结论
SLOT_DRAFT = "draft"                    # Writer 的报告草稿
SLOT_CRITIQUE = "critique"              # Critic 的审校意见
SLOT_FINAL = "final_report"             # 最终报告
SLOT_REVISION_HISTORY = "revision_history"   # 历次返工记录

# 每个槽位的"负责人"，用于文档化与调试时的合法性提示
SLOT_OWNERS = {
    SLOT_USER_QUERY: "user",
    SLOT_PLAN: "orchestrator",
    SLOT_EVIDENCE: "researcher",
    SLOT_ANALYSIS: "analyst",
    SLOT_DRAFT: "writer",
    SLOT_CRITIQUE: "critic",
    SLOT_FINAL: "orchestrator",
    SLOT_REVISION_HISTORY: "orchestrator",
}


@dataclass
class SlotEntry:
    """
    槽位的一条记录。

    【为什么槽位要存"历史"而不是只存最新值？】
        因为 Critic 打回重做时，我们需要对比"改之前"和"改之后"，
        也需要在必要时回滚。只存最新值就丢失了这些信息。
        version 从 1 开始递增，每次 write 都 +1。
    """
    value: Any
    writer: str                 # 谁写的
    version: int                # 第几版
    ts: float                   # 时间戳
    note: str = ""              # 写入备注（比如"根据 Critic 意见补充了扩展性维度"）


class Blackboard:
    """
    共享工作台。

    典型用法：
        bb = Blackboard()
        bb.write(SLOT_EVIDENCE, evidence_list, writer="researcher")
        ...
        ev = bb.read(SLOT_EVIDENCE)          # 拿到最新版
        old = bb.read(SLOT_EVIDENCE, version=1)   # 拿历史版本
    """

    def __init__(self) -> None:
        # key -> [SlotEntry(version=1), SlotEntry(version=2), ...]
        self._slots: dict[str, list[SlotEntry]] = {}
        # 全部消息按时间顺序
        self._messages: list[Message] = []
        # 消息订阅者：sender 之外，谁想被通知
        self._listeners: list[Callable[[Message], None]] = []

        import time
        self._now = time.time

    # ==========================================================================
    # 一、结构化状态读写
    # ==========================================================================

    def write(self, key: str, value: Any, *, writer: str, note: str = "") -> SlotEntry:
        """
        写入一个槽位（追加新版本）。

        注意：这里【不是覆盖】，而是追加。
        read() 默认返回最新版，但历史版本仍然保留。
        """
        import time

        history = self._slots.setdefault(key, [])
        entry = SlotEntry(
            value=value,
            writer=writer,
            version=len(history) + 1,
            ts=time.time(),
            note=note,
        )
        history.append(entry)

        # 如果这个槽位"不该"由该 Agent 写，给个提示但不阻止。
        # 【为什么只警告不阻止？】学习项目里，打破约定往往正是你想做的实验。
        # 硬性阻止会让你没法试"如果让 Writer 直接改证据会怎样"。
        owner = SLOT_OWNERS.get(key)
        if owner and owner != writer and CFG.log_level != "quiet":
            print(f"    [黑板提示] 槽位 '{key}' 的负责人是 '{owner}'，"
                  f"但 '{writer}' 写入了内容（仅提示，不阻止）")

        return entry

    def read(self, key: str, default: Any = None, *, version: int | None = None) -> Any:
        """
        读取槽位。
            version=None  -> 最新版
            version=N     -> 第 N 版（1-based）；不存在则返回 default
        """
        history = self._slots.get(key)
        if not history:
            return default
        if version is None:
            return history[-1].value
        idx = version - 1
        if 0 <= idx < len(history):
            return history[idx].value
        return default

    def read_entry(self, key: str) -> SlotEntry | None:
        """读取最新版的完整记录（含 writer / version / ts），调试时有用"""
        history = self._slots.get(key)
        return history[-1] if history else None

    def has(self, key: str) -> bool:
        return bool(self._slots.get(key))

    def version_count(self, key: str) -> int:
        return len(self._slots.get(key, []))

    # ==========================================================================
    # 二、消息通信
    # ==========================================================================

    def post(self, msg: Message) -> None:
        """
        发布一条消息。

        post 做了三件事：
            1. 存入消息流水（保证可复盘）
            2. 打印一行日志（保证运行时可观察）
            3. 通知订阅者（预留的扩展点：可以挂审计、告警等）
        """
        self._messages.append(msg)

        if CFG.log_level != "quiet":
            prefix = "  " * min(msg.round_no + 1, 3)   # 用缩进体现返工轮次层级
            print(f"{prefix}📨 {msg.brief()}")

        for listener in self._listeners:
            try:
                listener(msg)
            except Exception as e:      # noqa: BLE001
                # 监听器出问题绝不能影响主流程 —— 这是"旁路逻辑"的基本要求
                print(f"    [警告] 消息监听器异常: {e}")

    def subscribe(self, listener: Callable[[Message], None]) -> None:
        """注册消息监听器（本项目里 TraceRecorder 用它来落盘轨迹）"""
        self._listeners.append(listener)

    def inbox(self, recipient: str, *, kind: MsgKind | None = None,
              after_index: int = 0) -> list[Message]:
        """
        查收发给某人的消息。

        【after_index 的用途】实现"只看新消息"。
        Agent 在一轮循环里不应该反复处理同一条历史消息，
        所以调用方传上一次读到的位置，只拿增量。
        """
        out = []
        for msg in self._messages[after_index:]:
            if msg.recipient not in (recipient, "broadcast"):
                continue
            if kind is not None and msg.kind != kind:
                continue
            out.append(msg)
        return out

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def all_messages(self) -> list[Message]:
        """返回消息流水的副本，防止外部意外修改内部状态"""
        return list(self._messages)

    # ==========================================================================
    # 三、上下文构建（给 Agent 拼 prompt 用）
    # ==========================================================================

    def build_context(self, keys: list[str], *, max_chars_per_slot: int = 1500) -> str:
        """
        把若干槽位的内容拼成一段文本，用于塞进 Agent 的 prompt。

        【为什么要限制长度？】
            LLM 有上下文窗口上限，而且输入越长越贵、越容易"注意力涣散"。
            所以对每个槽位做截断是必要的工程处理。
            截断时保留头部（通常是结论/概览部分），并明确标注"已截断"，
            避免模型以为这就是全部内容。

        【为什么用 keys 列表让调用方指定，而不是无脑全塞？】
            这体现"最小必要上下文"原则：Researcher 不需要看 Writer 的草稿，
            塞进去只会干扰它，还可能让它偷懒照抄。
        """
        parts = []
        for key in keys:
            if not self.has(key):
                continue
            value = self.read(key)
            text = value if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False, indent=2
            )
            if len(text) > max_chars_per_slot:
                text = text[:max_chars_per_slot] + f"\n…（内容过长，已截断，原长 {len(text)} 字符）"
            parts.append(f"### {key}\n{text}")

        return "\n\n".join(parts) if parts else "（暂无可用上下文）"

    # ==========================================================================
    # 四、调试辅助
    # ==========================================================================

    def slot_names(self) -> list[str]:
        """
        返回所有已写入的槽位名。

        【为什么需要这个方法？（可视化场景）】
            黑板内部用 self._slots 这个私有字典存数据。
            Web 服务需要遍历所有槽位把它们推给前端，
            最省事的写法是直接访问 bb._slots —— 但那是【破坏封装】：
                1. 下划线前缀的含义就是"外部不要碰"
                2. 一旦内部存储结构改了（比如换成 OrderedDict、
                   或加一层缓存），所有直接访问 _slots 的地方都会坏
                3. 外部代码会开始依赖内部实现细节，耦合越来越深

            提供一个公开的只读方法，就等于明确了"这是允许的用法"，
            内部实现将来怎么改都不会影响调用方。
            【这是"最小公开接口"原则：内部随便重构，
              对外只暴露稳定的那部分。】
        """
        return list(self._slots.keys())

    def snapshot(self) -> dict:
        """
        返回所有槽位的完整快照（含版本历史），用于可视化。

        返回结构：
            {
              "evidence": {
                 "latest_version": 2,
                 "versions": [
                    {"version": 1, "writer": "researcher", "note": "...", "value": ...},
                    {"version": 2, ...},
                 ]
              },
              ...
            }

        【为什么要返回版本历史，而不只是最新值？】
            因为"数据如何演化"正是学习多智能体时最该看清的东西。
            比如 Writer 的草稿有两版（第 1 轮被打回、第 2 轮改进），
            只看最新版你就无法对比"返工到底改了什么"。
            把版本历史暴露出来，界面就能画出"数据流演变"。
        """
        out: dict = {}
        for key, history in self._slots.items():
            out[key] = {
                "latest_version": history[-1].version,
                "versions": [
                    {
                        "version": e.version,
                        "writer": e.writer,
                        "note": e.note,
                        "ts": e.ts,
                        "value": _jsonable(e.value),
                    }
                    for e in history
                ],
            }
        return out

    def dump_slots(self) -> str:
        """打印所有槽位概览，用于观察协作结束后系统的完整状态"""
        lines = ["─" * 70, "黑板状态快照", "─" * 70]
        for key, history in self._slots.items():
            latest = history[-1]
            value = latest.value
            size = len(value) if isinstance(value, (str, list, dict)) else 1
            note = f"  ← {latest.note}" if latest.note else ""
            lines.append(
                f"  {key:<20} v{latest.version}  by {latest.writer:<13} 大小={size}{note}"
            )
        lines.append("─" * 70)
        return "\n".join(lines)


def _jsonable(obj: Any) -> Any:
    """
    把任意对象转成可 JSON 序列化的形式。

    【为什么需要它？】黑板上存的是任意 Python 对象 ——
    可能是 dict、list、str，也可能是自定义类的实例。
    直接 json.dumps 遇到不可序列化的对象会抛 TypeError，
    导致整个快照推送失败。

    【这里的选择：宁可降级成 repr 字符串，也不要抛异常。】
        因为这是"调试/展示"用途的数据，
        显示成 "<Chunk object at 0x...>" 虽然不完美，但界面还能用；
        而抛异常会让整个可视化功能挂掉。
        【可观测性代码要"尽力而为"，不能因为数据格式奇怪就整体失败。】
    """
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except (TypeError, ValueError):
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        return repr(obj)
