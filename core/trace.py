"""
core/trace.py —— 轨迹记录器（学习多智能体的"黑匣子"）
================================================================================
【这个文件在架构里的位置】
    旁路模块。它订阅黑板上的所有消息和 Agent 的每一步思考，
    异步落盘成 trace.jsonl，但【完全不参与业务逻辑】。

【为什么我（作者）要额外加这个模块】
    学多智能体最容易卡住的地方是："信息到底是怎么在 Agent 之间流动的？"
    光看代码很难建立直觉 —— 因为代码是"静态的结构"，
    而多智能体的本质是"动态的过程"。

    所以需要一个能把过程完整记录下来的东西。跑一次任务后，
    你可以打开 trace.jsonl，看到类似这样的时间线：
        第 1 步  orchestrator 思考：需要拆解任务
        第 2 步  orchestrator 调用工具 plan_task
        第 3 步  orchestrator → researcher 派发任务 T1
        第 4 步  researcher 思考：先检索
        第 5 步  researcher 调用 search_local_docs("向量数据库")
        第 6 步  工具返回 4 个文档块
        ...
        第 N 步  critic 判定 revise，触发第 2 轮
        第 N+1 步 orchestrator → writer 返工要求

    这比读一万行代码都直观。

【格式选择：JSONL 而不是 JSON】
    JSONL = 每行一个独立的 JSON 对象。
    优点：
        - 可以边跑边写，不用等全部结束才落盘（进程崩了也不丢数据）
        - 可以用 grep / findstr 直接搜某一行
        - 可以流式处理超大文件，不必一次读进内存
    日志类数据几乎总是用 JSONL，这是个通用的工程习惯。

【设计要点：记录器绝不能让主流程崩溃】
    所有写入都包在 try/except 里。日志系统挂了不该影响业务 ——
    这是"可观测性组件"的铁律。

【v2 新增：事件订阅（为 Web 可视化服务）】
    原本这个类只干一件事：把事件写进 trace.jsonl 落盘。
    做网页可视化时需要一个"实时推送"的能力，最直接的想法是
    再写一个 WebTraceRecorder 类 —— 但那意味着：
        a) 所有埋点都要写两遍（一处落盘、一处推送）
        b) 以后加新事件类型，两处都要改，必然漏一处
        c) 埋点代码散落在各处，越来越难维护

    所以我改成【发布-订阅】结构：
        埋点代码只调用 self.tracer.record_event(...) 一次，
        记录器内部再分发给所有订阅者（落盘器、SSE 推送器、控制台…）。

    这样一来：
        - 埋点处完全不需要知道"有谁在消费这些事件"
        - 新增一种消费方式（比如推给 WebSocket、发给监控系统）
          只需要 subscribe 一个新的回调，不用动任何业务代码
        - 落盘和推送天然保证内容一致（同一份数据源）

    【这就是"观察者模式"在可观测性场景下的价值：
      让"产生事件"和"消费事件"彻底解耦。】
    它和黑板用订阅机制通知 TraceRecorder 是同一个思路 ——
    本项目里这个模式出现了两次，说明它在"一对多通知"场景下确实好用。
================================================================================
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from config import CFG


class TraceRecorder:
    """
    把运行过程记录到 trace.jsonl，并分发给所有订阅者。

    提供三类记录接口，对应三种可观测的粒度：
        record_event  —— 里程碑事件（任务开始/结束、阶段切换）
        record_react  —— ReAct 单步（思考/动作/观察），粒度最细
        record_message—— Agent 间消息（协作关系）

    另外实现了 __call__，所以它可以直接当黑板的订阅者使用：
        bb.subscribe(recorder)

    【订阅用法（Web 可视化就是这么接进来的）】
        recorder = TraceRecorder()
        recorder.subscribe(lambda rec: print(rec))     # 每产生一条事件就回调

        订阅者拿到的 rec 是完整的记录字典（和写进 JSONL 的是同一个对象），
        所以订阅者不需要再解析任何格式。
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        write_file: bool = True,
    ) -> None:
        self.path = path or CFG.trace_path
        # write_file=False 时只做分发、不落盘。
        # Web 模式下如果只想看实时推送，可以不产生文件。
        self.enabled = CFG.enable_trace and write_file

        # 每次运行都重新开始，避免和上次的记录混在一起
        if self.enabled:
            try:
                self.path.write_text("", encoding="utf-8")
            except OSError as e:
                print(f"[警告] 无法写入轨迹文件 {self.path}: {e}；本次不记录轨迹")
                self.enabled = False

        self._seq = 0     # 全局序号，用于排序和引用
        self._subscribers: list[Callable[[dict], None]] = []

        # 【为什么要加锁？】
        #   Web 版里，事件由"跑任务的线程"产生，而 SSE 连接由
        #   "Flask 的请求线程"消费。两个线程同时操作订阅者列表
        #   会有竞态问题（比如推送时列表正被修改）。
        #   用一把简单的锁保护住就够了 —— 这个场景锁竞争很低，
        #   性能完全不是问题，正确性才是。
        self._lock = threading.Lock()

    # --------------------------------------------------------------------------
    # 订阅管理
    # --------------------------------------------------------------------------
    def subscribe(self, callback: Callable[[dict], None]) -> Callable[[], None]:
        """
        注册一个订阅者。每产生一条事件，callback 会被调用一次。

        返回值是一个"取消订阅"的函数。
        【为什么要返回取消函数？】因为 Web 场景下浏览器可能随时关闭连接，
        不取消订阅的话，回调会一直堆积，造成内存泄漏。
        返回一个取消函数是最简单的清理手段。
        """
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def _publish(self, record: dict) -> None:
        """
        把一条记录分发给所有订阅者。

        【关键：订阅者抛异常绝不能影响其他订阅者，更不能影响主流程】
            如果 SSE 推送因为浏览器断开而抛错，
            不能连带把落盘也搞挂，也不能让 Agent 的执行中断。
            所以这里逐个 try/except。
            【这是"扇出"场景的通用防御手段：一个订阅者坏了，
              不应该让整个广播失败。】
        """
        with self._lock:
            targets = list(self._subscribers)   # 复制一份，避免迭代时被修改

        for cb in targets:
            try:
                cb(record)
            except Exception:      # noqa: BLE001
                # 静默忽略：可观测性组件不能反过来影响业务
                pass

    # --------------------------------------------------------------------------
    # 内部写入
    # --------------------------------------------------------------------------
    def _write(self, kind: str, **fields: Any) -> None:
        self._seq += 1
        record = {
            "seq": self._seq,
            "ts": round(time.time(), 4),
            "kind": kind,
            **fields,
        }

        # 先落盘（如果启用）
        if self.enabled:
            try:
                # 用 "a"(append) 模式打开：每次写一行就关，进程崩溃也不丢已写内容。
                # 频繁开关文件确实有开销，但日志量不大，换来的是可靠性。
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as e:
                # 写日志失败绝不能中断业务
                print(f"[警告] 轨迹写入失败: {e}")

        # 再分发（无论是否落盘都要分发 —— 实时可视化可能不需要文件）
        self._publish(record)

    # --------------------------------------------------------------------------
    # 三类记录接口
    # --------------------------------------------------------------------------
    def record_event(self, event: str, **detail: Any) -> None:
        """记录一个里程碑事件"""
        self._write("event", event=event, **detail)

    def record_react(
        self,
        *,
        agent: str,
        step: int,
        phase: str,          # "thought" | "action" | "observation" | "final"
        content: Any,
        extra: dict | None = None,
    ) -> None:
        """
        记录 ReAct 循环中的一步。

        【phase 的含义】
            thought     : 模型的思考内容
            action      : 决定调用哪个工具、参数是什么
            observation : 工具返回了什么
            final       : 给出了最终答案
        把这四个 phase 沿着时间轴排开，就是一个完整的 ReAct 轨迹 ——
        这正是你理解 ReAct 最需要的东西。
        """
        self._write(
            "react",
            agent=agent,
            step=step,
            phase=phase,
            content=content if isinstance(content, (str, int, float, bool, type(None))) else str(content),
            extra=extra or {},
        )

    def record_rag(self, **payload: Any) -> None:
        """
        记录一次检索的内部过程（混合检索的各阶段数据）。

        【为什么要单独一类，而不是复用 record_event？】
            因为语义不同：
                record_event 记录的是"里程碑"（流程开始、Agent 结束…），
                    它的特征是"发生了某件事"。
                record_rag 记录的是"结构化的过程数据"（两路召回了哪些块、
                    各自多少分、哪些被阈值刷掉…），
                    它的特征是"带一堆数值明细"。

            混在一起会有两个问题：
                a) 前端无法按类别筛选（想只看检索细节时得靠 event 名字猜）
                b) 两者携带的字段结构完全不同，消费方要写一堆 if 分支

            【给不同语义的数据分配不同的 kind，是可观测性的基本功。
              它让消费方能按类别精确订阅，而不必解析内容来判断类型。】
        """
        self._write("rag", **payload)

    def record_message(self, msg) -> None:
        """记录一条 Agent 间消息（接收 Message 对象）"""
        self._write(
            "message",
            sender=msg.sender,
            recipient=msg.recipient,
            msg_kind=msg.kind.value,
            round_no=msg.round_no,
            task_id=msg.task_id,
            payload_summary=msg.payload if _is_small(msg.payload) else "(内容较大，见黑板)",
        )

    def __call__(self, msg) -> None:
        """
        实现 __call__ 使实例可调用，于是能直接传给 bb.subscribe()。
        这是 Python 的一个便利特性：把对象当函数用（callable object）。
        """
        self.record_message(msg)

    # --------------------------------------------------------------------------
    def summary(self) -> str:
        if self.enabled:
            return f"轨迹已记录 {self._seq} 条 -> {self.path}"
        return f"轨迹已记录 {self._seq} 条（未落盘，仅实时分发）"


def _is_small(obj: Any, limit: int = 800) -> bool:
    """判断对象序列化后是否够小，够小才完整记进轨迹"""
    try:
        return len(json.dumps(obj, ensure_ascii=False)) <= limit
    except (TypeError, ValueError):
        return False
