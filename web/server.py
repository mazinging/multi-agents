"""
web/server.py —— 多智能体协作的可视化后端
================================================================================
【这个文件在做什么】
    启动一个本地 Web 服务，让你在浏览器里实时观看多智能体协作的全过程：
        - 时间线：谁在什么时候做了什么事（返工循环一眼可见）
        - ReAct：展开某个 Agent 的 Thought → Action → Observation 推理链
        - 黑板：数据如何在槽位间流动、版本如何演进
        - RAG：检索的内部计算（两路召回 → RRF → 阈值 → 最终结果）
        - 报告：带引用溯源，点击引用可跳转原文

【为什么用 Flask 而不是 FastAPI？】
    环境里已经装了 Flask 3.0.3，FastAPI 没装而且没网络装不了。
    更重要的是：【Flask 完全够用】。
    本项目的实时推送用的是 SSE（Server-Sent Events），
    它的本质就是"一个不关闭的 HTTP 响应，服务端持续往里写文本"，
    Flask 的生成器响应（generator response）原生支持这一点，
    不需要 WebSocket、不需要异步框架。
    【不要为了技术栈先进而选型，要为了"够用且能跑"而选型。】

【为什么用 SSE 而不是 WebSocket？】
    这是一个值得讲清楚的选型：
        SSE      : 单向（服务端 -> 浏览器），基于普通 HTTP，自动重连，实现极简
        WebSocket: 双向，需要握手协议，需要额外库

    我们的需求是【纯单向】的：服务端推送事件，浏览器只负责显示。
    用户的所有交互（启动任务）用普通 POST 请求就够了。
    所以 SSE 是正好合适的工具，用 WebSocket 属于杀鸡用牛刀。

    【工程判断：选能满足需求的最简单方案。】
    多引入一个 WebSocket 库，就多一份依赖、多一份部署复杂度、
    多一类出错可能。而它带来的双向能力我们根本用不上。

【整体数据流】
    浏览器 POST /api/run  ->  服务端在后台线程启动多智能体流程
                                    │
                                    ▼
                        每个埋点触发 tracer 分发事件
                                    │
                                    ▼
                    事件进入队列 -> SSE 线程取出 -> 推给浏览器
                                    │
                                    ▼
                          浏览器 JS 实时渲染

【一个关键的设计细节：为什么要用队列？】
    跑任务的线程和推送的线程是两个不同的线程。
    任务线程产生事件的速度可能比浏览器消费得快（尤其是 MockLLM，毫秒级）。
    如果直接推，慢客户端会阻塞任务线程 —— 【业务被可视化拖慢，
    这是绝对不能接受的】。

    所以中间加一个队列做缓冲：任务线程只管往队列里塞（永不阻塞），
    推送线程按自己的节奏取。这就是经典的"生产者-消费者"模式。

    另外每个连接有自己的队列：多个浏览器同时打开时，
    各自收到完整的事件流，互不干扰。
================================================================================
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

# 把项目根目录加入 sys.path，这样无论从哪里启动都能 import 到项目模块
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core  # noqa: F401  触发控制台编码修复
from config import CFG
from flask import Flask, Response, jsonify, request, send_from_directory

from agents import (
    AnalystAgent,
    CriticAgent,
    OrchestratorAgent,
    ResearcherAgent,
    WriterAgent,
)
from core.blackboard import Blackboard
from core.trace import TraceRecorder
from llm import build_llm
from rag import VectorStore

WEB_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=None)


# ==============================================================================
# 全局状态（单任务模式）
# ==============================================================================
#
# 【为什么用"一次只跑一个任务"的简单模型？】
#   因为这是教学工具，不是生产服务。
#   多任务并发需要处理：任务隔离、资源竞争、进度追踪…
#   这些都会让代码复杂度上升，而学习收益为零。
#   把复杂度留给真正需要它的地方 —— 这是很重要的取舍。
#
# 【但要注意：全局可变状态在多线程下是危险的】
#   所以下面所有对 STATE 的读写都加了锁。
#   不加锁会怎样？两个浏览器同时点"开始"，
#   可能一个任务的 tracer 被另一个覆盖，事件流错乱。
#   这类 bug 极难复现和调试，【在有并发的地方，一开始就加锁】。
# ==============================================================================
STATE = {
    "running": False,
    "store": None,          # 向量库（首次运行时构建，之后复用）
    "result": None,         # 最近一次运行的结果摘要
    "error": None,
    "started_at": None,
    "finished_at": None,
    "query": None,
}
STATE_LOCK = threading.Lock()

# 每个 SSE 连接一个队列。
# 用列表存所有活跃连接的队列，任务产生事件时广播给每一个。
SUBSCRIBERS: list[queue.Queue] = []
SUBSCRIBERS_LOCK = threading.Lock()


def broadcast(record: dict) -> None:
    """
    把一条轨迹事件广播给所有已连接的浏览器。

    【注意 put_nowait 和 QueueFull 的处理】
        每个队列都设了 maxsize。如果浏览器卡住不消费，
        队列会满。这时【必须丢弃新事件而不是阻塞】——
        因为阻塞会拖慢跑任务的线程，进而影响业务本身。
        可观测性永远不能反过来影响业务，这是铁律。
    """
    with SUBSCRIBERS_LOCK:
        targets = list(SUBSCRIBERS)

    for q in targets:
        try:
            q.put_nowait(record)
        except queue.Full:
            # 队列满说明该客户端消费太慢，丢弃这条事件即可。
            # 生产系统里这里应该记一个"丢了多少条"的指标，
            # 教学项目就简单处理。
            pass


def build_store(tracer: TraceRecorder) -> VectorStore:
    """
    构建向量库，并把检索埋点接到 tracer 上。

    【关键点：store.trace_hook】
        我们在 rag/store.py 里给 VectorStore 加了一个 trace_hook 参数。
        在这里把它设好，此后【每一次检索】都会自动上报内部过程，
        而 tools / agents 层完全不需要知道有埋点这回事。

        这就是依赖注入的好处：rag 层只管"在关键位置调用 hook"，
        至于 hook 干什么（记录？推送？打印？）由上层决定。
        命令行模式传 None，零开销；Web 模式传广播函数，实时可见。
    """
    # 把 RAG 埋点包装成一条标准轨迹记录。
    #
    # 【这里踩过一个坑，值得记录】
    #   我最初直接写 store = VectorStore(trace_hook=tracer.record_event)，
    #   看起来完全合理 —— record_event 接 **detail 参数，正好能收下埋点数据。
    #
    #   但它生成的是 kind="event", event="rag" 的记录，
    #   而前端的 handleEvent 是按 kind 分发的，
    #   它期望 kind="rag" 才是检索事件。
    #   结果：RAG 面板始终是空的，而且【没有任何报错】——
    #   事件确实发出来了，只是被归到了错误的类别里。
    #
    #   这就是"接口形状看起来兼容、语义却不一致"的典型 bug。
    #   record_event 的语义是"记录一个里程碑"，
    #   而检索埋点不是里程碑，是结构化的过程数据，应该有自己的一类。
    #
    #   教训：复用函数前要问"它的语义和我需要的一致吗"，
    #   而不只是"参数能不能传进去"。
    def rag_hook(payload: dict) -> None:
        tracer.record_rag(**payload)

    store = VectorStore(trace_hook=rag_hook)
    store.build_from_directory(verbose=False)
    return store


def run_pipeline(query: str) -> None:
    """
    在后台线程里跑一次完整的多智能体流程。

    【为什么必须放后台线程？】
        Flask 默认每个请求一个线程。如果直接在 POST /api/run 里跑流程，
        这个请求会一直不返回，浏览器会一直转圈，
        而且拿不到中间过程 —— 因为 SSE 连接是另一个请求。
        所以：POST 立刻返回"已启动"，真正的执行放到后台线程。
    """
    global STATE

    tracer = TraceRecorder(write_file=True)

    # 把 tracer 的所有事件接到广播函数上。
    # 【这一行是实时可视化的核心】
    #   因为埋点代码只调用了 tracer.record_xxx()，
    #   我们在这里 subscribe 一下，就把"写文件"和"推浏览器"
    #   两个消费方式同时接上了，业务代码一行都不用改。
    tracer.subscribe(broadcast)

    try:
        with STATE_LOCK:
            store = STATE["store"]

        if store is None:
            tracer.record_event("build_index_start")
            store = build_store(tracer)
            with STATE_LOCK:
                STATE["store"] = store
            tracer.record_event(
                "build_index_end",
                stats=store.stats(),
                outline=store.outline(),
            )

        # ---- 组装智能体（和 main.py 完全一致的模式）----
        llm = build_llm()
        blackboard = Blackboard()

        researcher = ResearcherAgent(llm, store, blackboard=blackboard, tracer=tracer)
        analyst = AnalystAgent(llm, store, blackboard=blackboard, tracer=tracer)
        writer = WriterAgent(llm, blackboard=blackboard, tracer=tracer)
        critic = CriticAgent(llm, store, blackboard=blackboard, tracer=tracer)

        orchestrator = OrchestratorAgent(
            llm,
            researcher=researcher,
            analyst=analyst,
            writer=writer,
            critic=critic,
            blackboard=blackboard,
            tracer=tracer,
        )

        # 把角色与工具权限也推给前端（用于绘制"角色卡片"）
        tracer.record_event(
            "agents_ready",
            agents=[
                {
                    "name": a.role,
                    "tools": a.registry.names if a.registry.names else [],
                    "steps": a.max_steps,
                }
                for a in (researcher, analyst, writer, critic)
            ],
        )

        result = orchestrator.run(query)

        # ---- 推送最终结果 ----
        tracer.record_event(
            "pipeline_summary",
            rounds=result.rounds,
            passed=result.passed,
            messages=result.messages,
            report=result.final_report,
            history=result.history,
            llm_stats=llm.stats(),
        )

        # 把黑板完整快照推给前端（供"黑板数据流"面板展示，
        # 包含每个槽位的版本历史 —— 那是观察返工改了什么的关键）
        tracer.record_event("blackboard_snapshot", slots=blackboard.snapshot())

        with STATE_LOCK:
            STATE["result"] = {
                "rounds": result.rounds,
                "passed": result.passed,
                "report": result.final_report,
                "history": result.history,
                "messages": result.messages,
            }

    except Exception as exc:      # noqa: BLE001
        # 【异常必须上报给前端】
        #   否则浏览器那边只会看到"事件流突然停了"，
        #   完全不知道为什么，非常难排查。
        #   把完整的 traceback 推过去，让错误在界面上可见。
        tb = traceback.format_exc()
        tracer.record_event("error", message=str(exc), traceback=tb)
        with STATE_LOCK:
            STATE["error"] = str(exc)

    finally:
        tracer.record_event("stream_end")
        with STATE_LOCK:
            STATE["running"] = False
            STATE["finished_at"] = time.time()


def _jsonable(obj):
    """把任意对象转成可 JSON 序列化的形式。

    【注意】core/blackboard.py 里也有一个同名函数，
    但那个处理的是"黑板槽位里的值"，是黑板的内部实现细节。
    这里处理的是"要推送给前端的事件字段"。
    两者服务不同场景，各自保留一份更清晰。

    【不过如果将来出现第三处，就应该提取成公共工具了 ——
      重复两次可以接受，三次就该抽象。】
    """
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except (TypeError, ValueError):
        return repr(obj)


# ==============================================================================
# 路由
# ==============================================================================
@app.route("/")
def index():
    """返回前端页面"""
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/api/config")
def api_config():
    """前端启动时拉取配置，用于显示当前的 mock/真实模型等信息"""
    return jsonify({
        "llm_mode": CFG.llm_mode,
        "llm_model": CFG.llm_model,
        "embedder_mode": CFG.embedder_mode,
        "embedding_dim": CFG.embedding_dim,
        "chunk_size": CFG.chunk_size,
        "chunk_overlap": CFG.chunk_overlap,
        "retrieval_top_k": CFG.retrieval_top_k,
        "max_react_steps": CFG.max_react_steps,
        "max_revision_rounds": CFG.max_revision_rounds,
    })


@app.route("/api/run", methods=["POST"])
def api_run():
    """
    启动一次任务。立刻返回，不等待执行完成。

    【为什么立刻返回？】见 run_pipeline 的说明 ——
    真正的执行在后台线程，进度通过 SSE 推送。
    这是"长任务"的标准交互模式：
        提交任务 -> 立即返回任务ID -> 通过推送/轮询获取进度
    """
    with STATE_LOCK:
        if STATE["running"]:
            return jsonify({"ok": False, "error": "已有任务正在执行，请等待完成"}), 409

        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"ok": False, "error": "问题不能为空"}), 400

        STATE["running"] = True
        STATE["error"] = None
        STATE["result"] = None
        STATE["query"] = query
        STATE["started_at"] = time.time()

    thread = threading.Thread(target=run_pipeline, args=(query,), daemon=True)
    thread.start()

    return jsonify({"ok": True})


@app.route("/api/stream")
def api_stream():
    """
    SSE 事件流。浏览器连上这里后，会持续收到轨迹事件。

    【SSE 的格式规定（必须严格遵守，否则浏览器不认）】
        每条消息形如：
            data: <内容>\n\n          <- 结尾必须是两个换行

        还可以带事件名和 id：
            event: message\n
            data: xxx\n\n

        这里我们用最简形式：所有事件都通过 data 传 JSON，
        由前端根据 JSON 里的 kind 字段自己分发。
        【为什么不用 event 字段区分？】因为 kind 已经在 JSON 里了，
        多一层机制只会增加不一致的风险。

    【为什么要发心跳（keepalive）？】
        如果长时间没有事件（比如真实模型调用要 30 秒），
        中间的网络设备或浏览器可能认为连接已死而断开。
        周期性发一个注释行（以 : 开头）可以保活。
        这是 SSE 的常规做法。
    """
    # 每个连接一个独立队列，保证多浏览器互不干扰
    q: queue.Queue = queue.Queue(maxsize=5000)
    with SUBSCRIBERS_LOCK:
        SUBSCRIBERS.append(q)

    def generate():
        try:
            # 先发一条"连接成功"，让前端知道可以开始了
            yield f"data: {json.dumps({'kind': 'connected', 'ts': time.time()})}\n\n"

            while True:
                try:
                    record = q.get(timeout=15.0)
                except queue.Empty:
                    # 15 秒没事件就发个心跳保活。
                    # 以冒号开头的行是 SSE 注释，浏览器会忽略。
                    yield ": keepalive\n\n"
                    continue

                yield f"data: {json.dumps(record, ensure_ascii=False)}\n\n"

                if record.get("kind") == "event" and record.get("event") == "stream_end":
                    # 任务结束，但【不关闭连接】——
                    # 用户可以再点一次"开始"，复用这个连接。
                    # 只发一条提示让前端更新按钮状态。
                    yield f"data: {json.dumps({'kind': 'idle', 'ts': time.time()})}\n\n"
        except GeneratorExit:
            # 浏览器主动断开
            pass
        finally:
            # 【必须清理】否则连接越积越多，内存泄漏
            with SUBSCRIBERS_LOCK:
                if q in SUBSCRIBERS:
                    SUBSCRIBERS.remove(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # 禁用 nginx 缓冲（部署时有用）
            "Connection": "keep-alive",
        },
    )


@app.route("/api/chunk/<chunk_id>")
def api_chunk(chunk_id: str):
    """
    按 chunk_id 返回原文块 —— 这是"引用溯源"能工作的基础。

    【为什么需要这个接口？】
        报告里点 [02_vector_db_cost#2] 时，前端要能显示出这个块的原文，
        这样你才能验证"报告里的这句话是不是真的来自这里"。
        这正是 RAG 防幻觉链条能被【人工核对】的关键一环：
        系统说"我有依据"，你得能一眼验证这个依据是不是真的。
    """
    with STATE_LOCK:
        store = STATE["store"]

    if store is None:
        return jsonify({"ok": False, "error": "索引尚未构建"}), 404

    chunk = store.get_chunk(chunk_id)
    if chunk is None:
        return jsonify({"ok": False, "error": f"未找到块 {chunk_id}"}), 404

    return jsonify({
        "ok": True,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "source": getattr(chunk, "source", ""),
        "position": getattr(chunk, "position", 0),
        "text": chunk.text,
    })


@app.route("/api/doc/<doc_id>")
def api_doc(doc_id: str):
    """返回某篇文档的全部块（供"精读"场景展示）"""
    with STATE_LOCK:
        store = STATE["store"]

    if store is None:
        return jsonify({"ok": False, "error": "索引尚未构建"}), 404

    chunks = store.get_doc_chunks(doc_id)
    if not chunks:
        return jsonify({"ok": False, "error": f"未找到文档 {doc_id}"}), 404

    return jsonify({
        "ok": True,
        "doc_id": doc_id,
        "chunks": [
            {"chunk_id": c.chunk_id, "position": getattr(c, "position", i), "text": c.text}
            for i, c in enumerate(chunks)
        ],
    })


@app.route("/api/outline")
def api_outline():
    """返回知识库目录（供 RAG 面板展示语料全貌）"""
    with STATE_LOCK:
        store = STATE["store"]

    if store is None:
        return jsonify({"ok": False, "error": "索引尚未构建"}), 404

    return jsonify({"ok": True, "outline": store.outline(), "stats": store.stats()})


@app.route("/api/status")
def api_status():
    """查询当前状态（前端刷新页面后用它恢复界面）"""
    with STATE_LOCK:
        return jsonify({
            "running": STATE["running"],
            "error": STATE["error"],
            "result": STATE["result"],
            "query": STATE["query"],
            "index_ready": STATE["store"] is not None,
        })


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    print("=" * 78)
    print("  多智能体协作可视化")
    print("=" * 78)
    print(f"""
  请在浏览器打开:  http://127.0.0.1:{port}

  当前配置:
    LLM 模式   : {CFG.llm_mode}""" + (
        "  (离线 MockLLM，零成本，毫秒级)"
        if CFG.llm_mode == "mock" else f"  ({CFG.llm_model})"
    ) + f"""
    向量化模式 : {CFG.embedder_mode}  (维度 {CFG.embedding_dim})
    检索 top_k : {CFG.retrieval_top_k}
    返工上限   : {CFG.max_revision_rounds}

  提示: 首次运行会自动构建索引（约 0.3 秒）。
        索引构建后会缓存在内存里，后续任务复用。

  按 Ctrl+C 停止服务。
""")

    # 【为什么 threads 模式的并发要显式打开？】
    #   SSE 连接会长期占用一个请求线程，同时跑任务的又是另一个线程。
    #   如果用单线程模式，SSE 一连上，其它所有请求（包括 POST /api/run）
    #   都会排队等待，界面直接卡死。
    #   这是 Flask 开发服务器的一个常见坑，必须显式设置 threaded=True。
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[已停止]")
        sys.exit(0)
