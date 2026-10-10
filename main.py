"""
main.py —— 程序入口：跑一次完整的多智能体协作任务
================================================================================
【这个文件的作用】
    把 rag / tools / agents / core 各层组装起来，跑一个完整的任务，
    并把全过程打印出来供你观察学习。

【建议的阅读顺序（配合本文件运行）】
    1. 先运行 `python main.py`，看完整输出，建立整体印象
    2. 再打开 trace.jsonl，看信息流动的完整时间线
    3. 然后按下述顺序读代码：
         config.py            <- 先知道有哪些开关
         rag/loader.py        <- RAG 第 1 步：切块
         rag/embedder.py      <- RAG 第 2 步：向量化
         rag/store.py         <- RAG 第 3 步：检索
         tools/rag_tools.py   <- 把检索包装成 Agent 能调用的工具
         agents/base.py       <- ★ ReAct 循环引擎（最核心）
         agents/*.py          <- 五个角色，看它们如何配置这个引擎
         agents/orchestrator.py <- 流程编排
         core/blackboard.py   <- 看 Agent 之间如何交换数据

【运行方式】
    # 离线模式（默认，零成本，推荐先跑这个）
    python main.py

    # 切换真实 LLM（需要先设 API Key）
    # PowerShell:
    $env:MA_LLM_MODE="openai"; $env:LLM_API_KEY="sk-xxx"; python main.py

    # 看更详细的日志（含完整 prompt）
    $env:MA_LOG_LEVEL="verbose"; python main.py

    # 自定义问题
    python main.py "你们的问题"

【关于 Python 解释器】
    本项目需要 numpy。如果你的默认 python 没有 numpy，
    请用有 numpy 的解释器运行，例如：
        D:\\anaconda\\python.exe main.py
================================================================================
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ------------------------------------------------------------------------------
# 把项目根目录加入 sys.path。
# 【为什么要这样做？】
#   因为我们用了 `from rag import ...` 这种"绝对导入"。
#   当你在项目目录下运行 `python main.py` 时，Python 会把 main.py 所在目录
#   加入 sys.path，所以能正常工作。
#   但如果你在别的目录用 `python path/to/main.py` 运行，就会 ImportError。
#   显式加入项目根目录，能保证无论从哪里运行都能找到模块。
#   【这是让脚本"可移植"的常见做法。】
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 导入 core 会触发 Windows 控制台编码修复（见 core/console.py）
import core  # noqa: F401,E402

from config import CFG  # noqa: E402
from agents import (  # noqa: E402
    AnalystAgent,
    CriticAgent,
    OrchestratorAgent,
    ResearcherAgent,
    WriterAgent,
)
from core.blackboard import Blackboard  # noqa: E402
from core.trace import TraceRecorder  # noqa: E402
from llm import build_llm  # noqa: E402
from rag import VectorStore  # noqa: E402

# 默认任务。
# 【为什么选这个任务？】它同时具备三个特征：
#     1. 有知识依赖      -> 必须查资料（RAG）
#     2. 需要多步查证    -> 边想边做（ReAct）
#     3. 工作可切分      -> 找资料/分析/写作/审校是不同认知模式（Multi-Agent）
# 这三个特征正好对应本项目要演示的三项技术。
DEFAULT_QUERY = (
    "我们是一个 5 人左右的小团队，想搭建一个中文企业知识库问答系统，"
    "没有专职运维人员，预算也比较有限。请问应该怎么选择向量数据库和相关技术方案？"
    "希望给出具体的选型建议和落地步骤。"
)


def banner(title: str, char: str = "=") -> None:
    """打印分节标题，让输出更有层次感"""
    print(f"\n{char * 78}")
    print(f"  {title}")
    print(f"{char * 78}")


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY

    # ==========================================================================
    # 阶段 1：环境与配置
    # ==========================================================================
    banner("多智能体 RAG-ReAct 协作系统", "█")

    print(f"""
  本示例演示三项技术的协作：
    RAG          -> 从本地知识库检索带来源的资料（防幻觉的基础）
    ReAct        -> 每个 Agent 内部都是 Thought/Action/Observation 循环
    Multi-Agent  -> 调研员 / 分析师 / 撰写者 / 审校者 分工协作，审校可打回重做

  当前配置：
    LLM 模式      : {CFG.llm_mode}""" + (
        "  (离线 MockLLM，零成本，输出稳定)"
        if CFG.llm_mode == "mock" else
        f"  ({CFG.llm_model} @ {CFG.llm_base_url})"
    ) + f"""
    Embedding 模式: {CFG.embedder_mode}  (维度 {CFG.embedding_dim})
    切块参数      : chunk_size={CFG.chunk_size}, overlap={CFG.chunk_overlap}
    检索 top_k    : {CFG.retrieval_top_k}
    ReAct 步数上限: {CFG.max_react_steps}
    审校返工上限  : {CFG.max_revision_rounds}
    轨迹记录      : {'开启 -> ' + CFG.trace_file if CFG.enable_trace else '关闭'}
""")

    print(f"  用户问题:\n    {query}\n")

    # ==========================================================================
    # 阶段 2：构建 RAG 索引
    # ==========================================================================
    banner("阶段 1/4：构建 RAG 索引（离线阶段）")

    print("  加载知识库文档并切块：")
    store = VectorStore()
    t0 = time.time()
    chunk_count = store.build_from_directory()
    build_time = time.time() - t0

    print(f"\n  {store.stats()}")
    print(f"  建索引耗时 {build_time:.2f}s")

    # ==========================================================================
    # 阶段 3：组装多智能体系统
    # ==========================================================================
    banner("阶段 2/4：组装多智能体系统")

    llm = build_llm()
    blackboard = Blackboard()
    tracer = TraceRecorder() if CFG.enable_trace else None

    # ---- 创建四个角色 ----
    # 【注意参数的不同，这体现了角色职责的边界】
    #   Researcher 需要 store（要检索）和工具
    #   Analyst    需要 store（要对比检索）
    #   Writer     不需要 store（它只加工已有信息，不检索）
    #   Critic     需要 store（要校验引用 ID 是否真实存在）
    researcher = ResearcherAgent(llm, store, blackboard=blackboard, tracer=tracer)
    analyst = AnalystAgent(llm, store, blackboard=blackboard, tracer=tracer)
    writer = WriterAgent(llm, blackboard=blackboard, tracer=tracer)
    critic = CriticAgent(llm, store, blackboard=blackboard, tracer=tracer)

    # ---- 创建协调者 ----
    orchestrator = OrchestratorAgent(
        llm,
        researcher=researcher,
        analyst=analyst,
        writer=writer,
        critic=critic,
        blackboard=blackboard,
        tracer=tracer,
    )

    # 打印各角色实际拿到的工具，让你看到"角色专精"是怎么落地的
    print("  角色与工具权限：")
    for agent in (researcher, analyst, writer, critic):
        tools = agent.registry.names if agent.registry.names else ["(无工具，纯生成)"]
        print(f"    {agent.role:<12} -> {', '.join(tools)}")

    # ==========================================================================
    # 阶段 4：执行协作流程
    # ==========================================================================
    banner("阶段 3/4：执行协作流程（在线阶段）")

    t0 = time.time()
    result = orchestrator.run(query)
    total_time = time.time() - t0

    # ==========================================================================
    # 阶段 5：结果与统计
    # ==========================================================================
    banner("阶段 4/4：最终报告")

    print(result.final_report)

    banner("运行统计")
    print(f"""
  流程耗时      : {total_time:.2f}s
  迭代轮数      : {result.rounds}  (含审校打回重做)
  审校结果      : {'✓ 通过' if result.passed else '✗ 未通过（已达返工上限）'}
  黑板消息      : {result.messages} 条
  LLM 调用统计  : {llm.stats()}
""")

    # 打印黑板状态快照，让你看到最终的数据全貌
    print(blackboard.dump_slots())

    # 打印每轮审校记录
    if result.history:
        print("\n  各轮审校记录：")
        for rec in result.history:
            print(f"    第 {rec['round']} 轮: {rec['verdict']} "
                  f"(得分 {rec['score']}) — {len(rec['issues'])} 个问题")

    if tracer:
        print(f"\n  {tracer.summary()}")

    # ==========================================================================
    # 保存最终报告
    # ==========================================================================
    out_path = ROOT / "report_output.md"
    out_path.write_text(result.final_report, encoding="utf-8")
    print(f"  📄 报告已保存: {out_path}")

    # ==========================================================================
    # 学习提示
    # ==========================================================================
    banner("接下来可以看什么", "─")
    print("""
  1. 打开 trace.jsonl，看信息流动的完整时间线：
       每条记录都有 seq 序号和 kind 字段：
         kind=event   里程碑（流程开始/结束、Agent 开始/结束）
         kind=react   ReAct 单步（thought / action / observation / final）<- 最有价值
         kind=message Agent 之间的通信
     建议用这样的命令过滤出某个 Agent 的推理轨迹：
         findstr /C:"researcher" trace.jsonl        (Windows cmd)
         Select-String "researcher" trace.jsonl     (PowerShell)

  2. 改配置做实验（改 config.py 或设环境变量）：
       MA_LOG_LEVEL=verbose   看每次送给 LLM 的完整 prompt
       MA_LLM_MODE=openai     切换真实模型，观察行为差异

  3. 跑检索质量评估，理解 RAG 的效果瓶颈：
       python eval_retrieval.py    测 Recall@5 和 MRR
       python tune_threshold.py    看阈值如何影响召回与噪音的权衡

  4. 动手改代码（建议的练习顺序）：
       a) 在 knowledge/ 里加一篇自己的文档，看它如何影响检索结果
       b) 改 config.py 里的 chunk_size / chunk_overlap，用 eval_retrieval.py 测效果变化
       c) 给 Researcher 加一个新工具（比如"统计某文档的字数"）
       d) 新增一个角色，比如 fact_checker，插到 writer 和 critic 之间
       e) 改 CFG.max_revision_rounds = 1，观察"只审一轮"对报告质量的影响
""")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n[中断] 用户取消执行")
        sys.exit(130)
    except Exception as exc:      # noqa: BLE001
        # 【顶层异常处理】
        #   把完整的堆栈打出来，方便定位问题。
        #   同时给一个友好的提示，因为初学者看到长堆栈容易不知所措。
        import traceback
        print("\n" + "=" * 78)
        print("  ❌ 程序执行出错")
        print("=" * 78)
        traceback.print_exc()
        print("\n  常见原因排查：")
        print("    1. 缺少 numpy -> 换用有 numpy 的解释器运行")
        print("       例如: D:\\anaconda\\python.exe main.py")
        print("    2. knowledge/ 目录为空或文档格式异常")
        print("    3. llm_mode='openai' 但未设置 LLM_API_KEY")
        print("    4. 控制台编码问题 -> 设置环境变量 NO_EMOJI=1 后重试")
        sys.exit(1)
