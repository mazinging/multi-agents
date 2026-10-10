"""
agents/__init__.py —— 多智能体包
================================================================================
【本包的结构，以及一个非常重要的设计决策】

    base.py         : BaseReActAgent —— ReAct 循环引擎（所有 Agent 共用）
    orchestrator.py : 协调者 —— 拆解任务、派活、汇总、判定返工
    researcher.py   : 调研员 —— 检索取证
    analyst.py      : 分析师 —— 横向对比分析
    writer.py       : 撰写者 —— 组织成报告
    critic.py       : 审校者 —— 挑错、评分、决定是否打回

【核心设计：所有 Agent 共用同一个 ReAct 引擎】
    这是本项目最重要的架构决策，请务必理解：

    常见误区是把每个 Agent 写成一个独立的类，各自实现一套循环逻辑。
    结果就是 5 份高度相似但细节不同的代码 —— 改一个 bug 要改 5 处，
    而且很难说清"这 5 个 Agent 到底有什么本质区别"。

    本项目的做法是：
        BaseReActAgent 负责【怎么想】（ReAct 循环机制）
        子类只定义【想什么、能做什么】：
            - role            角色名
            - system_prompt   角色的人设与任务目标
            - registry        它能用的工具子集
            - build_task()    怎么把上游数据变成自己的任务

    【这个设计带来的认知收获】
        你会发现"多智能体"并没有那么神秘 ——
        它本质上就是"同一个推理引擎 + 不同的角色设定 + 不同的工具权限"。
        真正困难的部分不是"让 Agent 会思考"（那是 ReAct 引擎的活），
        而是【角色怎么划分、信息怎么传递、质量怎么保证】。

    这正是本项目想让学习者体会到的东西：
        多智能体的复杂度在【协作设计】，不在【单个 Agent 的实现】。
================================================================================
"""

from agents.base import BaseReActAgent, AgentRunResult
from agents.orchestrator import OrchestratorAgent
from agents.researcher import ResearcherAgent
from agents.analyst import AnalystAgent
from agents.writer import WriterAgent
from agents.critic import CriticAgent

__all__ = [
    "BaseReActAgent",
    "AgentRunResult",
    "OrchestratorAgent",
    "ResearcherAgent",
    "AnalystAgent",
    "WriterAgent",
    "CriticAgent",
]
