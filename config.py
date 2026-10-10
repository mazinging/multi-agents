"""
config.py —— 全局配置中心
================================================================================
【这个文件在架构里的位置】
    最底层。所有模块（llm / rag / tools / agents）都从这里读配置。
    这样做的意义：想切换"离线 mock 模式"和"真实大模型模式"，
    只需要改这一个文件里的一行，上层几十个文件一行都不用动。

【核心设计原则：面向接口编程】
    本项目有两处"可替换实现"，都通过配置切换：
        1. LLM       : MockLLM（离线，零成本）      <-> OpenAICompatLLM（真模型）
        2. Embedder  : HashingEmbedder（离线，手写） <-> 真 Embedding API
    为什么这么设计？
        学习多智能体时，真正难的是"流程对不对"，而不是"模型聪不聪明"。
        Mock 模式让你能在零成本、毫秒级、输出稳定的前提下，把整个
        RAG -> ReAct -> 多 Agent 协作 的链路反复跑通、打断点、看日志。
        等流程完全理解了，再切真模型去观察"智能"带来的差异。
================================================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------------------------
# 项目根目录：__file__ 是本文件路径，.parent 就是项目根
# 用绝对路径而不是相对路径，是为了避免"在别的目录下运行脚本"导致找不到文件
# ------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    """
    用 dataclass 而不是普通 class 的原因：
        - 自动生成 __init__，字段一目了然
        - 默认值 + 类型标注 = 自带文档
        - 可以直接 print(cfg) 看到全部配置，方便调试
    """

    # ==========================================================================
    # 一、LLM 配置
    # ==========================================================================

    # 【最重要的开关】"mock" = 离线假模型；"openai" = 真模型
    # 建议学习顺序：先 mock 跑通 -> 再切 openai 对比差异
    llm_mode: str = "mock"

    # ---- 以下仅在 llm_mode == "openai" 时生效 ----
    # OpenAI 兼容协议意味着：DeepSeek / Moonshot / 本地 Ollama / vLLM 都能用，
    # 只要它们提供 /v1/chat/completions 这个接口
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", ""))
    llm_temperature: float = 0.2   # 调低温度：Agent 需要稳定、可复现的输出
    llm_timeout: int = 60

    # ==========================================================================
    # 二、Embedding 配置（RAG 的"向量化"环节）
    # ==========================================================================

    # "hashing" = 纯手写、离线、零依赖；"openai" = 调真 embedding 接口
    embedder_mode: str = "hashing"

    # 向量维度。手写 hashing 方案下，维度越大冲突越少但越占内存。
    # 512 对小语料（几十篇文档）足够了。
    embedding_dim: int = 512

    embedding_model: str = "text-embedding-3-small"   # 仅 embedder_mode=="openai" 时用

    # ==========================================================================
    # 三、RAG 切块配置
    # ==========================================================================

    chunk_size: int = 300      # 每个文本块的目标字符数
    chunk_overlap: int = 60    # 相邻块的重叠字符数

    # 为什么需要 overlap？
    #   如果一句话正好被切成两半，"前半句在块 A、后半句在块 B"，
    #   那么无论召回 A 还是 B，语义都是残缺的，检索质量会明显下降。
    #   让相邻块重叠一部分，能保证任何一句话至少完整地出现在某一个块里。

    retrieval_top_k: int = 4   # 每次检索返回几个最相似的块

    # ==========================================================================
    # 四、ReAct 循环配置
    # ==========================================================================

    # 单个 Agent 最多执行多少轮 Thought->Action->Observation。
    # 【必须有这个上限】否则模型可能陷入死循环（一直重复搜同一个词），
    # 导致无限调用、无限烧钱。这是 ReAct 实现里最容易被忽略的防护。
    max_react_steps: int = 6

    # 工具执行失败后，允许重试几次
    max_tool_retries: int = 2

    # 解析 LLM 输出失败时（格式不对），重试几次
    max_parse_retries: int = 2

    # ==========================================================================
    # 五、多 Agent 协作配置
    # ==========================================================================

    # Critic 审校不通过时，最多打回重做几轮。
    # 这个循环是"多 Agent 相对单 Agent 的真正增益点"：
    #   单 Agent 写完就结束了，没人检查；
    #   多 Agent 有独立视角的 Critic，能发现问题并触发返工。
    max_revision_rounds: int = 2

    # Critic 的通过分数线（0~100）
    critic_pass_score: int = 70

    # ==========================================================================
    # 六、日志与轨迹
    # ==========================================================================

    # 是否把每一步详细轨迹写入 trace.jsonl
    # 这份文件是学习时最有用的东西：可以完整复盘"信息如何在 Agent 间流动"
    enable_trace: bool = True
    trace_file: str = "trace.jsonl"

    # 控制台日志详细程度: "quiet" | "normal" | "verbose"
    # verbose 会把每次送给 LLM 的完整 prompt 也打出来
    log_level: str = "normal"

    # ==========================================================================
    # 七、路径
    # ==========================================================================

    @property
    def knowledge_dir(self) -> Path:
        return ROOT_DIR / "knowledge"

    @property
    def trace_path(self) -> Path:
        return ROOT_DIR / self.trace_file


# ------------------------------------------------------------------------------
# 全局单例配置对象
# 其他模块统一 `from config import CFG` 使用
# ------------------------------------------------------------------------------
CFG = Config()


# ------------------------------------------------------------------------------
# 支持环境变量覆盖：方便在不改代码的情况下临时切换模式
#   例如 PowerShell:  $env:MA_LLM_MODE="openai"; python main.py
# ------------------------------------------------------------------------------
if os.environ.get("MA_LLM_MODE"):
    CFG.llm_mode = os.environ["MA_LLM_MODE"]
if os.environ.get("MA_EMBEDDER_MODE"):
    CFG.embedder_mode = os.environ["MA_EMBEDDER_MODE"]
if os.environ.get("MA_LOG_LEVEL"):
    CFG.log_level = os.environ["MA_LOG_LEVEL"]
