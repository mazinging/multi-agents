"""
tools/base.py —— 工具抽象基类与注册表
================================================================================
【这个文件在架构里的位置】
    定义了"什么是工具"以及"怎么管理工具"。所有具体工具都继承 BaseTool。

【为什么要有 ToolResult 而不是直接返回字符串？】
    如果把"执行成功的结果"和"执行失败的错误"都揉成一个字符串返回，
    上层就没法区分这两种情况：
        - 成功但结果为空（"知识库确实没有这个内容"）
        - 失败（"参数传错了，你应该重试"）
    这两种情况对 Agent 的后续决策【完全不同】：
        成功但为空 -> 应该换个关键词再查，或者放弃
        失败       -> 应该修正参数后重试
    用结构化返回值明确区分，Agent 才能做出正确决策。
    这是"错误处理不能靠字符串约定"这一原则的体现。

【ToolRegistry 为什么要存在？】
    a) Agent 需要知道"有哪些工具可用"，才能选对工具。
       注册表能自动生成工具清单，塞进 prompt。
    b) 防止模型调用不存在的工具。模型有时会"发明"工具名，
       注册表能在执行前拦截并给出清晰的错误。
    c) 权限控制：不同角色的 Agent 可以只注册它能用的工具子集。
       这是多智能体里"角色专精"的技术实现手段。
================================================================================
"""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """
    工具执行结果。

    【字段设计说明】
        ok       : 是否成功。Agent 据此决定"重试"还是"继续"
        content  : 给模型看的文本内容（成功时的结果 / 失败时的错误说明）
        data     : 结构化数据（供程序消费，不进 prompt）。
                   比如检索到的 Chunk 对象列表 —— 程序需要用它做引用校验，
                   但不需要把整个对象塞进 prompt。
        meta     : 元信息（耗时、命中数等），用于日志和调试

    【为什么 content 和 data 要分开？】
        这是"给人/模型看"和"给程序用"两种用途的分离。
        模型需要的是简洁、带上下文的自然语言文本；
        程序需要的是精确的对象和字段。
        混在一起会导致：要么 prompt 里塞满 JSON（浪费 token 且模型难读），
        要么程序被迫解析自然语言（脆弱）。
        【这个分离在多智能体工程里非常关键】
    """
    ok: bool = True
    content: str = ""
    data: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, content: str, data: Any = None, **meta) -> "ToolResult":
        return cls(ok=True, content=content, data=data, meta=meta)

    @classmethod
    def failure(cls, content: str, **meta) -> "ToolResult":
        return cls(ok=False, content=content, data=None, meta=meta)


class BaseTool(ABC):
    """
    所有工具的基类。

    子类必须实现：
        name        : 工具名（模型调用时用的标识）
        description : 工具用途说明（会进 prompt，所以要写清楚"什么时候该用它"）
        parameters  : 参数说明（同样进 prompt）
        run          : 实际执行逻辑

    【为什么用 @property 而不是类属性定义 name/description？】
        两者都可以。用 property 的好处是子类可以动态生成
        （比如描述里带上当前知识库的文档列表），更灵活。
        本项目为了让子类写起来简洁，用类属性 + 抽象 property 的混合方式，
        实际上类属性就够了 —— 这里用抽象方法是为了强制子类必须定义，
        否则会在实例化时报错，而不是等到 prompt 拼装时才发现工具没有描述。
    """

    # 子类覆盖这三个
    name: str = ""
    description: str = ""
    parameters: str = ""

    @abstractmethod
    def run(self, arg: str) -> ToolResult:
        """
        执行工具。

        【为什么参数是单一的 str，而不是 **kwargs？】
            这是一个刻意的设计取舍：
                好处 —— 模型只需要输出一行文本作为参数，
                        解析极其简单，不会出现 JSON 格式错误。
                        （回想 llm.py 里 parse_json_loose 的复杂度，
                         就知道让模型输出结构化参数有多容易出错）
                代价 —— 复杂参数需要自己约定格式（比如用逗号分隔）。

            对本项目来说，所有工具都只需要一个"查询词"或"文档ID"，
            单一字符串参数完全够用，而且大幅提高了可靠性。
            【工程原则：为模型的可靠性做设计，而不是为理论上的优雅】
            如果你的工具真需要多参数，建议用 JSON 字符串，
            但一定要像 llm.py 那样写容错解析。
        """
        raise NotImplementedError

    def spec(self) -> str:
        """
        生成给模型看的工具说明。

        【格式为什么要这样设计？】
            用固定的小标题（用途/参数）而不用大段散文，
            因为结构化的描述更容易被模型正确理解。
            实测中，格式清晰的工具说明能明显降低模型选错工具的概率。
        """
        lines = [f"### 工具名: {self.name}", f"用途: {self.description}"]
        if self.parameters:
            lines.append(f"参数: {self.parameters}")
        return "\n".join(lines)

    def safe_run(self, arg: str) -> ToolResult:
        """
        带异常保护的执行入口。

        【为什么要有这一层？】
            Agent 不该因为一个工具抛异常就整个流程崩溃。
            工具内部可能有各种意外：文件不存在、编码错误、网络超时……

            这里统一捕获所有异常，转成 ok=False 的 ToolResult，
            并且【把异常信息转成对模型友好的说明】。
            同时保留 traceback 到 meta 里，供开发者调试。

            【关键设计】异常信息不能原样返回给模型。
            Python 的堆栈信息对模型毫无意义，反而会干扰它。
            我们要返回的是"发生了什么 + 你该怎么办"。
        """
        try:
            result = self.run(arg)
            # 防御性检查：子类可能忘了返回 ToolResult
            if not isinstance(result, ToolResult):
                return ToolResult.failure(
                    f"工具 {self.name} 返回值类型错误（期望 ToolResult，"
                    f"实际 {type(result).__name__}）。这是程序 bug，不是你的问题。"
                )
            return result

        except Exception as e:      # noqa: BLE001 —— 这里确实要捕获所有异常
            # 把异常转成模型能理解的说明
            friendly = _friendly_error(e, self.name)
            return ToolResult.failure(
                friendly,
                exception=type(e).__name__,
                detail=str(e),
                traceback=traceback.format_exc(),
            )


def _friendly_error(exc: Exception, tool_name: str) -> str:
    """
    把 Python 异常翻译成"模型能看懂并据此行动"的提示。

    【这个函数体现了"错误信息是 prompt 的一部分"这一理念】
        差的错误信息:  "KeyError: 'doc_03'"
        好的错误信息:  "未找到文档 doc_03。请先调用 search_local_docs
                        找到正确的文档 ID 再重试。"
        后者能引导模型自我纠正，前者只会让它困惑或重复犯错。
    """
    name = type(exc).__name__
    msg = str(exc)

    if name == "FileNotFoundError":
        return f"工具 {tool_name} 所需的文件不存在：{msg}。请检查路径或改用其他工具。"
    if name == "ValueError":
        return f"工具 {tool_name} 参数有问题：{msg}。请调整参数后重试。"
    if name == "KeyError":
        return f"工具 {tool_name} 找不到指定的资源：{msg}。请先用检索工具确认可用选项。"
    if name == "TimeoutError":
        return f"工具 {tool_name} 执行超时。可以重试一次，或改用其他方式获取信息。"
    if name in ("ConnectionError", "URLError"):
        return f"工具 {tool_name} 网络不可达。请改用本地工具获取信息。"

    return f"工具 {tool_name} 执行出错（{name}）：{msg}"


class ToolRegistry:
    """
    工具注册表。

    用法：
        registry = ToolRegistry()
        registry.register(SearchTool(store))
        ...
        result = registry.execute("search_local_docs", "向量数据库")
        print(registry.catalog())      # 生成给模型的工具清单
    """

    def __init__(self) -> None:
        # 用 dict 而不是 list：注册和查找都是 O(1)，
        # 而且天然保证工具名唯一（重复注册会覆盖，见下面的检查）
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        注册一个工具。

        【为什么要检查重复注册？】
            静默覆盖会导致很难查的 bug：你注册了新版工具，
            但旧版不知何时又被注册了一次，结果用的是旧版。
            直接报错能让问题在启动时就暴露。
        """
        if not tool.name:
            raise ValueError(f"工具 {type(tool).__name__} 没有定义 name 属性")
        if tool.name in self._tools:
            raise ValueError(
                f"工具名重复: '{tool.name}'。"
                f"已注册的类型是 {type(self._tools[tool.name]).__name__}，"
                f"本次要注册 {type(tool).__name__}。请改名或用不同的注册表。"
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def execute(self, name: str, arg: str) -> ToolResult:
        """
        执行指定工具。

        【这是防御模型"发明工具"的关键位置】
            模型有时会调用一个不存在的工具（尤其是在长对话里
            或者工具很多的时候）。这时不能让它崩，
            而要返回一个明确的错误 + 可用工具清单，
            引导它下一轮选对。
        """
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools.keys()) or "(无)"
            return ToolResult.failure(
                f"工具 '{name}' 不存在。可用的工具只有：{available}。"
                f"请从这些工具中选择一个。"
            )
        return tool.safe_run(arg)

    def subset(self, names: list[str]) -> "ToolRegistry":
        """
        创建一个只含指定工具的子注册表。

        【用途：实现角色权限隔离】
            调研员只需要检索工具，不需要"写报告"工具；
            撰写者需要"读证据"工具，不需要"检索"工具。
            给每个 Agent 只注册它该用的工具，有两个好处：
                1. prompt 更短（工具说明不用全塞），模型更聚焦
                2. 从机制上防止 Agent 越权做事
                   （比如 Writer 直接去检索而不看证据 —— 那会破坏分工）

            【注意】这不是安全边界（Python 里没有真正的私有），
            而是"降低模型犯错的概率"。这是给模型设计约束，
            和给攻击者设防是两回事。
        """
        sub = ToolRegistry()
        for n in names:
            tool = self._tools.get(n)
            if tool is None:
                raise ValueError(f"subset 请求了未注册的工具: '{n}'")
            sub.register(tool)
        return sub

    def catalog(self) -> str:
        """
        生成完整的工具清单，用于塞进 Agent 的 prompt。

        【为什么按注册顺序输出而不是排序？】
            保持注册顺序能让开发者通过调整注册顺序来控制
            工具在 prompt 里的呈现顺序。实测中，
            重要的工具放在前面更容易被模型选中。
            这是一个很简单但有效的提示工程技巧。
        """
        if not self._tools:
            return "（当前没有可用工具）"
        return "\n\n".join(tool.spec() for tool in self._tools.values())

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
