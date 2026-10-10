"""
core/console.py —— 控制台输出兼容层（Windows 必需）
================================================================================
【为什么需要这个文件？】
    Windows 的中文控制台默认编码是 GBK（代码页 936）。
    当程序往控制台打印 UTF-8 字符（尤其是 emoji，如 📄 🔍 ✅）时：
        UnicodeEncodeError: 'gbk' codec can't encode character ...
    程序直接崩溃。

    这是一个非常典型的"代码逻辑全对，但一跑就崩"的环境问题。
    解决办法有两种：
        1. 改系统设置（chcp 65001）—— 不可靠，换台机器又不行
        2. 在程序内把 stdout/stderr 重新包装成 UTF-8 —— 可靠，本项目采用
    另外还要处理"终端字体不支持 emoji"的情况，所以提供降级开关。

【工程原则：环境适配应该在程序启动时一次性做完，
  而不是在每个 print 处做 try/except】—— 后者会把业务代码弄得很脏。
================================================================================
"""

from __future__ import annotations

import io
import sys

# 是否使用 emoji。某些 Windows 终端（尤其是旧版 cmd）显示 emoji 会变成方块，
# 自动检测失败时可以通过环境变量 NO_EMOJI=1 关闭
_USE_EMOJI = True


def setup_console() -> None:
    """
    在程序最开头调用一次，修复 Windows 控制台编码问题。

    实现原理：
        sys.stdout 默认是 TextIOWrapper，绑定了系统的编码（GBK）。
        我们用 io.TextIOWrapper 把底层的二进制缓冲区重新包一层，
        强制指定 encoding="utf-8" 和 errors="replace"。

        errors="replace" 很关键：即使遇到无法编码的字符，
        也只是显示成 "?"，而不是抛异常让程序崩溃。
        对于日志输出，"显示难看" 远好于 "程序挂掉"。
    """
    global _USE_EMOJI

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue

        # 已经是 UTF-8 就不用处理（比如在 Linux/macOS 或已配置好的终端）
        enc = (getattr(stream, "encoding", None) or "").lower()
        if enc.replace("-", "") == "utf8":
            continue

        # 需要底层有 buffer 才能重新包装。
        # 在某些环境（比如被重定向到 StringIO，或 IDE 的伪终端）没有 buffer，
        # 这时跳过即可，不能强行赋值，否则会破坏输出。
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            continue

        try:
            setattr(
                sys,
                stream_name,
                io.TextIOWrapper(
                    buffer,
                    encoding="utf-8",
                    errors="replace",   # 关键：编码失败也不崩溃
                    line_buffering=True,
                ),
            )
        except (ValueError, OSError):
            # 重新包装失败（极少数环境），静默跳过。
            # 此时可能后续打印 emoji 会失败，所以顺手关掉 emoji
            _USE_EMOJI = False

    # 即使编码修好了，也可能遇到"终端字体不显示 emoji"的情况，
    # 所以再做一个可用性探测：往 stderr 试写一个 emoji，
    # 失败就把 emoji 全局关掉。用 stderr 是为了不污染 stdout。
    _probe_emoji()


def _probe_emoji() -> None:
    """
    探测当前输出流能否真正写出 emoji。

    【为什么需要这一步？】
        编码能通过（UTF-8 允许 emoji），不代表终端能显示 ——
        旧版 Windows 控制台会把 emoji 渲染成两个方块，比不显示还难看。
        所以这里实际写一次到 stderr 试探。写 stderr 而非 stdout，
        是为了不干扰可能被重定向的 stdout 数据流。
    """
    global _USE_EMOJI
    import os

    if os.environ.get("NO_EMOJI"):
        _USE_EMOJI = False
        return

    try:
        sys.stderr.write("\u200b")   # 零宽空格，安全且能触发编码检查
        sys.stderr.flush()
    except (UnicodeEncodeError, ValueError, OSError):
        _USE_EMOJI = False


# ------------------------------------------------------------------------------
# emoji 辅助：让业务代码统一调用 emoji("📄", "[doc]")，
# 由这里决定到底输出 emoji 还是 ASCII 替代文本
# ------------------------------------------------------------------------------
def emoji(symbol: str, fallback: str = "") -> str:
    """
    返回 emoji 或它的纯文本替代。

    用法：
        print(f"{emoji('📄', '[doc]')} 加载文档...")
    """
    import os

    if not _USE_EMOJI or os.environ.get("NO_EMOJI"):
        return fallback
    return symbol
