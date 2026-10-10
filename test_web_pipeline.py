"""
验证 Web 服务的完整链路：POST /api/run -> SSE 事件流 -> 各阶段数据齐全。

【为什么要写这个测试？】
    Web 功能涉及三个并发角色：
        1. Flask 请求线程（POST /api/run）
        2. 跑任务的后台线程
        3. SSE 推送线程
    它们之间靠队列和锁通信。这种结构最容易出的问题是
    "某类事件根本没被推出来" —— 而界面上只会表现为
    "某个面板一直是空的"，你很难判断是前端渲染问题还是后端没发。

    所以用一个脚本把事件流抓下来，直接检查每类事件是否都到了。
    这比开浏览器用肉眼看可靠得多。

    事实上这个测试立刻就抓到了一个真实 bug：
    RAG 埋点被错误地归类成了 kind="event"，导致前端 RAG 面板永远空白，
    而且【没有任何报错】。见 web/server.py 里 build_store 的注释。

【用法】
    先启动服务:  python web/server.py 5011
    再运行本测试: python test_web_pipeline.py [端口]
"""

import json
import sys
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: F401  触发控制台 UTF-8 编码修复

BASE = "http://127.0.0.1:" + (sys.argv[1] if len(sys.argv) > 1 else "5011")
QUERY = "我们是一个小团队，想搭建中文企业知识库问答系统，应该怎么选向量数据库？"

events = []
lock = threading.Lock()


def listen():
    """连上 SSE，把事件收进列表，直到这一轮结束"""
    req = urllib.request.Request(BASE + "/api/stream")
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                rec = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            with lock:
                events.append(rec)
            if rec.get("kind") == "idle":
                return


def post_run(query):
    body = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/api/run",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


t = threading.Thread(target=listen, daemon=True)
t.start()
time.sleep(0.6)   # 等 SSE 连上

print("POST /api/run ...")
print("  ", post_run(QUERY))

t.join(timeout=90)

with lock:
    got = list(events)

print(f"\n共收到 {len(got)} 条事件\n")

kinds = Counter(r.get("kind") for r in got)
print("按 kind 分类:", dict(kinds))

# ---- 1. ReAct 事件 ----
react = [r for r in got if r.get("kind") == "react"]
phases = Counter(r.get("phase") for r in react)
agents = Counter(r.get("agent") for r in react)
print(f"\nReAct 事件 {len(react)} 条, phase 分布: {dict(phases)}")
print(f"  涉及 Agent: {dict(agents)}")

# ---- 2. RAG 埋点 ----
rag = [r for r in got if r.get("kind") == "rag"]
stages = Counter(r.get("stage") for r in rag)
print(f"\nRAG 埋点 {len(rag)} 条, stage 分布: {dict(stages)}")

# ---- 3. 里程碑 ----
milestones = [r.get("event") for r in got if r.get("kind") == "event"]
print(f"\n里程碑事件: {milestones}")

# ---- 4. 黑板快照 ----
bb = [r for r in got
      if r.get("kind") == "event" and r.get("event") == "blackboard_snapshot"]
if bb:
    slots = bb[-1]["slots"]
    print(f"\n黑板快照: {len(slots)} 个槽位")
    for k, v in slots.items():
        print(f"    {k:<20} v{v['latest_version']}  ({len(v['versions'])} 个版本)")
else:
    print("\n[X] 没有收到黑板快照")

# ---- 5. 最终报告 ----
summary = [r for r in got
           if r.get("kind") == "event" and r.get("event") == "pipeline_summary"]
if summary:
    s = summary[-1]
    print(f"\n最终报告: {len(s['report'])} 字符 | {s['rounds']} 轮 | 通过={s['passed']}")
    print(f"  审校历史: {[(h['round'], h['verdict'], h['score']) for h in s['history']]}")
    print(f"  LLM 统计: {s.get('llm_stats')}")
else:
    print("\n[X] 没有收到最终报告")

# ---- 6. 错误 ----
errs = [r for r in got
        if r.get("kind") == "event" and r.get("event") == "error"]
if errs:
    print(f"\n[X] 出现错误: {errs[-1].get('message')}")
    print(errs[-1].get("traceback", "")[:1500])

# ---- 结论 ----
print("\n" + "=" * 70)
checks = {
    "ReAct 四阶段齐全(thought/action/observation/final)":
        set(phases) >= {"thought", "action", "observation", "final"},
    "四个角色都参与(researcher/analyst/writer/critic)":
        set(agents) >= {"researcher", "analyst", "writer", "critic"},
    "RAG 埋点齐全(recall/fuse/prune/final)":
        set(stages) >= {"recall", "fuse", "prune", "final"},
    "收到黑板快照": bool(bb),
    "收到最终报告": bool(summary),
    "无错误": not errs,
}
for name, ok in checks.items():
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")
print("=" * 70)
print("全部通过" if all(checks.values()) else "存在失败")
sys.exit(0 if all(checks.values()) else 1)
