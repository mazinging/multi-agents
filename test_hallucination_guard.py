"""
回归测试：验证程序化防幻觉机制真的能拦住编造的引用。

【这个测试为什么重要？】
    Critic 的 programmatic_check 是整个防幻觉链条的最后一道闸门。
    如果它失灵（比如正则写错、集合运算写反），
    编造的引用就会顺利流到最终报告里，而你不会收到任何警告。
    这是"安静地失效"的典型场景 —— 必须用测试固定住它的行为。
"""
import sys
sys.path.insert(0, r"E:\CodexProject\Multi_Agents")
import core
from rag import VectorStore
from agents.critic import CriticAgent
from llm import MockLLM

store = VectorStore()
store.build_from_directory(verbose=False)

critic = CriticAgent(MockLLM(), store)

# 证据范围内可用的引用（模拟 Researcher 实际看到的）
available = ["01_vector_db_overview#0", "02_vector_db_cost#2"]

cases = [
    ("全部有效引用",
     "结论 A [01_vector_db_overview#0]，结论 B [02_vector_db_cost#2]",
     [], []),
    ("包含编造的引用（知识库中不存在）",
     "结论 A [99_不存在文档#7]，结论 B [02_vector_db_cost#2]",
     ["99_不存在文档#7"], []),
    ("存在但超出调研范围",
     "结论 A [05_rag_architecture#3]",
     [], ["05_rag_architecture#3"]),
    ("无任何引用",
     "这是一段没有任何引用的结论。",
     [], []),
]

print("=" * 74)
ok = True
for name, draft, want_invalid, want_oos in cases:
    r = critic.programmatic_check(draft, available)
    good = (r["invalid"] == want_invalid and r["out_of_scope"] == want_oos)
    ok = ok and good
    print(f"{'✓' if good else '✗'} {name}")
    print(f"    invalid={r['invalid']}  越界={r['out_of_scope']}  总引用={r['total_citations']}")

# 关键测试：程序化校验必须能否决 LLM 的 pass 判定
print()
print("=" * 74)
print("关键测试：LLM 说 pass，但代码发现编造引用 -> 必须强制 revise")
critic.blackboard = None
from core.blackboard import Blackboard, SLOT_DRAFT, SLOT_EVIDENCE
bb = Blackboard()
bb.write(SLOT_DRAFT, {"content": "结论 [99_不存在文档#7]"}, writer="writer")
bb.write(SLOT_EVIDENCE, {"citations": available}, writer="researcher")
critic.blackboard = bb

result = critic.postprocess(
    '{"verdict": "pass", "score": 95, "issues": [], "suggestions": [], "reviewer_notes": "看起来不错"}',
    [],
)
final = bb.read("critique")
print(f"  LLM 原始判定 : pass (95分)")
print(f"  代码最终判定 : {final['verdict']}")
print(f"  判定依据     : {final.get('programmatic')}")
veto_ok = final["verdict"] == "revise"
ok = ok and veto_ok
print(f"  {'✓ 否决生效' if veto_ok else '✗ 否决失效！编造引用被放过'}")

print()
print("=" * 74)
print("全部通过 ✓" if ok else "存在失败 ✗")
sys.exit(0 if ok else 1)
