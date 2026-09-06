"""E15: 高阶顺序表示 - L2 三元组上下文 (双前缀 -> 下一个)。

四问: 基底参与=高阶结构计数; 外部 stub=无(只读分布不选句);
结论归谁=基底表示能力; 停点=分布反映 2 前缀上下文即可。

补充约束(已确认):
  1) 滑动窗口锁定: 相邻三元组, 非相邻不计数;
  2) 三元组存储膨胀观测(样本量 vs 不同三元组种类数);
  3) 边界声明: 只解决双前缀; 3 前缀以上(如 X 在 A,B 之前两格)不在 E15;
  4) 全程校验 L0/L1 基线未改(行为快照回归)。

场景(2 前缀窗口, 情景必须紧贴决策点):
  S1 上下文分叉: [X1,B,C]x100 + [X2,B,D]x100
      -> L1 只看 B: C=D 等强; L2 看 [X1,B]: 只有 C; [X2,B]: 只有 D;
  S2 前缀泛化: [A,B,C]/[A,B,D]/[Q,B,E] 各 x100
      -> [A,B] -> C=D, E=0; [Q,B] -> E, C/D=0;
  S3 能力边界: [X,A,B,C]x100 + [Y,A,B,D]x100 (X/Y 在 3 前缀外)
      -> [A,B] 仍 C=D 等强, L2 看不到 X/Y (记录, 不在 E15 解决);
  S4 存储膨胀观测; S5 滑动窗口锁定; S6 基线行为回归。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_layer import ContextLayer
from experiments.checker import Checker
from order_layer import OrderLayer
from synapse_net import SynapseNet


def run() -> dict:
    c = Checker()
    data: dict = {}

    # ---------- S5 滑动窗口锁定 ----------
    cl0 = ContextLayer()
    cl0.learn(["s0", "s1", "s2", "s3"])
    c.check(
        "S5 滑动窗口锁定: 只计相邻三元组 (s0,s1,s2)、(s1,s2,s3)",
        cl0.count("s0", "s1", "s2") == 1
        and cl0.count("s1", "s2", "s3") == 1
        and cl0.count("s0", "s1", "s3") == 0
        and cl0.count("s0", "s2", "s3") == 0,
        f"n(s0,s1,s2)=1, n(s1,s2,s3)=1; 非相邻 = 0",
    )

    # ---------- S1 上下文分叉 (情景紧贴决策点) ----------
    cl1 = ContextLayer()
    for _ in range(100):
        cl1.learn(["X1", "B", "C"])
        cl1.learn(["X2", "B", "D"])
    c.check(
        "S1 存储: n(X1,B,C)=100, n(X2,B,D)=100",
        cl1.count("X1", "B", "C") == 100 and cl1.count("X2", "B", "D") == 100,
        f"n(X1,B,C)={cl1.count('X1', 'B', 'C')}, n(X2,B,D)={cl1.count('X2', 'B', 'D')}",
    )
    q1 = cl1.query(["X1", "B"])
    q2 = cl1.query(["X2", "B"])
    c.check(
        "S1 L2: [X1,B] 只有 C; [X2,B] 只有 D (双前缀消歧)",
        set(q1) == {"C"} and math.isclose(q1["C"], 1.0, rel_tol=1e-9)
        and set(q2) == {"D"} and math.isclose(q2["D"], 1.0, rel_tol=1e-9),
        f"[X1,B]->{q1}; [X2,B]->{q2}",
    )
    # L1 对照: 只看当前 B
    l1 = OrderLayer()
    for _ in range(100):
        l1.learn(["X1", "B", "C"])
        l1.learn(["X2", "B", "D"])
    l1_b = l1.activate("B")
    c.check(
        "S1 L1 对照: 只看 B -> C=D 等强 (洞在 L1, L2 补上)",
        math.isclose(l1_b["C"], l1_b["D"], rel_tol=1e-9),
        f"C={l1_b['C']:.4f}, D={l1_b['D']:.4f}, ratio={l1_b['C'] / l1_b['D']:.4f}",
    )
    data["s1"] = {"query_X1B": q1, "query_X2B": q2, "L1_C": l1_b["C"], "L1_D": l1_b["D"]}

    # ---------- S2 前缀泛化 ----------
    cl2 = ContextLayer()
    for _ in range(100):
        cl2.learn(["A", "B", "C"])
        cl2.learn(["A", "B", "D"])
        cl2.learn(["Q", "B", "E"])
    q_ab = cl2.query(["A", "B"])
    q_qb = cl2.query(["Q", "B"])
    c.check(
        "S2 [A,B] -> C=D 等强、E=0; [Q,B] -> E、C/D=0 (第二前缀才够)",
        set(q_ab) == {"C", "D"}
        and math.isclose(q_ab["C"], 1.0, rel_tol=1e-9)
        and math.isclose(q_ab["D"], 1.0, rel_tol=1e-9)
        and set(q_qb) == {"E"},
        f"[A,B]->{q_ab}; [Q,B]->{q_qb}",
    )
    data["s2"] = {"query_AB": q_ab, "query_QB": q_qb}

    # ---------- S3 能力边界 (3 前缀以外) ----------
    cl3 = ContextLayer()
    for _ in range(100):
        cl3.learn(["X", "A", "B", "C"])
        cl3.learn(["Y", "A", "B", "D"])
    q3 = cl3.query(["A", "B"])
    c.check(
        "S3 能力边界: X/Y 在 3 前缀外 -> [A,B] 仍 C=D 等强 (记录, 不在 E15 解决)",
        set(q3) == {"C", "D"} and math.isclose(q3["C"], 1.0, rel_tol=1e-9)
        and math.isclose(q3["D"], 1.0, rel_tol=1e-9),
        f"[A,B]->{q3} (需 3 前缀才能看到 X/Y, 属未来课题)",
    )
    data["s3"] = {"query_AB": q3}

    # ---------- S4 存储膨胀观测 ----------
    alpha = "ABCDEFGH"
    sizes = []
    cl4 = ContextLayer()
    total = 0
    for i in range(2000):
        seq = [alpha[i % 8], alpha[(i * 3 + 1) % 8], alpha[(i * 5 + 2) % 8], alpha[(i * 7 + 3) % 8]]
        cl4.learn(seq)
        total += 1
        if total in (10, 100, 500, 2000):
            sizes.append({"samples": total, "prefix_types": cl4.prefix_types(), "triple_types": cl4.triple_types()})
    growth = [s["triple_types"] for s in sizes]
    c.check(
        "S4 存储观测: 三元组种类随样本量单调不减且 <= 8^3=512",
        all(growth[i] <= growth[i + 1] for i in range(len(growth) - 1))
        and max(growth) <= 512,
        "; ".join(f"{s['samples']}样本:{s['triple_types']}种" for s in sizes),
    )
    data["s4"] = sizes

    # ---------- S6 基线行为回归 (L0/L1 未改) ----------
    net = SynapseNet()
    net.learn(["A", "B"])
    acts0 = net.activate("A")
    base_ok = (
        net.node_count() == 2
        and math.isclose(acts0["B"], 0.9, rel_tol=1e-9)
    )
    c.check(
        "S6 基线回归: L0 空网络->learn([A,B])->B=0.9 仍成立 (未改动)",
        base_ok,
        f"nodes={net.node_count()}, a(B)={acts0.get('B', 0):.4f}",
    )

    return {
        "id": "exp15_higher_order_context",
        "title": "E15 高阶顺序: L2 三元组上下文 (双前缀->下一个, 不选句)",
        "passed": c.passed,
        "checks": c.checks,
        "data": data,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
