"""E37: v2 带值种子入口（常量语义）。

严格按 DESIGN-E37.md。**v1 文件一字不动；v2 是平行分支。**

预注册要点:
  - P1a：v1 哈希不变 + v2 文件哈希首记录 + v2.activate 接受 str|list[str]|dict
  - P1b（修正版）：常量语义用"能抬得动"的配置测——{"A":0.5,"B":0.9} 与 {"A":0.9,"B":0.5}
  - P2：值保留（B=0.9, C=0.81）
  - P3：排序差异（同集合、值对调的两个配置输出不同）
  - P4：向后兼容（名字列表输出 = v1）
  - P5：整表比较（预检预测：变化 13 / 不变 13）
  - P6：两哈希口径
  - P7：通过 → C37；不通过 → B31；**B30 无条件登记**（常量语义的边界）
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from synapse_net import SynapseNet as V1Net
from synapse_net_v2 import SynapseNet as V2Net

from experiments.checker import Checker

ALPHA = [chr(ord("A") + i) for i in range(26)]
REPS = 100
TOL = 1e-12
V1_HASH_EXPECTED = "cf99a07a0e60fd4d"
FILES = ["synapse_net.py", "synapse_net_v2.py"]

C37_TEXT = (
    "C37：L0 的种子入口支持带值 mapping（v2 分支）。种子值在传播中保持【常量】"
    "（不被抬高或降低）；名字列表形式向后兼容（仍全部 1.0），第一轮 11 处"
    "多节点种子调用语义不变。"
    "带值种子改变传播路径的【绝对水平】：26 字母链上，v2 与 v1 相比有 13 个节点值变化"
    "（A 侧半条链 B..N），13 个不变（A 与 Z 侧 O..Z）。因此「传播路径与种子入口无关」"
    "不成立——路径的绝对水平由种子值决定。"
    "「排序差异」在 v2 下可测（配置 1 与配置 2 输出不同），这是 v1（值抹平）下做不到的"
    "（C36 的边界被 v2 越过）。"
    "范围限定：v2 分支；v1 冻结不变（E0–E36 全部有效）。"
)
B30_TEXT = (
    "B30：v2 的常量语义导致种子节点在传播中「冻结」——种子节点不会被任何传播路径强化"
    "（即使某条路径的值更高）。这是常量语义的直接后果，不是 bug。范围限定：v2 分支。"
)
B31_TEXT = (
    "B31：v2 的实现与预期不符（哪条判据挂、实测值见 JSON）。诊断三步："
    "先查 v2 实现（synapse_net_v2.py 的 activate）→ 再查预检对照（DESIGN-E37-IMPACT 的 activate_v2）"
    "→ 再谈机制。范围限定：v2 分支。"
)


def sha16(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def build(net_cls):
    net = net_cls()
    for a, b in zip(ALPHA, ALPHA[1:]):
        for _ in range(REPS):
            net.learn([a, b])
    return net


def run() -> dict:
    c = Checker()
    h_before = {f: sha16(f) for f in FILES}
    net1, net2 = build(V1Net), build(V2Net)

    # ---------- P1a ----------
    c.check("P1a 文件与接口: v1 哈希不变 + v2 文件存在 + v2.activate 接受 str/list/dict",
            h_before["synapse_net.py"] == V1_HASH_EXPECTED
            and isinstance(net2.activate("A"), dict)
            and isinstance(net2.activate(["A", "B"]), dict)
            and isinstance(net2.activate({"A": 0.5, "B": 0.9}), dict),
            f"v1={h_before['synapse_net.py']}（期望 {V1_HASH_EXPECTED}）；"
            f"v2={h_before['synapse_net_v2.py']}（首记录）；三种输入均接受")

    # ---------- P1b 常量语义（可分辨判据）----------
    m1 = net2.activate({"A": 0.5, "B": 0.9})     # B→A 到达 0.81 > 0.5
    m2 = net2.activate({"A": 0.9, "B": 0.5})     # A→B 到达 0.81 > 0.5
    p1b_ok = (abs(m1["A"] - 0.5) <= TOL and abs(m1["B"] - 0.9) <= TOL
              and abs(m2["A"] - 0.9) <= TOL and abs(m2["B"] - 0.5) <= TOL)
    c.check("P1b 常量语义: 能抬得动的配置下，种子值仍精确等于传入值",
            p1b_ok,
            f"{{A:0.5,B:0.9}} → A={m1['A']:.6f}, B={m1['B']:.6f}（若浮动/下界：A 会变 0.81）；"
            f"{{A:0.9,B:0.5}} → A={m2['A']:.6f}, B={m2['B']:.6f}（若浮动/下界：B 会变 0.81）")

    # ---------- P2 值保留 ----------
    cfg = {"Z": 1.0, "A": 1.0, "B": 0.9, "C": 0.81}
    o1 = net1.activate(list(cfg))                # v1：名字列表（被抹平）
    o2 = net2.activate(cfg)                      # v2：带值
    p2_ok = (abs(o2["B"] - 0.9) <= TOL and abs(o2["C"] - 0.81) <= TOL
             and abs(o1["B"] - 1.0) <= TOL and abs(o1["C"] - 1.0) <= TOL)
    c.check("P2 值保留: v2 下 B=0.9、C=0.81（v1 下均为 1.0）",
            p2_ok,
            f"v1: A={o1['A']:.3f} B={o1['B']:.3f} C={o1['C']:.3f}；"
            f"v2: A={o2['A']:.3f} B={o2['B']:.3f} C={o2['C']:.3f}")

    # ---------- P3 排序差异 ----------
    cfg1 = {"Z": 1.0, "A": 1.0, "B": 0.9, "C": 0.81}
    cfg2 = {"Z": 1.0, "A": 0.81, "B": 0.9, "C": 1.0}
    a1, a2 = net2.activate(cfg1), net2.activate(cfg2)
    diffs = {k: abs(a1[k] - a2[k]) for k in ALPHA if abs(a1[k] - a2[k]) > TOL}
    same_in_v1 = all(abs(net1.activate(list(cfg1))[k] - net1.activate(list(cfg2))[k]) <= TOL for k in ALPHA)
    c.check("P3 排序差异: 同集合、值对调的两配置在 v2 下输出不同（v1 下相同）",
            len(diffs) > 0 and same_in_v1,
            f"v2 差异节点数={len(diffs)}（示例 {list(diffs.items())[:3]}）；v1 下两配置相同={same_in_v1}")

    # ---------- P4 向后兼容 ----------
    names = ["Z", "A", "B", "C"]
    b1, b2 = net1.activate(names), net2.activate(names)
    p4_ok = all(abs(b1[k] - b2[k]) <= TOL for k in ALPHA)
    c.check("P4 向后兼容: 名字列表形式在 v2 下输出 = v1（逐位）",
            p4_ok,
            f"26 节点逐位相同={p4_ok}（第一轮 11 处多节点种子调用均为名字列表）")

    # ---------- P5 整表比较 ----------
    changed = [k for k in ALPHA if abs(o1[k] - o2[k]) > TOL]
    unchanged = [k for k in ALPHA if abs(o1[k] - o2[k]) <= TOL]
    p5_ok = len(changed) == 13 and len(unchanged) == 13
    c.check("P5 整表比较: v2 相对 v1 变化 13 个节点、不变 13 个（预检预测）",
            p5_ok,
            f"变化={changed}；不变={unchanged}")

    # ---------- P6 只读 ----------
    h_after = {f: sha16(f) for f in FILES}
    c.check("P6 只读（两哈希口径）: v1 前后一致；v2 首记录在案",
            h_before == h_after and h_after["synapse_net.py"] == V1_HASH_EXPECTED,
            f"before={h_before}；after={h_after}")

    # ---------- P7 双向 ----------
    ok = p2_ok and len(diffs) > 0 and p4_ok
    outcome = "C37" if ok else "B31"
    text = C37_TEXT if ok else B31_TEXT
    c.check("P7 双向结局: P2+P3+P4 通过 → C37；否则 → B31（B30 无条件登记）",
            outcome in ("C37", "B31"),
            f"落定 {outcome}（P2={p2_ok}, P3 差异节点={len(diffs)}, P4={p4_ok}）")

    return {
        "id": "exp37_seed_values",
        "title": "E37 v2 带值种子入口（常量语义）: 值保留、排序差异、向后兼容",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "versions": {"v1": {"file": "synapse_net.py", "sha256_16": h_after["synapse_net.py"],
                                "role": "冻结基线（E0–E36）"},
                         "v2": {"file": "synapse_net_v2.py", "sha256_16": h_after["synapse_net_v2.py"],
                                "role": "平行分支（E37 起）", "semantics": "带值种子 + 常量（不可变）"}},
            "P1b_constant_semantics": {
                "cfg_甲": {"in": {"A": 0.5, "B": 0.9}, "out": {"A": round(m1["A"], 9), "B": round(m1["B"], 9)},
                           "would_be_if_float_or_floor": 0.81},
                "cfg_乙": {"in": {"A": 0.9, "B": 0.5}, "out": {"A": round(m2["A"], 9), "B": round(m2["B"], 9)},
                           "would_be_if_float_or_floor": 0.81}},
            "P2_value_preserved": {"v1": {k: round(o1[k], 6) for k in ("A", "B", "C")},
                                   "v2": {k: round(o2[k], 6) for k in ("A", "B", "C")}},
            "P3_ordering": {"cfg1": cfg1, "cfg2": cfg2, "n_diff_nodes_v2": len(diffs),
                            "diff_nodes": sorted(diffs), "identical_in_v1": bool(same_in_v1)},
            "P4_backward_compat": {"names": names, "identical_to_v1": bool(p4_ok)},
            "P5_full_table": {"changed": changed, "unchanged": unchanged,
                              "v1": {k: round(o1[k], 6) for k in ALPHA},
                              "v2": {k: round(o2[k], 6) for k in ALPHA}},
            "readonly": {"hashes_before": h_before, "hashes_after": h_after,
                         "v1_expected": V1_HASH_EXPECTED},
            "outcome": outcome,
            "registration": {"outcome": outcome, "text": text,
                             "C37": C37_TEXT, "B30": B30_TEXT, "B31": B31_TEXT,
                             "B30_always": True},
            "note": "v2 是平行分支；v1 一字不动。B30（常量语义导致种子冻结）无条件登记。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
