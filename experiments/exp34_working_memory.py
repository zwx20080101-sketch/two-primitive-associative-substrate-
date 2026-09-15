"""E34: 工作记忆（最小版本）—— 外部 W 层跨次状态携带。

严格按 DESIGN-E34.md。**无 numpy 依赖；不改任何层文件；不新增层。**

预注册要点:
  - W 是【实验侧读取端工具】（同 E14 的 sum 模式），不是 L5/S4，不进 LAYERS、不改 BOUNDARY；
  - W 规格：{node: value} + 容量 k=3；hold(acts) 取 value 最大的 3 个；seeds() 只返回节点名
    （不传 value —— 多种子入口会把值抹成 1.0）；
  - P3 核心：有 W 时 A=1.0、无 W 时 A=0.9^25=0.071790，差异 >= 0.5；
  - P4 消融：W 存在但 seeds() 不被调用 → A 回到 0.071790
    （与 P3b 数值相同但不是重复实验：一个"W 不存在"，一个"W 在读端被禁用"）。
"""

from __future__ import annotations

import json
from pathlib import Path

from synapse_net import SynapseNet

from experiments.checker import Checker
from experiments.exp31_auto_scale import layer_hashes

ALPHA = [chr(ord("A") + i) for i in range(26)]
REPS = 100
K = 3
DIFF_MIN = 0.5
TOL = 1e-12

C34_TEXT = (
    "C34：在零残留的联想基底上，外部 W 层可以保存上一次 activate 的输出快照（top-3），"
    "并在下一次 activate 时作为额外种子参与扩散（A 从 0.071790 变为 1.0）；"
    "W 不写 L0、不影响基底计数。"
    "定性：这是【实验侧读取端工具】，不是基底新能力——功劳归读取端。"
    "范围限定：两步依赖任务的输出差异由【多种子入口的值抹除效应】贡献，"
    "不是「基底长出了记忆」。"
)
B27_TEXT = (
    "B27：W 层的功能【数值幅度】依赖多种子入口的「值抹除效应」（种子一律置 1.0）；"
    "若无该效应，W 的功能仍然可见（传入的节点至少是它被传入的那个值，而不是 0.9^25），"
    "只是幅度变小。这是【工具效果依赖具体实现细节】的边界——W 的可见性幅度不是基底的属性。"
)


class WorkingMemory:
    """实验侧读取端工具（不是层）。容量 k，按 value 降序、同值按节点名升序取 top-k。"""

    def __init__(self, k: int = K):
        self.k = k
        self.items: dict[str, float] = {}

    def hold(self, acts: dict) -> None:
        top = sorted(acts.items(), key=lambda kv: (-kv[1], kv[0]))[: self.k]
        self.items = {n: float(v) for n, v in top}

    def seeds(self) -> list:
        return [n for n, _ in sorted(self.items.items(), key=lambda kv: (-kv[1], kv[0]))]

    def snapshot(self) -> dict:
        return dict(self.items)

    def clear(self) -> None:
        self.items = {}


def signature(net: SynapseNet) -> dict:
    """L0 签名：节点数 / 边数 / 边集合（排序逐项含计数）。"""
    edges = sorted((tuple(sorted(k)), int(v["count"])) for k, v in net.connections.items())
    return {"nodes": net.node_count(), "edges": net.edge_count(), "edge_items": edges}


def build_chain() -> SynapseNet:
    net = SynapseNet()
    for a, b in zip(ALPHA, ALPHA[1:]):
        for _ in range(REPS):
            net.learn([a, b])
    return net


def run() -> dict:
    c = Checker()
    hashes_before = layer_hashes()

    net = build_chain()
    sig_before = signature(net)
    W = WorkingMemory(K)

    # ---------- 步骤 1：activate(A) → W.hold ----------
    S1 = net.activate("A")
    W.hold(S1)
    w_after_hold = W.snapshot()

    # ---------- P3：有 W 的步骤 2 vs 无 W 的对照 ----------
    S0 = net.activate("Z")                                  # P3b：无 W（W 对象根本不存在于调用中）
    S2 = net.activate(["Z"] + W.seeds())                    # P3a：有 W（seeds() 参与种子）
    a_no_w = float(S0.get("A", 0.0))
    a_with_w = float(S2.get("A", 0.0))
    diff = a_with_w - a_no_w

    # ---------- P2（数据记录）：W 持存 ----------
    for seed in ("M", "A", "Z"):
        net.activate(seed)
    w_after_3 = W.snapshot()

    # ---------- P4：消融（W 存在，但 seeds() 不被调用）----------
    S4 = net.activate("Z")
    a_ablated = float(S4.get("A", 0.0))

    # ---------- P1a / P1b ----------
    hashes_after = layer_hashes()
    c.check("P1a 接口不改: 四个层文件 sha256 前后一致",
            hashes_before == hashes_after,
            f"before={hashes_before}；after={hashes_after}")

    sig_after = signature(net)
    c.check("P1b L0 不改: W 写读前后 节点数/边数/边集合签名 逐位相同",
            sig_before == sig_after,
            f"{sig_before['nodes']} 节点 / {sig_before['edges']} 边；"
            f"签名逐位相同={sig_before == sig_after}")

    c.check("P2（数据记录，不作独立判据）: 3 次 activate 后 W 内容逐位相同",
            True,
            f"hold 后={w_after_hold}；3 次 activate 后={w_after_3}；相同={w_after_hold == w_after_3}"
            f"（Python 对象的自然属性，不是实验结论）")

    c.check("P3 功能可见: 有 W 时 A=1.0；无 W 时 A=0.9^25=0.071790；差异 >= 0.5",
            abs(a_with_w - 1.0) <= TOL and abs(a_no_w - 0.9 ** 25) <= 1e-6 and diff >= DIFF_MIN,
            f"P3a 有 W: A={a_with_w:.6f}；P3b 无 W: A={a_no_w:.9f}（0.9^25={0.9**25:.9f}）；"
            f"差异={diff:.6f}")

    c.check("P4 消融: W 存在但 seeds() 不被调用 → A 退化到无 W 状态（0.071790）",
            abs(a_ablated - a_no_w) <= TOL,
            f"消融后 A={a_ablated:.9f}；与 P3b 数值相同={abs(a_ablated - a_no_w) <= TOL}"
            f"（两者不重复：P3b = W 不存在，P4 = W 在读端被禁用）")

    ok = (abs(a_with_w - 1.0) <= TOL and diff >= DIFF_MIN and abs(a_ablated - a_no_w) <= TOL)
    outcome = "C34" if ok else "B27"
    text = C34_TEXT if ok else B27_TEXT
    c.check("P5 双向结局: P3/P4 通过 → C34；不通过 → B27",
            outcome in ("C34", "B27"),
            f"落定 {outcome}（有 W A={a_with_w:.4f} / 无 W A={a_no_w:.4f} / 消融 A={a_ablated:.4f}）")

    return {
        "id": "exp34_working_memory",
        "title": "E34 工作记忆（最小版本）: 外部 W 层跨次状态携带",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "W_spec": {"capacity_k": K, "hold_rule": "value 最大的 k 个（同值按节点名升序）",
                       "seeds_rule": "只返回节点名（不传 value）",
                       "positioning": "实验侧读取端工具（同 E14 的 sum 模式）；不是 L5/S4"},
            "W_after_hold": {k: round(v, 6) for k, v in w_after_hold.items()},
            "W_after_3_activates": {k: round(v, 6) for k, v in w_after_3.items()},
            "W_unchanged": bool(w_after_hold == w_after_3),
            "P3a_with_W": {"seeds": ["Z"] + W.seeds(), "A": round(a_with_w, 9)},
            "P3b_no_W": {"seeds": ["Z"], "A": round(a_no_w, 9), "expected_0.9_25": round(0.9 ** 25, 9)},
            "P3_diff": round(diff, 9),
            "P4_ablation": {"seeds": ["Z"], "W_exists": True, "seeds_called": False,
                            "A": round(a_ablated, 9)},
            "tables": {
                "S0_no_W_activate(Z)": {k: round(v, 6) for k, v in sorted(S0.items())},
                "S1_step1_activate(A)": {k: round(v, 6) for k, v in sorted(S1.items())},
                "S2_with_W_activate([Z]+W.seeds())": {k: round(v, 6) for k, v in sorted(S2.items())},
            },
            "readonly": {"layer_hashes_before": hashes_before, "layer_hashes_after": hashes_after,
                         "L0_signature_same": sig_before == sig_after,
                         "L0_nodes": sig_before["nodes"], "L0_edges": sig_before["edges"]},
            "outcome": outcome,
            "registration": {"outcome": outcome, "text": text, "C34": C34_TEXT, "B27": B27_TEXT},
            "note": "W 不进 LAYERS、不改 BOUNDARY；基底对 W 一无所知（零残留，见设计 §0）。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
