"""E36: 顺序标记的适用边界（分类 ≠ 排序）。

严格按 DESIGN-E36.md。**无 numpy 依赖；不改任何层文件；不新增层。**

预注册要点:
  - W 扩展为 {node: (value, step)}；seeds(step=k) 只返回该步的痕迹（读取端过滤）；
  - 标签【进不了基底】（activate 只吃节点名，预检已证）→ 分类可测、排序不可测；
  - P4 是【负面记录】不是 FAIL：注入后所有痕迹在基底里都是 1.0；
  - P5 用"步2 独有节点 X"区分过滤与不过滤：X = 0.81（有过滤）vs 1.0（无过滤）；
  - 证伪时先查实现再谈机制（synapse_net.py:98 / seeds() 是否传值 / activate 内部保留值）。
"""

from __future__ import annotations

import json
from pathlib import Path

from synapse_net import SynapseNet

from experiments.checker import Checker
from experiments.exp31_auto_scale import layer_hashes
from experiments.exp34_working_memory import signature

ALPHA = [chr(ord("A") + i) for i in range(26)]
REPS = 100
K = 3
A_NO_INJECT = 0.9 ** 25        # 0.071789799
X_NO_FILTER = 1.0              # X 自己是种子
X_WITH_FILTER = 0.9 ** 2       # 0.81，X 距 Z 两跳
DIFF_MIN_P3 = 0.5
DIFF_MIN_P5 = 0.1
TOL = 1e-9

C36_TEXT = (
    "C36：外部 W 层的顺序标记【只作用于读取端过滤】（哪些痕迹被包含在种子集里），"
    "不影响基底内部——基底种子入口把值抹平（E34 已证），因此顺序标记不产生优先级排序。"
    "范围限定：分类 ≠ 排序；外部工具 ≠ 基底能力；排序差异需要值域编码，"
    "涉及改基底入口行为，属越权嫌疑，不在本实验。"
)
B29_TEXT = (
    "B29：按 step 分类的注入未能产生可测差异（具体偏差见 JSON）；"
    "顺序标记在该语料上不足以支撑「按来源分类」，需要重新定位机制。"
)


class TaggedWorkingMemory:
    """E36 版 W：在 E34 的 {node: value} 上加 step 字段（语义沿用，不重写 k/tie-break）。"""

    def __init__(self, k: int = K):
        self.k = k
        self.items: dict[str, tuple] = {}

    def hold(self, acts: dict, step: int) -> None:
        top = sorted(acts.items(), key=lambda kv: (-kv[1], kv[0]))[: self.k]
        for n, v in top:
            self.items[n] = (float(v), int(step))

    def seeds(self, step: int | None = None) -> list:
        rows = [(n, v, st) for n, (v, st) in self.items.items()
                if step is None or st == step]
        return [n for n, _, _ in sorted(rows, key=lambda r: (-r[1], r[0]))]

    def tags(self) -> dict:
        return dict(self.items)


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
    W = TaggedWorkingMemory(K)

    # ---------- 三步链 ----------
    s1 = net.activate("A")
    W.hold(s1, step=1)
    s2 = net.activate("Z")
    W.hold(s2, step=2)

    inj = net.activate(["Z"] + W.seeds(step=1))      # 只注入步1（有过滤）
    ctl = net.activate(["Z"])                        # 不注入
    allin = net.activate(["Z"] + W.seeds())          # 注入全集（无过滤，退化为 E35）

    a_inj, a_ctl = float(inj["A"]), float(ctl["A"])
    diff3 = a_inj - a_ctl
    x_filtered, x_unfiltered = float(inj["X"]), float(allin["X"])
    diff5 = x_unfiltered - x_filtered

    # P4 观测量：注入后各痕迹在基底里的值
    injected = ["Z"] + W.seeds(step=1)
    injected_values = {n: round(float(inj[n]), 9) for n in sorted(set(injected))}
    all_one = all(abs(v - 1.0) <= TOL for v in injected_values.values())

    # ---------- P1a / P1b ----------
    hashes_after = layer_hashes()
    c.check("P1a 接口不改: 四个层文件 sha256 前后一致",
            hashes_before == hashes_after, f"before==after: {hashes_before == hashes_after}")
    sig_after = signature(net)
    c.check("P1b L0 不改: 节点数/边数/边集合签名逐位相同",
            sig_before == sig_after,
            f"{sig_before['nodes']} 节点 / {sig_before['edges']} 边；签名相同={sig_before == sig_after}")

    # ---------- P2 标签逐项核对 ----------
    tags = W.tags()
    p2_ok = (tags == {"A": (1.0, 1), "B": (0.9, 1), "C": (0.81, 1),
                      "Z": (1.0, 2), "Y": (0.9, 2), "X": (0.81, 2)}
             and W.seeds(step=1) == ["A", "B", "C"]
             and W.seeds(step=2) == ["Z", "Y", "X"])
    c.check("P2 顺序标记正确: W 中每个节点带正确 step；seeds(step=1)/(2) 逐项一致",
            p2_ok,
            f"tags={ {k: (round(v, 4), st) for k, (v, st) in tags.items()} }；"
            f"seeds(1)={W.seeds(1)}；seeds(2)={W.seeds(2)}")

    # ---------- P3 分类差异可测 ----------
    c.check("P3 分类差异可测: seeds(step=1) 只返回步1 的 3 个；注入后 A=1.0；与不注入之差 >= 0.5",
            len(W.seeds(step=1)) == 3 and abs(a_inj - 1.0) <= TOL and diff3 >= DIFF_MIN_P3,
            f"seeds(step=1)={W.seeds(step=1)}；A(注入)={a_inj:.6f}；A(不注入)={a_ctl:.9f}；差异={diff3:.6f}")

    # ---------- P4 排序差异不可测（负面记录）----------
    c.check("P4（负面记录，不是 FAIL）: 注入后所有痕迹在基底里都是 1.0，无相对权重差异",
            all_one,
            f"注入集合={sorted(set(injected))}；基底中的值={injected_values}；"
            f"全部 1.0={all_one}"
            + ("" if all_one else " → 先查实现（synapse_net.py:98 / seeds 是否传值 / activate 内部保留值）"))

    # ---------- P5 消融：过滤 vs 不过滤（观测量 X）----------
    c.check("P5 消融: 无过滤时 X=1.0；有过滤时 X=0.81；两者差 >= 0.1",
            abs(x_unfiltered - X_NO_FILTER) <= TOL
            and abs(x_filtered - X_WITH_FILTER) <= 1e-9
            and diff5 >= DIFF_MIN_P5,
            f"X(有过滤)={x_filtered:.6f}（期望 0.9^2={X_WITH_FILTER:.6f}）；"
            f"X(无过滤)={x_unfiltered:.6f}（期望 1.0）；差={diff5:.6f}"
            + ("" if diff5 >= DIFF_MIN_P5 else " → 先查实现（同上三条）"))

    # ---------- P6 落定 ----------
    ok = (len(W.seeds(step=1)) == 3 and abs(a_inj - 1.0) <= TOL and diff3 >= DIFF_MIN_P3
          and diff5 >= DIFF_MIN_P5)
    outcome = "C36" if ok else "B29"
    text = C36_TEXT if ok else B29_TEXT
    c.check("P6 双向结局: P3 + P5 通过 → C36；不通过 → B29",
            outcome in ("C36", "B29"),
            f"落定 {outcome}（P3 差异={diff3:.4f}；P5 差异={diff5:.4f}）")

    return {
        "id": "exp36_sequence_tags",
        "title": "E36 顺序标记的适用边界: 分类 ≠ 排序",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "spec": {"K": K, "tie_break": "值降序；同值节点名升序（与 E35 §6 一致，此处复述以保证 E36 可独立复现）",
                     "W": "{node: (value, step)}；seeds(step=k) 只返回该步痕迹"},
            "depends_on": ["E34（W 层外部工具）", "E35（W 栈行为已验证）"],
            "tags": {k: [round(v, 6), st] for k, (v, st) in W.tags().items()},
            "seeds_step1": W.seeds(1), "seeds_step2": W.seeds(2), "seeds_all": W.seeds(),
            "P3": {"A_injected": round(a_inj, 9), "A_not_injected": round(a_ctl, 9),
                   "expected_A_not_injected": round(A_NO_INJECT, 9), "diff": round(diff3, 9)},
            "P4": {"injected_nodes": sorted(set(injected)), "values_in_substrate": injected_values,
                   "all_are_1.0": bool(all_one),
                   "diagnosis_order_if_violated": ["synapse_net.py:98 是否被改",
                                                   "seeds() 是否传了值而不是节点名",
                                                   "activate 内部是否在某种情形下保留了值"]},
            "P5": {"X_with_filter": round(x_filtered, 9), "expected_with_filter": round(X_WITH_FILTER, 9),
                   "X_without_filter": round(x_unfiltered, 9), "diff": round(diff5, 9)},
            "aux_interface_note": {"activate_dict_silently_accepted": True,
                                   "detail": "activate({'A':1}) 被接受：dict 被当成 iterable，只取 keys → 只激活 A",
                                   "registry": "写入 B21 补充实例（与 L2 的 {}、L4 的 None 同族）"},
            "readonly": {"layer_hashes_before": hashes_before, "layer_hashes_after": hashes_after,
                         "L0_signature_same": sig_before == sig_after,
                         "L0_nodes": sig_before["nodes"], "L0_edges": sig_before["edges"]},
            "outcome": outcome,
            "registration": {"outcome": outcome, "text": text, "C36": C36_TEXT, "B29": B29_TEXT},
            "note": "标签只活在 W 层内部；基底对它一无所知（预检已证 activate 只吃节点名）。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
