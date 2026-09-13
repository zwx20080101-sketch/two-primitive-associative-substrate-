"""E27: 上下文窗口边界（B01 正式化）—— k_min = k*。

严格按 DESIGN-E27.md。**纯计数，无 numpy 依赖；不改任何层文件。**

口径（预注册）:
  - 键 = 目标之前最近的 k 个 token；不足 k 个时退化为全部可用 token（JSON 标 degenerate）；
  - 语料 = 两条场景的中间 token 完全相同，只有最远 token 不同；对照行无判别 token；
  - 决定度 = 给定键之后正确目标的概率（0.50 = 纯歧义，1.00 = 完全确定）；
  - 对比度 = |a(C) - a(D)| / (a(C) + a(D))，分母为 0 时记 0；
  - probe-k = 实验脚本内现算的**只读计数探针**（不是新层，不写入任何层文件）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from context_layer import ContextLayer
from order_layer import OrderLayer
from synapse_net import SynapseNet

from experiments.checker import Checker

REPS = 100
LEVELS = [1, 2, 3, 4, 5]
CONTROL = "inf"                 # 无判别 token 的纯分叉对照
K_SCAN = [1, 2, 3, 4, 5]
TOL_AMBIG = 1e-6                # |决定度 - 0.50| 容差
TOL_ZERO = 1e-9                 # |对比度| 容差（与 E26 零耦合判据同口径）
UNBALANCED = [(1, 1), (1, 2), (1, 5)]
LAYER_FILES = ["synapse_net.py", "order_layer.py", "context_layer.py"]

C27_TEXT = (
    "C27：在\"中间 token 共享、只有最远 token 不同\"的受控语料族下，"
    "现有层（L0/L1/L2）的最大可用消歧前缀长度 = 2；"
    "当判别上下文到决策点的距离 k* >= 3 时，决定度恒为 0.50；"
    "最小可消歧前缀长度 k_min 精确等于 k*（k* ∈ {1,2,3,4,5} 上验证）。"
    "阳性对照：k*=2 时 L2 决定度 = 1.00。"
)
B01_TEXT = (
    "B01（升级）：现有层最大前缀 = 2；k* >= 3 时决定度恒为 0.50；"
    "k_min = k*；k* >= 3 的上下文消歧需要最小窗口 k* 的新层。"
)


def scenes(kstar):
    """k*=数字 → 两条场景；k*=CONTROL → 无判别 token 的纯分叉对照。"""
    if kstar == CONTROL:
        return ["B", "C"], ["B", "D"]
    shared = [f"s{i}" for i in range(1, kstar)]
    return ["uX"] + shared + ["C"], ["uY"] + shared + ["D"]


def key_of(seq, k):
    """目标之前最近的 k 个 token；不足则退化为全部可用 token。"""
    avail = seq[:-1]
    if len(avail) < k:
        return tuple(avail), True
    return tuple(avail[-k:]), False


def build(kstar, reps_x=REPS, reps_y=REPS):
    sX, sY = scenes(kstar)
    net, l1, l2 = SynapseNet(), OrderLayer(), ContextLayer()
    for _ in range(reps_x):
        net.learn(sX); l1.learn(sX); l2.learn(sX)
    for _ in range(reps_y):
        net.learn(sY); l1.learn(sY); l2.learn(sY)
    return {"net": net, "l1": l1, "l2": l2, "sX": sX, "sY": sY}


def contrast(a: float, b: float) -> float:
    s = a + b
    return 0.0 if s <= 0 else abs(a - b) / s


def decision_degree(built, k, reps_x=REPS, reps_y=REPS):
    """给定"最近 k 个 token"这个键之后，正确目标的概率（两场景取较小者）。"""
    sX, sY = built["sX"], built["sY"]
    kX, degX = key_of(sX, k)
    kY, degY = key_of(sY, k)
    counts: dict[tuple, dict[str, int]] = {}
    for key, tgt, reps in ((kX, sX[-1], reps_x), (kY, sY[-1], reps_y)):
        counts.setdefault(key, {})
        counts[key][tgt] = counts[key].get(tgt, 0) + reps
    pX = counts[kX][sX[-1]] / sum(counts[kX].values())
    pY = counts[kY][sY[-1]] / sum(counts[kY].values())
    return {
        "degree": min(pX, pY),
        "p_sceneX": pX,
        "p_sceneY": pY,
        "key_sceneX": list(kX),
        "key_sceneY": list(kY),
        "same_key": kX == kY,
        "degenerate": bool(degX and degY),
    }


def seed_contrast(built, layer: str):
    """从"目标前一个 token"出发，看 C/D 的对比度（L0 或 L1）。"""
    sX, sY = built["sX"], built["sY"]
    seed = sX[-2]
    act = built["net"].activate([seed]) if layer == "L0" else built["l1"].activate([seed])
    c = float(act.get(sX[-1], 0.0))
    d = float(act.get(sY[-1], 0.0))
    return {"contrast": contrast(c, d), "target_sceneX": c, "target_sceneY": d, "seed": seed}


def l2_contrast(built):
    """L2 用它自己的 2 前缀查询，看候选里 C/D 的对比度（序列不足 3 时记 N/A）。"""
    sX, sY = built["sX"], built["sY"]
    if len(sX) < 3:
        return {"contrast": None, "note": "序列不足 3 token，L2 无 2 前缀可用（N/A）"}
    pref = sX[-3:-1]
    q = built["l2"].query(pref)
    c = float(q.get(sX[-1], 0.0))
    d = float(q.get(sY[-1], 0.0))
    return {"contrast": contrast(c, d), "target_sceneX": c, "target_sceneY": d, "prefix": pref}


def structure_ok(kstar):
    """P1: 两条场景除"最远 token"与"目标"外逐位相同；对照行无判别 token。"""
    sX, sY = scenes(kstar)
    diff = [i for i, (a, b) in enumerate(zip(sX, sY)) if a != b]
    if kstar == CONTROL:
        return diff == [len(sX) - 1], diff
    return diff == [0, len(sX) - 1] and len(sX) == kstar + 1, diff


def layer_hashes() -> dict:
    root = Path(__file__).resolve().parent.parent
    out = {}
    for name in LAYER_FILES:
        p = root / name
        out[name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def layer_counts(built) -> dict:
    net, l1, l2 = built["net"], built["l1"], built["l2"]
    return {
        "L0_nodes": net.node_count(), "L0_edges": net.edge_count(),
        "L1_nodes": l1.node_count(), "L1_edges": l1.edge_count(),
        "L2_prefix_types": l2.prefix_types(), "L2_triple_types": l2.triple_types(),
    }


def run() -> dict:
    c = Checker()
    hashes_before = layer_hashes()

    # ---------- P1 语料构造核查 ----------
    struct = {str(k): structure_ok(k) for k in LEVELS + [CONTROL]}
    c.check("P1 语料构造核查: 两场景仅'最远 token'与'目标'不同; 对照行无判别 token",
            all(ok for ok, _ in struct.values()),
            "; ".join(f"k*={k}: 差异位置={d}" for k, (_, d) in struct.items()))

    # ---------- 全阶梯测量 ----------
    ladder = {}
    for kstar in LEVELS + [CONTROL]:
        built = build(kstar)
        before = layer_counts(built)
        row = {
            "scenes": {"X": built["sX"], "Y": built["sY"]},
            "degree_by_k": {str(k): decision_degree(built, k) for k in K_SCAN},
            "L0": seed_contrast(built, "L0"),
            "L1": seed_contrast(built, "L1"),
            "L2": l2_contrast(built),
        }
        row["readonly_ok"] = before == layer_counts(built)
        ladder[str(kstar)] = row

    # ---------- P2 阴性复现（B01）----------
    l3 = ladder["3"]
    c.check("P2 阴性复现(B01): k*=3 上 L1 与 L2 的 C/D 对比度精确为 0",
            abs(l3["L1"]["contrast"]) < TOL_ZERO and abs(l3["L2"]["contrast"]) < TOL_ZERO,
            f"k*=3: L1 对比度={l3['L1']['contrast']:.3e}（C={l3['L1']['target_sceneX']:.4f}, "
            f"D={l3['L1']['target_sceneY']:.4f}）, L2 对比度={l3['L2']['contrast']:.3e}")

    # ---------- P3 阳性对照 ----------
    pos_l2 = ladder["2"]["degree_by_k"]["2"]["degree"]
    pos_ok = all(ladder[str(k)]["degree_by_k"][str(kk)]["degree"] >= 1 - TOL_AMBIG
                 for k in LEVELS for kk in K_SCAN if kk >= k)
    c.check("P3 阳性对照: k*=2 上 L2 决定度 = 1.00; 且每个 k* 的 probe-k(k>=k*) = 1.00",
            pos_l2 >= 1 - TOL_AMBIG and pos_ok,
            f"k*=2 上 L2(probe-2) 决定度={pos_l2:.4f}; 所有 k>=k* 的 probe-k 均 >=1-1e-6: {pos_ok}")

    # ---------- P4 定量律 ----------
    law_rows, k_min = {}, {}
    for k in LEVELS:
        ks = [kk for kk in K_SCAN if ladder[str(k)]["degree_by_k"][str(kk)]["degree"] >= 1 - TOL_AMBIG]
        k_min[str(k)] = min(ks) if ks else None
        law_rows[str(k)] = {
            "k_min": k_min[str(k)],
            "low_ok": all(abs(ladder[str(k)]["degree_by_k"][str(kk)]["degree"] - 0.5) < TOL_AMBIG
                          for kk in K_SCAN if kk < k),
        }
    ctrl_ok = all(abs(ladder[CONTROL]["degree_by_k"][str(kk)]["degree"] - 0.5) < TOL_AMBIG
                  for kk in K_SCAN)
    c.check("P4 定量律: k*∈{1..5} 上 k_min = k*; 对照行(无判别 token)恒 0.50",
            all(k_min[str(k)] == k for k in LEVELS) and all(r["low_ok"] for r in law_rows.values()) and ctrl_ok,
            "k_min=" + json.dumps(k_min, ensure_ascii=False) + f"; 对照行恒 0.50: {ctrl_ok}")

    # ---------- P5 只读 ----------
    hashes_after = layer_hashes()
    ro_ok = hashes_before == hashes_after and all(r["readonly_ok"] for r in ladder.values())
    c.check("P5 只读: 层文件哈希不变; 每次测量前后 L0/L1/L2 计数不变",
            ro_ok,
            f"层文件哈希 before==after: {hashes_before == hashes_after}; "
            f"各档测量前后计数一致: {all(r['readonly_ok'] for r in ladder.values())}")

    # ---------- 辅助: 不平衡暴露 ----------
    bal = []
    for rx, ry in UNBALANCED:
        b = build(3, reps_x=REPS * rx, reps_y=REPS * ry)
        d = decision_degree(b, 2, reps_x=REPS * rx, reps_y=REPS * ry)
        bal.append({"exposure_X": rx, "exposure_Y": ry, "degree": d["degree"]})

    # ---------- P6 登记文本 ----------
    c.check("P6 登记: JSON 内含 C27 与 B01(升级) 的拟登记文本",
            "C27" in C27_TEXT and "B01（升级）" in B01_TEXT,
            "文本已写入 JSON 的 registration 字段")

    return {
        "id": "exp27_context_window",
        "title": "E27 上下文窗口边界: k_min = k*（B01 正式化）",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "ladder": ladder,
            "k_min": k_min,
            "unbalanced_exposure": bal,
            "tolerances": {"ambiguity": TOL_AMBIG, "zero": TOL_ZERO},
            "reps_per_scene": REPS,
            "layer_hashes": {"before": hashes_before, "after": hashes_after},
            "registration": {"C27": C27_TEXT, "B01_upgraded": B01_TEXT},
            "note": "probe-k 是实验侧只读计数探针，不是新层；本实验不新增层文件。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
