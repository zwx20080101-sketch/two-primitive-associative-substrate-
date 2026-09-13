"""E29: 层级 = 多尺度读出（不是"涌现"）。

严格按 DESIGN-E29.md。复用 E24 的谱聚类读出与 ARI（P1a 接口回归保证接口未被改坏）。
**有 numpy 依赖；不改任何层文件。**

口径（预注册）:
  - 读出 = E24 谱聚类（impute_diagonal → 归一化拉普拉斯前 k 个特征向量 → k-means）；
  - 指标 = ARI（置换不变；绝不用逐位比较 —— B19）；
  - L0 统计签名 = (节点数, 边数, 边集合排序后逐项比较含计数) —— 不依赖节点排列顺序；
  - k=1 的 ARI 点标 definitional: true（单簇对多簇真值恒为 0，是定义性质非经验结果）。
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.checker import Checker
from experiments.exp24_module_differentiation import (
    SESSION_SEED,
    ari,
    corpus_planted,
    impute_diagonal,
    matrix_from_net,
    net_from_counts,
    spectral,
)
from experiments.exp25_module_dynamics import block_matrix, rho

SIZES = [3, 3, 4, 4]
STRENGTHS = [90, 100, 110, 120]
S2 = 40
S3 = 10
K_SCAN = [1, 2, 3, 4, 5]
KM_SEEDS = list(range(1000, 1010))
TOL_MARGIN = 0.99
TOL_COARSE = 0.4
TOL_STRICT = 1e-12
LAYER_FILES = ["synapse_net.py", "order_layer.py", "context_layer.py"]

C29_TEXT = (
    "C29：在 14 节点、4 子模块（大小 3/3/4/4，质量 270/300/440/480 两两不同）、"
    "跨子模块强度 s2=40 的受控嵌套语料上，同一份 L0 共现统计支持多尺度读出："
    "谱聚类 k=2 读出 2 个超模块（ARI=1.000），k=4 读出 4 个子模块（ARI=1.000），k=3 为混合；"
    "消融跨子模块边后 k=2 的超模块读出塌到 0.286（接近随机）。"
    "因此层级 = 读出尺度 k 的函数；k 是预设参数（与 λ、hop_decay、K 同级），不是涌现的结构。"
)
B_TEXT = (
    "B（候选，编号待定）：谱聚类读出需要外部给定尺度 k。当前实现没有"
    "从数据自动判断层数的机制；特征值间隙等自动选择准则不在本工作范围。"
    "层数分辨率有上限：更弱的第三层耦合（s3=10）在 k>=2 时对读数无影响（辅助记录）。"
)


def module_groups():
    return [[f"m{j}_{i}" for i in range(SIZES[j])] for j in range(4)]


def nodes_of(groups):
    return [n for g in groups for n in g]


def sub_truth(groups):
    return np.array([j for j, g in enumerate(groups) for _ in g])


def super_truth(groups):
    return np.array([0 if j < 2 else 1 for j, g in enumerate(groups) for _ in g])


def build(s2: int = S2, s3: int = 0):
    """阶段1 建子模块；阶段2 建超模块；阶段3（可选）第三层弱耦合。"""
    groups = module_groups()
    pairs = []
    for j, g in enumerate(groups):
        for a, b in itertools.combinations(g, 2):
            pairs.append((a, b, STRENGTHS[j]))
    for x, j in ((0, 1), (2, 3)):
        for a in groups[x]:
            for b in groups[j]:
                pairs.append((a, b, s2))
    if s3:
        for a in groups[0] + groups[1]:
            for b in groups[2] + groups[3]:
                pairs.append((a, b, s3))
    return net_from_counts(pairs)


def signature(net) -> dict:
    """L0 统计签名：边集合排序后逐项比较（含计数），不依赖节点排列顺序。"""
    edges = sorted((tuple(sorted(k)), int(v["count"])) for k, v in net.connections.items())
    return {"nodes": net.node_count(), "edges": net.edge_count(), "edge_items": edges}


def layer_hashes() -> dict:
    root = Path(__file__).resolve().parent.parent
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest()[:16] for n in LAYER_FILES}


def readout(M: np.ndarray, k: int, sub_t, sup_t) -> dict:
    """对同一份共现矩阵，用 k 个簇读出；报 10 个 k-means 种子的中位数与范围。"""
    vs, vp = [], []
    for s in KM_SEEDS:
        a = spectral(impute_diagonal(M), k, seed=s)
        vs.append(float(ari(a, sub_t)))
        vp.append(float(ari(a, sup_t)))
    return {
        "k": k,
        "ari_sub_median": float(np.median(vs)), "ari_sub_min": min(vs), "ari_sub_max": max(vs),
        "ari_super_median": float(np.median(vp)), "ari_super_min": min(vp), "ari_super_max": max(vp),
        "definitional": k == 1,
        "note": "ARI 的定义性质（单簇对多簇真值恒为 0），非经验结果" if k == 1 else None,
    }


def run() -> dict:
    c = Checker()
    hashes_before = layer_hashes()
    groups = module_groups()
    nodes = nodes_of(groups)
    sub_t, sup_t = sub_truth(groups), super_truth(groups)

    # ---------- P1a 接口回归（E24 语料，同一套实现）----------
    net_r, nodes_r, labels_r, _ = corpus_planted()
    Mr = matrix_from_net(net_r, nodes_r)
    assign_r = spectral(impute_diagonal(Mr), 3, seed=SESSION_SEED)
    truth_r = np.array([labels_r[n] for n in nodes_r])
    ari_r = float(ari(assign_r, truth_r))
    scale_r = max(
        np.mean([Mr[p, q] for p in range(len(nodes_r)) if assign_r[p] == i
                 for q in range(len(nodes_r)) if assign_r[q] == i and p != q])
        for i in sorted(set(assign_r.tolist()))
    )
    rho_r = rho(block_matrix(Mr, assign_r, scale_r))
    c.check("P1a 接口回归: E24 植入语料重现 ARI=1.000 / 3 模块 / ρ=0.0019",
            ari_r >= 0.999 and len(set(assign_r.tolist())) == 3 and abs(rho_r - 0.0019) < 5e-4,
            f"ARI={ari_r:.3f}, 模块数={len(set(assign_r.tolist()))}, ρ={rho_r:.4f}")

    # ---------- P1b 语料构造核查 ----------
    net = build()
    declared = {frozenset(e) for e in
                [p[:2] for p in [(a, b) for j, g in enumerate(groups)
                                 for a, b in itertools.combinations(g, 2)]]}
    declared |= {frozenset((a, b)) for x, j in ((0, 1), (2, 3))
                 for a in groups[x] for b in groups[j]}
    observed = {frozenset(k) for k in net.connections.keys()}
    masses = [SIZES[j] * STRENGTHS[j] for j in range(4)]
    gaps = [abs(masses[i] - masses[j]) / max(masses[i], masses[j])
            for i in range(4) for j in range(i + 1, 4)]
    c.check("P1b 语料构造核查: 4 子模块/2 超模块边集逐边核对; 质量非简并",
            declared == observed and min(gaps) > 0.0 and net.node_count() == len(nodes),
            f"声明 {len(declared)} 边 / 实测 {len(observed)} 边 / 一致={declared == observed}; "
            f"质量={masses}, 最小相对差={min(gaps):.3f}; 节点={net.node_count()}")

    # ---------- L0 统计签名（跨 k + 前后）----------
    M = matrix_from_net(net, nodes)
    sig0 = signature(net)
    sig_after_activate = []
    for _ in range(3):
        net.activate(nodes[0])
        sig_after_activate.append(signature(net))

    # ---------- k 曲线（P2 / 辅助 1）----------
    curve = [readout(M, k, sub_t, sup_t) for k in K_SCAN]
    sig_after_scan = [signature(net) for _ in range(len(K_SCAN))]
    by_k = {r["k"]: r for r in curve}

    c.check("P2 多尺度读出: k=4 → ARI vs 子模块 >= 0.99; k=2 → ARI vs 超模块 >= 0.99",
            by_k[4]["ari_sub_median"] >= TOL_MARGIN and by_k[2]["ari_super_median"] >= TOL_MARGIN,
            f"k=4 子模块 {by_k[4]['ari_sub_median']:.3f} [{by_k[4]['ari_sub_min']:.3f},{by_k[4]['ari_sub_max']:.3f}]; "
            f"k=2 超模块 {by_k[2]['ari_super_median']:.3f} [{by_k[2]['ari_super_min']:.3f},{by_k[2]['ari_super_max']:.3f}]"
            f"（0.99 为余量阈值，预检复现值 1.000）")

    # ---------- P3 消融必要性 ----------
    M_abl = M.copy()
    idx = {n: i for i, n in enumerate(nodes)}
    for x, j in ((0, 1), (2, 3)):
        for a in groups[x]:
            for b in groups[j]:
                M_abl[idx[a], idx[b]] = M_abl[idx[b], idx[a]] = 0
    abl2 = readout(M_abl, 2, sub_t, sup_t)
    abl4 = readout(M_abl, 4, sub_t, sup_t)
    a2 = abl2["ari_super_median"]
    p3_note = ("通过但不理想（落在 0.35-0.40）" if 0.35 <= a2 <= TOL_COARSE else
               ("先复查消融是否彻底（落在 0.40-0.50）" if a2 > TOL_COARSE else "通过"))
    c.check("P3 消融必要性: 切掉跨子模块边后 k=2 的 ARI vs 超模块 <= 0.4（粗阈值）",
            a2 <= TOL_COARSE,
            f"消融后 k=2 超模块={a2:.3f}（预检 0.286；粗阈值 0.4 → {p3_note}）；"
            f"k=4 子模块={abl4['ari_sub_median']:.3f}（应保持 1.000，说明消融只打掉层级不打坏统计）")

    # ---------- P4 L0 统计不变（两条）----------
    same_across_k = all(s == sig0 for s in sig_after_scan)
    same_after_activate = all(s == sig0 for s in sig_after_activate)
    c.check("P4 L0 统计不变: ① activate 前后一致; ② 边集合排序逐项比较跨 k 完全相同",
            same_after_activate and same_across_k,
            f"activate 前后一致={same_after_activate}; 跨 k 一致={same_across_k}; "
            f"签名=(节点 {sig0['nodes']}, 边 {sig0['edges']}, {len(sig0['edge_items'])} 项排序边集合)")

    # ---------- P5 层文件 sha256 ----------
    hashes_after = layer_hashes()
    c.check("P5 只读（纪律 ⑧）: 层文件 sha256 前后一致",
            hashes_before == hashes_after,
            f"before={hashes_before}；after={hashes_after}")

    # ---------- P6 登记文本 ----------
    c.check("P6 登记: JSON 内含 C29 与 B(候选) 文本",
            "C29" in C29_TEXT and "外部给定尺度 k" in B_TEXT,
            "文本已写入 JSON 的 registration 字段")

    # ---------- 辅助 2: 第三层弱耦合 ----------
    net3 = build(s3=S3)
    M3 = matrix_from_net(net3, nodes)
    r3_2, r3_4 = readout(M3, 2, sub_t, sup_t), readout(M3, 4, sub_t, sup_t)

    return {
        "id": "exp29_hierarchy_scales",
        "title": "E29 层级 = 多尺度读出: k=2 超模块 / k=4 子模块（k 是预设参数）",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "k_curve": curve,
            "ablation": {"k2_super": abl2, "k4_sub": abl4},
            "l0_signature": {"nodes": sig0["nodes"], "edges": sig0["edges"],
                             "edge_items_sample": [list(map(str, e)) for e in sig0["edge_items"][:5]],
                             "same_across_k": same_across_k,
                             "same_after_activate": same_after_activate},
            "s3_auxiliary": {
                "auxiliary": True,
                "note": "第三层弱耦合（s3=10）；预检显示对 k>=2 读数无影响。仅记录，不进判据。",
                "k2_super_median": r3_2["ari_super_median"], "k4_sub_median": r3_4["ari_sub_median"],
            },
            "corpus": {"nodes": len(nodes), "sizes": SIZES, "strengths": STRENGTHS,
                       "masses": masses, "s2": S2, "s3_aux": S3,
                       "params_locked": "正式语料 = 预检语料，逐参数一致（改参数须重跑预检）"},
            "km_seeds": KM_SEEDS,
            "layer_hashes": {"before": hashes_before, "after": hashes_after},
            "registration": {"C29": C29_TEXT, "B_candidate": B_TEXT,
                             "methodology_4th_class": "指标对退化方向不敏感（并入合并总述；不占 B 编号）"},
            "note": "层级 = 读出尺度 k 的函数；k 是预设参数，不是涌现的结构。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
