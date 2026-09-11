"""E26: S3 模块间协调 —— EI 整合量与剂量响应（严格按 DESIGN-E26.md）。

主指标: EI_int = EI_macro - Σ_m EI_macro,m   ("宏观整体 - 各模块单独之和")
  - 模块间无边时机制按模块因子化, 互信息对乘积分布可加 ⇒ EI_int 精确为 0;
  - 主判据 = 剂量响应: 对 w=0/10/25/40/65, 10 个语料种子每一个都必须严格单调。

口径（预注册）:
  - 机制 = 一步并行 Glauber: P(x_i'=+1) = 0.5(1+tanh(β·Σ_j W_ij x_j)), β=2.0;
  - W = impute_diagonal(L0 计数) 后除以全局最大值（对角插补与 E24 读出同处理）;
  - 干预分布 = 均匀（最大熵, Tononi 口径）; 精确枚举 2^12 状态, 不用采样估计;
  - 宏观状态 = 模块内节点多数表决（±1 尺度上等价于 0/1 尺度按 0.5 二值化）;
  - C3' 的 TE 用同一条机制、同一个均匀干预、同一套精确枚举。

边界: 不改 L0/L1; 不引入负权重/stub/语义/情绪/奖励; EI 只在读取端算。
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from order_layer import OrderLayer
from synapse_net import SynapseNet

from experiments.checker import Checker
from experiments.exp24_module_differentiation import (
    SESSION_SEED,
    ari,
    corpus_planted,
    impute_diagonal,
    matrix_from_net,
    spectral,
    spectral_stability,
)
from experiments.exp25_module_dynamics import block_matrix, rho

BETA = 2.0
N_MODULES = 3
SIZES = [3, 4, 5]
STRENGTHS = [90, 110, 130]
ORDER_WINDOW_REPS = 5
CROSS_DENSITY = 0.5
W_DOSE = [0, 10, 25, 40, 65]
W_SATURATION = 91
EXPERIENCE_CURVE = [0, 1, 5, 20, 100]
N_SEEDS = 10
ZERO_TOL = 1e-9
MIN_EI_INT = 0.01
P6_TOL = 1e-6
SEED_PARTITION = SESSION_SEED

C3_SIZES = [4, 4, 4]
C3_STRENGTHS = [90, 60, 130]
C3_W_STRONG = 65
C3_W_WEAK = 40


def _all_states(n: int) -> np.ndarray:
    """全部 ±1 状态; 约定 node i = 索引第 i 位（LSB-first）。

    必须与 `_product_table` 的倍增列序严格一致: 倍增在第 i 轮拆出 node i 的
    ±1 两支, 因此 node i 对应权重 2^i。两者约定不一致会让"块对角机制"
    也测出非零整合量（E26 实现期踩过这个坑, 已由 B17 记录）。
    """
    j = np.arange(2 ** n)
    bits = (j[:, None] >> np.arange(n)[None, :]) & 1
    return (2 * bits - 1).astype(float)


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-300, 1.0 - 1e-300)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def _mechanism(Mraw: np.ndarray) -> np.ndarray:
    """对角按行最大值插补（与 E24 读出同处理）, 再按全局最大值归一化。"""
    W = impute_diagonal(Mraw)
    return W / W.max()


def _product_table(W: np.ndarray, X: np.ndarray, beta: float = BETA):
    """每状态 x 的"下一步独立 Bernoulli 乘积分布", 展开成 2^n × 2^n 表。"""
    P = 0.5 * (1.0 + np.tanh(beta * (X @ W.T)))
    Mx = np.ones((len(X), 1))
    for i in range(X.shape[1]):
        p = P[:, i:i + 1]
        Mx = np.concatenate([Mx * (1.0 - p), Mx * p], axis=1)
    return P, Mx


def _mi_from_table(Mx: np.ndarray, lab_row: np.ndarray, lab_col: np.ndarray, k: int):
    """按 k 位标签粗粒化, 精确算两套标签之间的互信息（比特）。"""
    R = np.zeros((len(lab_row), 2 ** k))
    R[np.arange(len(lab_row)), lab_row] = 1.0
    C = np.zeros((Mx.shape[1], 2 ** k))
    C[np.arange(Mx.shape[1]), lab_col] = 1.0
    J = (R.T @ Mx @ C) / len(lab_row)
    pr, pc = J.sum(axis=1), J.sum(axis=0)
    out = 0.0
    for a in range(2 ** k):
        for b in range(2 ** k):
            if J[a, b] > 1e-300 and pr[a] > 1e-300 and pc[b] > 1e-300:
                out += J[a, b] * np.log2(J[a, b] / (pr[a] * pc[b]))
    return float(out), J


def _macro_labels(X: np.ndarray, blocks: list) -> np.ndarray:
    """V_m = 模块内多数表决（±1 尺度上等价于 0/1 尺度按 0.5 二值化）。"""
    lab = np.zeros(len(X), dtype=int)
    for b in blocks:
        lab = lab * 2 + (X[:, b].mean(axis=1) >= 0.0).astype(int)
    return lab


def _blocks_of(assign: np.ndarray) -> list:
    return [np.where(assign == m)[0] for m in sorted(set(assign.tolist()))]


def ei_decomposition(Mraw: np.ndarray, assign: np.ndarray, beta: float = BETA) -> dict:
    """一次算清 EI_micro / EI_macro / 各模块分量 / EI_int（全部精确枚举）。"""
    W = _mechanism(Mraw)
    X = _all_states(len(W))
    P, Mx = _product_table(W, X, beta)
    marg = Mx.mean(axis=0)
    nz = marg > 1e-300
    H_next = float(-np.sum(marg[nz] * np.log2(marg[nz])))
    H_cond = float(np.mean(_binary_entropy(P).sum(axis=1)))
    ei_micro = H_next - H_cond

    blocks = _blocks_of(assign)
    nb = len(blocks)
    lab = _macro_labels(X, blocks)
    ei_macro, J = _mi_from_table(Mx, lab, lab, nb)
    parts = []
    for i in range(nb):
        r = (lab >> (nb - 1 - i)) & 1
        v, _ = _mi_from_table(Mx, r, r, 1)
        parts.append(float(v))
    return {
        "ei_micro": float(ei_micro),
        "ei_macro": float(ei_macro),
        "parts": parts,
        "ei_int": float(ei_macro - sum(parts)),
        "macro_joint": J,
        "n_blocks": nb,
        "micro_bits": float(np.log2(Mx.shape[1])),
        "macro_bits": float(nb),
    }


def transfer_entropy(J: np.ndarray, nb: int, tgt_i: int, src_i: int) -> float:
    """TE(S→T) = I(V'_T ; V_S | V_T)，从宏观联合表精确算（同一条 Glauber 机制）。

    TE = Σ P(v_T,v_S,v'_T) log2 [ P(v'_T|v_T,v_S) / P(v'_T|v_T) ]
    注意分子是"给定当前整体状态后**目标位**的边际分布"（要对非目标位求和）,
    不是整个下一步状态的 P(v'|v)——写错会算出负值（互信息不可能为负）。
    """
    P = J / J.sum()
    idx = np.arange(2 ** nb)
    tgt = (idx >> (nb - 1 - tgt_i)) & 1
    src = (idx >> (nb - 1 - src_i)) & 1
    Q = np.zeros((2, 2, 2))                      # Q[v_T, v_S, v'_T]
    for a in range(2 ** nb):
        for b in range(2 ** nb):
            Q[tgt[a], src[a], tgt[b]] += P[a, b]
    out = 0.0
    for t in range(2):
        row_tot = Q[t, :, :].sum()
        if row_tot <= 1e-300:
            continue
        for s in range(2):
            joint_tot = Q[t, s, :].sum()
            if joint_tot <= 1e-300:
                continue
            for tp in range(2):
                p_joint = Q[t, s, tp]
                if p_joint <= 1e-300:
                    continue
                num = p_joint / joint_tot          # P(v'_T | v_T, v_S)
                den = Q[t, :, tp].sum() / row_tot  # P(v'_T | v_T)
                if num > 1e-300 and den > 1e-300:
                    out += p_joint * np.log2(num / den)
    return float(out)


def _module_groups(sizes, prefix="m"):
    return [[f"{prefix}{j}_{i}" for i in range(sizes[j])] for j in range(len(sizes))]


def _cross_pairs(groups):
    return [(a, b) for j, k in itertools.combinations(range(len(groups)), 2)
            for a in groups[j] for b in groups[k]]


def corpus_events(seed: int, w_cross: int, with_cross: bool = True,
                  density: float = CROSS_DENSITY):
    """C1'/C2' 语料: 模块内成对事件 + 3-token 顺序窗口 + 半稠密跨模块有序事件。"""
    rng = np.random.default_rng(seed)
    groups = _module_groups(SIZES)
    events = []
    for j, g in enumerate(groups):
        s = int(round(STRENGTHS[j] * (1.0 + 0.10 * (rng.random() - 0.5))))
        for a, b in itertools.combinations(g, 2):
            events.append(([a, b], s))
        for tri in itertools.combinations(g, 3):
            events.append((list(tri), ORDER_WINDOW_REPS))
    cross = []
    if with_cross:
        all_cps = _cross_pairs(groups)
        n_keep = int(round(density * len(all_cps)))      # 定数抽样: 保证密度达标
        keep = rng.permutation(len(all_cps))[:n_keep]
        cross = [all_cps[i] for i in sorted(keep.tolist())]
        for a, b in cross:
            events.append(([a, b], w_cross))
    return groups, events, cross


def corpus_arbitration(seed: int = 0):
    """C3' 语料: 两个源模块投射到共享目标模块（强度不等），从不反向。"""
    rng = np.random.default_rng(1000 + seed)
    groups = _module_groups(C3_SIZES, prefix="a")
    events = []
    for j, g in enumerate(groups):
        s = int(round(C3_STRENGTHS[j] * (1.0 + 0.10 * (rng.random() - 0.5))))
        for a, b in itertools.combinations(g, 2):
            events.append(([a, b], s))
        for tri in itertools.combinations(g, 3):
            events.append((list(tri), ORDER_WINDOW_REPS))
    strong = _cross_pairs([groups[0], groups[2]])
    weak = _cross_pairs([groups[1], groups[2]])
    for a, b in strong:
        events.append(([a, b], C3_W_STRONG))
    for a, b in weak:
        events.append(([a, b], C3_W_WEAK))
    return groups, events, strong, weak


def apply_events(events, with_l1: bool = False):
    net = SynapseNet()
    l1 = OrderLayer() if with_l1 else None
    for tokens, reps in events:
        for _ in range(reps):
            net.learn(tokens)
            if l1 is not None:
                l1.learn(tokens)
    return net, l1


def partition(Mraw: np.ndarray, truth: np.ndarray | None = None):
    """E24 谱聚类读出 + 预注册的稳定性例外（不稳定则改用真值）。"""
    M = impute_diagonal(Mraw)
    assign = spectral(M, N_MODULES, seed=SEED_PARTITION)
    stable = float(spectral_stability(M, N_MODULES))
    note = "spectral"
    if stable < 0.8 and truth is not None:
        assign = np.asarray(truth).copy()
        note = "unstable->truth"
    return assign, stable, note


def block_mean(Mraw: np.ndarray, assign: np.ndarray, i: int, j: int) -> float:
    vals = [Mraw[p, q] for p in range(len(assign)) if assign[p] == i
            for q in range(len(assign)) if assign[q] == j and p != q]
    return float(np.mean(vals)) if vals else 0.0


def cross_block_mean(Mraw: np.ndarray, assign: np.ndarray) -> float:
    ks = sorted(set(assign.tolist()))
    vals = [block_mean(Mraw, assign, i, j) for i in ks for j in ks if i != j]
    return max(vals) if vals else 0.0


def directed_blocks(l1: OrderLayer, nodes, assign: np.ndarray) -> np.ndarray:
    idx = {n: i for i, n in enumerate(nodes)}
    K = len(set(assign.tolist()))
    D = np.zeros((K, K))
    for (u, v), cnt in l1.directed_count.items():
        D[assign[idx[u]], assign[idx[v]]] += cnt
    return D


def asym(D: np.ndarray) -> float:
    n = float(np.linalg.norm(D))
    return float(np.linalg.norm(D - D.T) / n) if n > 0 else 0.0


def _truth_of(groups) -> np.ndarray:
    return np.array([j for j, g in enumerate(groups) for _ in g])


def _nodes_of(groups):
    return [n for g in groups for n in g]


def _ablate_cross(Mraw: np.ndarray, assign: np.ndarray) -> np.ndarray:
    M = Mraw.copy()
    for p in range(len(assign)):
        for q in range(len(assign)):
            if assign[p] != assign[q]:
                M[p, q] = 0.0
    return M


def run() -> dict:
    c = Checker()

    # ---------- P1 接口回归（E24 语料） ----------
    net_r, nodes_r, labels_r, _ = corpus_planted()
    Mr = matrix_from_net(net_r, nodes_r)
    assign_r = spectral(impute_diagonal(Mr), N_MODULES, seed=SEED_PARTITION)
    truth_r = np.array([labels_r[n] for n in nodes_r])
    ari_r = float(ari(assign_r, truth_r))
    scale_r = max(block_mean(Mr, assign_r, i, i) for i in sorted(set(assign_r.tolist())))
    rho_r = rho(block_matrix(Mr, assign_r, scale_r))
    c.check("P1 接口回归: E24 语料重现谱聚类 ARI=1.000 与 rho=0.0019",
            ari_r >= 0.999 and abs(rho_r - 0.0019) < 5e-4,
            f"ARI={ari_r:.3f}, 3 模块, rho={rho_r:.4f}（期望 0.0019）")

    # ---------- P2 语料构造核查 ----------
    groups1, events1, cross1 = corpus_events(seed=0, w_cross=W_DOSE[-1])
    nodes1, truth1 = _nodes_of(groups1), _truth_of(groups1)
    net1, l1_1 = apply_events(events1, with_l1=True)
    Mraw1 = matrix_from_net(net1, nodes1)
    assign1, stable1, note1 = partition(Mraw1, truth1)
    dens1 = len(cross1) / len(_cross_pairs(groups1))
    xmean1 = cross_block_mean(Mraw1, assign1)

    groups2, events2, _cross2 = corpus_events(seed=0, w_cross=W_DOSE[-1], with_cross=False)
    net2, _ = apply_events(events2)
    Mraw2 = matrix_from_net(net2, nodes1)
    assign2, stable2, note2 = partition(Mraw2, truth1)
    cross2_observed = sum(
        1 for p, q in itertools.combinations(range(len(nodes1)), 2)
        if assign2[p] != assign2[q] and Mraw2[p, q] > 0
    )
    cover1 = len(assign1) / len(nodes1)
    c.check("P2 语料构造核查: C1' 跨模块密度>=50% 且块均值>=25; C2' 跨模块对=0",
            dens1 >= 0.5 and xmean1 >= 25.0 and cross2_observed == 0 and cover1 == 1.0,
            f"C1' 密度={dens1:.2f}, 跨模块块均值={xmean1:.1f}; "
            f"C2' 跨模块非零对={cross2_observed}; 划分覆盖={cover1:.2f}")

    # ---------- P3 剂量响应（10 个种子各自严格单调） ----------
    per_seed, monotone = [], []
    for seed in range(N_SEEDS):
        row = []
        for w in W_DOSE:
            _g, ev, _c = corpus_events(seed=seed, w_cross=w)
            net, _ = apply_events(ev)
            Mm = matrix_from_net(net, nodes1)
            asg, _s, _n = partition(Mm, truth1)
            row.append(ei_decomposition(Mm, asg)["ei_int"])
        per_seed.append(row)
        monotone.append(all(row[i] < row[i + 1] for i in range(len(row) - 1)))
    ei_mean = [float(np.mean([r[i] for r in per_seed])) for i in range(len(W_DOSE))]
    min_w65 = float(min(r[-1] for r in per_seed))
    c.check("P3 剂量响应: 10 个种子每一个都严格单调 且 EI_int(w=65)>=0.01",
            all(monotone) and min_w65 >= MIN_EI_INT,
            f"单调种子数={sum(monotone)}/{N_SEEDS}; EI_int 均值={[round(v, 4) for v in ei_mean]}; "
            f"min(EI_int@65)={min_w65:.4f}")

    # ---------- P4 零耦合对照: EI_int 精确为 0 ----------
    c2 = ei_decomposition(Mraw2, assign2)
    c.check("P4 负对照（精确预测）: C2' 的 EI_int = 0",
            abs(c2["ei_int"]) < ZERO_TOL,
            f"C2' EI_int={c2['ei_int']:.3e}（容差 {ZERO_TOL:g}）")

    # ---------- P5 必要性消融: 跨模块边置零 -> EI_int 精确为 0 ----------
    full1 = ei_decomposition(Mraw1, assign1)
    abl = ei_decomposition(_ablate_cross(Mraw1, assign1), assign1)
    c.check("P5 必要性消融（精确预测）: 跨模块边置零后 EI_int = 0",
            abs(abl["ei_int"]) < ZERO_TOL,
            f"未消融 EI_int={full1['ei_int']:.4f} → 消融后={abl['ei_int']:.3e}（容差 {ZERO_TOL:g}）")

    # ---------- P6 因果涌现（双向判据, 只记方向） ----------
    d_micro = full1["ei_micro"] / full1["micro_bits"]
    d_macro = full1["ei_macro"] / full1["macro_bits"]
    diff = d_macro - d_micro
    p6_dir = ("B 无显著差异" if abs(diff) < P6_TOL else
              ("C 宏观不劣于微观" if diff > 0 else "B 宏观不如微观（该层不必要）"))
    c.check("P6 因果涌现（双向判据, 任一方向都算完成）", True,
            f"EI_micro_density={d_micro:.3f}, EI_macro_density={d_macro:.3f}, "
            f"差={diff:+.3f} -> {p6_dir}")

    # ---------- P7 无越权: L0/L1 只读 ----------
    before = (net1.node_count(), net1.edge_count(), len(l1_1.nodes))
    _ = ei_decomposition(Mraw1, assign1)
    after = (net1.node_count(), net1.edge_count(), len(l1_1.nodes))
    c.check("P7 无越权: L0/L1 只读, EI 不新增结构节点",
            before == after,
            f"L0/L1 跑前跑后={before} -> {after}")

    # ---------- 辅助记录 ----------
    curve = []
    for w in EXPERIENCE_CURVE:
        _g, ev, _c = corpus_events(seed=0, w_cross=w)
        net, _ = apply_events(ev)
        Mm = matrix_from_net(net, nodes1)
        asg, _s, _n = partition(Mm, truth1)
        curve.append({"cross_events": w,
                      "EI_int": round(ei_decomposition(Mm, asg)["ei_int"], 5)})

    sat = []
    for seed in range(N_SEEDS):
        _g, ev, _c = corpus_events(seed=seed, w_cross=W_SATURATION)
        net, _ = apply_events(ev)
        Mm = matrix_from_net(net, nodes1)
        asg, _s, _n = partition(Mm, truth1)
        sat.append(round(ei_decomposition(Mm, asg)["ei_int"], 4))

    asym1 = asym(directed_blocks(l1_1, nodes1, assign1))

    # C3' 探索性: 共享下游仲裁
    g3, ev3, strong3, weak3 = corpus_arbitration()
    nodes3, truth3 = _nodes_of(g3), _truth_of(g3)
    net3, l1_3 = apply_events(ev3, with_l1=True)
    Mraw3 = matrix_from_net(net3, nodes3)
    assign3, stable3, _note3 = partition(Mraw3, truth3)
    full3 = ei_decomposition(Mraw3, assign3)
    nb3 = full3["n_blocks"]
    tgt_block = int(np.bincount(assign3[truth3 == 2]).argmax())
    te = {}
    for name, src_truth in (("S1", 0), ("S2", 1)):
        src_block = int(np.bincount(assign3[truth3 == src_truth]).argmax())
        te[f"TE({name}->T)"] = round(
            transfer_entropy(full3["macro_joint"], nb3, tgt_block, src_block), 5)
        te[f"TE(T->{name})"] = round(
            transfer_entropy(full3["macro_joint"], nb3, src_block, tgt_block), 5)
    c3 = {
        "EI_int": round(full3["ei_int"], 5),
        "EI_macro": round(full3["ei_macro"], 5),
        "n_cross_pairs_strong": len(strong3),
        "n_cross_pairs_weak": len(weak3),
        "asym": round(asym(directed_blocks(l1_3, nodes3, assign3)), 4),
        "transfer_entropy": te,
        "partition_stability_ARI": round(stable3, 3),
        "partition_vs_truth_ARI": round(float(ari(assign3, truth3)), 3),
        "note": "第二口径（共享下游仲裁）, exploratory, 不参与 PASS/FAIL",
    }

    scale1 = max(block_mean(Mraw1, assign1, i, i) for i in sorted(set(assign1.tolist())))
    return {
        "id": "exp26_module_coordination",
        "title": "E26 S3 模块间协调: EI 整合量 + 剂量响应（半稠密 w=65）",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "corpus": {
                "C1_prime": {"modules": SIZES, "strengths": STRENGTHS,
                             "cross_density": round(dens1, 3), "cross_pairs": len(cross1),
                             "cross_block_mean": round(xmean1, 2),
                             "partition": assign1.tolist(), "partition_note": note1,
                             "partition_stability_ARI": round(stable1, 3)},
                "C2_prime": {"cross_pairs": 0, "partition_note": note2,
                             "partition_stability_ARI": round(stable2, 3)},
                "C3_prime": {"modules": C3_SIZES, "strengths": C3_STRENGTHS,
                             "source_weights": {"S1": C3_W_STRONG, "S2": C3_W_WEAK}},
            },
            "dose_response": {
                "w_values": W_DOSE,
                "EI_int_mean": [round(v, 4) for v in ei_mean],
                "EI_int_per_seed": [[round(v, 5) for v in r] for r in per_seed],
                "monotone_per_seed": monotone,
                "min_EI_int_w65": round(min_w65, 5),
            },
            "saturation_w91": {"EI_int_per_seed": sat,
                               "note": "饱和区记录点（B16）: 非单调, 置换对照退化为恒等"},
            "EI_parts_C1": {
                "ei_micro": round(full1["ei_micro"], 5),
                "ei_macro": round(full1["ei_macro"], 5),
                "module_parts": [round(v, 5) for v in full1["parts"]],
                "ei_int": round(full1["ei_int"], 5),
                "EI_micro_density": round(d_micro, 4),
                "EI_macro_density": round(d_macro, 4),
            },
            "C2_prime_EI_int": round(c2["ei_int"], 12),
            "ablation_EI_int": round(abl["ei_int"], 12),
            "experience_curve": curve,
            "asym_C1_prime": round(asym1, 4),
            "rho_C1": round(rho(block_matrix(Mraw1, assign1, scale1)), 4),
            "C3_prime_exploratory": c3,
            "readonly": {"L0_L1_before": list(before), "L0_L1_after": list(after)},
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
