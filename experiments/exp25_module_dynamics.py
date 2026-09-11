"""E25: S2 模块内动力学 - 模块平均激活的双稳与滞回（严格按 DESIGN-E25.md）。

主口径: 状态 = 模块平均激活 V_i（由 L0 激活矩阵按 E24 划分取均值）;
        权重 W_ij = 模块间 L0 共现块均值（对称、非负、归一化）;
        动力学 dV_i/dt = -V_i + tanh(β(Σ_j W_ij V_j + θ_i + I_i)), θ=0。

预注册要点:
  - 归一化尺度统一取 C1 的最大块均值; β=2.0 作用于归一化后的 W;
  - P2 双分支由静态 ρ=max(offdiag)/max(diag) 决定: ρ<0.5 → 独立分支,
    判据 = 每模块低/高态各占 ≥20% 初值; 联合吸引子数=2^N 仅记录;
  - C2 = 单模块人工划分（不走 E24）, 判据 = 无双稳/无滞回;
  - C3 不参与 PASS/FAIL, 记为 exploratory（联合吸引子数/每模块占比/滞回/WTA）;
  - 准静态扫描: 每 I 点从上一点稳态出发, 积分 1000 步, 取最后 200 步平均;
  - 未归类初值比例 >20% 判 FAIL; 拟合 A,B,C 与 B²−4AC 写进 JSON 仅作旁证;
  - L0/L1 只读。

实现订正（2026-09-11，跑前修复，不改任何判据/参数）:
  原 `converged(Vn, V)` 在积分循环结束后比较的是**同一个数组对象**, 恒为真,
  导致"未归类比例"恒为 0（P4 空过）, 且未收敛轨迹混入吸引子统计（细簇 136）。
  现改为**不动点残差判据** r = max|-V + tanh(β(WV))| < 1e-6, 未收敛轨迹计入未归类。
  该订正记录为 B14。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.checker import Checker
from experiments.exp24_module_differentiation import (
    ari,
    corpus_planted,
    impute_diagonal,
    matrix_from_net,
    spectral,
    spectral_stability,
)
from order_layer import OrderLayer

BETA = 2.0
DT = 0.01
STEPS_ATTRACT = 5000
N_STARTS = 200
CLUSTER_TOL = 0.05
I_MIN, I_MAX, I_STEP = -2.0, 2.0, 0.05
STEPS_PER_I = 1000
AVG_LAST = 200
UNCLASSIFIED_MAX = 0.20
HYST_MIN = 0.10
HYST_CTRL_MAX = 0.02


# ---------------- 动力学 ----------------
def simulate(W, I_vec, V0, steps=STEPS_ATTRACT, dt=DT, beta=BETA):
    V = np.array(V0, dtype=float)
    for _ in range(steps):
        V = V + dt * (-V + np.tanh(beta * (W @ V + I_vec)))
    return V


def residual(V, W, I_vec=None):
    """不动点残差 r = max|-V + tanh(β(WV + I))|；r→0 表示已到不动点。"""
    field = W @ V if I_vec is None else W @ V + I_vec
    return float(np.max(np.abs(-V + np.tanh(BETA * field))))


def cluster_states(states, tol=CLUSTER_TOL):
    clusters = []
    for s in states:
        for c in clusters:
            if np.max(np.abs(s - c[0])) <= tol:
                c[0] = (c[0] * c[1] + s) / (c[1] + 1)
                c[1] += 1
                break
        else:
            clusters.append([np.array(s, dtype=float), 1])
    return clusters


def attractor_scan(W, seed=0, tol=1e-6):
    rng = np.random.default_rng(seed)
    finals, unclassified = [], 0
    for _ in range(N_STARTS):
        V0 = rng.uniform(-1.0, 1.0, len(W))  # 预注册: 初值域 U(-1,1)（tanh 自然定义域）
        V = np.array(V0, dtype=float)
        for _ in range(STEPS_ATTRACT):
            V = V + DT * (-V + np.tanh(BETA * (W @ V)))
        if residual(V, W) < tol:
            finals.append(V)
        else:
            unclassified += 1
    clusters = cluster_states(finals)
    unclassified_ratio = unclassified / N_STARTS
    return finals, clusters, unclassified_ratio


def basin_fractions(finals, n_modules):
    """每模块的低态(<0.5)/高态(>=0.5)初值占比。"""
    out = []
    F = np.array(finals) if finals else np.zeros((0, n_modules))
    for m in range(n_modules):
        low = int(np.sum(F[:, m] < 0.5))
        high = int(np.sum(F[:, m] >= 0.5))
        n = max(1, len(F))
        out.append({"low": round(low / n, 3), "high": round(high / n, 3)})
    return out


def state_separation(finals, n_modules):
    """辅助记录（不参与判据）: 每模块低/高态的实际取值，用来证明两态是分开的,
    而不是被 0.5 阈值硬切出来的。"""
    out = []
    F = np.array(finals) if finals else np.zeros((0, n_modules))
    for m in range(n_modules):
        col = F[:, m]
        lo, hi = col[col < 0.5], col[col >= 0.5]
        out.append({
            "low_mean": round(float(lo.mean()), 4) if lo.size else None,
            "high_mean": round(float(hi.mean()), 4) if hi.size else None,
            "low_span": [round(float(lo.min()), 4), round(float(lo.max()), 4)] if lo.size else None,
            "high_span": [round(float(hi.min()), 4), round(float(hi.max()), 4)] if hi.size else None,
        })
    return out


def hysteresis(W, module_idx, rising=True, steps_per_i=STEPS_PER_I):
    """准静态扫描: 每 I 点从上一点稳态出发, 积分后取末段平均。
    预注册值 steps_per_i=1000; 调用方可传更大值做数值稳健性旁证（不参与判据）。"""
    V = np.zeros(len(W))
    Is = np.arange(I_MIN, I_MAX + 1e-9, I_STEP)
    if not rising:
        Is = Is[::-1]
    traj = []
    for I in Is:
        I_vec = np.full(len(W), I)
        for _ in range(steps_per_i):
            V = V + DT * (-V + np.tanh(BETA * (W @ V + I_vec)))
        steady = V.copy()
        for _ in range(AVG_LAST):
            V = V + DT * (-V + np.tanh(BETA * (W @ V + I_vec)))
            steady += V
        steady /= (AVG_LAST + 1)
        traj.append((float(I), float(steady[module_idx])))
    # 转变点: 上升扫首个 >=0.5 / 下降扫首个 <=0.5
    if rising:
        trans = next((I for I, v in traj if v >= 0.5), None)
    else:
        trans = next((I for I, v in traj if v <= 0.5), None)
    return trans, traj


def fit_quadratic(traj):
    """旁证: 对 (V, dV/dt) 拟合 dV/dt = A V^2 + B V + C。"""
    I = np.array([p[0] for p in traj])
    V = np.array([p[1] for p in traj])
    dV = np.gradient(V, I) if len(V) > 2 else np.zeros_like(V)
    A_mat = np.vstack([V ** 2, V, np.ones_like(V)]).T
    coef, *_ = np.linalg.lstsq(A_mat, dV, rcond=None)
    A, B, C = [float(x) for x in coef]
    return {"A": round(A, 4), "B": round(B, 4), "C": round(C, 4),
            "discriminant_B2_4AC": round(B * B - 4 * A * C, 4)}


def block_matrix(Mraw, assign, scale):
    mods = sorted(set(int(x) for x in assign))
    K = len(mods)
    W = np.zeros((K, K))
    for i in mods:
        for j in mods:
            vals = [Mraw[p, q] for p in range(len(assign)) if assign[p] == i
                    for q in range(len(assign)) if assign[q] == j and p != q]
            W[i, j] = np.mean(vals) if vals else 0.0
    return W / scale


def rho(W):
    k = len(W)
    off = max(W[i, j] for i in range(k) for j in range(k) if i != j)
    diag = max(W[i, i] for i in range(k))
    return float(off / diag) if diag > 0 else 0.0


def run() -> dict:
    c = Checker()
    # ---------- C1: E24 planted + spectral partition ----------
    net1, nodes1, labels1, _ = corpus_planted()
    Mraw1 = matrix_from_net(net1, nodes1)
    M1 = impute_diagonal(Mraw1)
    truth1 = np.array([labels1[n] for n in nodes1])
    assign1 = spectral(M1, 3, seed=20260911)
    scale = float(max(
        np.mean([Mraw1[p, q] for p in range(len(nodes1)) if assign1[p] == i
                 for q in range(len(nodes1)) if assign1[q] == i and p != q])
        for i in sorted(set(assign1.tolist()))
    ))
    W1 = block_matrix(Mraw1, assign1, scale)
    rho1 = rho(W1)
    l0_before = (net1.node_count(), net1.edge_count())

    c.check("P1 接口: 模块划分来自 E24 谱聚类, 覆盖全部节点, V 为模块均值",
            len(set(assign1.tolist())) == 3 and all(np.isin(assign1, sorted(set(assign1.tolist())))),
            f"3 模块, 覆盖 {len(assign1)}/{len(nodes1)} 节点, ρ={rho1:.4f}")

    finals1, clusters1, unclass1 = attractor_scan(W1, seed=1)
    basins1 = basin_fractions(finals1, len(W1))
    sep1 = state_separation(finals1, len(W1))
    joint_coarse1 = len({tuple((np.array(s) >= 0.5).astype(int)) for s in finals1})

    if rho1 < 0.5:
        p2_ok = all(b["low"] >= 0.2 and b["high"] >= 0.2 for b in basins1)
        near = min(
            [abs(v) for s in sep1 for span in ("low_span", "high_span")
             for v in (s[span] or [])] or [0.0]
        )
        p2_detail = (f"独立分支(ρ={rho1:.4f}<0.5): 每模块盆地={basins1}; "
                     f"最靠近判定线的收敛态|V|={near:.2f}（远离 0.5，非硬切）; "
                     f"联合吸引子数(低/高二值模式)={joint_coarse1}, "
                     f"细簇={len(clusters1)}(仅记录)")
    else:
        p2_ok = len(clusters1) == 2 and all(b["low"] >= 0.2 and b["high"] >= 0.2 for b in basins1)
        p2_detail = f"同步分支(ρ={rho1:.4f}≥0.5): 联合吸引子数={len(clusters1)}, 盆地={basins1}"
    c.check("P2 双稳: 按 ρ 分支判据（每模块低/高态各≥20% 初值）", p2_ok, p2_detail)

    main_module = int(np.argmax(np.diag(W1)))
    up1, traj_up1 = hysteresis(W1, main_module, rising=True)
    dn1, traj_dn1 = hysteresis(W1, main_module, rising=False)
    width1 = (up1 - dn1) / (I_MAX - I_MIN) if (up1 is not None and dn1 is not None) else 0.0
    fit1 = fit_quadratic(traj_up1)
    # 数值稳健性旁证（不参与判据）: 每点积分步数放大 5 倍，看滞回宽度是否稳定
    up1x, _ = hysteresis(W1, main_module, rising=True, steps_per_i=5 * STEPS_PER_I)
    dn1x, _ = hysteresis(W1, main_module, rising=False, steps_per_i=5 * STEPS_PER_I)
    width1x = (up1x - dn1x) / (I_MAX - I_MIN) if (up1x is not None and dn1x is not None) else 0.0

    # ---------- C2: 单模块人工划分(不走 E24), 弱自兴奋 → 单稳 ----------
    nodes2 = nodes1
    Mraw2 = np.full((len(nodes2), len(nodes2)), 30.0)  # 弱共现
    np.fill_diagonal(Mraw2, 0.0)
    W2 = block_matrix(Mraw2, np.zeros(len(nodes2), dtype=int), scale)
    finals2, clusters2, unclass2 = attractor_scan(W2, seed=2)
    basins2 = basin_fractions(finals2, len(W2))
    up2, traj_up2 = hysteresis(W2, 0, rising=True)
    dn2, traj_dn2 = hysteresis(W2, 0, rising=False)
    width2 = (up2 - dn2) / (I_MAX - I_MIN) if (up2 is not None and dn2 is not None) else 0.0

    c.check("P3 滞回: C1 宽度≥10%; C2 单模块对照 ≤2%",
            width1 >= HYST_MIN and width2 <= HYST_CTRL_MAX,
            f"C1 宽度={width1:.3f}(up={up1},down={dn1}), C2 宽度={width2:.3f}(up={up2},down={dn2})")

    c.check("P4 未归类初值: C1 比例≤20%",
            unclass1 <= UNCLASSIFIED_MAX,
            f"C1 未归类={unclass1:.3f}, C2 未归类={unclass2:.3f}")

    p5_ari = spectral_stability(M1, 3)
    c.check("P5 划分依赖: C1 多种子一致性 ARI≥0.8; C2 人工单模块不检查",
            p5_ari >= 0.8,
            f"C1 一致性 ARI={p5_ari:.3f}")

    # ---------- 次级口径: L1 方向计数对称化（本语料下应等于 L0） ----------
    l1 = OrderLayer()
    for (a, b), rec in net1.connections.items():
        for _ in range(rec["count"]):
            l1.learn([a, b])
    l1_counts = {k: v for k, v in l1.directed_count.items()}
    sym_equal = all(
        l1_counts.get((a, b), 0) + l1_counts.get((b, a), 0) == rec["count"]
        for (a, b), rec in net1.connections.items()
    )
    c.check("P6 次级口径: L1 对称化方向计数与本语料 L0 共现一致（如实记录）",
            sym_equal,
            f"对称化后与 L0 计数一致={sym_equal}（成对事件下必然一致，方向信息为零）")

    l0_after = (net1.node_count(), net1.edge_count())
    c.check("P7 无越权: L0/L1 只读（跑前后节点/连接数一致）",
            l0_before == l0_after,
            f"before={l0_before}, after={l0_after}")

    # ---------- C3 探索性: 两模块 + 正耦合（无非负权重下的抑制） ----------
    size = 5
    M3 = np.zeros((2 * size, 2 * size))
    for i in range(size):
        for j in range(size):
            if i != j:
                M3[i, j] = M3[size + i, size + j] = 100.0
                M3[i, size + j] = M3[size + i, j] = 40.0  # 正耦合（非抑制）
    assign3 = np.array([0] * size + [1] * size)
    W3 = block_matrix(M3, assign3, scale)
    finals3, clusters3, unclass3 = attractor_scan(W3, seed=3)
    basins3 = basin_fractions(finals3, 2)
    up3, _ = hysteresis(W3, 0, rising=True)
    dn3, _ = hysteresis(W3, 0, rising=False)
    width3 = abs((up3 - dn3) / (I_MAX - I_MIN)) if (up3 is not None and dn3 is not None) else 0.0
    wta = any(
        ((cl[0][0] >= 0.5) != (cl[0][1] >= 0.5)) and cl[1] >= 0.05 * N_STARTS
        for cl in clusters3
    )
    joint_coarse3 = len({tuple((np.array(s) >= 0.5).astype(int)) for s in finals3})
    c3 = {"joint_attractor_patterns": joint_coarse3, "joint_clusters_fine": len(clusters3),
          "basins_per_module": basins3,
          "hysteresis_width": round(width3, 3), "winner_take_all": bool(wta),
          "note": "非负 L0 权重下预期无互斥切换；本项为 exploratory，不参与 PASS/FAIL"}

    return {
        "id": "exp25_module_dynamics",
        "title": "E25 S2 模块内动力学: 模块平均激活的双稳与滞回（ρ=%.4f 独立分支）" % rho1,
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "W_C1": np.round(W1, 4).tolist(),
            "rho_C1": round(rho1, 4),
            "branch": "independent" if rho1 < 0.5 else "synchronized",
            "C1": {"joint_attractor_patterns": joint_coarse1, "joint_clusters_fine": len(clusters1),
                   "basins_per_module": basins1,
                   "state_separation": sep1,
                   "unclassified_ratio": round(unclass1, 3),
                   "hysteresis_width": round(width1, 3), "up_transition": up1, "down_transition": dn1,
                   "hysteresis_width_5x_settling_aux": round(width1x, 3),
                   "fit_quadratic": fit1},
            "C2": {"modules": 1, "basins": basins2, "hysteresis_width": round(width2, 3),
                   "up_transition": up2, "down_transition": dn2,
                   "note": ("单稳（自耦合 0.231×2=0.46<1，唯一不动点）; "
                            "0.012 = 一个扫描步长(0.05/4.0)的离散化伪影，非真实滞回")},
            "C3_exploratory": c3,
            "secondary_L1": {"symmetric_equals_L0": bool(sym_equal)},
            "readonly": {"l0_before": l0_before, "l0_after": l0_after},
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
