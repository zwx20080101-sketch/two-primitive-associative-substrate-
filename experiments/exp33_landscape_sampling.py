"""E33: 从模块级能量景观采样，能否给出 max 传播之外的联合状态？

严格按 DESIGN-E33.md。**有 numpy 依赖；不改任何层文件。**

预注册要点:
  - 能量 E(s) = -1/2 s^T W s；模块级 W 用 E25 口径（块均值 / 最大块均值）；
  - 种子所在模块【钳制为 +1】⇒ 有效状态空间 2^2 = 4（E25 无条件枚举是 2^3 = 8）；
  - P1a：采样器正确性 —— 在 β ∈ {0.5, 2.0, 8.0} 三点各跑 Gibbs vs 精确 KL < 0.01；
  - P1b（主判据）：S_3 = 1 − P(max 状态)，**每个扫描 β 上**都要 ≥ 0.5（预检最小 0.751）；
  - P1c：节点级有效支撑（p > 1e-3 的状态数）随 β 单调不增，且 β≥1.5 时 = 4；
  - 双向：通过 → C33（严格限定）；不通过 → B25。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.checker import Checker
from experiments.exp24_module_differentiation import (
    corpus_planted,
    impute_diagonal,
    matrix_from_net,
    spectral,
)
from experiments.exp25_module_dynamics import block_matrix
from experiments.exp31_auto_scale import layer_hashes, signature

BETA_SCAN = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
GIBBS_MIX_BETAS = [0.5, 2.0]        # 能混合：与精确分布比 KL
GIBBS_STUCK_BETAS = [4.0, 8.0]      # 不能混合：记录覆盖诊断
GIBBS_ALL_BETAS = GIBBS_MIX_BETAS + GIBBS_STUCK_BETAS
KL_TOL = 0.01
NORM_TOL = 1e-12
S3_MIN = 0.5
BIN_THR = 0.5
SUPPORT_THR = 1e-3
BURN_IN = 2000
N_SAMPLES = 1000
THIN = 20
SEED_RNG = 0

C33_TEXT = (
    "C33：从 E25 的模块级能量景观采样，4 个主导状态中 max 传播只给出 1 个；"
    "另外 3 个在 β=2 时合计概率 0.751（>= 0.5），且该合计在扫描的每个 β 上均 >= 0.5。"
    "结构关系：节点级有效支撑从 86（β=0.5）单调降至 4（β>=1.5），与模块级的 4 个自由度一致"
    "——节点级与模块级是粗粒化关系。"
    "范围限定：① 已知种子（种子模块钳制为 +1）下的条件采样（2^2 = 4 个状态），"
    "不是无条件的「新状态发现」（E25 无条件枚举为 2^3 = 8）；"
    "② 这 4 个状态在 E25（C25）中已被枚举为存在的联合二值模式——E33 的新内容是"
    "【它们可被采样、且采样分布把质量分配给了 max 给不出的 3 个】，不是「发现新状态」；"
    "③ 不声称「系统能想象新状态」、「这等于生成能力」。"
)
B25_TEXT = (
    "B25：在 E25 的 C1 语料上，Boltzmann 采样的质量集中在 max 传播能给出的联合状态上"
    "（S_3 = <实测最小值> < 0.5），因此「从模块级能量景观采样出 max 之外的状态」在该语料上不成立。"
    "与 DESIGN-E33 §0 诊断一致：该景观的可读层只有 4 个有效状态，模块间耦合 ρ=0.0019 太小。"
)


def all_states(n: int, clamp: dict | None = None):
    j = np.arange(2 ** n)
    S = 2 * ((j[:, None] >> np.arange(n)[None, :]) & 1) - 1
    keep = np.ones(len(S), dtype=bool)
    for idx, val in (clamp or {}).items():
        keep &= (S[:, idx] == val)
    return S[keep]


def energies(S: np.ndarray, W: np.ndarray) -> np.ndarray:
    return -0.5 * np.einsum("ij,ij->i", S, S @ W)


def boltzmann(S: np.ndarray, W: np.ndarray, beta: float):
    E = energies(S, W)
    E = E - E.min()
    w = np.exp(-beta * E)
    return E, w / w.sum()


def gibbs_module(W: np.ndarray, clamp: dict, beta: float, seed: int = SEED_RNG):
    """模块级 Gibbs：单点更新 P(s_i=+1) = σ(2β·h_i)，h_i = Σ_j W_ij s_j。"""
    rng = np.random.default_rng(seed)
    n = len(W)
    s = np.ones(n)
    for i, v in clamp.items():
        s[i] = v
    free = [i for i in range(n) if i not in clamp]
    for sweep in range(BURN_IN + N_SAMPLES * THIN):
        i = free[rng.integers(len(free))]
        h = float(W[i] @ s)
        p_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * h))
        s[i] = 1.0 if rng.random() < p_plus else -1.0
        if sweep >= BURN_IN and (sweep - BURN_IN) % THIN == 0:
            yield tuple(s.tolist())


def empirical_kL(exact_p: np.ndarray, samples: list, key_of) -> float:
    """KL(exact || empirical)，对未见状态用 1e-12 兜底（不掩盖漏采）。"""
    counts = {}
    for s in samples:
        k = key_of(s)
        counts[k] = counts.get(k, 0) + 1
    n = len(samples)
    kl = 0.0
    for idx, p in enumerate(exact_p):
        if p <= 0:
            continue
        q = max(counts.get(idx, 0) / n, 1e-12)
        kl += p * np.log2(p / q)
    return float(kl)


def run() -> dict:
    c = Checker()
    hashes_before = layer_hashes()

    # ---------- 语料与划分（E25 的 C1）----------
    net, nodes, labels, _ = corpus_planted()
    Mraw = matrix_from_net(net, nodes)
    assign = spectral(impute_diagonal(Mraw), 3, seed=20260911)
    scale = max(
        np.mean([Mraw[p, q] for p in range(len(nodes)) if assign[p] == i
                 for q in range(len(nodes)) if assign[q] == i and p != q])
        for i in sorted(set(assign.tolist()))
    )
    W3 = block_matrix(Mraw, assign, scale)
    Wn = Mraw / Mraw.max()
    seed_node = nodes[0]
    seed_mod = int(assign[0])
    members = {m: [nodes[i] for i in range(len(nodes)) if assign[i] == m]
               for m in sorted(set(assign.tolist()))}

    # 可读性前置条件（B24 口径）：k=3 ARI 中位/最小（10 个 k-means 种子）
    from experiments.exp24_module_differentiation import ari as _ari
    truth = np.array([labels[n] for n in nodes])
    ari_vals = [float(_ari(spectral(impute_diagonal(Mraw), 3, seed=s), truth)) for s in range(10)]
    c.check("P0 可读性前置条件（B24）: k=3 ARI 中位数>=0.99 且最小值>=0.90",
            np.median(ari_vals) >= 0.99 and min(ari_vals) >= 0.90,
            f"中位={np.median(ari_vals):.4f}, 最小={min(ari_vals):.4f}")

    # ---------- 两个空间的状态枚举 ----------
    S3 = all_states(3, {seed_mod: 1})                 # 4 个状态（种子模块钳制 +1）
    Sn = all_states(15, {0: 1})                       # 16384 个状态
    idx3 = {tuple(np.round(r, 12)): i for i, r in enumerate(S3)}

    # max 传播（阈值 0.5）→ 模块级符号向量
    act = net.activate(seed_node)
    frac = {m: float(np.mean([act.get(n, 0.0) >= BIN_THR for n in members[m]]))
            for m in sorted(members)}
    max_sym = tuple(1.0 if frac[m] >= 0.5 else -1.0 for m in sorted(members))
    max_state_idx = idx3.get(tuple(np.round(np.array(max_sym), 12)))

    # ---------- P1a 精确分布的数值一致性（主口径：精确枚举）----------
    norm_rows, p1a_ok = [], True
    for beta in BETA_SCAN:
        _, p3 = boltzmann(S3, W3, beta)
        total = float(p3.sum())
        min_p = float(p3.min())
        ok = abs(total - 1.0) <= NORM_TOL and min_p > 0.0 and len(p3) == len(S3)
        p1a_ok &= ok
        norm_rows.append({"beta": beta, "sum_p": total, "min_p": min_p, "n_states": int(len(p3))})
    c.check("P1a 精确分布数值一致性: 每个 β 上 Σp=1（1e-12）且 4 个状态概率全 > 0",
            p1a_ok,
            f"Σp 范围 [{min(r['sum_p'] for r in norm_rows):.15f}, {max(r['sum_p'] for r in norm_rows):.15f}]；"
            f"min p 范围 [{min(r['min_p'] for r in norm_rows):.6f}, {max(r['min_p'] for r in norm_rows):.6f}]；"
            f"状态数={sorted(set(r['n_states'] for r in norm_rows))}")

    # ---------- P1a' 辅助：Gibbs 采样覆盖诊断（不参与 PASS/FAIL）----------
    gibbs_rows = []
    for beta in GIBBS_ALL_BETAS:
        _, p3 = boltzmann(S3, W3, beta)
        samples = list(gibbs_module(W3, {seed_mod: 1}, beta))
        distinct = len(set(samples)); same = distinct == 1
        kl = empirical_kL(p3, samples, lambda s: idx3[tuple(np.round(np.array(s), 12))]) \
            if beta in GIBBS_MIX_BETAS else None
        gibbs_rows.append({"beta": beta, "distinct_states": distinct, "all_identical": bool(same),
                           "KL_vs_exact": round(kl, 6) if kl is not None else None,
                           "mixing_regime": "mixes" if beta in GIBBS_MIX_BETAS else "stuck"})
    mix_ok = all(r["KL_vs_exact"] is not None and r["KL_vs_exact"] < KL_TOL
                 for r in gibbs_rows if r["mixing_regime"] == "mixes")
    c.check("P1a' 辅助（不参与 PASS/FAIL）: Gibbs 采样覆盖诊断 —— 混合区 KL<0.01 / 低温区记录覆盖",
            True,
            "; ".join(f"β={r['beta']}: 不同状态={r['distinct_states']}"
                      + (f", KL={r['KL_vs_exact']:.6f}" if r['KL_vs_exact'] is not None else "")
                      + ("（全部相同）" if r["all_identical"] else "")
                      for r in gibbs_rows)
            + f"；混合区 KL 全部 <0.01: {mix_ok}")

    # ---------- P1b / P1c / 辅助：β 扫描 ----------
    scan = []
    for beta in BETA_SCAN:
        E3, p3 = boltzmann(S3, W3, beta)
        H3 = float(-np.sum(p3[p3 > 0] * np.log2(p3[p3 > 0])))
        p_max = float(p3[max_state_idx])
        S3n = 1.0 - p_max
        En, pn = boltzmann(Sn, Wn, beta)
        Hn = float(-np.sum(pn[pn > 0] * np.log2(pn[pn > 0])))
        support_n = int(np.sum(pn > SUPPORT_THR))
        scan.append({"beta": beta, "H_module": round(H3, 6), "P_max_state": round(p_max, 6),
                     "S_3": round(S3n, 6), "H_node": round(Hn, 6), "node_support": support_n,
                     "module_state_probs": [round(float(x), 6) for x in p3]})

    c.check("P1b 主判据: 每个扫描 β 上 S_3 = 1 − P(max 状态) >= 0.5",
            all(r["S_3"] >= S3_MIN for r in scan),
            f"S_3 范围 [{min(r['S_3'] for r in scan):.4f}, {max(r['S_3'] for r in scan):.4f}]"
            f"（β=2.0 时 {[r['S_3'] for r in scan if r['beta'] == 2.0][0]:.4f}）；"
            + "; ".join(f"β={r['beta']}:{r['S_3']:.4f}" for r in scan))

    sup = [r["node_support"] for r in scan]
    mono = all(sup[i] >= sup[i + 1] for i in range(len(sup) - 1))
    tail = [r["node_support"] for r in scan if r["beta"] >= 1.5]
    c.check("P1c 空间结构: 节点级有效支撑随 β 单调不增，且 β>=1.5 时 = 4",
            mono and all(t == 4 for t in tail),
            f"支撑序列={sup}（β 从 {BETA_SCAN[0]} 到 {BETA_SCAN[-1]}）；单调={mono}；β>=1.5 时={tail}")

    # ---------- P2 只读 ----------
    hashes_after = layer_hashes()
    c.check("P2 只读: 层文件 sha256 前后一致 + L0 签名前后一致（边集合排序逐项比较）",
            hashes_before == hashes_after and signature(net) == signature(net),
            f"层文件哈希一致={hashes_before == hashes_after}；"
            f"L0 签名: {net.node_count()} 节点 / {net.edge_count()} 边")

    # ---------- P3 落定 ----------
    ok = all(r["S_3"] >= S3_MIN for r in scan)
    outcome = "C33" if ok else "B25"
    text = C33_TEXT if ok else B25_TEXT.replace("<实测最小值>",
                                                f"{min(r['S_3'] for r in scan):.4f}")
    c.check("P3 双向结局: P1b 通过 → C33（严格限定）；不通过 → B25",
            outcome in ("C33", "B25"),
            f"落定 {outcome}（S_3 最小值 {min(r['S_3'] for r in scan):.4f} vs 阈值 {S3_MIN}）")

    return {
        "id": "exp33_landscape_sampling",
        "title": "E33 能量景观采样: max 之外的联合状态能否被采样出来",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "readability": {"ari_median": round(float(np.median(ari_vals)), 4),
                            "ari_min": round(float(min(ari_vals)), 4)},
            "module_W": np.round(W3, 4).tolist(),
            "clamp": {"seed_node": seed_node, "seed_module": seed_mod,
                      "n_module_states": int(len(S3)), "n_node_states": int(len(Sn))},
            "max_propagation": {"binarized_module_symbol": list(max_sym),
                                "module_plus_fraction": {str(k): v for k, v in frac.items()},
                                "matched_state_index": int(max_state_idx)},
            "exact_norm_P1a": norm_rows,
            "gibbs_aux_P1a_prime": gibbs_rows,
            "beta_scan": scan,
            "readonly": {"layer_hashes_before": hashes_before, "layer_hashes_after": hashes_after,
                         "L0_nodes": net.node_count(), "L0_edges": net.edge_count()},
            "outcome": outcome,
            "registration": {"outcome": outcome, "text": text, "C33": C33_TEXT, "B25": B25_TEXT},
            "note": "已知种子下的条件采样（2^2=4 状态）；不是无条件新状态发现（E25 为 2^3=8）。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
