"""E32: B23 的层数对照实验（三个准则 × 四个嵌套层数）。

严格按 DESIGN-E32.md。**有 numpy 依赖；不改任何层文件。**

预注册要点:
  - 四个嵌套层数 L=1（基线对照）/2/3/4，块强度为等差序列（写死在 §1）；
  - 三准则（eigengap / modularity / BIC）**全部测**，每格给 k̂ 或 null+reason；
  - 命中分级：主真值 / 中层 / 粗层 / 未命中；L=1 是基线，不参与 B23 判定；
  - P3：近零特征值个数（λ<1e-6）必须精确等于最粗层块数 bL（阈值依据见 §1.1 的预检实测）；
  - 双向结局三分支：A 升级 / B-1 收窄·偏移一层 / B-2 收窄·仅 L=2；
  - 准则是 import 自 E31 的单一实现（不重写 —— B17 的教训）。
"""

from __future__ import annotations

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
from experiments.exp31_auto_scale import criteria, layer_hashes, signature

NEAR_ZERO = 1e-6
K_MAX = 8
S_LEVEL = {2: 40, 3: 16, 4: 6}
REASON_ENUM = ["computation_error", "criteria_undefined", "out_of_range"]

# ---- 语料定义（逐参数写死，见 DESIGN-E32 §1）----
CORPORA = {
    1: {"sizes": [4, 4, 4], "strengths": [90, 100, 110], "levels": [], "b": [3]},
    2: {"sizes": [3, 3, 3, 3], "strengths": [90, 100, 110, 120],
        "levels": [[[0, 1], [2, 3]]], "b": [4, 2]},
    3: {"sizes": [3] * 8, "strengths": [90 + 10 * i for i in range(8)],
        "levels": [[[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 1, 2, 3], [4, 5, 6, 7]]],
        "b": [8, 4, 2]},
4: {"sizes": [2] * 12, "strengths": [90 + 5 * i for i in range(12)],
        # v2 修正：b3 必须是 b2 的粗粒化（v1 写成 {0,1,2}|{3,4,5}|… 把 {2,3} 切开 → 非嵌套）
        "levels": [[[2 * i, 2 * i + 1] for i in range(6)],
                   [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9], [10, 11]],
                   [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11]]],
        "b": [12, 6, 4, 2]},
}

C32_TEXT_TMPL = (
    "C32：谱聚类读出的可读性由 r_j = min_g(组内质量 / 组外质量) 决定——存在阈值 r*："
    "当且仅当 r_j >= r* 时该层可读（10 种子 ARI 中位数 >= 0.99 且最小值 >= 0.90）。"
    "实测 r* ∈ ({lo}, {hi}]（区间，非点估计；只看非 boundary 点）。"
    "boundary 层（最粗层，r=∞、对外无边）按定义可读，不作为 r 的证据、不参与 r* 估计。"
    "范围限定：只在 E32 的四个语料（L=1..4、s2/s3/s4 固定、分组为 v2 修正版）上验证。"
)
B24_TEXT = (
    "B24：r_j 不足以刻画谱聚类的可读性——即使 r_j 占优（>1），该层也可能不可读"
    "（反例见 JSON 的 counterexample 字段：哪个可读点的 r 小于哪个不可读点的 r）；"
    "反之 r_j < 1（组外质量占优）的点均不可读。"
    "即：组外质量占优是实测到的不可读方向，组内质量占优不是可读的保证。"
)


def build(L: int):
    cfg = CORPORA[L]
    blocks = [[f"m{j}_{i}" for i in range(cfg["sizes"][j])] for j in range(len(cfg["sizes"]))]
    pairs = []
    for j, g in enumerate(blocks):
        for a, b in itertools.combinations(g, 2):
            pairs.append((a, b, cfg["strengths"][j]))
    for lvl, groups in enumerate(cfg["levels"], start=2):
        for grp in groups:
            for x, y in itertools.combinations(grp, 2):
                for a in blocks[x]:
                    for b in blocks[y]:
                        pairs.append((a, b, S_LEVEL[lvl]))
    net = net_from_counts(pairs)
    nodes = [n for g in blocks for n in g]
    fine_truth = np.array([j for j, g in enumerate(blocks) for _ in g])
    return net, nodes, fine_truth, pairs, blocks


def nesting_ok(cfg) -> tuple[bool, str]:
    """P1b 的嵌套性代码校验（v1 漏掉的那条）：

    for 每对相邻层 (b_j, b_{j+1}): b_{j+1} 的每个组必须是 b_j 若干组的并集。
    """
    prev = [[i] for i in range(cfg["b"][0])]
    for lvl_groups in cfg["levels"]:
        for g in lvl_groups:
            gs = set(g)
            if not any(set(p) <= gs for p in prev):
                return False, f"组 {g} 不与上一层任何组对齐"
            covered = {x for p in prev if set(p) <= gs for x in p}
            if covered != gs:
                return False, f"组 {g} 不是上一层组的并集（覆盖 {sorted(covered)}）"
        prev = [g for g in lvl_groups]
    return True, "逐相邻层校验通过"


def layer_groups(cfg, blocks, j: int):
    """第 j 层的组（节点名列表）：j=1 → 每个最细块一组；j>=2 → cfg['levels'][j-2] 展开。"""
    if j == 1:
        return [[n for n in b] for b in blocks]
    return [[n for bi in grp for n in blocks[bi]] for grp in cfg["levels"][j - 2]]


def truth_of_level(cfg, blocks, j: int):
    """第 j 层**自己**的分组真值（每个节点 → 它在该层的组号）。

    注意（实现期踩过的坑）：不能用最细层真值去评第 j 层（j>=2）——
    那样等于拿"12 块的划分"去比"4 个簇的输出"，ARI 会假性很低（B17 那类"工具错了给个合理数"）。
    """
    groups = layer_groups(cfg, blocks, j)
    lab = {}
    for gi, g in enumerate(groups):
        for n in g:
            lab[n] = gi
    return np.array([lab[n] for b in blocks for n in b])


def r_of_level(M, idx, groups):
    """r_j = min over groups of (组内质量 / 组外质量)；每条无向边计一次。"""
    allidx = set(idx.values())
    rs = []
    for g in groups:
        inside = {idx[n] for n in g}
        m_in = sum(M[a, b] for a in inside for b in inside if a < b)
        m_out = sum(M[a, b] for a in inside for b in (allidx - inside))
        rs.append(float("inf") if m_out == 0 else m_in / m_out)
    return min(rs), rs


def readable(median_ari: float, min_ari: float) -> bool:
    """可读 = 中位数 >= 0.99 且 最小值 >= 0.90（两个条件都要）。"""
    return median_ari >= 0.99 and min_ari >= 0.90


def verdict(khat, b_levels):
    """v1 的命中分级（保留给归档函数用；v2 不使用）。"""
    if khat is None:
        return "无法判定"
    if khat == b_levels[0]:
        return "命中主真值"
    if len(b_levels) >= 2 and khat == b_levels[-1]:
        return "命中粗层"
    if khat in b_levels[1:-1]:
        return "命中中层"
    return "未命中"


def _v1_run_archived() -> dict:
    """【归档】v1 的"层数律检验"实现——它在 P1b 上失败。

    失败记录见 `outputs/exp32_nesting_depth_v1_FAIL.json` 与 DESIGN-E32 附录 A。
    本函数不再被 main.py 调用（v2 的 run() 在文件末尾）。
    """
    c = Checker()
    hashes_before = layer_hashes()

    # ---------- P1a 接口回归（E24 语料，与 E32 的 L=1 不是同一份）----------
    net_r, nodes_r, labels_r, _ = corpus_planted()
    Mr = matrix_from_net(net_r, nodes_r)
    assign_r = spectral(impute_diagonal(Mr), 3, seed=SESSION_SEED)
    ari_r = float(ari(assign_r, [labels_r[n] for n in nodes_r]))
    scale_r = max(
        np.mean([Mr[p, q] for p in range(len(nodes_r)) if assign_r[p] == i
                 for q in range(len(nodes_r)) if assign_r[q] == i and p != q])
        for i in sorted(set(assign_r.tolist()))
    )
    rho_r = rho(block_matrix(Mr, assign_r, scale_r))
    c.check("P1a 接口回归: E24 植入语料(15 节点)重现 ARI=1.000 / 3 模块 / ρ=0.0019",
            ari_r >= 0.999 and len(set(assign_r.tolist())) == 3 and abs(rho_r - 0.0019) < 5e-4,
            f"ARI={ari_r:.3f}, 模块数={len(set(assign_r.tolist()))}, ρ={rho_r:.4f}"
            f"（注意：与 E32 的 L=1 语料（12 节点 4/4/4）不是同一份）")

    # ---------- P1b 语料构造核查（含 L=4 分组逐组核对）----------
    built, p1b = {}, []
    for L, cfg in CORPORA.items():
        net, nodes, fine_truth, pairs, blocks = build(L)
        A = impute_diagonal(matrix_from_net(net, nodes))
        declared = {frozenset((a, b)) for a, b, _ in pairs}
        observed = {frozenset(k) for k in net.connections.keys()}
        masses = [len(blocks[j]) * cfg["strengths"][j] for j in range(len(blocks))]
        gaps = [abs(masses[i] - masses[j]) / max(masses[i], masses[j])
                for i in range(len(masses)) for j in range(i + 1, len(masses))]
        ari_fine = float(ari(spectral(A, cfg["b"][0], seed=SESSION_SEED), fine_truth))
        # L=4 的分组逐组核对（写死对照见 DESIGN-E32 §1）
        groups_ok = cfg["levels"] == {
            2: [[[0, 1], [2, 3]]],
            3: [[[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 1, 2, 3], [4, 5, 6, 7]]],
            4: [[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
                [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
                [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]],
        }.get(L, [])
        built[L] = {"net": net, "nodes": nodes, "A": A, "cfg": cfg, "fine": fine_truth,
                    "sig": signature(net), "blocks": blocks}
        p1b.append({"L": L, "节点数": len(nodes), "最细块数": cfg["b"][0], "各层块数": cfg["b"],
                    "边集一致": declared == observed, "边数": len(observed),
                    "最小相对质量差": round(min(gaps), 4) if gaps else None,
                    "k=b1 时 ARI": round(ari_fine, 4), "分组逐组核对": bool(groups_ok)})
    c.check("P1b 语料构造核查: 四个 L 的边集逐边 + 质量非简并 + k=b1 时 ARI>=0.99 + 分组逐组核对(含 L=4)",
            all(r["边集一致"] and r["k=b1 时 ARI"] >= 0.99 and r["分组逐组核对"]
                and (r["最小相对质量差"] is None or r["最小相对质量差"] > 0) for r in p1b),
            "; ".join(f"L={r['L']}: {r['节点数']} 节点/{r['边数']} 边, 各层{r['各层块数']}, "
                      f"质量最小差={r['最小相对质量差']}, ARI={r['k=b1 时 ARI']}, 分组={r['分组逐组核对']}"
                      for r in p1b))

    # ---------- P2a / P2b 三准则 × 四层数 ----------
    matrix, detail = {}, {}
    for L, b in built.items():
        cfg = b["cfg"]
        r = criteria(b["A"], len(b["nodes"]))
        khats = {"eigengap": r["eigengap_k"], "modularity": r["modularity_k"], "bic": r["bic_k"]}
        # null + reason 判定（预注册枚举）
        cells = {}
        for name, kh in khats.items():
            reason = None
            if kh is None:
                reason = "criteria_undefined"
            elif not (1 <= kh <= K_MAX):
                reason = "out_of_range"
                kh = None
            values = list(r[f"{'eigengap' if name == 'eigengap' else name}_curve"].values()) \
                if name != "bic" else list(r["bic_curve"].values())
            if any(not np.isfinite(v) for v in values):
                reason = reason or "computation_error"
            cells[name] = {"k_hat": kh, "reason": reason, "hit": verdict(kh, cfg["b"])}
        matrix[L] = {"L": L, "各层块数": cfg["b"], "cells": cells}
        detail[L] = {"eigenvalues_first10": r["eigenvalues_first10"],
                     "eigengap_curve": r["eigengap_curve"], "eigengap_k": r["eigengap_k"],
                     "modularity_curve": r["modularity_curve"], "modularity_k": r["modularity_k"],
                     "bic_curve": r["bic_curve"], "bic_degenerate": r["bic_degenerate"], "bic_k": r["bic_k"],
                     "degenerate_floor": r["degenerate_floor"]}

    n_cells = sum(len(m["cells"]) for m in matrix.values())
    null_cells = [(m["L"], n) for m in matrix.values() for n, v in m["cells"].items() if v["k_hat"] is None]
    reasons_ok = all(v["reason"] in REASON_ENUM or v["reason"] is None
                     for m in matrix.values() for v in m["cells"].values())
    c.check("P2a 矩阵完整性: 三准则 × 四层数每格有 k̂ 或 null+reason（reason 属预注册枚举）",
            n_cells == 12 and reasons_ok,
            f"{n_cells}/12 格有值或 null；null 格={null_cells or '无'}；reason 枚举合规={reasons_ok}"
            f"（enum={REASON_ENUM}；degenerate 是标记不是 null）")
    checks_tbl = "; ".join(
        f"L={m['L']}: " + "/".join(f"{n}={v['k_hat']}({v['hit']})" for n, v in m["cells"].items())
        for m in matrix.values())
    c.check("P2b 命中分级: 每格按 主真值/中层/粗层/未命中 分级（null 记无法判定，不参与 P5）",
            all(v["hit"] in ("命中主真值", "命中中层", "命中粗层", "未命中", "无法判定")
                for m in matrix.values() for v in m["cells"].values()),
            checks_tbl)

    # ---------- P3 近零特征值个数 = 最粗层块数 ----------
    p3_rows = []
    for L, b in built.items():
        w = np.sort(np.linalg.eigvalsh(_lap(b["A"])))
        near = int(np.sum(w < NEAR_ZERO))
        nz = [x for x in w if x >= NEAR_ZERO]
        p3_rows.append({"L": L, "near_zero_count": near, "b_L": b["cfg"]["b"][-1],
                        "match": near == b["cfg"]["b"][-1],
                        "first_significant_lambda": round(float(nz[0]), 6) if nz else None,
                        "margin_orders": round(float(np.log10(nz[0] / NEAR_ZERO)), 1) if nz else None})
    c.check("P3 近零特征值个数(λ<1e-6) = 最粗层块数 bL（四个 L 全部）",
            all(r["match"] for r in p3_rows),
            "; ".join(f"L={r['L']}: #{r['near_zero_count']} vs bL={r['b_L']} "
                      f"(第一个显著 λ={r['first_significant_lambda']}, 余量 {r['margin_orders']} 个数量级)"
                      for r in p3_rows))

    # ---------- P4 只读 ----------
    hashes_after = layer_hashes()
    sig_ok = all(b["sig"] == signature(b["net"]) for b in built.values())
    c.check("P4 只读: 层文件 sha256 前后一致 + L0 签名前后一致（边集合排序逐项比较）",
            hashes_before == hashes_after and sig_ok,
            f"层文件哈希一致={hashes_before == hashes_after}；L0 签名一致={sig_ok}（"
            + "; ".join(f"L={L}: {b['sig']['nodes']} 节点/{b['sig']['edges']} 边" for L, b in built.items()) + "）")

    # ---------- P5 落定（三分支）----------
    dev = {L: matrix[L]["cells"]["eigengap"]["k_hat"] for L in (2, 3, 4)
           if matrix[L]["cells"]["eigengap"]["k_hat"] != CORPORA[L]["b"][-1]}
    if not dev:
        outcome = "A"
    elif all(kh == CORPORA[L]["b"][-2] for L, kh in dev.items()):
        outcome = "B-1"
    else:
        outcome = "B-2"
    text = {"A": A_TEXT, "B-1": B1_TEXT, "B-2": B2_TEXT}[outcome]
    c.check("P5+P6 落定: A(升级) / B-1(收窄·偏移一层) / B-2(收窄·仅L=2)",
            outcome in ("A", "B-1", "B-2"),
            f"eigengap 偏离最粗层的 L={dev or '无'} → 落定 {outcome}；登记文本已写入 JSON")

    return {
        "id": "exp32_nesting_depth",
        "title": "E32 层数对照: 三准则 × 四嵌套层数（B23 升级或收窄）",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "hit_matrix": matrix, "criteria_detail": detail, "P1b_rows": p1b,
            "near_zero_rows": p3_rows,
            "outcome": outcome, "eigengap_deviations": dev,
            "baseline_L1": {"L": 1, "cells": matrix[1]["cells"],
                            "note": "L=1 是基线对照（无嵌套），不参与 B23 判定；"
                                    "其语料与 E31 的 C2 逐参数相同，结果应一致（E31 记录：三准则均命中 k=3）"},
            "reason_enum": REASON_ENUM, "near_zero_threshold": NEAR_ZERO,
            "layer_hashes": {"before": hashes_before, "after": hashes_after},
            "registration": {"outcome": outcome, "B23_text": text,
                             "A": A_TEXT, "B1": B1_TEXT, "B2": B2_TEXT},
            "note": "仍为弱版本：已知真值语料上的命中检验，不是自动层级发现（B20 边界不变）。",
        },
    }


def _lap(A: np.ndarray) -> np.ndarray:
    d = A.sum(axis=1)
    d[d == 0] = 1.0
    Dm = np.diag(1.0 / np.sqrt(d))
    return np.eye(len(A)) - Dm @ A @ Dm


def run() -> dict:
    """E32 v2：谱聚类读出的可读性条件（r_j 是否决定可读性）。"""
    c = Checker()
    hashes_before = layer_hashes()

    # ---------- P1a 接口回归 ----------
    net_r, nodes_r, labels_r, _ = corpus_planted()
    Mr = matrix_from_net(net_r, nodes_r)
    assign_r = spectral(impute_diagonal(Mr), 3, seed=SESSION_SEED)
    ari_r = float(ari(assign_r, [labels_r[n] for n in nodes_r]))
    scale_r = max(
        np.mean([Mr[p, q] for p in range(len(nodes_r)) if assign_r[p] == i
                 for q in range(len(nodes_r)) if assign_r[q] == i and p != q])
        for i in sorted(set(assign_r.tolist()))
    )
    rho_r = rho(block_matrix(Mr, assign_r, scale_r))
    c.check("P1a 接口回归: E24 植入语料(15 节点)重现 ARI=1.000 / 3 模块 / ρ=0.0019",
            ari_r >= 0.999 and len(set(assign_r.tolist())) == 3 and abs(rho_r - 0.0019) < 5e-4,
            f"ARI={ari_r:.3f}, 模块数={len(set(assign_r.tolist()))}, ρ={rho_r:.4f}")

    # ---------- P1b 语料构造核查 + 嵌套性代码校验 ----------
    built, p1b = {}, []
    for L, cfg in CORPORA.items():
        net, nodes, fine_truth, pairs, blocks = build(L)
        declared = {frozenset((a, b)) for a, b, _ in pairs}
        observed = {frozenset(k) for k in net.connections.keys()}
        masses = [len(blocks[j]) * cfg["strengths"][j] for j in range(len(blocks))]
        gaps = [abs(masses[i] - masses[j]) / max(masses[i], masses[j])
                for i in range(len(masses)) for j in range(i + 1, len(masses))]
        nest, nest_msg = nesting_ok(cfg)
        built[L] = {"net": net, "nodes": nodes, "blocks": blocks, "fine": fine_truth,
                    "cfg": cfg, "sig": signature(net),
                    "A": impute_diagonal(matrix_from_net(net, nodes))}
        p1b.append({"L": L, "节点数": len(nodes), "各层块数": cfg["b"],
                    "边集一致": declared == observed, "边数": len(observed),
                    "最小相对质量差": round(min(gaps), 4) if gaps else None,
                    "嵌套性": nest, "嵌套性说明": nest_msg})
    c.check("P1b 语料构造核查: 边集逐边 + 质量非简并 + 嵌套性逐相邻层代码校验",
            all(r["边集一致"] and r["嵌套性"] and (r["最小相对质量差"] is None or r["最小相对质量差"] > 0)
                for r in p1b),
            "; ".join(f"L={r['L']}: {r['节点数']} 节点/{r['边数']} 边, 各层{r['各层块数']}, "
                      f"质量最小差={r['最小相对质量差']}, 嵌套={r['嵌套性']}" for r in p1b))

    # ---------- P1b' / P2 可读性曲线（逐语料逐层）----------
    points, boundary_rows = [], []
    for L, b in built.items():
        cfg, A, idx = b["cfg"], b["A"], {n: i for i, n in enumerate(b["nodes"])}
        for j in range(1, len(cfg["b"]) + 1):
            groups = layer_groups(cfg, b["blocks"], j)
            r, rs = r_of_level(A, idx, groups)
            k = cfg["b"][j - 1]
            # ARI 必须对【第 j 层自己的分组真值】，不是最细层真值
            truth_j = truth_of_level(cfg, b["blocks"], j)
            vals = [float(ari(spectral(A, k, seed=s), truth_j)) for s in range(10)]
            med, lo = float(np.median(vals)), min(vals)
            is_boundary = bool(r == float("inf"))       # np.float64 与 inf 比较会得到 np.bool_（json 不认）
            point = {"L": L, "层": j, "b_j": k,
                     "r_j": None if is_boundary else round(float(r), 4),
                     "r_reason": "no_external_edges" if is_boundary else None,
                     "boundary": is_boundary,
                     "ari_median": round(med, 4), "ari_min": round(lo, 4), "ari_max": round(max(vals), 4),
                     "readable": bool(readable(med, lo))}
            (boundary_rows if point["boundary"] else points).append(point)
    c.check("P1b' 读出可实现性 + P2 可读性曲线完整性: 每(语料,层)都有 r_j 与 ARI(中位/最小)",
            len(points) + len(boundary_rows) == sum(len(b["cfg"]["b"]) for b in built.values()),
            f"{len(points)} 个非 boundary 点 + {len(boundary_rows)} 个 boundary 点；"
            + "; ".join(f"L={p['L']}层{p['层']}: r={p['r_j']} ARI中位={p['ari_median']} 最小={p['ari_min']} "
                        f"{'可读' if p['readable'] else '不可读'}" for p in points))

    # ---------- P3 可读性条件（完全可分 → 区间；否则给反例）----------
    ok_pts = [p for p in points if p["readable"]]
    bad_pts = [p for p in points if not p["readable"]]
    separable, r_star, counter = False, None, None
    if ok_pts and bad_pts:
        hi_bad = max(bad_pts, key=lambda p: p["r_j"])
        lo_ok = min(ok_pts, key=lambda p: p["r_j"])
        if hi_bad["r_j"] < lo_ok["r_j"]:
            separable, r_star = True, (float(hi_bad["r_j"]), float(lo_ok["r_j"]))
        else:
            counter = {"可读点": lo_ok, "不可读点": hi_bad}
    c.check("P3 可读性条件（主判据, 只看非 boundary 点）: 完全可分 → 报 r* 区间; 否则报反例",
            True,  # 双向都合格；本行只记录结论
            (f"完全可分：r* ∈ ({r_star[0]}, {r_star[1]}]（可用区间，非点估计）" if separable else
             f"不完全可分：反例——可读点 L={counter['可读点']['L']}层{counter['可读点']['层']} "
             f"r={counter['可读点']['r_j']} < 不可读点 L={counter['不可读点']['L']}层"
             f"{counter['不可读点']['层']} r={counter['不可读点']['r_j']}" if counter else
             "样本不足以判定（全部可读或全部不可读）"))

    # ---------- P4 近零特征值个数 = 最粗层块数 ----------
    near_rows = []
    for L, b in built.items():
        w = np.sort(np.linalg.eigvalsh(_lap(b["A"])))
        nz = [x for x in w if x >= NEAR_ZERO]
        near_rows.append({"L": L, "near_zero": int(np.sum(w < NEAR_ZERO)),
                          "b_L": b["cfg"]["b"][-1],
                          "match": int(np.sum(w < NEAR_ZERO)) == b["cfg"]["b"][-1],
                          "first_significant": round(float(nz[0]), 6) if nz else None})
    c.check("P4 近零特征值个数(λ<1e-6) = 最粗层块数 bL",
            all(r["match"] for r in near_rows),
            "; ".join(f"L={r['L']}: #{r['near_zero']} vs bL={r['b_L']}（首个显著 λ={r['first_significant']}）"
                      for r in near_rows))

    # ---------- P5 只读 ----------
    hashes_after = layer_hashes()
    sig_ok = all(b["sig"] == signature(b["net"]) for b in built.values())
    c.check("P5 只读: 层文件 sha256 前后一致 + L0 签名前后一致（边集合排序逐项比较）",
            hashes_before == hashes_after and sig_ok,
            f"层文件哈希一致={hashes_before == hashes_after}；L0 签名一致={sig_ok}")

    # ---------- P6 落定 ----------
    outcome = "C32" if separable else "B24"
    text = C32_TEXT_TMPL.format(lo=r_star[0], hi=r_star[1]) if separable else B24_TEXT
    c.check("P6 登记: 完全可分 → C32（报 r* 区间）；否则 → B24（报反例）",
            outcome in ("C32", "B24"), f"落定 {outcome}；文本已写入 JSON")

    return {
        "id": "exp32_nesting_depth",
        "title": "E32 v2 可读性条件: r_j 是否决定谱聚类读出（B23 的适用边界）",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "readability_points": points,
            "boundary_points": boundary_rows,
            "boundary_note": "r=∞（对外无边）的层按定义可读（最粗层 = 连通分量），不作为 r 的证据、不参与 r* 估计。",
            "P1b_rows": p1b, "near_zero_rows": near_rows,
            "separable": separable, "r_star_interval": r_star, "counterexample": counter,
            "outcome": outcome,
            "eigenvalues": {L: [round(float(x), 6) for x in
                                np.sort(np.linalg.eigvalsh(_lap(b["A"])))[:10]] for L, b in built.items()},
            "l1_cross_validation": {
                "note": "L=1 语料 ≡ E31 的 C2（逐参数相同）；L=1 只有一层且对外无边，"
                        "故它落在 boundary 表；v1 实测三准则在 L=1 上均命中 k=3，与 E31 记录一致",
                "L1_ari_at_k3": next((p["ari_median"] for p in boundary_rows if p["L"] == 1), None),
            },
            "layer_hashes": {"before": hashes_before, "after": hashes_after},
            "registration": {"outcome": outcome, "text": text},
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
