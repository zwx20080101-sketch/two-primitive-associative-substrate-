"""E31: 自动尺度选择（弱版本）—— 准则能否命中植入的 k。

严格按 DESIGN-E31.md。**有 numpy 依赖；不改任何层文件。**

预注册要点:
  - 弱版本：语料是植入的，"正确的 k" 已知 → 检验准则能否命中；
  - 严格命中：k̂ == 主真值；"命中粗层"单列记录（不算命中也不算失败）；
  - 跨语料：至少一个准则在三个语料（主真值 4 / 3 / 5）上全部命中；
  - BIC 的 d_k = k（E24 用归一化拉普拉斯的前 k 个特征向量 ⇒ 喂给 k-means 的就是 k 个坐标；
    该矩阵第一特征向量 ∝ sqrt(度)，携带度信息，不是常数向量）；
  - 过分割保护：k > floor(n/2) 的 BIC 标 degenerate，不参与 argmin；
  - 双向结局：命中 → C31；未跨语料命中 → B22（措辞限定"在当前这三个语料结构上"）。
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
    kmeans,
    matrix_from_net,
    net_from_counts,
    spectral,
)
from experiments.exp25_module_dynamics import block_matrix, rho

K_MAX = 8
KM_SEEDS = list(range(10))
S2 = 40
TOL_TIE = 1e-12
LAYER_FILES = ["synapse_net.py", "order_layer.py", "context_layer.py", "chunk_layer.py"]

# 语料定义（大小 / 强度 / 超模块分组 / 主真值 / 粗层）
CORPORA = {
    "C1": {"sizes": [3, 3, 4, 4], "strengths": [90, 100, 110, 120],
           "supers": [[0, 1], [2, 3]], "k_true": 4, "k_coarse": 2,
           "desc": "4 子模块 + 2 超模块（沿用 E29 语料）"},
    "C2": {"sizes": [4, 4, 4], "strengths": [90, 100, 110],
           "supers": [], "k_true": 3, "k_coarse": 1,
           "desc": "3 子模块单层（无上层耦合）"},
    "C3": {"sizes": [3, 3, 3, 3, 3], "strengths": [90, 100, 110, 120, 130],
           "supers": [[0, 1, 2], [3, 4]], "k_true": 5, "k_coarse": 2,
           "desc": "5 子模块 + 2 超模块"},
}

C31_TEXT = (
    "C31：在三个植入真值互不相同（k=4 / 3 / 5）的受控语料上，<准则名> 给出的 k̂ "
    "与主真值精确相等（3/3）。范围限定：这是在已知真值语料上的「命中检验」，"
    "不是自动层级发现；无真值语料上的稳定性不在本工作范围（见 B20）。"
)
B22_TEXT = (
    "B22：在当前这三个语料结构上（C1 = 4 子模块/2 超模块、C2 = 3 子模块单层、"
    "C3 = 5 子模块/2 超模块），eigengap / modularity / BIC 三个准则均未能跨语料命中主真值。"
    "范围限定：本结论只覆盖这三个语料结构——不写成「当前准则不足以自动定尺度」。"
    "E29 的「k 必须外部给定」（B20）在本实验范围内仍然成立。"
)


def layer_hashes() -> dict:
    root = Path(__file__).resolve().parent.parent
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest()[:16] for n in LAYER_FILES}


def build(cfg: dict):
    n_sub = len(cfg["sizes"])
    sub = [[f"m{j}_{i}" for i in range(cfg["sizes"][j])] for j in range(n_sub)]
    pairs = []
    for j, g in enumerate(sub):
        for a, b in itertools.combinations(g, 2):
            pairs.append((a, b, cfg["strengths"][j]))
    for grp in cfg["supers"]:
        for x, y in itertools.combinations(grp, 2):
            for a in sub[x]:
                for b in sub[y]:
                    pairs.append((a, b, S2))
    net = net_from_counts(pairs)
    nodes = [n for g in sub for n in g]
    sub_truth = np.array([j for j, g in enumerate(sub) for _ in g])
    if cfg["supers"]:
        label = {}
        for i, grp in enumerate(cfg["supers"]):
            for j in grp:
                label[j] = i
        sup_truth = np.array([label[j] for j, g in enumerate(sub) for _ in g])
    else:
        sup_truth = None
    return net, nodes, sub_truth, sup_truth, pairs


def signature(net) -> dict:
    edges = sorted((tuple(sorted(k)), int(v["count"])) for k, v in net.connections.items())
    return {"nodes": net.node_count(), "edges": net.edge_count(), "edge_items": edges}


def laplacian(A: np.ndarray):
    d = A.sum(axis=1)
    d[d == 0] = 1.0
    Dm = np.diag(1.0 / np.sqrt(d))
    L = np.eye(len(A)) - Dm @ A @ Dm
    w, V = np.linalg.eigh(L)
    return w, V


def sse_of(X: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for c in set(labels.tolist()):
        pts = X[labels == c]
        if len(pts):
            total += float(((pts - pts.mean(axis=0)) ** 2).sum())
    return total


def criteria(A: np.ndarray, n: int) -> dict:
    """三个准则的完整扫描曲线（公式见设计 §3）。"""
    w, V = laplacian(A)
    eigengap = {}
    for k in range(1, K_MAX + 1):
        if k < len(w):
            eigengap[k] = float(w[k] - w[k - 1])   # 1-indexed: λ_{k+1} − λ_k
    k_eg = max(eigengap, key=lambda k: (eigengap[k], -k))

    mod = {}
    for k in range(1, K_MAX + 1):
        assign = spectral(A, k, seed=SESSION_SEED)
        mod[k] = float(modularity(A, assign, k))
    k_md = max(mod, key=lambda k: (mod[k], -k))

    degen_floor = n // 2
    bic, degenerate = {}, {}
    for k in range(1, K_MAX + 1):
        if k > min(K_MAX, n):
            continue
        best = min(
            sse_of(V[:, :k], kmeans(V[:, :k], k, seed=s)) for s in KM_SEEDS
        )
        best = max(best, 1e-300)
        bic[k] = float(n * k * np.log(best / (n * k)) + k * (k + 1) * np.log(n))
        degenerate[k] = bool(k > degen_floor)
    valid = {k: v for k, v in bic.items() if not degenerate[k]}
    k_bic = min(valid, key=lambda k: (valid[k], k)) if valid else None

    return {
        "eigenvalues_first10": [float(x) for x in w[:10]],
        "eigengap_curve": eigengap, "eigengap_k": int(k_eg),
        "modularity_curve": mod, "modularity_k": int(k_md),
        "bic_curve": bic, "bic_degenerate": degenerate, "bic_k": int(k_bic) if k_bic else None,
        "degenerate_floor": degen_floor,
    }


def modularity(A: np.ndarray, assign, k: int) -> float:
    m2 = float(A.sum())                 # 2m（A 含 impute 后的对角）
    if m2 <= 0:
        return 0.0
    d = A.sum(axis=1)
    q = 0.0
    for i in range(len(A)):
        for j in range(len(A)):
            if assign[i] == assign[j]:
                q += A[i, j] - d[i] * d[j] / m2
    return q / m2


def run() -> dict:
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
    c.check("P1a 接口回归: E24 植入语料重现 ARI=1.000 / 3 模块 / ρ=0.0019",
            ari_r >= 0.999 and len(set(assign_r.tolist())) == 3 and abs(rho_r - 0.0019) < 5e-4,
            f"ARI={ari_r:.3f}, 模块数={len(set(assign_r.tolist()))}, ρ={rho_r:.4f}")

    # ---------- P1b 语料构造核查（含主真值 4/3/5 设计先验）----------
    built, p1b_rows = {}, []
    for name, cfg in CORPORA.items():
        net, nodes, sub_t, sup_t, pairs = build(cfg)
        A = impute_diagonal(matrix_from_net(net, nodes))
        declared = {frozenset((a, b)) for a, b, _ in pairs}
        observed = {frozenset(k) for k in net.connections.keys()}
        masses = [cfg["sizes"][j] * cfg["strengths"][j] for j in range(len(cfg["sizes"]))]
        gaps = [abs(masses[i] - masses[j]) / max(masses[i], masses[j])
                for i in range(len(masses)) for j in range(i + 1, len(masses))]
        ari_true = float(ari(spectral(A, cfg["k_true"], seed=SESSION_SEED), sub_t))
        built[name] = {"net": net, "nodes": nodes, "A": A, "sub_t": sub_t, "sup_t": sup_t,
                       "cfg": cfg, "sig": signature(net)}
        p1b_rows.append({
            "语料": name, "描述": cfg["desc"], "节点数": len(nodes),
            "边集一致": declared == observed, "声明边": len(declared), "实测边": len(observed),
            "质量": masses, "最小相对质量差": round(min(gaps), 4),
            "主真值": cfg["k_true"], "粗层": cfg["k_coarse"],
            "k=主真值 时 ARI vs 子模块": round(ari_true, 4),
        })
    k_trues = [CORPORA[n]["k_true"] for n in ("C1", "C2", "C3")]
    c.check("P1b 语料构造核查: 边集逐边 + 质量非简并 + k=主真值 ARI>=0.99 + 主真值 4/3/5 互不相同（设计先验）",
            all(r["边集一致"] and r["最小相对质量差"] > 0 and r["k=主真值 时 ARI vs 子模块"] >= 0.99
                for r in p1b_rows) and k_trues == [4, 3, 5],
            "; ".join(f"{r['语料']}: {r['节点数']} 节点, {r['声明边']} 边一致={r['边集一致']}, "
                      f"质量最小差={r['最小相对质量差']}, k_true={r['主真值']}, ARI={r['k=主真值 时 ARI vs 子模块']}"
                      for r in p1b_rows))

    # ---------- 三准则 ----------
    results, matrix = {}, {}
    for name, b in built.items():
        cfg = b["cfg"]
        r = criteria(b["A"], len(b["nodes"]))
        # 命中判定（严格）
        def verdict(khat):
            if khat is None:
                return "未命中"
            if khat == cfg["k_true"]:
                return "命中"
            if cfg["k_coarse"] >= 2 and khat == cfg["k_coarse"]:
                return "命中粗层"
            return "未命中"
        r["hit"] = {"eigengap": verdict(r["eigengap_k"]),
                    "modularity": verdict(r["modularity_k"]),
                    "bic": verdict(r["bic_k"])}
        results[name] = r
        matrix[name] = {"k_true": cfg["k_true"], "k_coarse": cfg["k_coarse"],
                        "eigengap_k": r["eigengap_k"], "eigengap_hit": r["hit"]["eigengap"],
                        "modularity_k": r["modularity_k"], "modularity_hit": r["hit"]["modularity"],
                        "bic_k": r["bic_k"], "bic_hit": r["hit"]["bic"]}

    winners = [cr for cr in ("eigengap", "modularity", "bic")
               if all(results[n]["hit"][cr] == "命中" for n in ("C1", "C2", "C3"))]
    c.check("P2 跨语料命中（主判据）: 至少一个准则在三个语料上都命中主真值",
            len(winners) > 0,
            f"跨语料命中的准则={winners or '无'}；"
            + "; ".join(f"{n}: " + "/".join(f"{cr}={results[n][cr + '_k']}({results[n]['hit'][cr]})"
                                            for cr in ("eigengap", "modularity", "bic"))
                        for n in ("C1", "C2", "C3")))

    # ---------- P4 只读 ----------
    hashes_after = layer_hashes()
    sig_ok = all(b["sig"] == signature(b["net"]) for b in built.values())
    c.check("P4 只读: 层文件 sha256 前后一致 + L0 签名前后一致（边集合排序逐项比较）",
            hashes_before == hashes_after and sig_ok,
            f"层文件哈希一致={hashes_before == hashes_after}；"
            f"L0 签名一致={sig_ok}（"
            + "; ".join(f"{n}: {b['sig']['nodes']} 节点/{b['sig']['edges']} 边" for n, b in built.items()) + "）")

    # ---------- P5 双向结局 + P6 登记 ----------
    outcome = "C" if winners else "B"
    registered = ["C31"] if outcome == "C" else ["B22"]
    c.check("P5+P6 双向结局与登记: 命中 → C31；未跨语料命中 → B22",
            (outcome == "C" and registered == ["C31"]) or (outcome == "B" and registered == ["B22"]),
            f"落定={outcome} → 登记 {registered}"
            + (f"（跨语料命中准则：{winners}）" if winners else "（无准则跨语料命中）"))

    return {
        "id": "exp31_auto_scale",
        "title": "E31 自动尺度选择（弱版本）: 三准则在三个植入语料上的命中检验",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "hit_matrix": matrix,
            "criteria_detail": {n: results[n] for n in results},
            "P1b_rows": p1b_rows,
            "outcome": outcome,
            "cross_corpus_winners": winners,
            "k_max": K_MAX, "km_seeds": KM_SEEDS, "tie_tol": TOL_TIE,
            "layer_hashes": {"before": hashes_before, "after": hashes_after},
            "registration": {"outcome": outcome, "registered": registered,
                             "C31": C31_TEXT, "B22": B22_TEXT,
                             "C31_resolved": C31_TEXT.replace("<准则名>", " / ".join(winners)),
                             "note": "落定 C31 时 <准则名> 已替换为实际跨语料命中的准则；落定 B22 时不适用"},
            "note": "弱版本：在已知真值语料上的命中检验，不是自动层级发现（B20 仍然有效）。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
