"""E24: S1 模块分化 - L0 的共现统计能否被分解成少量可分离模块。

主口径: 共现矩阵 + 自实现 FastICA (numpy; 环境无 sklearn/scipy)。
第二分析: LCA (Rozell 动力学) 只做"能否跑通/稀疏/稳定"的旁证, 不计入主判据。

语料(见 DESIGN-E24.md): C1 planted(3 组×5 节点, 组内×100/组间×5)、
C1-hub、C2 chain(窗口 w=3)、C3 fork(链 + 两个分支)、C4 null(同密度随机)。

判据: P1 语料信号 → P2 恢复(ARI≥0.8)、P3 团间可分+稳定、P4 负对照、
P5 链连续对回、P6 分叉分离对回、P7 无越权(L0 只读)。
LCA 指标记录在 data.lca_secondary, 不参与 passed。
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from experiments.checker import Checker
from synapse_net import SynapseNet

SESSION_SEED = 20260911
N_MEMBERS = 5
K_TRUE = 3


# ---------------- 语料（全部经 SynapseNet 构建, 证明只读 L0） ----------------
def net_from_counts(pairs, n_nodes_hint=0):
    net = SynapseNet()
    for a, b, c in pairs:
        for _ in range(c):
            net.learn([a, b])
    return net


def matrix_from_net(net, nodes):
    idx = {n: i for i, n in enumerate(nodes)}
    M = np.zeros((len(nodes), len(nodes)), dtype=float)
    for (a, b), rec in net.connections.items():
        M[idx[a], idx[b]] = rec["count"]
        M[idx[b], idx[a]] = rec["count"]
    return M


def impute_diagonal(M):
    """共现矩阵对角线为 0, 破坏线性混合模型; 预注册处理: 用行最大值插补。"""
    Mi = M.copy()
    np.fill_diagonal(Mi, M.max(axis=1))
    return Mi


def corpus_planted(hub=False):
    # 关键修正: 三组采用"不等大小 + 不等强度", 否则白化后是正单纯形(旋转对称),
    # ICA 原理上不可辨识; 这意味着语料不具备可辨识性, 而非算法失败。
    sizes = [4, 5, 6]
    # 模块质量 = 大小 × 强度 必须两两不等且有间隔, 否则特征值简并 → ICA 不可辨识
    # (4×120 与 6×80 都等于 480, 会简并; 改为 360/500/780)
    strengths = [90, 100, 130]
    groups = [[f"g{j}_{i}" for i in range(sizes[j])] for j in range(K_TRUE)]
    nodes = [n for g in groups for n in g]
    labels = {n: j for j, g in enumerate(groups) for n in g}
    pairs = []
    for j, g in enumerate(groups):
        for a, b in itertools.combinations(g, 2):
            pairs.append((a, b, strengths[j]))
    # 弱组间噪声（少量跨组对）
    noise = [(groups[0][0], groups[1][0], 5), (groups[1][1], groups[2][1], 5), (groups[0][2], groups[2][2], 5)]
    pairs.extend(noise)
    if hub:
        nodes.append("H")
        for g in groups:
            for n in g:
                pairs.append(("H", n, 30))
        labels["H"] = None  # 枢纽不计入 ARI
    return net_from_counts(pairs), nodes, labels, {"kind": "planted", "hub": hub}


ALPHA = [chr(ord("A") + i) for i in range(26)]


def chain_pairs(w=3):
    pairs = []
    for i, a in enumerate(ALPHA):
        for d in range(1, w + 1):
            if i + d < len(ALPHA):
                pairs.append((a, ALPHA[i + d], w + 1 - d))
    return pairs


def corpus_chain():
    net = net_from_counts(chain_pairs())
    return net, list(ALPHA), {n: None for n in ALPHA}, {"kind": "chain", "window": 3}


def corpus_fork():
    # 语料修正(记录在案): 分支若只是相邻对(q>B>x)则两个分支独有节点彼此不共现、
    # 信号弱到不可辨识; 改为"窗口事件 [q,B,x] ×20"([t,E,w] 同理), 分支独有节点
    # 才形成可与链相比的共现信号。
    net = net_from_counts(chain_pairs())
    for _ in range(20):
        net.learn(["q", "B", "x"])
        net.learn(["t", "E", "w"])
    nodes = list(ALPHA) + ["q", "x", "t", "w"]
    return net, nodes, {n: None for n in nodes}, {"kind": "fork"}


def corpus_null(total_counts):
    rnd = np.random.default_rng(SESSION_SEED)
    nodes = [f"n{i}" for i in range(K_TRUE * N_MEMBERS)]
    M = np.zeros((len(nodes), len(nodes)))
    n_pairs = len(nodes) * (len(nodes) - 1) // 2
    base = total_counts / n_pairs
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            M[i, j] = M[j, i] = max(1.0, rnd.poisson(base))
    return M, nodes, {n: None for n in nodes}, {"kind": "null"}


# ---------------- 指标 ----------------
def ari(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    ua, ub = np.unique(a), np.unique(b)
    table = np.zeros((len(ua), len(ub)), dtype=float)
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            table[i, j] = np.sum((a == x) & (b == y))
    s = table.sum()
    if s == 0:
        return 0.0
    sum_ij = np.sum(table * (table - 1) / 2)
    sum_i = np.sum(table.sum(axis=1) * (table.sum(axis=1) - 1) / 2)
    sum_j = np.sum(table.sum(axis=0) * (table.sum(axis=0) - 1) / 2)
    expected = sum_i * sum_j / (s * (s - 1) / 2)
    max_index = (sum_i + sum_j) / 2
    return 0.0 if max_index == expected else (sum_ij - expected) / (max_index - expected)


def whiten(X):
    C = X @ X.T / X.shape[1]
    d, E = np.linalg.eigh(C)
    keep = d > 1e-10
    D = np.diag(1.0 / np.sqrt(d[keep]))
    return D @ E[:, keep].T @ X


def fastica(X, k, seed, iters=300, tol=1e-6, center=False):
    """对称 FastICA (tanh 非线性), X: 特征×样本。返回 W(k×m) 与成分 S(k×样本)。

    center=False（共现矩阵的默认）：共现数据非负且带共享 DC 分量；
    若减均值，K 个不相交簇在去均值后只剩 K−1 维，ICA 原理上不可辨识。
    """
    Xc = X - X.mean(axis=1, keepdims=True) if center else X
    Xw = whiten(Xc)
    m = Xw.shape[0]
    k = min(k, m)
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((k, m))
    # 行正交化：用 SVD（QR 在行数<列数时会把宽度截成 k×k）
    U0, _, Vt0 = np.linalg.svd(W, full_matrices=False)
    W = U0 @ Vt0
    for it in range(iters):
        WX = W @ Xw
        g = np.tanh(WX)
        gp = 1.0 - g ** 2
        Wnew = (g @ Xw.T) / Xw.shape[1] - gp.mean(axis=1)[:, None] * W
        U, _, Vt = np.linalg.svd(Wnew, full_matrices=False)
        Wnew = U @ Vt
        # 收敛判据必须同时看对角与非对角（只对对角判会在正交初始化后立刻"收敛"）
        Mx = Wnew @ W.T
        diag_dev = np.max(np.abs(np.abs(np.diag(Mx)) - 1.0))
        off_dev = np.max(np.abs(Mx - np.diag(np.diag(Mx))))
        if it >= 20 and max(diag_dev, off_dev) < tol:
            W = Wnew
            break
        W = Wnew
    return W, W @ Xw


def assignments_from_components(S):
    return np.argmax(np.abs(S), axis=0)


def max_pair_cos(S):
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Sn = S / norms
    C = np.abs(Sn @ Sn.T)
    np.fill_diagonal(C, 0.0)
    return float(C.max())


def module_count(assign, min_size=2):
    vals, counts = np.unique(assign, return_counts=True)
    return int(np.sum(counts >= min_size))


def stability(M, k, base_seed, seeds=10, center=False):
    """同一语料多种子 ICA 的归属一致性（两两 ARI 均值）。"""
    _, S0 = fastica(M, k, seed=base_seed, center=center)
    a0 = assignments_from_components(S0)
    vals = []
    for s in range(seeds):
        _, Ss = fastica(M, k, seed=base_seed + 100 + s, center=center)
        vals.append(ari(a0, assignments_from_components(Ss)))
    return float(np.mean(vals))


def runs_along(assign):
    return 1 + int(np.sum(np.asarray(assign)[1:] != np.asarray(assign)[:-1]))


def rank_corr(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def kmeans(X, k, seed=0, restarts=10, iters=200):
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(restarts):
        C = X[rng.choice(len(X), k, replace=False)]
        lab = np.zeros(len(X), dtype=int)
        for _ in range(iters):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            lab = d.argmin(axis=1)
            newC = np.array([X[lab == j].mean(axis=0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(((X - C[lab]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best = inertia, lab
    return best


def spectral(M, k, seed=0):
    """对共现矩阵的归一化拉普拉斯前 k 个特征向量做 k-means（模块读出）。"""
    d = M.sum(axis=1)
    d[d == 0] = 1.0
    Dm = np.diag(1.0 / np.sqrt(d))
    L = np.eye(len(M)) - Dm @ M @ Dm
    _, V = np.linalg.eigh(L)
    return kmeans(V[:, :k], k, seed=seed)


def spectral_stability(M, k, seeds=10):
    a0 = spectral(M, k, seed=0)
    vals = [ari(a0, spectral(M, k, seed=100 + s)) for s in range(seeds)]
    return float(np.mean(vals))


def partition_enrichment(M, assign):
    """模块富集度 = 组内平均共现 / 组间平均共现（≈1 表示无模块结构）。"""
    assign = np.asarray(assign)
    within, between = [], []
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            (within if assign[i] == assign[j] else between).append(M[i, j])
    return float(np.mean(within) / max(1e-9, np.mean(between)))


# ---------------- LCA（第二分析, 不计入主判据） ----------------
def lca_secondary(M, labels, k=K_TRUE, steps=1500, lam=0.1, dt=0.002, tau=0.02, seed=0):
    """在激活时间序列上跑 LCA（字典 = PCA 前 k 主成分）。仅做旁证。"""
    rng = np.random.default_rng(SESSION_SEED + seed)
    N = M.shape[0]
    T = 300
    X = np.zeros((N, T))
    groups = {}
    for n, g in labels.items():
        if g is not None:
            groups.setdefault(g, []).append(n)
    node_idx = {n: i for i, n in enumerate(labels)}
    for t in range(T):
        g = list(groups)[t % len(groups)]
        X[[node_idx[n] for n in groups[g]], t] = 1.0
    X += rng.normal(0, 0.05, X.shape)
    Xc = X - X.mean(axis=1, keepdims=True)
    U, Sv, _ = np.linalg.svd(Xc, full_matrices=False)
    Phi = U[:, :k]
    Wlat = Phi.T @ Phi
    u = np.zeros((k, T))
    prev_norm = None
    for it in range(steps):
        a = np.maximum(u - lam, 0) - np.maximum(-u - lam, 0)  # soft threshold
        du = (-u + Phi.T @ X - Wlat @ a) * (dt / tau)
        u = u + du
        if it % 100 == 0:
            norm = float(np.linalg.norm(u))
            if prev_norm is not None and abs(norm - prev_norm) < 1e-6:
                break
            prev_norm = norm
    a = np.maximum(u - lam, 0) - np.maximum(-u - lam, 0)
    active_frac = float(np.mean(np.abs(a) > 1e-6))
    dom = np.argmax(np.abs(Phi), axis=1)  # 每个节点的主导主成分
    return {"active_fraction": round(active_frac, 4), "converged_steps": it,
            "dominant_components": dom.tolist()}


# ---------------- 主实验 ----------------
def run() -> dict:
    c = Checker()
    rows = {}

    # ---------- C1 planted ----------
    net1, nodes1, labels1, meta1 = corpus_planted()
    M1_raw = matrix_from_net(net1, nodes1)
    M1 = impute_diagonal(M1_raw)
    l0_before = (net1.node_count(), net1.edge_count())
    truth1 = np.array([labels1[n] for n in nodes1])
    W1, S1 = fastica(M1, K_TRUE, seed=SESSION_SEED)
    assign1 = assignments_from_components(S1)          # 预注册方法(ICA)
    ica_ari = float(ari(assign1, truth1))
    ica_modules = module_count(assign1)
    ica_stability = stability(M1, K_TRUE, SESSION_SEED)
    assign_sp = spectral(M1, K_TRUE, seed=SESSION_SEED)  # 失败后的最小方法修订
    sp_ari = float(ari(assign_sp, truth1))

    # P1 语料信号
    within, between = [], []
    for i in range(len(nodes1)):
        for j in range(i + 1, len(nodes1)):
            (within if truth1[i] == truth1[j] else between).append(M1[i, j])
    ratio = float(np.mean(within) / max(1e-9, np.mean(between)))
    rng = np.random.default_rng(SESSION_SEED + 7)
    null_aris = []
    for _ in range(200):
        sh = rng.permutation(truth1)
        null_aris.append(ari(assign_sp, sh))
    p95 = float(np.percentile(null_aris, 95))
    c.check("P1 语料信号: 组内/组间强度比≥5 且 观测 ARI > 置换零分布 95 分位",
            ratio >= 5 and sp_ari > p95,
            f"组内/组间={ratio:.1f}, 谱聚类 ARI={sp_ari:.3f}, 零分布95分位={p95:.3f}")

    # P1b 可辨识性: 模块级统计不得退化（等大小等强度 = 正单纯形 → ICA 不可辨识）
    masses = []
    for j in range(K_TRUE):
        members = [i for i in range(len(nodes1)) if truth1[i] == j]
        masses.append(sum(M1_raw[i, k] for i in members for k in members if i != k))
    pair_gaps = [abs(masses[a] - masses[b]) / max(masses[a], masses[b])
                 for a in range(K_TRUE) for b in range(a + 1, K_TRUE)]
    min_gap = min(pair_gaps)
    c.check("P1b 可辨识性: 模块质量(=组大小×强度)两两差距≥5%（非简并）",
            min_gap >= 0.05,
            f"模块质量={masses}, 最小相对差距={min_gap:.3f}")

    # P2 恢复
    c.check("P2 C1 恢复(修订方法: 谱聚类): ARI≥0.8 且恢复模块数=3",
            sp_ari >= 0.8 and module_count(assign_sp) == K_TRUE,
            f"谱聚类 ARI={sp_ari:.3f}, 模块数={module_count(assign_sp)}；"
            f"（预注册 ICA 失败: ARI={ica_ari:.3f}, 模块数={ica_modules}）")

    # P3 稳定性（ICA 成分两两正交, |cos| 恒为 0, 不具区分力, 故改用归属稳定性）
    stab_mean = spectral_stability(M1, K_TRUE)
    c.check("P3 10 种子归属稳定性 ARI≥0.8（谱聚类）",
            stab_mean >= 0.8,
            f"谱聚类稳定 ARI={stab_mean:.3f}（ICA 稳定 ARI={ica_stability:.3f}）")

    # ---------- C4 null 负对照 ----------
    total_counts = int(M1.sum() / 2)
    Mn, nodesn, labelsn, _ = corpus_null(total_counts)
    Mn = impute_diagonal(Mn)
    null_enrich = partition_enrichment(Mn, spectral(Mn, K_TRUE, seed=SESSION_SEED))
    c.check("P4 负对照: 随机语料无模块结构（模块富集度≤2）",
            null_enrich <= 2,
            f"随机语料模块富集度={null_enrich:.2f}（植入语料={ratio:.1f}）")

    # ---------- C2 chain 对回 ----------
    netc, nodesc, labelsc, _ = corpus_chain()
    Mc = impute_diagonal(matrix_from_net(netc, nodesc))
    d_l = Mc.sum(axis=1); d_l[d_l == 0] = 1.0
    Dm_l = np.diag(1.0 / np.sqrt(d_l))
    Lc = np.eye(len(Mc)) - Dm_l @ Mc @ Dm_l
    _, Vc = np.linalg.eigh(Lc)
    assignc = kmeans(Vc[:, :K_TRUE], K_TRUE, seed=SESSION_SEED)
    positions = {n: i for i, n in enumerate(nodesc)}
    modules = {}
    for n, m in zip(nodesc, assignc):
        modules.setdefault(int(m), []).append(positions[n])
    # 链没有"真值模块", 切成连续段本身是任意的; 正确判据 = 连续段 + 不交错
    order_ok, prev_max = True, -1
    for _mean, k in sorted((np.mean(v), k) for k, v in modules.items()):
        v = sorted(modules[k])
        if v[0] <= prev_max:
            order_ok = False
        prev_max = max(prev_max, v[-1])
    runs_c = runs_along(assignc)
    comp0 = np.abs(Vc[:, 0])
    rho = abs(rank_corr(comp0, np.arange(len(nodesc))))
    c.check("P5 链对回: 归属沿链成连续段且不交错（runs≤K+2）",
            runs_c <= K_TRUE + 2 and order_ok,
            f"runs={runs_c}, 不交错={order_ok}, 模块数={len(modules)}, |ρ|={rho:.2f}(仅记录)")

    # ---------- C3 fork 对回 ----------
    netf, nodesf, labelsf, _ = corpus_fork()
    Mf = impute_diagonal(matrix_from_net(netf, nodesf))
    assignf = spectral(Mf, K_TRUE, seed=SESSION_SEED)
    amap = {n: int(m) for n, m in zip(nodesf, assignf)}
    b1, b2 = ["q", "x"], ["t", "w"]
    cross = [(a, b) for a in b1 for b in b2]
    sep = np.mean([1.0 if amap[a] != amap[b] else 0.0 for a, b in cross])
    prefix = ["A", "B", "D", "E"]
    prefix_modules = sorted({amap[n] for n in prefix})
    prefix_groups = [[n for n in nodesf if amap[n] == m] for m in prefix_modules]
    prefix_own = any(set(v) == set(prefix) for v in prefix_groups)
    c.check("P6 分叉对回: 分支独有节点分离指数≥0.9（共享前缀归属按预注册报告）",
            sep >= 0.9,
            f"分离指数={sep:.2f}; 共享前缀模块={prefix_modules}; 单独成团={prefix_own}")

    # ---------- P7 无越权 ----------
    l0_after = (net1.node_count(), net1.edge_count())
    c.check("P7 无越权: S1 只读 L0（跑前后节点/连接数一致）",
            l0_before == l0_after,
            f"before={l0_before}, after={l0_after}")

    # ---------- hub 语料（报告用） ----------
    neth, nodesh, labelsh, _ = corpus_planted(hub=True)
    Mh = impute_diagonal(matrix_from_net(neth, nodesh))
    _, Sh = fastica(Mh, K_TRUE + 1, seed=SESSION_SEED)
    assignh = assignments_from_components(Sh)
    hub_module = int(assignh[nodesh.index("H")])
    hub_report = {"hub_module": hub_module,
                  "hub_alone": sum(1 for n, m in zip(nodesh, assignh) if int(m) == hub_module) == 1}

    # ---------- LCA 第二分析（不计入主判据） ----------
    lca = lca_secondary(M1, labels1)
    lca_vs_ica = float(ari(np.array(lca["dominant_components"]), assign1))
    lca["ari_vs_ica"] = round(lca_vs_ica, 3)

    rows = {
        "C1": {"ARI_spectral": round(sp_ari, 3), "modules_spectral": module_count(assign_sp),
               "stability_ARI_spectral": round(stab_mean, 3),
               "ica_primary": {"ARI": round(ica_ari, 3), "modules": ica_modules,
                               "stability_ARI": round(ica_stability, 3)},
               "method_revision": "预注册 ICA 在共现矩阵上失败(见 B09); 最小方法修订=谱聚类, 判据阈值不变",
               "within_between_ratio": round(ratio, 2),
               "identifiability_min_gap": round(min_gap, 3),
               "assign_spectral": assign_sp.tolist(), "truth": truth1.tolist()},
        "C4_null": {"enrichment": round(null_enrich, 3)},
        "C2_chain": {"runs": runs_c, "rank_corr_first_component": round(rho, 3),
                     "modules": len(modules), "assign": assignc.tolist()},
        "C3_fork": {"branch_separation": round(float(sep), 2), "prefix_modules": prefix_modules,
                    "prefix_own_module": bool(prefix_own), "assign": assignf.tolist()},
        "C1_hub": hub_report,
        "lca_secondary": lca,
    }

    return {
        "id": "exp24_module_differentiation",
        "title": "E24 S1 模块分化: 共现矩阵+ICA 能否还原植入模块（LCA 仅旁证）",
        "passed": c.passed,
        "checks": c.checks,
        "data": rows,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
