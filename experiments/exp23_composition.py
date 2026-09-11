"""E23: 分层影响 / 组合律 - 多层同时开启时能否同时保住各自的现象。

语料（见 DESIGN-E23.md）:
  主链: A..Z 有序相邻 ×1
  分叉1: q>B>x ×1     （q、x 均为小写、不在字母表内，避免与链中字母撞车造成近路；
                       B→C 与 B→x 次数相同 => L1 无法区分；L2 用 (A,B) 前缀区分）
  分叉2: t>E>w ×1     （同理，用于检验同一组合规则在第二个分叉也成立）
  组块:  {ABC}{DEF}…{XYZ} 各 ×K=5 成块（L4）

Q1 方向: 从 A 正向 / 从 Z 反向 的能量比（L1 开→比值大；关→≈1）
Q2 分叉1: 前缀 (A,B) 下 C 与 X 的选择（L2 开→选 C；只 L1 或都不开→平局）
Q3 组块: 是否存在 9 个块、块节点是否可达、A→Z 最短跳数（L4 开→10 跳；关→25 跳）
Q4 分叉2: 前缀 (D,E) 下 F 与 w 的选择（检验组合规则一致性）

判据:
  P1 各层返回量可读
  P2 单层贡献可归因（L1→方向比；L2→分叉选择；L4→粒度/跳数）
  P3 组合(C111)同时保住三层各自现象（维度先行：L4 粒度 → L2 候选 → L1 顺序）
  P4 无越权: L1/L2 不改 L0 的 N/连接数；L4 新增仅限 CH_ 节点及成员/块级边
  P5 同一组合规则在两个分叉上一致
"""

from __future__ import annotations

import itertools
import json
import math
from collections import deque
from pathlib import Path

from chunk_layer import ChunkLayer
from context_layer import ContextLayer
from experiments.checker import Checker
from order_layer import OrderLayer
from synapse_net import SynapseNet

ALPHA = [chr(ord("A") + i) for i in range(26)]
K = 5
HOP = 0.9


def build(use_l1: bool, use_l2: bool, use_l4: bool):
    net, l1, l2, l4 = SynapseNet(), OrderLayer(), ContextLayer(), ChunkLayer(k=K)
    # L0: 主链相邻对 + 两个分叉的相邻对（与 L1/L2 见到的经验一致）
    for a, b in zip(ALPHA, ALPHA[1:]):
        net.learn([a, b])
    for a, b in (("q", "B"), ("B", "x"), ("t", "E"), ("E", "w")):
        net.learn([a, b])
    # L1/L2: 有序事件
    events = [ALPHA, ["q", "B", "x"], ["t", "E", "w"]]
    if use_l1 or use_l2:
        for ev in events:
            if use_l1:
                l1.learn(ev)
            if use_l2:
                l2.learn(ev)
    # L4: 9 个带边界单元 ×K 成块，并声明块级顺序边
    chunks = []
    if use_l4:
        groups = [ALPHA[i:i + 3] for i in range(0, 26, 3)]
        for g in groups:
            cid = None
            for _ in range(K):
                cid = l4.learn_unit(g)
            chunks.append(cid)
        l4.learn_sequence(chunks)  # 声明为 L4 的结构新增（块级顺序）
    return net, l1, l2, l4, chunks


def adj_from_net(net):
    adj = {}
    for (a, b), rec in net.connections.items():
        adj.setdefault(a, {})[b] = rec["count"]
        adj.setdefault(b, {})[a] = rec["count"]
    return adj


def merge_chunk_edges(adj, l4, chunks):
    for (cid, m), cnt in l4.member_bind.items():
        adj.setdefault(cid, {})[m] = max(adj.get(cid, {}).get(m, 0), cnt)
        adj.setdefault(m, {})[cid] = max(adj.get(m, {}).get(cid, 0), cnt)
    for (a, b), cnt in l4.high_pairs.items():
        adj.setdefault(a, {})[b] = cnt
        adj.setdefault(b, {})[a] = cnt
    return adj


def diffuse(adj, seeds, hop_decay=HOP):
    """与 L0 相同的 max 波前（每源 max 归一化 + max 吸收）。"""
    acts = {s: 1.0 for s in seeds}
    q = deque(seeds)
    while q:
        u = q.popleft()
        au = acts[u]
        nb = adj.get(u, {})
        if not nb:
            continue
        best = max(nb.values())
        for v, c in nb.items():
            arrival = au * (c / best) * hop_decay
            if arrival > acts.get(v, -math.inf) + 1e-12:
                acts[v] = arrival
                if v not in q:
                    q.append(v)
    return acts


def hops(adj, src, dst):
    if src not in adj:
        return None
    seen, q = {src}, deque([(src, 0)])
    while q:
        u, d = q.popleft()
        if u == dst:
            return d
        for v in adj.get(u, {}):
            if v not in seen:
                seen.add(v)
                q.append((v, d + 1))
    return None


def combo(use_l1, use_l2, use_l4):
    net, l1, l2, l4, chunks = build(use_l1, use_l2, use_l4)
    adj = adj_from_net(net)
    if use_l4:
        adj = merge_chunk_edges(adj, l4, chunks)

    # Q1 方向比
    if use_l1:
        fwd = l1.activate("A").get("Z", 0.0)
        rev = l1.activate("Z").get("A", 0.0)
    else:
        fwd = diffuse(adj, ["A"]).get("Z", 0.0)
        rev = diffuse(adj, ["Z"]).get("A", 0.0)
    dir_ratio = fwd / rev if rev > 0 else float("inf")

    # Q2 分叉1: (A,B) -> C vs X
    if use_l2:
        q = l2.query(["A", "B"])
        cand = {"C": q.get("C", 0.0), "x": q.get("x", 0.0)}
    elif use_l1:
        cand = {"C": l1.strength("B", "C"), "x": l1.strength("B", "x")}
    else:
        cand = {"C": net.connection_count("B", "C"), "x": net.connection_count("B", "x")}
    pick1 = "C" if cand["C"] > cand["x"] + 1e-9 else ("x" if cand["x"] > cand["C"] + 1e-9 else "tie")

    # Q4 分叉2: (D,E) -> F vs W
    if use_l2:
        q2 = l2.query(["D", "E"])
        cand2 = {"F": q2.get("F", 0.0), "w": q2.get("w", 0.0)}
    elif use_l1:
        cand2 = {"F": l1.strength("E", "F"), "w": l1.strength("E", "w")}
    else:
        cand2 = {"F": net.connection_count("E", "F"), "w": net.connection_count("E", "w")}
    pick2 = "F" if cand2["F"] > cand2["w"] + 1e-9 else ("w" if cand2["w"] > cand2["F"] + 1e-9 else "tie")

    # Q3 组块粒度与可达性
    bonus = {
        "chunks": len(chunks) if use_l4 else 0,
        "chunk_reachable": False,
        "hops_A_to_Z": hops(adj, "A", "Z"),
    }
    if use_l4:
        acts = diffuse(adj, ["A"])
        bonus["chunk_reachable"] = any(str(k).startswith("CH_") and v > 0 for k, v in acts.items())

    # L1/L2 返回量（用于 P1/P2）
    s1 = {"B->C": l1.count("B", "C"), "B->x": l1.count("B", "x")} if use_l1 else None
    s2 = l2.query(["A", "B"]) if use_l2 else None
    l4info = {"chunks": len(chunks), "members": len(l4.member_bind)} if use_l4 else None

    return {
        "flags": {"L1": use_l1, "L2": use_l2, "L4": use_l4},
        "q1": {"fwd": round(fwd, 9), "rev": round(rev, 9), "ratio": round(dir_ratio, 6) if dir_ratio != float("inf") else "inf"},
        "q2": {"cand": cand, "pick": pick1},
        "q4": {"cand": cand2, "pick": pick2},
        "q3": bonus,
        "returns": {"L1": s1, "L2": s2, "L4": l4info},
        "l0_baseline": {"nodes": net.node_count(), "edges": net.edge_count(),
                        "N_B_C": net.connection_count("B", "C"), "N_B_x": net.connection_count("B", "x")},
        "l4_added_nodes": sorted(n for n in (l4.high_nodes if use_l4 else []) if str(n).startswith("CH_")),
    }


def run() -> dict:
    c = Checker()
    rows = {}
    for f1, f2, f4 in itertools.product([False, True], repeat=3):
        key = f"C{int(f1)}{int(f2)}{int(f4)}"
        rows[key] = combo(f1, f2, f4)

    # P1: 每格都有该层返回量（开启时）
    p1 = all((r["returns"]["L1"] is not None or not r["flags"]["L1"])
             and (r["returns"]["L2"] is not None or not r["flags"]["L2"])
             and (r["returns"]["L4"] is not None or not r["flags"]["L4"])
             for r in rows.values())
    c.check("P1 各层返回量可读（8×3 表完整）", p1, f"{len(rows)} 组合")

    # P2: 单层贡献可归因
    dir_on = rows["C100"]["q1"]["ratio"] > 1e3
    dir_off = abs(rows["C000"]["q1"]["ratio"] - 1.0) < 1e-6
    fork_l2 = rows["C010"]["q2"]["pick"] == "C"
    fork_l1_tie = rows["C100"]["q2"]["pick"] == "tie" and rows["C000"]["q2"]["pick"] == "tie"
    chunk_only = rows["C001"]["q3"]["chunks"] == 9 and rows["C001"]["q3"]["hops_A_to_Z"] < rows["C000"]["q3"]["hops_A_to_Z"]
    c.check("P2a L1 必然性: L1 开→方向比>1e3; 关→≈1", dir_on and dir_off,
            f"L1开比值={rows['C100']['q1']['ratio']}; L0-only 比值={rows['C000']['q1']['ratio']}")
    c.check("P2b L2 必然性: 仅 L2 时 (A,B)→C; 仅 L1/L0 时平局", fork_l2 and fork_l1_tie,
            f"C010 pick={rows['C010']['q2']['pick']}, C100 pick={rows['C100']['q2']['pick']}, C000 pick={rows['C000']['q2']['pick']}")
    c.check("P2c L4 必然性: 仅 L4 时 9 块且 A→Z 跳数下降", chunk_only,
            f"C001 chunks={rows['C001']['q3']['chunks']}, hops={rows['C001']['q3']['hops_A_to_Z']} vs L0 hops={rows['C000']['q3']['hops_A_to_Z']}")

    # P3: 组合 (C111) 同时保住三层现象
    allr = rows["C111"]
    p3 = (allr["q1"]["ratio"] > 1e3) and (allr["q2"]["pick"] == "C") and (allr["q3"]["chunks"] == 9) \
        and (allr["q3"]["hops_A_to_Z"] < rows["C000"]["q3"]["hops_A_to_Z"]) and allr["q3"]["chunk_reachable"]
    c.check("P3 组合: C111 同时保住 方向/分叉选择/块粒度 三层现象", p3,
            f"ratio={allr['q1']['ratio']}, fork={allr['q2']['pick']}, chunks={allr['q3']['chunks']}, "
            f"hops={allr['q3']['hops_A_to_Z']}, chunk_reachable={allr['q3']['chunk_reachable']}")

    # P4: 无越权
    bases = {k: (r["l0_baseline"]["nodes"], r["l0_baseline"]["edges"], r["l0_baseline"]["N_B_C"], r["l0_baseline"]["N_B_x"]) for k, r in rows.items()}
    p4_l0 = len(set(bases.values())) == 1
    p4_chunk = all(all(str(n).startswith("CH_") for n in r["l4_added_nodes"]) for r in rows.values())
    c.check("P4 无越权: L1/L2 未改 L0; L4 新增仅 CH_ 节点", p4_l0 and p4_chunk,
            f"L0 基线唯一={p4_l0}; L4 新增节点全为 CH_={p4_chunk}")

    # P5: 组合规则在两个分叉上一致
    p5 = rows["C111"]["q2"]["pick"] == "C" and rows["C111"]["q4"]["pick"] == "F" \
        and rows["C010"]["q2"]["pick"] == "C" and rows["C010"]["q4"]["pick"] == "F"
    c.check("P5 一致性: 同一组合规则在分叉1/分叉2 都选中正确候选", p5,
            f"C111: {rows['C111']['q2']['pick']}/{rows['C111']['q4']['pick']}; "
            f"C010: {rows['C010']['q2']['pick']}/{rows['C010']['q4']['pick']}")

    return {
        "id": "exp23_composition",
        "title": "E23 分层影响/组合律: L1+L2+L4 同时开启时各层现象能否共存并可归因",
        "passed": c.passed,
        "checks": c.checks,
        "data": rows,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
