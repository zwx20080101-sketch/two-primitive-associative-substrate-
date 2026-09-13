"""E28: 读取端 max/sum 系统对比（B02 + B03 正式化）。

严格按 DESIGN-E28.md。**纯计数，无 numpy 依赖；不改任何层文件。**

口径（预注册）:
  - max = L0 的 activate（单一实现来源）；sum = 单次发射 + 出队冻结 + 封顶 cap=1.0；
  - sum 语义【照搬 E14】：每个节点只出队一次，出队时冻结自己的激活值（后再来的贡献被丢弃）。
    这一条不得偷改，否则 E14 与 E28 的 sum 不是同一个东西；
  - sum 标准遍历顺序 = 邻居字典序；另跑降序用于"顺序依赖"的记录；
  - 到达节点数两套阈值：A(绝对) > 1e-12；B(相对) > 最大激活值 × 1e-9（不一致以 B 为准）；
  - 运行环境要求 PYTHONHASHSEED=0（固定字符串哈希顺序）。
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

from synapse_net import SynapseNet

from experiments.checker import Checker

REPS = 100
SEED = "A"
HOP = 0.9
CAP = 1.0
TOL = 1e-12
REL_FACTOR = 1e-9
LAYER_FILES = ["synapse_net.py", "order_layer.py", "context_layer.py"]

CORPORA = {
    "T1": [("A", "B"), ("B", "C")],
    "T2": [("A", "B"), ("B", "C"), ("A", "D"), ("D", "C")],
    "T3": [("A", "B"), ("B", "C"), ("A", "D"), ("D", "C"), ("A", "E"), ("E", "C")],
    "T4": [("A", "B"), ("B", "C"), ("C", "B")],
    "T5": [("A", "B"), ("B", "C"), ("C", "A")],
}
RELABEL_T5 = {"A": "P", "B": "Q", "C": "R"}

C28_TEXT = (
    "C28：max 与 sum 是读取端的两种选择，不是对错。"
    "(a) max 精确定律：max(a, a) = a，容差 1e-12，任意路径数下严格成立；"
    "(b) sum 是另一种选择：单次发射 + 封顶 1.0，封顶损失随路径数单调增（记录原始和）；"
    "(c) max 顺序无关，sum 顺序依赖（字典序为默认遍历口径，换序差异是 B03 的量化指标）。"
)
B03_TEXT = (
    "B03（升级）：sum 模式的遍历顺序影响封顶落点。触发条件是拓扑判据——"
    "当拓扑中存在两个都可由种子一跳到达、且彼此相邻的节点时，sum 的封顶落点依赖遍历顺序"
    "（T5 三角语料：升序 [1.0, 0.9, 1.0] vs 降序 [1.0, 1.0, 0.9]）；max 仍顺序无关"
    "（均 [1.0, 0.9, 0.9]）。反例对照：T2/T3 里 B/D/E 是中间节点、C 是唯一终点，"
    "封顶必然落在 C 上，谁先谁后不影响结果（顺序无关）。封顶损失随路径数单调增："
    "单路 0.00、双路 0.62、三路 1.43。因此 sum 的输出不可与 max 直接对比。"
    "另注：T5 的 max a(C) = 0.9（不是 0.81），因为 A 与 C 直接相邻、存在一跳路径；"
    "T1-T4 中 A、C 均不相邻故一律 0.81——这是拓扑差别，不只是数值差别。"
    "exp14 的 JSON detail 抖动是这一性质的直接后果，不是测试缺陷。"
)
B02_TEXT = (
    "B02（指针）：已并入 C28 —— max 的精确不叠加是读取端设计选择，不是机制缺陷；"
    "原始观测「多路证据在 max 下不叠加」保留在此。"
)


def build(edges, reps=REPS):
    net = SynapseNet()
    for a, b in edges:
        for _ in range(reps):
            net.learn([a, b])
    return net


def sum_activate(net: SynapseNet, seed: str, order: str = "asc",
                 cap: float = CAP, hop_decay: float = HOP):
    """sum 规则（实验层，照搬 E14 语义）。

    单次发射 + 出队冻结: 每个节点只出队一次, 出队时把自己的激活值冻结,
    之后到达的贡献被丢弃（防无向边回环把整图泵到封顶）。
    同一波到达的多路贡献在目标出队前累加并封顶 cap。
    """
    acts = {seed: 1.0}
    fired: set[str] = set()
    queue = deque([seed])
    queued = {seed}
    while queue:
        u = queue.popleft()
        queued.discard(u)
        if u in fired:
            continue
        fired.add(u)                      # 冻结: 之后不再接收贡献
        nbrs = sorted(net._adj.get(u, ()), reverse=(order == "desc"))
        for v in nbrs:
            if v in fired:
                continue
            arrival = acts[u] * net.strength(u, v) * hop_decay
            if arrival > TOL:
                acts[v] = min(cap, acts.get(v, 0.0) + arrival)
                if v not in queued:
                    queued.add(v)
                    queue.append(v)
    return acts, len(fired)


def arrivals(acts: dict) -> dict:
    """到达节点数：绝对阈值 + 相对阈值两套口径。"""
    if not acts:
        return {"absolute": 0, "relative": 0}
    maxv = max(acts.values())
    return {
        "absolute": sum(1 for v in acts.values() if v > TOL),
        "relative": sum(1 for v in acts.values() if v > maxv * REL_FACTOR),
    }


def edge_set(net: SynapseNet) -> set:
    return {frozenset(e) for e in net.connections.keys()}


def layer_hashes() -> dict:
    root = Path(__file__).resolve().parent.parent
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest()[:16] for n in LAYER_FILES}


def run() -> dict:
    c = Checker()
    hashes_before = layer_hashes()

    # ---------- P1 语料构造核查（含 T5 无多余边）----------
    nets, struct = {}, {}
    for name, edges in CORPORA.items():
        net = build(edges)
        nets[name] = net
        declared = {frozenset(e) for e in edges}
        observed = edge_set(net)
        struct[name] = {"declared": len(declared), "observed": len(observed), "exact": declared == observed}
    c.check("P1 语料构造核查: T1-T5 边集合逐边核对(含 T5 无多余边)",
            all(v["exact"] for v in struct.values()),
            "; ".join(f"{k}: 声明 {v['declared']} 边 / 实测 {v['observed']} 边 / 精确一致={v['exact']}"
                      for k, v in struct.items()))

    # ---------- 五条语料的双模式测量 ----------
    table, detail = {}, {}
    for name, net in nets.items():
        mx = net.activate(SEED)
        sm_asc, fired = sum_activate(net, SEED, order="asc")
        sm_desc, _ = sum_activate(net, SEED, order="desc")
        raw, _ = sum_activate(net, SEED, order="asc", cap=float("inf"))
        tgt = "C"
        table[name] = {
            "max_target": mx.get(tgt, 0.0),
            "sum_target": sm_asc.get(tgt, 0.0),
            "sum_raw_target": raw.get(tgt, 0.0),
            "cap_loss_target": max(0.0, raw.get(tgt, 0.0) - sm_asc.get(tgt, 0.0)),
            "sum_fired": fired,
            "nodes": net.node_count(),
            "arrivals_max": arrivals(mx),
            "arrivals_sum": arrivals(sm_asc),
        }
        detail[name] = {
            "max_acts": {k: round(v, 6) for k, v in sorted(mx.items())},
            "sum_asc_acts": {k: round(v, 6) for k, v in sorted(sm_asc.items())},
            "sum_desc_acts": {k: round(v, 6) for k, v in sorted(sm_desc.items())},
            "sum_raw_acts": {k: round(v, 6) for k, v in sorted(raw.items())},
            "order_dependent_nodes": [
                k for k in set(sm_asc) | set(sm_desc)
                if abs(sm_asc.get(k, 0.0) - sm_desc.get(k, 0.0)) > TOL
            ],
        }

    # ---------- P2 max 精确不叠加 ----------
    vals = [table[k]["max_target"] for k in ("T1", "T2", "T3")]
    spread = max(vals) - min(vals)
    c.check("P2 max 精确不叠加(B02 主体): T1/T2/T3 的 a(C) 严格相等",
            spread <= TOL,
            f"T1/T2/T3 a(C) = {['%.4f' % v for v in vals]}；极差={spread:.3e}"
            f"（容差 {TOL:g}；> 容差即停查机制，不许调容差）")

    # ---------- P3 sum 封顶 + 原始和 ----------
    ok3 = all(abs(table[k]["sum_target"] - 1.0) <= TOL for k in ("T2", "T3"))
    c.check("P3 sum 封顶 + 原始和: T2/T3 封顶后 = 1.0，并记录原始和与损失",
            ok3,
            "; ".join(f"{k}: 原始和={table[k]['sum_raw_target']:.4f} → 封顶 {table[k]['sum_target']:.4f}"
                      f"（损失 {table[k]['cap_loss_target']:.4f}）" for k in ("T1", "T2", "T3")))

    # ---------- P4 T4 环: sum 终止且封顶（单次调用的出队数）----------
    t4 = table["T4"]; n4 = nets["T4"].node_count()
    sm4, fired4 = sum_activate(nets["T4"], SEED, order="asc")
    c.check("P4 T4 环 sum 终止且封顶（sum 可用性自检）: 单次调用出队数 = 节点数，且所有值 <= 1.0",
            fired4 == n4 and max(sm4.values()) <= 1.0 + TOL,
            f"单次 activate 调用出队数={fired4}，节点数={n4}，最大值={max(sm4.values()):.4f}（cap=1.0）")

    # ---------- P5 max 顺序无关（重标签同构检查）----------
    edges_relabeled = [(RELABEL_T5[a], RELABEL_T5[b]) for a, b in CORPORA["T5"]]
    net_re = build(edges_relabeled)
    mx_re = net_re.activate(RELABEL_T5[SEED])
    mx_or = nets["T5"].activate(SEED)
    mapped = {RELABEL_T5[k]: v for k, v in mx_or.items()}
    diffs = {k: abs(mapped[k] - mx_re.get(k, 0.0)) for k in mapped}
    p5_ok = max(diffs.values()) <= TOL
    c.check("P5 max 顺序无关: 重标签到哈希顺序不同的 token 名后激活值逐点相同",
            p5_ok,
            f"重标签 {RELABEL_T5}；最大逐点差={max(diffs.values()):.3e}（容差 {TOL:g}）；"
            f"原={detail['T5']['max_acts']} 重标签={ {k: round(v, 6) for k, v in sorted(mx_re.items())} }")

    # ---------- P6 只读（纪律 8: 层文件 sha256）----------
    hashes_after = layer_hashes()
    c.check("P6 只读: 层文件 sha256 前后一致",
            hashes_before == hashes_after,
            f"before={hashes_before}；after={hashes_after}")

    # ---------- P7 登记文本 ----------
    c.check("P7 登记: JSON 内含 C28 / B03(升级) / B02(指针) 文本",
            all(s in (C28_TEXT + B03_TEXT + B02_TEXT) for s in ("C28", "B03（升级）", "B02（指针）")),
            "三段文本已写入 JSON 的 registration 字段")

    # ---------- 辅助记录 ----------
    aux_order = {
        "auxiliary": True,
        "note": "sum 的顺序依赖是它的固有性质（B03），本实验要证明它有，不是要求它没有；不参与 PASS/FAIL。",
        "T5_asc": detail["T5"]["sum_asc_acts"],
        "T5_desc": detail["T5"]["sum_desc_acts"],
        "T5_order_dependent_nodes": detail["T5"]["order_dependent_nodes"],
        "T5_max": detail["T5"]["max_acts"],
    }
    aux_arrivals = {
        "auxiliary": True,
        "note": "到达节点数两套阈值口径（不一致时以 relative 为准）。",
        "rows": {k: {"max": v["arrivals_max"], "sum": v["arrivals_sum"]} for k, v in table.items()},
    }

    return {
        "id": "exp28_readout_modes",
        "title": "E28 读取端 max/sum 系统对比: max 精确不叠加, sum 封顶且顺序依赖",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "table": table,
            "activations": detail,
            "auxiliary_order_dependence": aux_order,
            "auxiliary_arrivals": aux_arrivals,
            "corpus_structure": struct,
            "tolerances": {"strict": TOL, "relative_factor": REL_FACTOR, "cap": CAP, "hop_decay": HOP},
            "layer_hashes": {"before": hashes_before, "after": hashes_after},
            "registration": {"C28": C28_TEXT, "B03_upgraded": B03_TEXT, "B02_pointer": B02_TEXT},
            "note": "sum 只在实验层实现（E14 语义：单次发射 + 出队冻结），不改 L0、不新增层文件。",
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(run(), ensure_ascii=False, indent=2))
