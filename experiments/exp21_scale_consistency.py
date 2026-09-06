"""E21: V1.2 规模一致性 - 关键现象在 1x/10x/100x 实例规模下是否仍成立。

六个现象各自在"共享同层、标签互不冲突"的网络上累加实例,
规模 S in {1,10,100}; 指标 = 通过率(判据成立的实例数/S)。

套件级判据:
  S=1: 六现象通过率 = 1.0 (复现原始判据);
  S=10/100: 通过率 >= 0.99;
  全程不改层文件; 附章 L2/L4 快基准(旁证, 不进主判据)。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from chunk_layer import ChunkLayer
from context_layer import ContextLayer
from experiments.checker import Checker
from order_layer import OrderLayer
from synapse_net import SynapseNet

OUT = Path(__file__).resolve().parent.parent / "outputs" / "exp21_scale.json"
SCALES = [1, 10, 100]


def _tok(prefix: str, i: int, j: int) -> str:
    return f"{prefix}{i}_{j}"


# ---------------- 现象实现 ----------------
def build_direction(s: int):
    l1 = OrderLayer()
    for i in range(s):
        ls = [_tok("d", i, j) for j in range(5)]
        for _ in range(100):
            l1.learn(ls)
    return l1


def dir_ok(layer: OrderLayer, i: int) -> bool:
    ls = [_tok("d", i, j) for j in range(5)]
    try:
        fwd = layer.activate(ls[0])[ls[-1]]
        rev = layer.activate(ls[-1])[ls[0]]
        return fwd / rev > 100
    except KeyError:
        return False


def build_translation(s: int):
    net = SynapseNet()
    for i in range(s):
        hub = _tok("t", i, 0)
        c1 = [_tok("t", i, 1), _tok("t", i, 2), _tok("t", i, 3)]
        c2 = [_tok("t", i, 4), _tok("t", i, 5), _tok("t", i, 6)]
        for _ in range(100):
            net.learn([hub, *c1])
            net.learn([hub, *c2])
    return net


def trans_ok(net: SynapseNet, i: int) -> bool:
    c1, c2 = _tok("t", i, 1), _tok("t", i, 4)
    try:
        return c2 in net.activate(c1)
    except KeyError:
        return False


def build_hub(s: int):
    net = SynapseNet()
    for i in range(s):
        a, b, c, d, e = (_tok("h", i, j) for j in range(5))
        for _ in range(100):
            net.learn([a, b])
            net.learn([b, c])
            net.learn([a, d])
            net.learn([d, e])
    return net


def hub_ok(net: SynapseNet, i: int) -> bool:
    a, b, c, d, e = (_tok("h", i, j) for j in range(5))
    try:
        acts = net.activate(a)
        return (
            math.isclose(acts[b], 0.9, rel_tol=1e-9)
            and math.isclose(acts[d], 0.9, rel_tol=1e-9)
            and math.isclose(acts[c], 0.81, rel_tol=1e-9)
            and math.isclose(acts[e], 0.81, rel_tol=1e-9)
        )
    except KeyError:
        return False


def build_completion(s: int):
    net = SynapseNet()
    for i in range(s):
        chain = [_tok("p", i, j) for j in range(5)]
        for _ in range(100):
            for a, b in zip(chain, chain[1:]):
                net.learn([a, b])
    return net


def comp_ok(net: SynapseNet, i: int) -> bool:
    chain = [_tok("p", i, j) for j in range(5)]
    try:
        acts = net.activate([chain[0], chain[-1]])
        return acts.get(chain[2], 0.0) >= 0.8
    except KeyError:
        return False


def build_double(s: int):
    cl = ChunkLayer(k=5)
    cids = []
    for i in range(s):
        unit = [_tok("w", i, 0), _tok("w", i, 1), _tok("w", i, 1), _tok("w", i, 2)]
        cid = None
        for _ in range(100):
            cid = cl.learn_unit(unit)
        cids.append(cid)
    return cl, cids


def dbl_ok(cl: ChunkLayer, cids: list, i: int) -> bool:
    expected = [_tok("w", i, 0), _tok("w", i, 1), _tok("w", i, 1), _tok("w", i, 2)]
    try:
        return cl.spell(cids[i]) == expected
    except KeyError:
        return False


def build_role(s: int):
    l2 = ContextLayer()
    for i in range(s):
        a, b, v = _tok("r", i, 0), _tok("r", i, 1), _tok("r", i, 2)
        for _ in range(100):
            l2.learn([a, v, b])
            l2.learn([b, v, a])
    return l2


def role_ok(l2: ContextLayer, i: int) -> bool:
    a, b, v = _tok("r", i, 0), _tok("r", i, 1), _tok("r", i, 2)
    q1, q2 = l2.query([a, v]), l2.query([b, v])
    return set(q1) == {b} and set(q2) == {a}


PHENOMENA = {
    "方向": (build_direction, dir_ok, False),
    "翻译": (build_translation, trans_ok, False),
    "枢纽": (build_hub, hub_ok, False),
    "补全": (build_completion, comp_ok, False),
    "双写": (build_double, dbl_ok, True),
    "角色": (build_role, role_ok, False),
}


def run() -> dict:
    c = Checker()
    rates = {}
    for name, (builder, checker, special) in PHENOMENA.items():
        row = {}
        for s in SCALES:
            obj = builder(s)
            if special:
                cl, cids = obj
                row[s] = sum(1 for i in range(s) if checker(cl, cids, i)) / s
            else:
                row[s] = sum(1 for i in range(s) if checker(obj, i)) / s
        rates[name] = row

    ok_s1 = all(math.isclose(rates[n][1], 1.0, abs_tol=1e-9) for n in PHENOMENA)
    ok_big = all(rates[n][10] >= 0.99 and rates[n][100] >= 0.99 for n in PHENOMENA)
    c.check(
        "S=1: 六现象通过率全部 = 1.0 (原始判据复现)",
        ok_s1,
        "; ".join(f"{n}={rates[n][1]}" for n in PHENOMENA),
    )
    c.check(
        "S=10 与 S=100: 六现象通过率全部 >= 0.99 (无规模退化)",
        ok_big,
        "; ".join(f"{n}:10x={rates[n][10]},100x={rates[n][100]}" for n in PHENOMENA),
    )

    # ---------- 附章: L2/L4 快基准 ----------
    annex = {}
    for units in (1000, 10000):
        l2 = ContextLayer()
        t0 = time.perf_counter()
        for i in range(units):
            l2.learn([f"a{i % 8}", f"b{(i * 3 + 1) % 8}", f"c{(i * 5 + 2) % 8}", f"d{(i * 7 + 3) % 8}"])
        learn_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        for i in range(units):
            l2.query([f"a{i % 8}", f"b{(i * 3 + 1) % 8}"])
        query_s = time.perf_counter() - t0
        annex[f"L2_{units}u"] = {
            "learn_s": round(learn_s, 3),
            "query_s": round(query_s, 3),
            "triple_types": l2.triple_types(),
        }
    for words in (1000, 5000):
        cl = ChunkLayer(k=5)
        t0 = time.perf_counter()
        for i in range(words):
            unit = [f"l{i}_0", f"l{i}_1", f"l{i}_1", f"l{i}_2"]
            for _ in range(5):
                cl.learn_unit(unit)
        build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        for i in range(min(words, 500)):
            cl.spell(cl.chunk_ids()[i])
        spell_s = time.perf_counter() - t0
        annex[f"L4_{words}w"] = {
            "build_s": round(build_s, 3),
            "spell500_s": round(spell_s, 3),
            "chunks": len(cl.chunks),
        }
    c.check(
        "附章: L2/L4 快基准已采集 (旁证, 不进主判据)",
        all(v is not None for v in annex.values()),
        f"L2(10k): {annex.get('L2_10000u', {})}; L4(5k): {annex.get('L4_5000w', {})}",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rates": rates, "annex": annex}, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "id": "exp21_scale_consistency",
        "title": "E21 V1.2 规模一致性: 六现象 1x/10x/100x 通过率 + L2/L4 快基准",
        "passed": c.passed,
        "checks": c.checks,
        "data": {"rates": rates, "annex": annex},
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
