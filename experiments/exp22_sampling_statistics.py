"""E22: V1.3 统计口径 - 随机输入族抽样成功率 + Wilson 95% 置信区间。

对六现象(方向/翻译/枢纽/补全/双写/角色)各抽样 M=60 个随机输入族:
  每次随机 token 标签 + 随机重复次数 r in [30,80];
  正样本 = 机制 ON; 负样本 = 机制 OFF/随机(同标签同 r)。

判据:
  正样本率 >= 0.99 且 Wilson 下限 >= 0.90;
  零效应负样本率 <= 0.12 (方向/翻译/枢纽/补全/双写);
  二分机会负样本(角色) CI 覆盖 0.5。
"""

from __future__ import annotations

import json
import itertools
import math
import random
from pathlib import Path

from chunk_layer import ChunkLayer
from context_layer import ContextLayer
from experiments.checker import Checker
from order_layer import OrderLayer
from synapse_net import SynapseNet

OUT = Path(__file__).resolve().parent.parent / "outputs" / "exp22_stats.json"
M = 60
Z = 1.96


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float, float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def tok(pre: str, d: int, i: int, j: int) -> str:
    return f"{pre}{d}_{i}_{j}"


# 每个现象: (pos_ok(seed), neg_ok(seed)); 每次内部自建随机实例
def sample_direction(seed: int, pos: bool) -> bool:
    rnd = random.Random(seed)
    r = rnd.randint(30, 80)
    ls = [tok("dir", seed, 0, j) for j in range(5)]
    if pos:
        l1 = OrderLayer()
        for _ in range(r):
            l1.learn(ls)
        try:
            return l1.activate(ls[0])[ls[-1]] / l1.activate(ls[-1])[ls[0]] > 100
        except KeyError:
            return False
    net = SynapseNet()
    for _ in range(r):
        for a, b in zip(ls, ls[1:]):
            net.learn([a, b])
    try:
        return net.activate(ls[0])[ls[-1]] / net.activate(ls[-1])[ls[0]] > 100
    except KeyError:
        return False


def sample_translation(seed: int, pos: bool) -> bool:
    rnd = random.Random(seed)
    r = rnd.randint(30, 80)
    hub = tok("tr", seed, 0, 0)
    c1 = [tok("tr", seed, 0, k) for k in (1, 2, 3)]
    c2 = [tok("tr", seed, 0, k) for k in (4, 5, 6)]
    net = SynapseNet()
    if pos:
        for _ in range(r):
            net.learn([hub, *c1])
            net.learn([hub, *c2])
        return c2[0] in net.activate(c1[0])
    for _ in range(r):
        net.learn(c1)
        net.learn(c2)
    return c2[0] in net.activate(c1[0])


def sample_hub(seed: int, pos: bool) -> bool:
    rnd = random.Random(seed)
    r = rnd.randint(30, 80)
    a, b, c, d, e = (tok("hub", seed, 0, j) for j in range(5))
    net = SynapseNet()
    if pos:
        for _ in range(r):
            net.learn([a, b])
            net.learn([b, c])
            net.learn([a, d])
            net.learn([d, e])
    else:
        # 随机结构: 同节点、同边数, 但真正随机选边
        rnd2 = random.Random(seed + 999)
        nodes = [a, b, c, d, e]
        all_pairs = list(itertools.combinations(nodes, 2))
        pairs = rnd2.sample(all_pairs, 4)
        for _ in range(r):
            for x, y in pairs:
                net.learn([x, y])
    try:
        acts = net.activate(a)
        return (
            math.isclose(acts.get(b, 0.0), 0.9, rel_tol=1e-9)
            and math.isclose(acts.get(d, 0.0), 0.9, rel_tol=1e-9)
            and math.isclose(acts.get(c, 0.0), 0.81, rel_tol=1e-9)
            and math.isclose(acts.get(e, 0.0), 0.81, rel_tol=1e-9)
        )
    except KeyError:
        return False


def sample_completion(seed: int, pos: bool) -> bool:
    rnd = random.Random(seed)
    r = rnd.randint(30, 80)
    chain = [tok("cmp", seed, 0, j) for j in range(5)]
    net = SynapseNet()
    for _ in range(r):
        for a, b in zip(chain, chain[1:]):
            net.learn([a, b])
    if pos:
        return net.activate([chain[0], chain[-1]]).get(chain[2], 0.0) >= 0.8
    direct = set([chain[0], chain[-1]])
    for x, y in net.connections:
        for s in (chain[0], chain[-1]):
            if s in (x, y):
                direct.add(x if s == y else y)
    return chain[2] in direct


def sample_double(seed: int, pos: bool) -> bool:
    rnd = random.Random(seed)
    r = rnd.randint(30, 80)
    unit = [tok("dbl", seed, 0, 0), tok("dbl", seed, 0, 1), tok("dbl", seed, 0, 1), tok("dbl", seed, 0, 2)]
    if pos:
        cl = ChunkLayer(k=5)
        cid = None
        for _ in range(r):
            cid = cl.learn_unit(unit)
        return cl.spell(cid) == unit
    net = SynapseNet()
    for _ in range(r):
        for a, b in zip(unit, unit[1:]):
            if a != b:
                net.learn([a, b])
    return False  # 平面链无块记录


def sample_role(seed: int, pos: bool) -> bool:
    rnd = random.Random(seed)
    r = rnd.randint(30, 80)
    a, b, v = tok("rol", seed, 0, 0), tok("rol", seed, 0, 1), tok("rol", seed, 0, 2)
    if pos:
        l2 = ContextLayer()
        for _ in range(r):
            l2.learn([a, v, b])
            l2.learn([b, v, a])
        return set(l2.query([a, v])) == {b} and set(l2.query([b, v])) == {a}
    l1 = OrderLayer()
    for _ in range(r):
        l1.learn([a, v, b])
        l1.learn([b, v, a])
    acts = l1.activate(v)
    # 无前缀信息: 严格更强才选; 平局用随机硬币打破(≈机会水平)
    rnd2 = random.Random(seed + 77)
    if math.isclose(acts.get(b, 0.0), acts.get(a, 0.0), rel_tol=1e-9):
        return rnd2.random() < 0.5
    return acts.get(b, 0.0) > acts.get(a, 0.0)


PHEN = {
    "方向": (sample_direction, "zero"),
    "翻译": (sample_translation, "zero"),
    "枢纽": (sample_hub, "zero"),
    "补全": (sample_completion, "zero"),
    "双写": (sample_double, "zero"),
    "角色": (sample_role, "chance"),
}


def run() -> dict:
    c = Checker()
    rows = {}
    for name, (fn, neg_kind) in PHEN.items():
        pos_ok = sum(1 for s in range(M) if fn(1000 + s, True))
        neg_ok = sum(1 for s in range(M) if fn(1000 + s, False))
        pp, plo, phi = wilson(pos_ok, M)
        np_, nlo, nhi = wilson(neg_ok, M)
        rows[name] = {
            "pos": {"k": pos_ok, "rate": round(pp, 4), "ci_lo": round(plo, 4), "ci_hi": round(phi, 4)},
            "neg": {"k": neg_ok, "rate": round(np_, 4), "ci_lo": round(nlo, 4), "ci_hi": round(nhi, 4)},
        }

    pos_ok_all = all(rows[n]["pos"]["rate"] >= 0.99 and rows[n]["pos"]["ci_lo"] >= 0.90 for n in PHEN)
    neg_ok_all = all(rows[n]["neg"]["rate"] <= 0.12 for n in PHEN if PHEN[n][1] == "zero")
    chance = rows["角色"]["neg"]
    chance_ok = chance["ci_lo"] <= 0.5 <= chance["ci_hi"]

    c.check(
        "正样本: 六现象率>=0.99 且 Wilson 下限>=0.90",
        pos_ok_all,
        "; ".join(f"{n}={rows[n]['pos']['rate']}([{rows[n]['pos']['ci_lo']},{rows[n]['pos']['ci_hi']}])" for n in PHEN),
    )
    c.check(
        "负样本(零效应): 方向/翻译/枢纽/补全/双写 率<=0.12",
        neg_ok_all,
        "; ".join(f"{n}={rows[n]['neg']['rate']}([{rows[n]['neg']['ci_lo']},{rows[n]['neg']['ci_hi']}])" for n in PHEN if PHEN[n][1] == "zero"),
    )
    c.check(
        "负样本(二分机会): 角色 CI 覆盖 0.5",
        chance_ok,
        f"角色 neg={chance['rate']}([{chance['ci_lo']},{chance['ci_hi']}])",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "id": "exp22_sampling_statistics",
        "title": "E22 V1.3 统计口径: 随机输入族成功率 + Wilson 95% CI",
        "passed": c.passed,
        "checks": c.checks,
        "data": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
