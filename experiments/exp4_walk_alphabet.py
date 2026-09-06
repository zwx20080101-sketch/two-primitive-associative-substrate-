"""E4: 26 字母逐对强化 + 链式走读实验。

协议 (教师-网络交互, 全部发生在实验侧, 底层仍是空白起步):
  第 1 轮(增量学习):
    发 A        -> 网络回复 A            -> 教师把 [A, B] 一起发回(绑定)
    发 B        -> 网络回复 B            -> 教师把 [B, C] 一起发回(绑定)
    ...直到 [Y, Z]
  之后 N-1 轮(强化): 反复发送相邻字母对。

连接上记录的是"这两个字母被一起提起的次数" N (共现累计, 永不删除)。
测试:
  1) 直接打 A, 信号能否到达整条链的末端 Z;
  2) 在末端把 Z 打回去, 能否反向走到起点 A;
  3) 读取端走读器逐次拿"最强未访问节点"当下一步, 能否完整输出 A..Z。

注意: 走读器是实验侧的观测工具, 不属于底层; 底层只做共现绑定与激活扩散。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.checker import Checker
from synapse_net import SynapseNet

ALPHABET = [chr(ord("A") + i) for i in range(26)]


def _walk(net: SynapseNet, start: str) -> list[str]:
    """读取端走读器: 每一步激活当前字母, 选激活最强的未访问邻居走下去。

    这不是底层机制——底层只负责给出全量激活; 选"下一步"属于上层/实验侧。
    """
    seq = [start]
    seen = {start}
    current = start
    while True:
        acts = net.activate(current)
        candidates = {k: v for k, v in acts.items() if k not in seen and v > 1e-9}
        if not candidates:
            break
        nxt = max(sorted(candidates), key=lambda k: candidates[k])
        seq.append(nxt)
        seen.add(nxt)
        current = nxt
    return seq


def run(passes: int = 100) -> dict:
    net = SynapseNet()  # 绝对干净: 内部零数据
    c = Checker()

    c.check(
        "底层启动为空: 0 nodes / 0 connections / 无任何预设",
        net.node_count() == 0 and net.edge_count() == 0 and net.list_nodes() == [],
        f"nodes={net.node_count()}, connections={net.edge_count()}",
    )

    # ---- 第 1 轮: 增量交互学习 -------------------------------------
    for i in range(25):
        net.learn([ALPHABET[i]])             # 教师发单个字母(只创建/刷新节点, 不产生连接)
        net.activate(ALPHABET[i])            # 网络回复该字母(实验侧观察, 不写底层)
        net.learn([ALPHABET[i], ALPHABET[i + 1]])  # 教师把相邻字母一起发回 -> 绑定
    net.learn(["Z"])                         # 收尾: Z 单独出现一次

    # ---- 之后: 强化学习 -------------------------------------------
    for _ in range(passes - 1):
        for i in range(25):
            net.learn([ALPHABET[i], ALPHABET[i + 1]])

    c.check(
        "经验后: 恰好 26 nodes / 25 connections (无预置字母)",
        net.node_count() == 26 and net.edge_count() == 25,
        f"nodes={net.node_count()}, connections={net.edge_count()}",
    )
    c.check(
        "只有相邻对连接; A-Z、A-C 等从未共现 -> 无连接",
        net.connection_count("A", "Z") == 0 and net.connection_count("A", "C") == 0,
        "N(A,Z)=0, N(A,C)=0",
    )

    counts = {f"{a}-{b}": net.connection_count(a, b) for a, b in zip(ALPHABET, ALPHABET[1:])}
    uniform = all(v == passes for v in counts.values())
    c.check(
        f"每条相邻边都被一起提起 {passes} 次 (N 永久累计)",
        uniform,
        "示例: " + ", ".join(f"{k}:{v}" for k, v in list(counts.items())[:4]) + " ...",
    )

    # ---- 测试 1: 直接打 A, 信号能否到达末端 Z ---------------------
    forward_acts = net.activate("A")
    z_expected = 0.9 ** 25
    c.check(
        "直接打 A: 全链 26 个字母全部亮起",
        set(forward_acts) == set(ALPHABET),
        f"keys={''.join(sorted(forward_acts))}",
    )
    c.check(
        "直接打 A: 信号到达末端 Z (0.9^25)",
        math.isclose(forward_acts["Z"], z_expected, rel_tol=1e-9),
        f"a(Z)={forward_acts['Z']:.9f} (期望 {z_expected:.9f})",
    )

    # ---- 测试 2: 在末端把 Z 打回去 ---------------------------------
    reverse_acts = net.activate("Z")
    a_expected = 0.9 ** 25
    c.check(
        "打回 Z: 信号反向到达起点 A",
        math.isclose(reverse_acts["A"], a_expected, rel_tol=1e-9),
        f"a(A)={reverse_acts['A']:.9f} (期望 {a_expected:.9f})",
    )

    # ---- 测试 3: 走读器完整输出 ------------------------------------
    fwd = _walk(net, "A")
    rev = _walk(net, "Z")
    c.check(
        "走读器从 A 出发完整输出 A..Z",
        "".join(fwd) == "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "".join(fwd),
    )
    c.check(
        "走读器从 Z 打回完整输出 Z..A",
        "".join(rev) == "ZYXWVUTSRQPONMLKJIHGFEDCBA",
        "".join(rev),
    )

    # ---- 激活读出顺序(同一份全量输出, 按激活强度降序) -------------
    ranked = sorted(forward_acts.items(), key=lambda kv: (-kv[1], kv[0]))
    c.check(
        "激活强度降序读出 = 完整正向序列 A..Z",
        "".join(k for k, _ in ranked) == "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "".join(k for k, _ in ranked),
    )

    return {
        "id": "exp4_walk_alphabet",
        "title": f"E4 逐对强化 + 链式走读 (相邻对各强化 {passes} 次)",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "network": {
                "nodes": net.node_count(),
                "connections": net.edge_count(),
            },
            "counts": counts,  # 每条边的"被一起提起次数"
            "activation": {ch: round(forward_acts[ch], 9) for ch in ALPHABET},
            "forward_walk": "".join(fwd),
            "reverse_walk": "".join(rev),
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
