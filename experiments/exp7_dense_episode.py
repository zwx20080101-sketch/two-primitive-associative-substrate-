"""E7: 密集瞬间 - 一次摔倒为什么记得住 (带时间维, 不同时注入)。

模型: 真实世界是三维+时间, 经验按时间流注入:
  平日 100 天: 每天一次正常跑步 [快,风,稳,加速], t=1..100
  摔倒那一天:  一次摔倒 = 约 1.5 秒内的高密度脉冲流, 每个脉冲有自己的 t
               (失衡前兆 -> 失衡 -> 空中 -> 撞击 -> 痛风暴 -> 后续站起)
对照组: 一次"轻轻一碰" [轻碰,轻疼] x1, t=200

测量:
  A) 摔倒前: 打"快"只有日常邻居, 无 摔/疼 (此刻还没有这条路径);
  B) 一次摔倒后: 疼 成为枢纽 (邻居数 >= 4);
  C) 打"摔" -> 疼 强 (0.9);
  D) 打"空中" -> 疼 第二跳出现 (0.81) = 预警;
  E) 打"快" -> 疼 中等偏弱 (0.36, 快不是疼的最强邻居);
  F) 对照组: 轻碰只留一条边, 无枢纽无预警;
  G) 闭包: 任何激活输出都曾是 learn 过的节点;
  H) 时间维: 每个连接带 last_seen; 日常边旧(t<=100), 摔倒边新(t>101);
  I) L1 顺序层: 快->不稳->失衡->空中->摔->疼 的分向计数成立, 打空中能顺向到达疼。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attention import attention_filter
from experiments.checker import Checker
from order_layer import OrderLayer
from synapse_net import SynapseNet


def _degree(net: SynapseNet, node: str) -> int:
    return sum(1 for a, b in net.connections if node in (a, b))


def run() -> dict:
    net = SynapseNet()
    order = OrderLayer()
    c = Checker()
    stages: list[dict] = []

    # ---------- 平日 100 天: 每天一次正常跑步 ----------
    for day in range(1, 101):
        net.learn(["快", "风", "稳", "加速"], t=float(day))

    before = net.activate("快")
    c.check(
        "摔倒前: 打快只有日常邻居, 摔/疼不存在 (还没有这条路径)",
        set(before) == {"快", "风", "稳", "加速"},
        f"keys={sorted(before)}",
    )

    # ---------- 摔倒那一天: 时间轴上的脉冲流 ----------
    t = 101.0
    pulse_plan = [
        ("失衡前兆-不稳", ("快", "不稳"), 30),
        ("失衡前兆-加速不稳", ("加速", "不稳"), 20),
        ("重心失衡", ("不稳", "失衡"), 30),
        ("离地空中", ("失衡", "空中"), 30),
        ("撞击:空中->摔", ("空中", "摔"), 30),
        ("撞击:速度仍在->摔", ("快", "摔"), 20),
        ("撞击:加速仍在->摔", ("加速", "摔"), 10),
        ("撞击:触地", ("摔", "地面"), 20),
        ("痛风暴:摔<->疼", ("摔", "疼"), 60),
        ("痛风暴:疼->摔(反馈)", ("疼", "摔"), 40),
        ("痛风暴:快仍活跃<->疼", ("快", "疼"), 40),
        ("痛风暴:加速衰减<->疼", ("加速", "疼"), 10),
        ("痛风暴:地面<->疼", ("地面", "疼"), 30),
        ("后续:站起<->疼", ("站起", "疼"), 30),
        ("后续:疼->站起(持续疼)", ("疼", "站起"), 30),
    ]
    for name, pair, n in pulse_plan:
        t0 = t
        for _ in range(n):
            t += 0.001
            net.learn(list(pair), t=t)
            order.learn(list(pair), t=t)
        stages.append({"stage": name, "pair": pair, "pulses": n, "t_start": round(t0, 3), "t_end": round(t, 3)})

    # ---------- A) 枢纽 ----------
    deg_pain = _degree(net, "疼")
    c.check(
        "A) 一次摔倒后: 疼 成为枢纽 (邻居数>=4)",
        deg_pain >= 4,
        f"疼 的邻居={sorted(n for n in net.nodes if net.connection_count('疼', n) > 0)}, degree={deg_pain}",
    )

    # ---------- B/C/D/E) 各线索取疼 ----------
    a_shuai = net.activate("摔")
    a_kong = net.activate("空中")
    a_kuai = net.activate("快")
    c.check(
        "B) 打 摔 -> 疼 0.9 (密集共现, 最强)",
        math.isclose(a_shuai["疼"], 0.9, rel_tol=1e-9),
        f"a(疼|摔)={a_shuai['疼']:.6f}",
    )
    c.check(
        "C) 打 空中 -> 疼 0.81 (第二跳: 还没落地就预警)",
        math.isclose(a_kong["疼"], 0.81, rel_tol=1e-9),
        f"a(疼|空中)={a_kong['疼']:.6f}",
    )
    c.check(
        "D) 打 快 -> 疼 0.36 (可达但弱: 快不是疼的最强邻居)",
        math.isclose(a_kuai["疼"], 0.36, rel_tol=1e-9),
        f"a(疼|快)={a_kuai['疼']:.6f}",
    )
    hi = attention_filter(a_kong, threshold=0.7)
    c.check(
        "C2) 空中预警: 疼 0.81 >= 0.7 被放出 (低于 0.85 的强信号线)",
        "疼" in hi and set(hi) <= set(net.list_nodes()),
        f"visible={sorted(hi)}",
    )

    # ---------- F) 对照: 轻轻一碰 ----------
    net2 = SynapseNet()
    net2.learn(["轻碰", "轻疼"], t=200.0)
    deg_light = _degree(net2, "轻疼")
    c.check(
        "F) 轻碰一次: 只留 1 条边, 无枢纽/无第二跳",
        deg_light == 1,
        f"轻疼 degree={deg_light}; keys(轻碰)={sorted(net2.activate('轻碰'))}",
    )

    # ---------- G) 闭包 ----------
    learned = set(net.nodes)
    all_acts = {**a_shuai, **a_kong, **a_kuai}
    c.check(
        "G) 闭包: 所有激活输出都曾是学过的节点 (无中生有判据)",
        set(all_acts) <= learned,
        f"输出 {len(all_acts)} 节点, 全部在 learned 集合内",
    )

    # ---------- H) 时间维 ----------
    last_daily = net.last_seen("快", "风")
    last_fall = net.last_seen("摔", "疼")
    c.check(
        "H) 时间维: 连接带 last_seen; 日常边旧(t=100), 摔倒边新(t>101)",
        last_daily == 100.0 and last_fall > 101.0,
        f"last_seen(快,风)={last_daily}, last_seen(摔,疼)={last_fall:.3f}",
    )

    # ---------- I) L1 顺序层: 分向计数与顺向预警 ----------
    dir_ok = (
        order.count("快", "不稳") == 30
        and order.count("不稳", "失衡") == 30
        and order.count("失衡", "空中") == 30
        and order.count("空中", "摔") == 30
        and order.count("摔", "疼") == 60
    )
    l1_air = order.activate("空中")
    c.check(
        "I) L1 顺序链成立: 快->不稳->失衡->空中->摔->疼 分向计数",
        dir_ok and math.isclose(l1_air.get("疼", 0.0), 0.81, rel_tol=1e-9),
        "counts: 快->不稳 30, 不稳->失衡 30, 失衡->空中 30, 空中->摔 30, 摔->疼 60; "
        f"L1 a(疼|空中)={l1_air.get('疼', 0):.6f}",
    )

    return {
        "id": "exp7_dense_episode",
        "title": "E7 密集瞬间: 一次摔倒=时间轴上的脉冲流 (不同时注入)",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "timeline": stages,
            "time_metadata": {
                "last_seen(快,风)": last_daily,
                "last_seen(摔,疼)": round(last_fall, 4),
            },
            "degrees": {
                "疼(摔倒)": deg_pain,
                "轻疼(轻碰对照)": deg_light,
            },
            "activation_from_shuai": {k: round(v, 6) for k, v in sorted(a_shuai.items())},
            "activation_from_kongzhong": {k: round(v, 6) for k, v in sorted(a_kong.items())},
            "activation_from_kuai": {k: round(v, 6) for k, v in sorted(a_kuai.items())},
            "directed_counts": {
                "快->不稳": order.count("快", "不稳"),
                "不稳->失衡": order.count("不稳", "失衡"),
                "失衡->空中": order.count("失衡", "空中"),
                "空中->摔": order.count("空中", "摔"),
                "摔->疼": order.count("摔", "疼"),
                "疼->摔": order.count("疼", "摔"),
            },
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
