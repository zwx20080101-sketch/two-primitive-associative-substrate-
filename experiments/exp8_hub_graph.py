"""E8: 连接是图不是线 - 共享枢纽能否支撑多条句子。

训练(每句 100 次, 只喂相邻对/顺序对):
  狗咬猫   -> [狗,咬] [咬,猫]      L1: [狗,咬,猫]
  狗追老鼠 -> [狗,追] [追,老鼠]    L1: [狗,追,老鼠]
  猫追老鼠 -> [猫,追] [追,老鼠]    L1: [猫,追,老鼠]

重点: 追-老鼠 被两句共用(200); 狗 连着 咬/追; 追 连着 狗/猫/老鼠。
判据(预先注册):
  S1 结构: 5 节点 5 边; N(追,老鼠)=200; N(狗,猫)=0; N(咬,老鼠)=0;
  S2 枢纽: degree(狗)=2, degree(追)=3;
  S3 打狗: 咬=追=0.9, 猫=老鼠=0.81 (两句候选并行, 不选边);
  S4 打咬: 猫 0.9 > 老鼠 0.729 (本从句更近);
  S5 打老鼠: 狗=猫=0.405 (主语分不清, 诚实歧义);
  S6 L1: 方向只来自经验; 反向计数=0;
  S7 组合: 走读器从狗跨枢纽走出 狗咬猫追老鼠(新排列, 无新节点, 闭包)。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.checker import Checker
from order_layer import OrderLayer
from synapse_net import SynapseNet


def _degree(net: SynapseNet, node: str) -> int:
    return sum(1 for a, b in net.connections if node in (a, b))


def _walk(net, start: str) -> list[str]:
    """读取端走读器: 每步取激活最强的未访问邻居 (与 E4/E5 同规则)。"""
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


def _train_once(net: SynapseNet, order: OrderLayer, sentence: list[str]) -> None:
    for a, b in zip(sentence, sentence[1:]):
        net.learn([a, b])
    order.learn(sentence)


def run() -> dict:
    net = SynapseNet()
    order = OrderLayer()
    for _ in range(100):
        _train_once(net, order, ["狗", "咬", "猫"])
        _train_once(net, order, ["狗", "追", "老鼠"])
        _train_once(net, order, ["猫", "追", "老鼠"])
    c = Checker()

    # ---------- S1 结构 ----------
    c.check(
        "S1 结构: 5 nodes / 5 edges",
        net.node_count() == 5 and net.edge_count() == 5,
        f"nodes={net.node_count()}, edges={net.edge_count()}",
    )
    c.check(
        "S1 共用边: N(追,老鼠)=200; 未共现 N(狗,猫)=0, N(咬,老鼠)=0",
        net.connection_count("追", "老鼠") == 200
        and net.connection_count("狗", "猫") == 0
        and net.connection_count("咬", "老鼠") == 0,
        f"N(追,老鼠)={net.connection_count('追', '老鼠')}, "
        f"N(狗,猫)={net.connection_count('狗', '猫')}, "
        f"N(咬,老鼠)={net.connection_count('咬', '老鼠')}",
    )

    # ---------- S2 枢纽 ----------
    c.check(
        "S2 枢纽: degree(狗)=2, degree(追)=3",
        _degree(net, "狗") == 2 and _degree(net, "追") == 3,
        f"degree(狗)={_degree(net, '狗')}, degree(追)={_degree(net, '追')}",
    )

    # ---------- S3 打狗: 候选并行 ----------
    a_gou = net.activate("狗")
    ok_s3 = (
        math.isclose(a_gou["咬"], 0.9, rel_tol=1e-9)
        and math.isclose(a_gou["追"], 0.9, rel_tol=1e-9)
        and math.isclose(a_gou["猫"], 0.81, rel_tol=1e-9)
        and math.isclose(a_gou["老鼠"], 0.81, rel_tol=1e-9)
    )
    c.check(
        "S3 打狗: 咬=追=0.9, 猫=老鼠=0.81 (两句候选并行, 不选边)",
        ok_s3,
        f"咬={a_gou['咬']:.4f}, 追={a_gou['追']:.4f}, "
        f"猫={a_gou['猫']:.4f}, 老鼠={a_gou['老鼠']:.4f}",
    )

    # ---------- S4 打咬: 本从句更强 ----------
    a_yao = net.activate("咬")
    c.check(
        "S4 打咬: 猫 0.9 > 老鼠 0.729 (本从句更近)",
        math.isclose(a_yao["猫"], 0.9, rel_tol=1e-9)
        and math.isclose(a_yao["老鼠"], 0.729, rel_tol=1e-9)
        and a_yao["猫"] > a_yao["老鼠"],
        f"猫={a_yao['猫']:.4f}, 老鼠={a_yao['老鼠']:.4f}",
    )

    # ---------- S5 打老鼠: 主语诚实歧义 ----------
    a_laoshu = net.activate("老鼠")
    c.check(
        "S5 打老鼠: 追 0.9; 狗=猫=0.405 (主语分不清)",
        math.isclose(a_laoshu["追"], 0.9, rel_tol=1e-9)
        and math.isclose(a_laoshu["狗"], 0.405, rel_tol=1e-9)
        and math.isclose(a_laoshu["猫"], 0.405, rel_tol=1e-9),
        f"追={a_laoshu['追']:.4f}, 狗={a_laoshu['狗']:.4f}, 猫={a_laoshu['猫']:.4f}",
    )

    # ---------- S6 L1 方向只来自经验 ----------
    dir_ok = (
        order.count("狗", "咬") == 100
        and order.count("咬", "猫") == 100
        and order.count("狗", "追") == 100
        and order.count("猫", "追") == 100
        and order.count("追", "老鼠") == 200
        and order.count("咬", "狗") == 0
        and order.count("猫", "咬") == 0
        and order.count("老鼠", "追") == 0
    )
    c.check(
        "S6 L1 方向: 顺向计数与经验一致, 反向=0",
        dir_ok,
        "狗->咬 100, 咬->猫 100, 狗->追 100, 猫->追 100, 追->老鼠 200; 反向 0",
    )

    # ---------- S7 组合走读 + 闭包 ----------
    walk = _walk(net, "狗")
    c.check(
        "S7 走读: 跨枢纽走出 狗咬猫追老鼠 (旧零件的新排列)",
        walk == ["狗", "咬", "猫", "追", "老鼠"],
        "".join(walk),
    )
    learned = set(net.nodes)
    c.check(
        "S7 闭包: 全程无新节点; N(咬,老鼠)=0 说明是组合不是直连",
        set(walk) <= learned and net.connection_count("咬", "老鼠") == 0,
        f"walk={''.join(walk)}",
    )

    return {
        "id": "exp8_hub_graph",
        "title": "E8 连接是图不是线: 共享枢纽支撑多条句子",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "counts": {
                "狗-咬": net.connection_count("狗", "咬"),
                "咬-猫": net.connection_count("咬", "猫"),
                "狗-追": net.connection_count("狗", "追"),
                "猫-追": net.connection_count("猫", "追"),
                "追-老鼠": net.connection_count("追", "老鼠"),
                "狗-猫": net.connection_count("狗", "猫"),
                "咬-老鼠": net.connection_count("咬", "老鼠"),
            },
            "degrees": {"狗": _degree(net, "狗"), "追": _degree(net, "追")},
            "activation_from_狗": {k: round(v, 6) for k, v in sorted(a_gou.items())},
            "activation_from_咬": {k: round(v, 6) for k, v in sorted(a_yao.items())},
            "activation_from_老鼠": {k: round(v, 6) for k, v in sorted(a_laoshu.items())},
            "directed_counts": {
                "狗->咬": order.count("狗", "咬"),
                "咬->猫": order.count("咬", "猫"),
                "狗->追": order.count("狗", "追"),
                "猫->追": order.count("猫", "追"),
                "追->老鼠": order.count("追", "老鼠"),
            },
            "walk_from_狗": "".join(walk),
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
