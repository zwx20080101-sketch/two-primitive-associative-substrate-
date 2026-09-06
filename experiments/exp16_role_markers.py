"""E16: 角色标记归属判定 - 带施事/受事信息能否区分"狗咬猫"vs"猫咬狗"。

四问: 基底参与=把语序/标记绑定成可检索结构; 外部 stub=提供标记系统与分词
(输入侧); 结论归谁=基底能否区分(可验) + 标记系统属外部; 停点=给出
"谁咬谁"的可检索分布即可, 不做语义判断。

判据(预先注册):
  S1-1 L1 方向分账: n(狗->咬)=n(咬->猫)=n(猫->咬)=n(咬->狗)=100;
  S1-2 完整句取回: 打狗->咬 0.9 -> 猫 0.81 (句1); 打猫->咬 -> 狗 0.81 (句2);
  S1-3 中位词歧义: 只打"咬" -> 狗≈猫 等强 (诚实边界);
  S1-4 L2 前缀消歧: [狗,咬] -> 只有猫; [猫,咬] -> 只有狗;
  S3   格标记: [狗,把,猫,咬]/[猫,把,狗,咬] -> [狗,把]猫, [猫,把]狗,
        [把,猫]咬, [把,狗]咬;
  S6   闭包与基线回归: 无新节点; L0/L1/L2 行为不变。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_layer import ContextLayer
from experiments.checker import Checker
from order_layer import OrderLayer
from synapse_net import SynapseNet


def run() -> dict:
    c = Checker()
    data: dict = {}

    # ---------- S1 语序角色 (双向都学) ----------
    l1 = OrderLayer()
    l2 = ContextLayer()
    for _ in range(100):
        l1.learn(["狗", "咬", "猫"])
        l1.learn(["猫", "咬", "狗"])
        l2.learn(["狗", "咬", "猫"])
        l2.learn(["猫", "咬", "狗"])

    c.check(
        "S1-1 方向分账: 四个方向各 100",
        l1.count("狗", "咬") == 100
        and l1.count("咬", "猫") == 100
        and l1.count("猫", "咬") == 100
        and l1.count("咬", "狗") == 100,
        f"n(狗->咬)={l1.count('狗', '咬')}, n(咬->猫)={l1.count('咬', '猫')}, "
        f"n(猫->咬)={l1.count('猫', '咬')}, n(咬->狗)={l1.count('咬', '狗')}",
    )

    a_gou = l1.activate("狗")
    a_mao = l1.activate("猫")
    c.check(
        "S1-2 完整句取回: 打狗->咬0.9->猫0.81; 打猫->咬0.9->狗0.81",
        math.isclose(a_gou["咬"], 0.9, rel_tol=1e-9)
        and math.isclose(a_gou["猫"], 0.81, rel_tol=1e-9)
        and math.isclose(a_mao["咬"], 0.9, rel_tol=1e-9)
        and math.isclose(a_mao["狗"], 0.81, rel_tol=1e-9),
        f"打狗: 咬={a_gou['咬']:.2f}, 猫={a_gou['猫']:.2f}; "
        f"打猫: 咬={a_mao['咬']:.2f}, 狗={a_mao['狗']:.2f}",
    )

    a_yao = l1.activate("咬")
    c.check(
        "S1-3 中位词歧义(诚实边界): 只打'咬' -> 狗≈猫 等强",
        math.isclose(a_yao["狗"], a_yao["猫"], rel_tol=1e-9),
        f"狗={a_yao['狗']:.4f}, 猫={a_yao['猫']:.4f}",
    )

    q_gou_yao = l2.query(["狗", "咬"])
    q_mao_yao = l2.query(["猫", "咬"])
    c.check(
        "S1-4 L2 前缀消歧: [狗,咬]->只有猫; [猫,咬]->只有狗",
        set(q_gou_yao) == {"猫"} and math.isclose(q_gou_yao["猫"], 1.0, rel_tol=1e-9)
        and set(q_mao_yao) == {"狗"} and math.isclose(q_mao_yao["狗"], 1.0, rel_tol=1e-9),
        f"[狗,咬]->{q_gou_yao}; [猫,咬]->{q_mao_yao}",
    )
    data["s1"] = {
        "from_gou": {k: round(v, 4) for k, v in sorted(a_gou.items())},
        "from_mao": {k: round(v, 4) for k, v in sorted(a_mao.items())},
        "from_yao": {k: round(v, 4) for k, v in sorted(a_yao.items())},
        "L2_gou_yao": q_gou_yao,
        "L2_mao_yao": q_mao_yao,
    }

    # ---------- S3 格标记 (把=受事标记, 输入侧提供) ----------
    lm = ContextLayer()
    for _ in range(100):
        lm.learn(["狗", "把", "猫", "咬"])
        lm.learn(["猫", "把", "狗", "咬"])
    q1 = lm.query(["狗", "把"])
    q2 = lm.query(["猫", "把"])
    q3 = lm.query(["把", "猫"])
    q4 = lm.query(["把", "狗"])
    c.check(
        "S3 格标记存储: n(狗,把,猫)=100, n(猫,把,狗)=100",
        lm.count("狗", "把", "猫") == 100 and lm.count("猫", "把", "狗") == 100,
        f"n(狗,把,猫)={lm.count('狗', '把', '猫')}, n(猫,把,狗)={lm.count('猫', '把', '狗')}",
    )
    c.check(
        "S3 格标记区分: [狗,把]->猫; [猫,把]->狗; [把,猫]->咬; [把,狗]->咬",
        set(q1) == {"猫"} and set(q2) == {"狗"}
        and set(q3) == {"咬"} and set(q4) == {"咬"},
        f"[狗,把]->{q1}; [猫,把]->{q2}; [把,猫]->{q3}; [把,狗]->{q4}",
    )
    data["s3"] = {"q_dog_ba": q1, "q_cat_ba": q2, "q_ba_cat": q3, "q_ba_dog": q4}

    # ---------- S6 闭包与基线回归 ----------
    learned = set(lm.nodes)
    all_outputs = set(q1) | set(q2) | set(q3) | set(q4) | set(q_gou_yao) | set(q_mao_yao)
    c.check(
        "S6 闭包: 无新节点; 标记'把'来自输入侧, 基底未发明角色概念",
        all_outputs <= learned,
        f"outputs={sorted(all_outputs)}, learned={sorted(learned)}",
    )
    net = SynapseNet()
    net.learn(["A", "B"])
    acts0 = net.activate("A")
    c.check(
        "S6 基线回归: L0 learn([A,B])->B=0.9 仍成立 (本实验未改层文件)",
        math.isclose(acts0["B"], 0.9, rel_tol=1e-9),
        f"a(B)={acts0.get('B', 0):.4f}",
    )

    return {
        "id": "exp16_role_markers",
        "title": "E16 角色标记归属判定: 语序/格标记能否区分 狗咬猫 vs 猫咬狗",
        "passed": c.passed,
        "checks": c.checks,
        "data": data,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
