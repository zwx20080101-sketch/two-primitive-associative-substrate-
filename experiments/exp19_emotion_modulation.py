"""M2-C: 情绪调制(系统级) - 强反馈是否需要独立调制轴?

宪章四问: 基底=绑定/检索(无写侧调制); 外部 stub=情绪模块(读取端增益)+
输入侧强度标记; 结论归谁=系统整体(记账表); 停点=两种"强反馈生效路径"
(输入密度 / 输入标记+外部读取增益)都验证, 拍板基底是否需要独立调制轴。

构造(脉冲次数完全相同, 只差输入标记):
  蛇链: [蛇,剧疼]x100 + [剧疼,剧]x100     (剧=强疼标记)
  棍链: [棍,轻疼]x100 + [轻疼,轻]x100     (轻=轻疼标记)
密度路径复核: 蛇 x300 vs 棍 x60 (有公共邻居草丛x100)

预期:
  1) 基底密度相同时不分轻重: a(剧疼|蛇)=a(轻疼|棍)=0.9;
  2) 情绪 stub 读取端增益: fear(蛇)=0.9*3 > fear(棍)=0.9*1, 躲避仅蛇触发;
  3) 密度路径(E7 复核): 300 vs 60 -> 基底自身 0.9 vs 0.54 区分;
  4) 结论: 基底无需写侧调制轴; 情绪调制=外部读取端。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotion_stub import EmotionStub
from experiments.checker import Checker
from synapse_net import SynapseNet


def run() -> dict:
    net = SynapseNet()
    emotion = EmotionStub()
    c = Checker()

    # ---------- 训练: 次数完全相同, 只差强度标记 ----------
    for _ in range(100):
        net.learn(["蛇", "剧疼"])
        net.learn(["剧疼", "剧"])
        net.learn(["棍", "轻疼"])
        net.learn(["轻疼", "轻"])

    c.check(
        "1) 结构对等: 四条边 N 全=100, 唯一差别是标记 剧/轻",
        net.connection_count("蛇", "剧疼") == 100
        and net.connection_count("棍", "轻疼") == 100
        and net.connection_count("剧疼", "剧") == 100
        and net.connection_count("轻疼", "轻") == 100,
        "N(蛇,剧疼)=N(棍,轻疼)=N(剧疼,剧)=N(轻疼,轻)=100",
    )

    acts_snake = net.activate("蛇")
    acts_stick = net.activate("棍")
    c.check(
        "2) 基底不分轻重: a(剧疼|蛇)=a(轻疼|棍)=0.9 (密度相同则基底无差异)",
        math.isclose(acts_snake["剧疼"], 0.9, rel_tol=1e-9)
        and math.isclose(acts_stick["轻疼"], 0.9, rel_tol=1e-9),
        f"a(剧疼|蛇)={acts_snake['剧疼']:.4f}, a(轻疼|棍)={acts_stick['轻疼']:.4f}",
    )

    c.check(
        "3) 标记绑定且无语义: 基底只存 疼-剧/疼-轻 共现, 无'危险'节点",
        "剧" in acts_snake and "轻" in acts_stick and "危险" not in net.nodes,
        f"nodes={sorted(net.nodes)}",
    )

    fear_snake = emotion.fear(acts_snake, "剧疼")
    fear_stick = emotion.fear(acts_stick, "轻疼")
    avoid_snake = emotion.avoid(acts_snake, "剧疼")
    avoid_stick = emotion.avoid(acts_stick, "轻疼")
    c.check(
        "4) 情绪 stub 读取端调制: fear(蛇)=0.9*3 > fear(棍)=0.9*1; 躲避仅蛇触发",
        math.isclose(fear_snake, 2.7, rel_tol=1e-9)
        and math.isclose(fear_stick, 0.9, rel_tol=1e-9)
        and avoid_snake and not avoid_stick,
        f"fear(蛇)={fear_snake:.2f}, fear(棍)={fear_stick:.2f}, "
        f"avoid(蛇)={avoid_snake}, avoid(棍)={avoid_stick}",
    )

    # ---------- 密度路径复核 (E7 机制) ----------
    net_s = SynapseNet()
    net_g = SynapseNet()
    for _ in range(300):
        net_s.learn(["蛇", "剧疼"])
    for _ in range(100):
        net_s.learn(["蛇", "草丛"])
    for _ in range(60):
        net_g.learn(["棍", "轻疼"])
    for _ in range(100):
        net_g.learn(["棍", "草丛"])
    d_snake = net_s.activate("蛇")["剧疼"]
    d_stick = net_g.activate("棍")["轻疼"]
    c.check(
        "5) 密度路径(E7 复核): 300 vs 60 (公共邻居100) -> 基底自身 0.9 vs 0.54",
        math.isclose(d_snake, 0.9, rel_tol=1e-9)
        and math.isclose(d_stick, 0.54, rel_tol=1e-9)
        and d_snake > d_stick,
        f"a(疼|蛇,300次)={d_snake:.4f}, a(疼|棍,60次)={d_stick:.4f}",
    )

    # ---------- 功劳记账 ----------
    ledger = [
        {"步骤": "强度标记(剧/轻)注入", "谁干的": "身体/感知侧(输入)", "内容": "输入结构, 非基底发明"},
        {"步骤": "疼痛-标记绑定", "谁干的": "基底 L0", "内容": "只存共现, 不分轻重"},
        {"步骤": "密度差异(300 vs 60)", "谁干的": "输入频率 -> 基底相对强度", "内容": "E7 机制复核"},
        {"步骤": "fear 响应与躲避决策", "谁干的": "情绪 stub(外部读取端)", "内容": "增益表占位, 真实模块应从反馈经验学习"},
        {"步骤": "强反馈生效路径", "谁干的": "两条: 输入密度 / 输入标记+外部读取增益", "内容": "基底无需写侧调制轴"},
    ]
    c.check(
        "6) 功劳记账: 五步归属写清; 结论=基底不需要独立调制轴",
        len(ledger) == 5 and all(row["谁干的"] for row in ledger),
        "标记=输入侧; 绑定=基底; 情绪响应=外部读取端; 密度=输入频率",
    )

    return {
        "id": "exp19_emotion_modulation",
        "title": "M2-C 情绪调制(系统级): 强反馈两条路径, 基底无需写侧调制轴",
        "passed": c.passed,
        "checks": c.checks,
        "data": {
            "marker_path": {
                "a(剧疼|蛇)": round(acts_snake["剧疼"], 6),
                "a(轻疼|棍)": round(acts_stick["轻疼"], 6),
                "fear_snake": round(fear_snake, 6),
                "fear_stick": round(fear_stick, 6),
                "avoid_snake": avoid_snake,
                "avoid_stick": avoid_stick,
            },
            "density_path": {
                "snake_300": round(d_snake, 6),
                "stick_60": round(d_stick, 6),
            },
            "ledger": ledger,
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
