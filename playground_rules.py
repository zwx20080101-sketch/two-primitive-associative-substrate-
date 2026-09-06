"""Synapse-Net 规则沙箱（Playground Rules）

用途：像设计实验一样，用代码写"你的上层规则"，然后观察网络涌现出什么。
底层（L0）仍是共现绑定 + 激活扩散；L1/L2/L4 是上层表示层；本脚本只做演示，
不修改任何层文件。运行：python playground_rules.py
"""

from __future__ import annotations

from context_layer import ContextLayer
from order_layer import OrderLayer
from chunk_layer import ChunkLayer
from synapse_net import SynapseNet

K = 5  # 成块阈值（预设）


class Toy:
    """轻量封装：一个事件喂给对应层。"""

    def __init__(self):
        self.l0 = SynapseNet()
        self.l1 = OrderLayer()
        self.l2 = ContextLayer()
        self.l4 = ChunkLayer(k=K)

    def sync(self, tokens):
        """同步窗口 -> L0 两两绑定。"""
        self.l0.learn(tokens)

    def seq(self, tokens):
        """有序序列 -> L0 相邻对 + L1 方向 + L2 三元组。"""
        self.l0.learn(tokens)
        self.l1.learn(tokens)
        self.l2.learn(tokens)

    def unit(self, tokens):
        """带边界单元 -> L4 成块计数。"""
        self.l4.learn_unit(tokens)

    def activate(self, node):
        """L0 激活扩散，返回全量 {节点: 激活值}。"""
        return self.l0.activate(node)

    def query(self, prefix):
        """L2 查询：前两项 -> 下一候选强度分布。"""
        return self.l2.query(prefix)

    def spell(self, cid):
        """L4 拼写记录。"""
        return self.l4.spell(cid)

    def stats(self):
        return {
            "nodes": self.l0.node_count(),
            "connections": self.l0.edge_count(),
            "directed": self.l1.edge_count(),
            "triples": self.l2.triple_types(),
            "chunks": len(self.l4.chunks),
        }


def trace(toy: Toy, seed: str):
    """按激活降序打印输入->输出链路。"""
    acts = toy.activate(seed)
    chain = sorted(acts.items(), key=lambda kv: -kv[1])
    return " ⇒ ".join(f"{k}({v:.3f})" if v < 1 else f"{k}(种子)" for k, v in chain)


def custom_rule(toy: Toy):
    """← 你的自定义规则写在这里。改完直接 python playground_rules.py 看涌现。"""
    # 示例：反复输入 "A B C" 与 "A B D"，看 L2 双前缀是否能区分。
    for _ in range(100):
        toy.seq(["A", "B", "C"])
        toy.seq(["A", "B", "D"])
    print("custom 三元组 [A,B] 的候选：", toy.query(["A", "B"]))
    print("custom 三元组 [X,B] 不可用提示：", toy.query(["X", "B"]))
    print("custom 激活动画示例：", trace(toy, "A"))


def main() -> None:
    toy = Toy()

    print("== E6a 双前缀消歧 ==")
    for _ in range(100):
        toy.seq(["X1", "B", "C"])
        toy.seq(["X2", "B", "D"])
    print("  L2 查询 [X1,B] ->", toy.query(["X1", "B"]))
    print("  L2 查询 [X2,B] ->", toy.query(["X2", "B"]))
    print("  L0 激活 B 的中位词歧义：", toy.activate("B"))

    toy2 = Toy()
    print("\n== E5 Hello 正倒 ==")
    for _ in range(100):
        toy2.seq(["H", "e", "l", "l", "o"])
    print("  正走链路：", trace(toy2, "H"))
    print("  倒走链路：", trace(toy2, "o"))
    fwd = toy2.l1.activate("H")["o"]
    rev = toy2.l1.activate("o")["H"]
    print(f"  L1 方向层：正走到 o={fwd:.4f}，倒走到 H={rev:.4g}（正强反弱 = 方向不对称）")

    toy3 = Toy()
    print("\n== E12 组块双写 ==")
    for _ in range(100):
        toy3.unit(["H", "e", "l", "l", "o"])
    cid = next(iter(toy3.l4.chunks))
    print("  spell(", cid, ") =", toy3.spell(cid))

    print("\n== 自定义规则：custom_rule(toy) ==")
    custom_rule(Toy())

    print("\n当前包含层：L0/L1/L2/L4；修改 custom_rule 即可探索新的涌现。")


if __name__ == "__main__":
    main()
