"""外部情绪模块 stub（占位实现，真实情绪模块未来替换）。

宪章定位：本文件属于"外部模块"，不是基底。
职责（读取端调制）：读基底激活图 + 输入侧的强度标记，输出情绪响应
（fear = 激活值 x 强度增益；decision = 是否超过躲避阈值）。
增益表当前为占位常量（真实模块应从身体反馈经验中学习该表）。
不写回基底、不改任何连接。
"""

from __future__ import annotations


class EmotionStub:
    """占位情绪模块：读取端增益表 + 阈值决策。"""

    def __init__(self, gain_table: dict[str, float] | None = None, avoid_threshold: float = 2.0):
        # 占位增益: "剧"强度反馈在情绪读取端放 3 倍, "轻" 1 倍
        self.gain = gain_table if gain_table is not None else {"剧": 3.0, "轻": 1.0}
        self.avoid_threshold = avoid_threshold

    def fear(self, activations: dict[str, float], pain_node: str) -> float:
        """fear = a(疼) x 当前同时激活的强度标记增益(取最大)。"""
        base = activations.get(pain_node, 0.0)
        gain = 1.0
        for marker, g in self.gain.items():
            if marker in activations and activations[marker] > 1e-9:
                gain = max(gain, g)
        return base * gain

    def avoid(self, activations: dict[str, float], pain_node: str) -> bool:
        return self.fear(activations, pain_node) >= self.avoid_threshold
