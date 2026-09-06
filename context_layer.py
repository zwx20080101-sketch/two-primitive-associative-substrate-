"""Layer 2 (L2): 上下文/高阶顺序表示层 (Context Layer) - 最小实现。

定位: 独立外挂层, 不改 L0/L1 基线。
规则(滑动窗口, 已锁定):
  对有序事件 [s0, s1, ..., sk], 每个相邻窗口记一次三元组:
    (s[i], s[i+1]) -> s[i+2]  =>  n(a, b, c) += 1
  非相邻不计数, 不滑动窗口模式之外的第二套逻辑。

查询: 给定前两项 (a, b), 返回"下一个候选"的强度分布
      (计数按该前缀的最大值归一化, 0..1)。

边界: 只输出候选分布, 不选候选; 只解决"双前缀"上下文,
      更长 N 元(3 前缀以上)时序依赖不在本层范围。
"""

from __future__ import annotations


class ContextLayer:
    def __init__(self):
        self.counts: dict[tuple[str, str], dict[str, int]] = {}  # (a,b) -> {c: n}
        self.nodes: set[str] = set()
        self._clock = 0.0

    # ------------------------------------------------------------------
    # 学习: 滑动相邻三元组
    # ------------------------------------------------------------------
    def learn(self, ordered_event: list[str], t=None) -> None:
        if t is None:
            self._clock += 1.0
        else:
            t = float(t)
            if t > self._clock:
                self._clock = t
        self.nodes.update(ordered_event)
        for i in range(len(ordered_event) - 2):
            a, b, c = ordered_event[i], ordered_event[i + 1], ordered_event[i + 2]
            bucket = self.counts.setdefault((a, b), {})
            bucket[c] = bucket.get(c, 0) + 1

    # ------------------------------------------------------------------
    # 查询: 给定前两项 -> 下一候选强度分布 (计数 max 归一化)
    # ------------------------------------------------------------------
    def count(self, a: str, b: str, c: str) -> int:
        return self.counts.get((a, b), {}).get(c, 0)

    def candidates(self, prefix: list[str]) -> dict[str, int]:
        """返回该前缀下每个候选的原始计数(未出现前缀 -> 空)。"""
        if len(prefix) != 2:
            raise ValueError(f"L2 只支持 2 前缀查询, 收到 {len(prefix)} 个")
        return dict(self.counts.get((prefix[0], prefix[1]), {}))

    def query(self, prefix: list[str]) -> dict[str, float]:
        """候选强度分布: count / max(count), 未知前缀返回空。"""
        raw = self.candidates(prefix)
        if not raw:
            return {}
        best = max(raw.values())
        return {c: n / best for c, n in raw.items()}

    # ------------------------------------------------------------------
    # 观测(存储膨胀监控)
    # ------------------------------------------------------------------
    def prefix_types(self) -> int:
        return len(self.counts)

    def triple_types(self) -> int:
        """不同 (a,b,c) 三元组种类数, 用于存储膨胀观测。"""
        return sum(len(bucket) for bucket in self.counts.values())
