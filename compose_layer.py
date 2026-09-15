"""Layer 5 (L5): 组合层 (Compose Layer) — 两个模式共现绑定产生组合符号。

与 L4（chunk_layer.py）**同构**：自有节点集 + 自有边 + 同一条扩散规则（max 吸收，hop_decay）。
与 L4 的差异（三点，见 DESIGN-E42 §2）：
  ① 触发条件 = **coactivation**（两个模式同时激活），不是 L4 的 repeat（重复 >= K）；
  ② **成员边显式建**（C ↔ P∪Q 全部成员，边权 1.0）—— L4 的块节点 activate 时【不展开成员】，
     若不给成员边，E42 的 P3（可分性）直接不成立；
  ③ 每个组合符号带 **origins 字段**（= "coactivation"）—— L4 的 CH_n 没有该字段。

**不改 L0/L1/L2/L4；不改 v1/v2/v3。** 本层是 L0 之外的新增结构（同 L4 的 CH 节点）。

符号 id 用 `CMP_n`（不写 `C_n`）—— 语料里的成员节点名可能就叫 "C"，避免命名撞车。
"""

from __future__ import annotations

import math
from collections import deque

COACTIVATION = "coactivation"


class ComposeLayer:
    """L5：共现绑定层。触发 = 两个模式同时激活；符号自带 origins 与内部值。"""

    def __init__(self):
        self.combinations: dict[str, list] = {}          # cid -> 成员（升序去重）
        self.combine_for_members: dict[frozenset, str] = {}
        self.origins: dict[str, str] = {}                # cid -> "coactivation"
        self.internal_values: dict[str, dict] = {}       # cid -> {member: value}
        self.high_pairs: dict[tuple, int] = {}           # 无向相邻（规范序）
        self.high_nodes: set[str] = set()
        self._next_id = 1

    # ------------------------------------------------------------------
    # 组合的形成：两个模式【同时激活】
    # ------------------------------------------------------------------
    def coactivate(self, P: dict, Q: dict, rule: str = "prod") -> str:
        """两个模式同时激活 → 产生（或复用）组合符号。

        rule = "prod" ：内部值 = v1 × v2（只留共现成员的值；非共享成员 = 0）
        rule = "keep" ：内部值 = 1.0（丢值序，辅助口径）

        【语义（显式声明，E42 定死）】符号的 id 由【成员集合】决定（同一成员集 → 同一 id）；
        内部值【每次调用覆写】（last-write-wins，与 E34 的 W.hold 同族）。
        因此：用不同的 rule 或不同的模式值对【同一成员集】反复 coactivate，符号的值会变。
        若需要"符号冻结"，调用方应另用独立实例（E42 的 P4 就是这么做的）。
        """
        members = sorted(set(P) | set(Q))
        if rule == "prod":
            vals = {m: float(P.get(m, 0.0)) * float(Q.get(m, 0.0)) for m in members}
        elif rule == "keep":
            vals = {m: 1.0 for m in members}
        else:
            raise ValueError(f"unknown rule: {rule!r} (use 'prod' or 'keep')")

        key = frozenset(members)
        cid = self.combine_for_members.get(key)
        if cid is None:
            cid = f"CMP_{self._next_id}"
            self._next_id += 1
            self.combinations[cid] = members
            self.combine_for_members[key] = cid
            self.origins[cid] = COACTIVATION
            self.high_nodes.add(cid)
            self.high_nodes.update(members)
            # 成员边【显式建】：每条计 1 次 → strength = 1/1 = 1.0（L4 的归一化口径）
            for m in members:
                self.high_pairs[self._key(cid, m)] = 1
        self.internal_values[cid] = dict(vals)
        return cid

    def internal(self, cid: str) -> dict:
        """组合符号的内部值（逐位可查 —— P4 的直接判据对象）。"""
        return dict(self.internal_values[cid])

    def members_of(self, cid: str) -> list:
        return list(self.combinations[cid])

    # ------------------------------------------------------------------
    # 边与归一化（与 L4 同一口径）
    # ------------------------------------------------------------------
    @staticmethod
    def _key(a: str, b: str) -> tuple:
        return (a, b) if a < b else (b, a)

    def count_pair(self, a: str, b: str) -> int:
        return self.high_pairs.get(self._key(a, b), 0)

    def _neighbors(self, u: str) -> set:
        out = set()
        for (a, b) in self.high_pairs:
            if a == u:
                out.add(b)
            elif b == u:
                out.add(a)
        return out

    def _strength(self, u: str, v: str) -> float:
        uv = self.count_pair(u, v)
        best = uv
        for w in self._neighbors(u):
            best = max(best, self.count_pair(u, w))
        return uv / best if best > 0 else 0.0

    # ------------------------------------------------------------------
    # 激活（与 L0/L4 同一条规则：max 吸收 + hop_decay）
    # ------------------------------------------------------------------
    def activate(self, start: str | list[str], hop_decay: float = 0.9) -> dict:
        """组合层全量激活（含组合符号与它的成员节点）。"""
        seeds = [start] if isinstance(start, str) else list(start)
        for s in seeds:
            if s not in self.high_nodes:
                raise KeyError(f"node not in compose layer: {s!r}")
        acts: dict[str, float] = {s: 1.0 for s in seeds}
        queue = deque(seeds)
        queued = set(seeds)
        while queue:
            u = queue.popleft()
            queued.discard(u)
            au = acts[u]
            for v in self._neighbors(u):
                arrival = au * self._strength(u, v) * hop_decay
                if arrival > acts.get(v, -math.inf) + 1e-12:
                    acts[v] = arrival
                    if v not in queued:
                        queued.add(v)
                        queue.append(v)
        return acts

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def combination_ids(self) -> list:
        return list(self.combinations)

    def origin_of(self, cid: str):
        return self.origins.get(cid)

    def degree(self, node: str) -> int:
        return len(self._neighbors(node))
