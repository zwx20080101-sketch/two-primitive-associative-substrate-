"""Layer 4 (L4): 组块层 (Chunk Layer) - 最小实现。

原则与 L0/L1 一致, 只是换了一层尺度:
  同一段"带边界的有序序列"(unit) 在经验中出现 >= K 次 -> 自动获得一个块节点;
  块节点内部保存"顺序记录"(允许重复元素, 如 HELLO 的双写 l);
  块节点可当普通原子参与更高层序列(句子)的共现。

本层不改 L0/L1; 块只由见过的成员组成(无中生有闭包继续适用);
块名是机器号 CH_n, 标签不携带语义。
"""

from __future__ import annotations

import math
from collections import deque


class ChunkLayer:
    def __init__(self, k: int = 5):
        self.k = k
        self.unit_counts: dict[tuple, int] = {}
        self.chunks: dict[str, list] = {}       # chunk_id -> 有序成员表(允许重复)
        self.chunk_for_unit: dict[tuple, str] = {}
        self.member_bind: dict[tuple[str, str], int] = {}  # (chunk, member) 绑定次数
        self.high_pairs: dict[tuple[str, str], int] = {}   # 块级相邻共现(无向, 规范序)
        self.high_nodes: set[str] = set()
        self._next_id = 1
        self._clock = 0.0

    # ------------------------------------------------------------------
    # 时钟
    # ------------------------------------------------------------------
    def _tick(self, t):
        if t is None:
            self._clock += 1.0
            return self._clock
        t = float(t)
        if t > self._clock:
            self._clock = t
        return t

    # ------------------------------------------------------------------
    # 块的形成: 带边界的有序序列重复 >= K 次
    # ------------------------------------------------------------------
    def learn_unit(self, unit: list, t=None) -> str | None:
        """unit = 一次完整的带边界单元(如一个词/短语)。返回块 id(若已成块)。"""
        self._tick(t)
        seq = tuple(unit)
        self.unit_counts[seq] = self.unit_counts.get(seq, 0) + 1
        if seq not in self.chunk_for_unit and self.unit_counts[seq] >= self.k:
            cid = f"CH_{self._next_id}"
            self._next_id += 1
            self.chunks[cid] = list(seq)  # 顺序记录(含重复)
            self.chunk_for_unit[seq] = cid
            self.high_nodes.add(cid)
        cid = self.chunk_for_unit.get(seq)
        if cid is not None:
            for m in dict.fromkeys(seq):  # 去重成员绑定
                key = (cid, m)
                self.member_bind[key] = self.member_bind.get(key, 0) + 1
        return cid

    def spell(self, chunk_id: str) -> list:
        """块级书写/回放: 按顺序记录返回成员表(双写 l 在此还原)。"""
        if chunk_id not in self.chunks:
            raise KeyError(f"unknown chunk: {chunk_id}")
        return list(self.chunks[chunk_id])

    # ------------------------------------------------------------------
    # 块级序列: 句子 = 原子(词或块)的有序共现
    # ------------------------------------------------------------------
    def learn_sequence(self, atoms: list[str], t=None) -> None:
        self._tick(t)
        self.high_nodes.update(atoms)
        for a, b in zip(atoms, atoms[1:]):
            key = (a, b) if a < b else (b, a)
            self.high_pairs[key] = self.high_pairs.get(key, 0) + 1

    def count_pair(self, a: str, b: str) -> int:
        key = (a, b) if a < b else (b, a)
        return self.high_pairs.get(key, 0)

    # ------------------------------------------------------------------
    # 块级激活(与 L0 同一扩散规则)
    # ------------------------------------------------------------------
    def _neighbors(self, u: str) -> set[str]:
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

    def activate(self, start: str | list[str], hop_decay: float = 0.9) -> dict[str, float]:
        """块级全量激活(只含原子/块, 不含展开成员; 展开用 spell)。"""
        seeds = [start] if isinstance(start, str) else list(start)
        for s in seeds:
            if s not in self.high_nodes:
                raise KeyError(f"node not in chunk layer: {s!r}")
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
    def chunk_ids(self) -> list[str]:
        return list(self.chunks)

    def degree(self, node: str) -> int:
        return len(self._neighbors(node))
