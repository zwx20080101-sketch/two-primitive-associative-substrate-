"""复杂度基线测量 (M3-4)。

测 L0 与 L1 在 节点规模 200/1000/3000/6000 下的:
  - 建网事件数(learn 次数) 与 单次 learn 平均耗时;
  - 单次 activate 耗时(中位数, 从 n0 全图扩散);
  - 峰值内存(不含边生成阶段)。

图结构: 环 + 确定性随机弦(平均度约 4), 保证全图连通。
结果写入 outputs/benchmark.json 并打印表格。
"""

from __future__ import annotations

import json
import random
import statistics
import time
import tracemalloc
from pathlib import Path

from order_layer import OrderLayer
from synapse_net import SynapseNet

SCALES_L0 = [200, 1000, 3000, 6000]
SCALES_L1 = [200, 1000, 3000, 6000]  # L1 已加邻接索引(2026-09-06), 与 L0 同尺度
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def gen_edges(n: int, seed: int = 42) -> list[tuple[int, int]]:
    rnd = random.Random(seed)
    edges = set()
    for i in range(n):
        edges.add(tuple(sorted((i, (i + 1) % n))))          # 环, 保证连通
        j = (i + 3 + rnd.randrange(0, max(1, n // 50))) % n  # 确定性弦
        if j != i:
            edges.add(tuple(sorted((i, j))))
    return sorted(edges)


def bench_l0(n: int) -> dict:
    edges = gen_edges(n)
    tracemalloc.start()
    net = SynapseNet()
    t0 = time.perf_counter()
    for a, b in edges:
        net.learn([f"n{a}", f"n{b}"])
    build_s = time.perf_counter() - t0
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    net.activate("n0")  # 预热 + 连通性验证
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        net.activate("n0")
        times.append(time.perf_counter() - t0)
    return {
        "layer": "L0",
        "nodes": n,
        "edges": len(edges),
        "learn_events": len(edges),
        "build_s": round(build_s, 4),
        "learn_avg_us": round(build_s * 1e6 / len(edges), 2),
        "activate_ms": round(statistics.median(times) * 1e3, 3),
        "peak_mem_mb": round(peak_mb, 2),
    }


def bench_l1(n: int) -> dict:
    edges = gen_edges(n)
    tracemalloc.start()
    layer = OrderLayer()
    t0 = time.perf_counter()
    for a, b in edges:
        layer.learn([f"n{a}", f"n{b}"])
    build_s = time.perf_counter() - t0
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    layer.activate("n0")
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        layer.activate("n0")
        times.append(time.perf_counter() - t0)
    return {
        "layer": "L1",
        "nodes": n,
        "edges": len(edges),
        "learn_events": len(edges),
        "build_s": round(build_s, 4),
        "learn_avg_us": round(build_s * 1e6 / len(edges), 2),
        "activate_ms": round(statistics.median(times) * 1e3, 3),
        "peak_mem_mb": round(peak_mb, 2),
    }


def main() -> None:
    rows = []
    for n in SCALES_L0:
        rows.append(bench_l0(n))
        print(f"L0 n={n} done", flush=True)
    for n in SCALES_L1:
        rows.append(bench_l1(n))
        print(f"L1 n={n} done", flush=True)

    print(f"{'layer':<4} {'nodes':>6} {'edges':>7} {'learn/ev':>9} {'learn_avg(us)':>13} "
          f"{'activate(ms)':>13} {'peak(MB)':>9}")
    for r in rows:
        print(f"{r['layer']:<4} {r['nodes']:>6} {r['edges']:>7} {r['learn_events']:>9} "
              f"{r['learn_avg_us']:>13} {r['activate_ms']:>13} {r['peak_mem_mb']:>9}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "benchmark.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved -> outputs/benchmark.json")


if __name__ == "__main__":
    main()
