<<<<<<< HEAD
# Synapse-Net V0

本仓库是 Synapse-Net V0 的参考实现。文档：
[DESIGN.md](./DESIGN.md)（实验设计）、[STATUS.md](./STATUS.md)（状态与计划）、
[LAYERS.md](./LAYERS.md)（层次规划）、[BOUNDARY.md](./BOUNDARY.md)（边界宪章）、
[ROADMAP.md](./ROADMAP.md)（拆分路线图与上量 gate）、
[REVIEW-M1-M2.md](./REVIEW-M1-M2.md)（M1+M2 阶段性评审）、
[FORMULAS.md](./FORMULAS.md)（公式冻结，M3-1）、
[INTERFACE.md](./INTERFACE.md)（接口契约，M3-2）、
[PARAMETERS.md](./PARAMETERS.md)（参数表，M3-3）、
[BENCHMARK.md](./BENCHMARK.md)（复杂度基线，M3-4）、
[GATE-REVIEW.md](./GATE-REVIEW.md)（上量 gate 评审）、
[RESEARCH-PROPOSAL.md](./RESEARCH-PROPOSAL.md)（研究提案：科学验证）、
[PAPER-OUTLINE.md](./PAPER-OUTLINE.md)（V2 论文大纲）、
[CLAIMS-REGISTRY.md](./CLAIMS-REGISTRY.md)（结论-证据登记表）、
[BOUNDARIES-QUICK.md](./BOUNDARIES-QUICK.md)（边界速查）。

中文开源研究报告（完整版）：
[REPORT-zh.md](./REPORT-zh.md)（Markdown）·
[REPORT-zh.pdf](./outputs/pdf/REPORT-zh.pdf)（PDF，由 `python make_pdf.py` 生成）。

```text
synapse_net.py    # L0 共现基底(冻结): learn(共现绑定) / activate(激活扩散)
order_layer.py    # L1 方向/顺序层(含有序自转移=重复标记)
context_layer.py  # L2 上下文层(双前缀三元组)
chunk_layer.py    # L4 组块层(成块/顺序记录/块级序列)
attention.py      # 读取端: attention_filter 纯函数
language_stub.py  # 外部 stub: 语言规则(叠词判定/渲染)
motor_stub.py     # 外部 stub: 动作/书写(块 -> 字母输出)
emotion_stub.py   # 外部 stub: 情绪(读取端增益/阈值)
experiments/      # E0-E22 实验
main.py           # 实验入口
benchmark.py      # 复杂度基准
outputs/          # 每次运行的 JSON 报告(数据备份)
```

底层 `synapse_net.py` 已冻结，不再修改；新能力一律以"上一层"的形式添加。

## 运行

```powershell
python main.py            # 全部实验
python main.py exp1_chain # 只跑一个实验
```

可用实验：`exp0_smoke`、`exp1_chain`、`exp2_strength`、`exp3_threshold`、
`exp4_walk_alphabet`、`exp5_direction_hello`、`exp6_fork_context`、
`exp6b_translation_synonym`、`exp7_dense_episode`、`exp8_hub_graph`、
`exp9_partial_completion`、`exp10_recency_lambda`、`exp11_interference`、
`exp12_chunk_nesting`、`exp13_repeat_marker`、`exp14_multi_path_evidence`、
`exp15_higher_order_context`、`exp16_role_markers`、`exp17_system_reduplication`、
`exp18_spelling_output`、`exp19_emotion_modulation`、`exp20_ablation_suite`、
`exp21_scale_consistency`、`exp22_sampling_statistics`。

底层核心接口：

```python
net = SynapseNet()            # 空网络: 0 nodes / 0 connections
net.learn(["A", "B"])         # 共现绑定 (事件内两两连接)
acts = net.activate("A")      # 全量激活扩散, 弱节点也返回
attention_filter(acts, threshold=0.5)   # 读取端过滤
```

Python 不在 PATH 时可用捆绑运行时：

```powershell
& "C:\Users\A\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" main.py exp1_chain
```

## 复现与发布

```powershell
python main.py            # 全量实验 E0-E22(确定性, 逐项 PASS)
python main.py exp20_ablation_suite   # 只跑单个实验
python benchmark.py       # 复杂度基准 -> outputs/benchmark.json
python verify_claims.py   # 会审: 草稿数值/红线/数据一致性 77 项核对
python scan_norms.py      # 措辞规范扫描
python build_report.py    # 组装 REPORT-zh.md(章节草稿合并)
python make_pdf.py        # 生成 outputs/pdf/REPORT-zh.pdf
```

- 依赖：仅 Python 标准库，无第三方包；
- 数据：每次运行会覆盖 `outputs/<exp>.json`（确定性生成，内容可复现）；
- 额外套件数据：`outputs/exp20_ablation.json`、`exp21_scale.json`、
  `exp22_stats.json` 由对应实验模块自行写出；
- 发布前检查：① `python main.py` 全 PASS；② 结论与
  [CLAIMS-REGISTRY.md](./CLAIMS-REGISTRY.md) 逐条一致；
  ③ git tag 快照已建立（当前 v0.9.0，发布前再打 v1.0.0）。
=======
# two-primitive-associative-substrate-
A minimal two‑primitive associative substrate, controlled empirical test suite (E0‑E22), visualization playground.双原语联想基底，受控实证测试套件与可视化探索玩具
>>>>>>> 91b8d5a976d9c3adf42eafba2a7c6a51ccbef833
