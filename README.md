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
数据整理指南（给 AI）：[SKILL-ai-data.md](./SKILL-ai-data.md)；
输入格式：[INPUT-FORMAT.md](./INPUT-FORMAT.md)。
未覆盖/待探索缺口：[OPEN-QUESTIONS.md](./OPEN-QUESTIONS.md)。
后续开发计划（含三条外部建议评估）：[NEXT-PLAN.md](./NEXT-PLAN.md)。

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
`exp21_scale_consistency`、`exp22_sampling_statistics`、`exp23_composition`。
`exp24_module_differentiation`（numpy 依赖）。
`exp25_module_dynamics`（numpy 依赖：模块平均激活的双稳与滞回）。
`exp26_module_coordination`（numpy 依赖：EI 整合量与剂量响应，精确枚举 2^12 状态）。

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

## 交互演示（网页玩具）

**快速体验**（别人怎么直接用）：

1. 在线：把仓库用 GitHub Pages 部署后，直接访问
   `https://<你的用户名>.github.io/<仓库名>/`（根目录会进入演示页）；
2. 本地：克隆后双击打开 [playground.html](./playground.html)（或根目录 `index.html`）。
   单文件、无外部依赖，任何现代浏览器直接打开即可。
3. 三步上手：顶部"实验场景"选一个（如 E5 或 E19）→ 点"激活"看波纹逐跳扩散 →
   读"本次实验结果"（激活数、因果 Token 序列、能量流与"证明了什么"）。
   也可在底部"批量输入"按格式（`@t` 时间 / `[ ]` 同时 / `>` 先后 / `{ }` 单元）写自己的实验数据。

已包含的可复现实验（对应仓库 CLAIMS-REGISTRY 的 C/B 条目）：E1 链、E2 重复强度、
E4 对称双向、E5 Hello 正倒、E6a 分叉、E15 双前缀、E16 角色、E7 密集(近似)、
E12 组块双写、E18 拼写、E17 叠词外推、E19 情绪。

可转发的一句话：

> 我们开源了 Synapse-Net——一个只用"共现绑定 + 激活扩散"两条规则、叠加若干上层表示层
> 与外部模块的最小联想记忆基底。这个交互页从空白网络开始，能载入我们跑过的实验
> （26 字母链、Hello 正倒、双前缀、角色、组块双写、叠词、情绪等），点激活看信号
> 像波浪一样逐跳扩散，并输出本次结果、因果能量流与"这实验证明了什么"；
> 底层 L0 冻结、上层与外部模块不冒充基底，所有结论都带可溯源数据。
> 用浏览器打开即可体验，也可以写带时间戳的事件流让 AI 一键复现。

（其余细节见下方。）

- 视觉：人脑形网络（聚合成脑叶团块）、肉粉/奶油配色、画布主屏；激活=波纹扩散；
- 实验场景下拉：E1/E2/E4/E5/E6a/E15/E16/E7(近似)/E12/E18/E17/E19，
  载入后显示该实验的**权威成果**（来自 CLAIMS-REGISTRY 与 outputs JSON）；
- 激活后输出**本次实验结果 + 能量流**：种子、激活节点数、传导次数、最深跳、
  峰值能量、逐跳能量轨迹，直至"能量衰减耗尽"；
- 副本：保存/恢复/删除快照，可"保存→继续训练→再保存"形成多副本；支持导出/导入 JSON；
- 配置："保存配置 / 配置"——导出/导入**你选的选项**（启用哪些上层 L1/L2/L4、
  hop_decay、阈值、场景、种子、当前网络、副本备份），一键还原，下次不用重选；
- 批量输入（格式见 [INPUT-FORMAT.md](./INPUT-FORMAT.md)）：
  `@t` 时间戳、`[ ]` 同时、`>` 先后、`{ }` 整体单元、`;`/换行隔断，一键载入；
- 时间戳实时显示（时钟 / 每条边最近共现 t），作为"因果=时间先后"的载体；
- 上层 L1/L2/L4 与外部 stub（拼写/叠词/情绪）为可开关图层与按钮；
  边界：上层按"上层表示层 / 外部 stub"呈现，不冒充基底；
   λ=0（W=N）、hop_decay、K、threshold 均为预设值；不模拟真实时间轴/多模态。
   （本轮冻结：网页玩具 v3；已知未覆盖的"分层影响/组合律"仅记录，尚未实现。）

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
