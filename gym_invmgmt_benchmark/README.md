# FBiT × gym-invmgmt

这是一个完全独立的实验目录：它将原始 FBiT 方法适配到论文
[`gym-invmgmt`](https://arxiv.org/abs/2605.11355) 的 `CoreEnv`、训练协议和
26 场景评估合同。原始
`algos/FBiT.py` 不会被导入或修改。

## 实验口径

- 信息层级：**Blind**。策略只接收已经实现的需求、欠货、库存、在途量、
  时间和 goodwill，不读取需求模型参数或未来需求。
- 训练：base 和 serial 拓扑各一个 generalist；每个默认 200,000
  环境步，按 episode 随机化需求效应、goodwill、履约模式、噪声和
  M5-like demand。
- 评估：论文的 22 个核心场景和 4 个 MARL 补充行，共 26 场景；
  每个场景使用 10 个 canonical seeds。
- 结果：官方 artifact 保存在 `vendor/gym_invmgmt_paper/results/`，
  FBiT 结果单独写入本目录，合并操作不会覆盖官方 CSV。

上游版本和来源见 [UPSTREAM.md](UPSTREAM.md)。

## 环境

```bash
python -m pip install -e ".[dev]"
```

`torch-geometric` 在当前系统会提示 `pyg-lib`/`torch-sparse` 的 GLIBC
可选加速库不可用，但 `GATv2Conv` 的 PyTorch fallback 已通过测试。
FBiT 流程不重新求解 Oracle，因此不要求安装 `pulp`。

## 运行

### Blind 基线的逐边尺度重训

如需重训 Blind 基线，请先安装额外依赖：

```bash
python -m pip install -e ".[dev,aligned-blind]"
```

`scripts/train_blind_edge_caps.py` 使用论文仓库发布的训练构建器，
可从头训练 PPO-MLP-B、PPO-GNN-B、ST-PPO-B 和 SAC-B。DAgger-B
暂不包含在该训练入口中：上游
`r2barati/gym-invmgmt-paper` 的公开版本
`0ec993a9dad0dd220787e77e77868d10ad5ab799` 只发布了 DAgger 评估代理、
checkpoint 和结果缓存，没有发布 DAgger 数据聚合训练程序。
`benchmarks/train_gnn_il.py` 是 Oracle BC + PPO fine-tuning 的 GNN-IL，
不是 DAgger-B，因此不会被当作 DAgger-B 代用。
逐边尺度评估的 `--method all` 因此只包含上述四种可重训方法；
`--method DAgger-B` 仅为作者训练源码生成的新 artifact 保留。

### FBiT 训练与评估

分别或顺序训练两个拓扑：

```bash
python scripts/train_fbit.py \
  --topology all --steps 200000 --seed 0
```

中断后从 `checkpoints/fbit_<topology>_blind_resume.pt` 续跑：

```bash
python scripts/train_fbit.py \
  --topology base --steps 200000 --seed 0 --resume
```

完成两个 final checkpoint 后评估并合并：

```bash
python scripts/evaluate_fbit.py --scenarios all --seeds 10
python scripts/merge_with_paper.py
python scripts/verify_results.py
```

一键流程会自动选择空闲显存最多的两个 GPU 并行训练；GPU 不足时顺序执行：

```bash
python run_fbit_benchmark.py --steps 200000
```

可用 `--devices 4,5` 明确指定物理 GPU，或用 `--devices cpu` 强制 CPU。

## 输出

```text
checkpoints/fbit_base_blind_final.pt
checkpoints/fbit_serial_blind_final.pt
checkpoints/*_resume.pt
training_logs/fbit_<topology>_blind/
results/cache_v2/bench_FBiT.csv
results/benchmark_with_fbit.csv
```

部署和论文评估使用 `*_final.pt`；`*_best.pt` 只是训练诊断。缓存按
`ScenarioKey + Seed` 幂等续跑，`--force` 才重算所选场景。

## 验证

快速测试：

```bash
PYTHONPATH="$PWD" python -m pytest -q -m "not slow"
```

包含 CPU 单 episode 训练 smoke test：

```bash
PYTHONPATH="$PWD" python -m pytest -q
```

完整实验验收时，`bench_FBiT.csv` 应有 260 个唯一场景—seed 行且无失败；
`benchmark_with_fbit.csv` 应有 26 行，并包含 `FBiT_*` 指标。

## 统计分析

完成完整评估后，可运行配对统计分析：

```bash
python scripts/statistical_analysis.py
```

脚本按 `ScenarioKey + Seed` 配对，以场景为统计块，输出场景块置换检验
p 值、两层 bootstrap 95% 置信区间、场景级配对 Cohen's dz 效应量以及
Holm 校正后的 p 值。结果位于 `results/statistical_analysis/`。
