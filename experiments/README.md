# 实验记录

`snapshot_20261021/manifest.json` 记录每份归档文件的来源路径、大小、SHA256 和快照时间。
训练中的 JSONL 仅保留完整行；这是一份静态快照，后续训练不会自动同步到 GitHub。

| 标识 | 实验 |
| --- | --- |
| 01 | R2Dreamer，无 context |
| 03 | state4，recurrent 注入，无交换损失 |
| 09 | deep 注入，交换损失权重 1 |
| 10 | DALI joint deep 注入，交换损失权重 1 |
| 11 | DALI shallow posterior 注入，交换损失权重 1 |
| dreamerv3 | 另一个 DreamerV3 基线 |
| T_mass_friction | T 路口上的 R2Dreamer baseline / 11，从头训练 |

前六组为环岛实验。T 路口实验使用 default/M/L，质量和 wheel_friction
分别取标称值的 0.6–1.4 倍，间隔 0.1；最大前轮转角固定为各车型标称值。
车流密度 0.2。T baseline 已完成，11 在本次快照时仍在训练，详见归档的 status.json。
提前终止的旧 T 路口实验不作为这组正式结果。

`runs/` 保存原始训练 metrics.jsonl、Hydra 配置和可用的逐回合评估记录。
`analysis/` 保存世界模型、context 交换、驾驶稳定性报告及主要图表。
`T_validation/` 保存当前 T 实验状态和已有验证记录。
历史配置中的绝对路径仅用于溯源，迁移后应设置新的 logdir。

训练曲线使用 `episode/eval_*`，TensorBoard EMA=0.99，并做初始偏差校正；
不是训练 episode/score。单训练种子，图中不绘制跨种子置信带。
独立驾驶评估与训练中的 20 回合验证分开记录。
全回合展示为按用户要求事后选择的案例，不能代替整体统计。
03 与 09 同时改变注入方式和损失，不能将差异单独归因于交换损失。
辅助响应预测误差也不能直接等同于 RSSM 未来观测重建精度。

权重、TensorBoard event、回放缓存、原始轨迹 NPZ 和 MetaDrive assets 未上传。
因此这里可以审查配置、数值和已有图表；重新运行策略评估还需要对应 checkpoint，
部分分析还需要原始轨迹/缓存语料。当前源码是最新实现，历史 frozen source 未逐份打包。
要做新的独立快照，可在有完整 logdir 的原机运行：

```bash
python scripts/export_experiment_records.py --output experiments/snapshot_NEW_DATE
```
