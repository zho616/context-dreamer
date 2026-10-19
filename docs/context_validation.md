# Context migration validation — 2026-10-15

## 正式训练启动时的编译修复

- 首次 size12M / batch16×64 正式训练在首次更新时触发 PyTorch 2.8 Inductor 错误：
  FP32 context attention 被降低为 FP16，而输出投影权重仍为 FP32；此前小尺寸编译测试未触发。
- `HistoryEncoder.forward` 改为 `torch.compiler.disable`，保留现有 FP32 autocast 作用域、
  参数、数学结构与辅助梯度；主世界模型仍按原配置 compile/AMP。
- 修复后31项测试再次通过，两种方法的小模型各12次 GPU 编译完整更新通过。
  日志 `/tmp/context-training-fp32-tests.log`、`/tmp/context-training-fp32-compile.log`。
- 失败目录 `logdir/context_chassis_500k_20261015_102744` 保留；修复后新建
  `logdir/context_chassis_500k_20261015_103141`，两种 context 方法各500K步、seed0、串行执行。
  完整配置、冻结源码和运行状态均存于实验目录。

## 循环输入条件化：当前默认 recurrent_concat

- 默认由 `adarms_deter` 改为 `recurrent_concat`，迁移 cRSSM / DALI 官方 RSSM 循环输入拼接机制。
  在本仓库 block-GRU 中扩展 stochastic 输入投影；不改原门控和策略输入维数。
- 两个旧模式仍可显式选择；已有 checkpoint 按各自保存的配置恢复，注入模式之间不可直接 strict 互载。
- 31项单元测试通过，覆盖两种方法的 baseline、循环拼接、head拼接、AdaRMS 训练通路。
  新增 context 改变 h/prior/posterior、WM梯度更新 context 投影且不反传 E、过滤/想象一致及 reset 隔离检查。
- context 关闭时，与迁移前快照的556项初始参数、指标和梯度逐项 `torch.equal` 完全一致。
- 两种 context 方法的小模型 `torch.compile(reduce-overhead)` 各12次 GPU AMP 完整更新通过；
  所有指标有限、辅助 D 参数实际更新。未测试默认全尺寸编译性能。
- 三车型随机底盘、size12M、四组各96环境步真实 MetaDrive 短训练全部正常退出，包含评估和 checkpoint。
- 两种新 context 模型的真实训练 checkpoint 均 strict 恢复成功、优化器状态非空，恢复后动作有限；
  日志 `/tmp/recurrent-concat-restore.log`。
- 记录：`/tmp/recurrent-concat-tests.log`、`/tmp/recurrent-concat-compile.log`、
  `/tmp/recurrent_concat_chassis_smoke_20261015/status.json`。
- 这些验证说明运行通路可用，不构成驾驶收益、长期训练稳定性或底盘泛化的实验证据。
  已完成的500K随机底盘基线不含 context，不为本次新默认注入提供性能结论。

## 三车型离散底盘随机化

- 训练池default/m/l；质量和转向scale独立从0.8/0.9/1.0/1.1/1.2选择，最终物理参数为整数。
- 30项测试通过，覆盖原有路径、实际刚体质量、实际车轮角度、参数整局恒定、物理控制响应、
  地图/底盘采样独立性、重置复用及s/xl独立评估池。
- 对评估中的log_chassis常量按每局终点值统计，避免按步累加；不作为网络输入或重建目标。
- R2Dreamer/DreamerV3各自关闭/开启context的四组size12M随机底盘短训练各96步，包含评估与checkpoint保存。
- 记录：`/tmp/r2-chassis-four-arm-final/status.json`、`/tmp/chassis-all-tests-final.log`。
- 底盘随机化为独立开关，原metadrive/metadrive_context配置仍默认关闭。
- 尚未据此开展500K长训练或得出驾驶性能/底盘适应收益结论。

## 两维状态 + AdaRMS 更新

- 状态/预测目标 `[a_long, yaw_rate]`，K5 token4，context8。
- 25项测试通过（原19项+context6项），覆盖两种方法与两种注入方式。
- 新增调制零初始化与原Deter精确等价、prior/posterior条件响应、梯度隔离及reset一致性检查。
- context关闭后，再次与迁移前快照比较556项参数/指标/梯度：全部完全一致。
- 默认size12M四组各96步真实MetaDrive短训练完成，两维context使用adarms_deter。
- 两种context方法真实checkpoint strict恢复并成功输出动作；全部参数有限，零初始化调制权重已实际更新。
- 两种context方法小模型 `torch.compile(reduce-overhead)` 各12次GPU AMP更新通过，指标有限且D参数实际改变。
  日志 `/tmp/context2-compile.log`，尺寸同下方历史编译测试；未测默认全尺寸编译性能。
- 本轮记录：`/tmp/r2-context2-four-arm-smoke/status.json`、`/tmp/context2-all-tests.log`、`/tmp/context2-restore.log`。
- 两维旧拼接对照通过单元训练更新；未另跑长训练或作性能排序。

## 前版六维 + prior/posterior 拼接（历史记录）

环境：`/home/server/anaconda3/envs/r2dreamer/bin/python`，PyTorch 2.8，MetaDrive，CUDA。

| 检查 | 结果 |
|---|---|
| `python -m unittest discover -s tests` | 24 tests passed，包括原有19项及新增5项 |
| 修改前快照 vs context 关闭 | R2Dreamer/DreamerV3 初始化 state dict、loss/metrics、梯度共556项 `torch.equal` 一致 |
| 真实 MetaDrive context 采样 | reset空历史，连续8步物理量有限，K5右对齐，动作与目标状态一致 |
| GPU AMP | 两种方法，各自开/关 context，完整 update 输出有限 |
| `torch.compile(mode='reduce-overhead')` | 两种 context 方法，小模型各12次 update；指标有限且D参数实际改变 |
| 默认 size12M 四组短训练 | 每组96环境步；R2Dreamer、R2Dreamer+context、DreamerV3、DreamerV3+context全部正常退出 |
| checkpoint 恢复 | 两种 context 方法的真实训练 latest.pt 均 strict 加载、优化器状态非空、动作输出有限 |
| `git diff --check` | 通过 |

四组运行包含并行环境、CPU replay、GPU训练、episode reset、评估、best.pt/latest.pt保存。
短训练参数：2环境，batch4×8，imag_horizon15，train_ratio8，time_limit16，compile=false。
编译测试使用 deter32 / hidden16 / stoch4 / discrete4、batch2×7、imag_horizon3；
不等价于默认大批次全尺寸编译性能测试。

运行记录位于 `/tmp/r2-context-four-arm-smoke/status.json`，各子目录有 metrics.jsonl、console.log及checkpoint。
补充日志：`/tmp/context-all-tests-final.log`、`/tmp/context-compile.log`、`/tmp/context-restore.log`。
临时文件会被系统清理；本记录保留验证范围。尚未开展多seed长训练、驾驶收益比较或JAX权重数值对齐。
