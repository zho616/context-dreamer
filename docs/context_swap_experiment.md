# State4 recurrent：同底盘 context 交换实验

2026-10-20 更新：用户要求停止，本轮训练已中断，最后日志47082步；
最后周期latest保存于40002步。原冻结源码/配置/日志保留，不改成新参数。
后续Deep、交换权重1的准备见 `context_swap_deep1.md`，尚未启动正式训练。

## 基线与范围

对应 TensorBoard `03_500K_state4_recurrent`：
`logdir/context_kinematics_500k_20261015_180340/source`。

正式实验：`logdir/context_swap_state4_20261020_run1/training`。
从头训练，不从 best 权重续训。使用原冻结源码，仅添加交换损失和配对元数据，
另采用当前 trainer/train 的周期 latest 保存逻辑。逐文件 SHA256 见实验
`provenance.json`，模型补丁见 `swap.patch`。

保持 state4=[纵向投影加速度,右正横摆角速度,平面速率,右正前轮角]、
K5/token6/context8、原 Transformer 编码器/MLP残差预测器、
recurrent_concat、动态想象和原五步辅助损失。此次不加入自行车模型，
不更改加速度定义，不添加 Actor/Critic context 条件化。

## 损失与配对

`loss = 原训练损失 + 0.1 * context_swap`。

对于实际转移 `(s_t,u_t,s_{t+1})`，选择同底盘另一真实 episode 的 context：
`L_swap = mean(sum(abs(predict(c_donor,s_t,u_t)-s_{t+1}) / STATE_SCALE))`。
状态尺度与原辅助损失一致，单步监督；当前状态和动作保持不变。
梯度通过供体 context 回到编码器，并更新原预测器。

- 精确匹配 `(model_id,实际mass,实际max_steering)`；此实验其他力/摩擦设置由车型预设固定。
- 严格跨真实 episode；要求供体和受体历史均有完整五步。
- `ctx_episode_uid=[env_id,reset_counter]` 仅作配对元数据。
  原 replay `episode` 是固定 worker 流 ID，不能用于此判定。
- 模型/质量/转向标签仅选择供体，不输入 encoder、predictor、RSSM。
- 排除 reset、terminal、last、collision、非有限状态和跨 episode 目标。
- 在当前 batch 的候选窗口中随机选择供体；每个有效窗口都可作为受体，
  因此是双向的随机交换，而不是每次强制成对互换。
- 没有跨 episode 精确匹配时该受体不产生交换损失；不放宽底盘条件。

默认 `model.context.swap_weight=0`，不启用新损失；正式实验设为 0.1。
配对随机数会改变整体随机数流，不能把相同 seed 理解成逐步相同训练轨迹。
旧 replay 缺少真实 episode ID，不可直接用于启用交换损失的训练。

## 训练与检查

500K环境步、seed0、8训练环境、ratio512、CPU replay500K、默认AMP/compile；
车型DEFAULT/M/L，质量和最大转角各5档，共75个配置。
每10K评估20回合选best，每20K保存latest，训练后best在新地图/S/XL各测100回合。
结果与历史单seed基线比较，尚不足以单独支持统计稳健的改进结论。

监测 `train/loss/context_swap`、`train/context/swap_pairs`、
`swap_eligible`、`swap_coverage`、`swap_own_l1`、`swap_zero_l1`、
`swap_gap`。own/zero均在相同配对受体集合上比较。
零context仅作诊断、不训练；低覆盖或模型忽略context须结合指标判断。
跨底盘交换诊断尚未加入在线指标，后续评估应单独比较。

启动前验证：

- `PYTHONPATH=/home/server/projects/metadrive .../python -m unittest discover -s tests`：
  40项通过，包括CUDA AMP交换梯度、动作对齐、同回合拒绝、精确参数匹配、
  缺失配对/无效数据、边界检查。
- 96步真实MetaDrive smoke完成并保存best/latest。
  smoke为保证交换路径被执行，单独使用固定DEFAULT/1.0/1.0底盘；
  有效配对14–22个、覆盖率100%，不代表正式75配置覆盖率或训练效果。

准备命令：

```bash
python scripts/prepare_context_swap_experiment.py --output <fresh_experiment_root>
PYTHONPATH=/home/server/projects/metadrive python scripts/train_metadrive_compare.py \
  --logdir <fresh_experiment_root>/training \
  --source-from <fresh_experiment_root>/prepared_source \
  --methods r2dreamer --context-only --chassis-randomization \
  --context-swap-weight 0.1 --steps 500000 --evaluate-best
```

上述为原0.1实验的历史记录。准备脚本随后已更新，默认准备Deep/权重1，
显式复制当前context相关模块并输出实际差异与SHA256；不能用新版默认值重现旧启动参数。
