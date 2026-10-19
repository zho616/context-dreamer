# MetaDrive 环岛训练（LiDAR）

使用已有的 Conda 环境：

```bash
conda activate r2dreamer
cd /home/server/projects/r2dreamer
python train.py --config-name metadrive
```

## 输入、场景与资源

- 单智能体连续控制，动作 `[steering, throttle/brake]`，范围 `[-1, 1]`。
- 使用 MetaDrive 的 `LidarStateObservation`：240 束、50 米范围的环绕 LiDAR 扫描，
  加 19 维车辆状态与导航，共 259 维 `float32`。这里“雷达”指模拟 LiDAR；
  不提供毫米波雷达的多普勒速度或 RCS。`num_others=0` 不附加周围车辆的显式状态。
- 使用 MLP 编码器，保留 R2-Dreamer 表征损失；不启用 CNN、相机或窗口渲染。
- 地图 `block_sequence: O`，包含初始直路和双车道环岛。其他车辆由 MetaDrive 控制。
- 默认训练地图种子 `0–99`，评估 `10000–10019`；全局 `seed` 同时偏移这两个区间。
  各进程有独立、可复现的场景采样 RNG。
- 默认交通密度 `0.2`，交通模式 `trigger`；每次决策执行 5 个 0.02 秒物理步，
  即 10 Hz 控制。`action_repeat=1`；每回合最多 1000 个决策，训练预算 50 万个决策（所有训练进程合计）。
- 默认 8 个训练进程、2 个评估进程，`size12M` 模型、CUDA 训练。
  每 10000 个合计训练 steps 评估 20 个回合，每个评估进程计入 10 回合；
  训练开始时也评估一次。统计按 20 个回合平均，评估数据不写入训练回放。
  `env.eval_episode_num` 设置回合总数，`env.eval_env_num` 设置评估并行数，
  `env.env_num` 设置训练并行数。每次评估重置采样序列，两个进程交错遍历相同的 20 个评估种子。
  经验回放放在 CPU，容量 50 万条，包含状态和潜变量，避免大量占用 16 GB 显存。
  `train_ratio=512`，即默认 16×64 批次下，每 2 个合计环境 steps 调用一次更新。
  回放存储会按最大容量分配内存。
- 到达目标、碰撞或驶离道路是真正终止；时间上限是截断，允许价值自举。

MetaDrive LiDAR 模拟周围车辆的扫描；车道边界、道路和目标路线信息由状态与导航部分提供。
官方说明：[Observation](https://metadrive-simulator.readthedocs.io/en/latest/obs.html)。
地图生成说明：[Environments](https://metadrive-simulator.readthedocs.io/en/latest/rl_environments.html)。

## 当前论文对齐奖励与随机出发

当前默认 `reward.mode=paper`，每步奖励为
`R = Δs + 0.1 × (v / 50 km/h) + R_term`。`Δs` 使用导航参考车道上的有符号纵向位移，
不使用旧加法奖励的朝向二值门控，也没有横向惩罚或额外时间惩罚。
到达目的地时 `R_term=+10`，碰撞时为 `-20`，驶出道路时为 `-10`；
事件项加到当步进度和速度奖励上，不覆盖它们。事件优先级：成功、碰撞、驶出道路。
时间上限截断没有额外终止奖励。车辆最大速度配置为 50 km/h。

`spawn.randomize=true` 时，每个训练回合随机选择四个入口之一、两条车道之一、
车道纵向位置 5–25 米（离车道末端至少 5 米）、横向偏移 ±0.5 米，
以及其他三个出口之一。同一个训练地图再次被抽中时，起终点组合和位置也会重新采样。
评估保留相同的随机出发规则，但每个地图种子的路线/出发位置固定，便于比较 checkpoint。
20 个固定评估场景已验证覆盖全部四个入口。

多入口模式会在回合开始激活原生 trigger 生成的环岛交通，保留密度及生成位置，
避免只有原始入口能够触发交通流。生成交通后会清除距离自车初始位置不足 8 米的背景车，
防止随机出发与原生交通摆放重叠，产生第一步碰撞。旧训练配置没有 `spawn` 字段时仍采用原始入口行为。

最大步数 500k、replay buffer 500k（CPU）、train_ratio 512，
训练并行环境 8，评估并行环境 2、每轮 20 回合、间隔 10k 步。
模型损失权重不变。已保存 checkpoint 的测试默认使用其自身训练配置。

## 之前的加法奖励与换道

奖励通过本项目的 `RoundaboutEnv` 子类实现，不修改共享的 MetaDrive 源码。
`reward.mode=additive` 使用下述加法奖励；`reward.mode=metadrive` 调用安装版本的原生奖励函数，
无独立横向惩罚，终止事件覆盖当步密集奖励。
`R = R1 + R2 + R3 + R4`，权重在 `configs/env/metadrive_lidar.yaml` 的 `reward` 中配置：

- `R1 = progress × Δs × I_pos`，默认 `progress=1.0`。
- `R2 = speed × (v/v_max) × I_pos`，此前 `speed=0.1`，车辆最大速度 `v_max=80 km/h`。
- `I_pos` 为 0 或 1，由车辆朝向与所选参考车道切向是否同向决定。
- `R3 = -lateral × |y| / lane_width`，此前 `lateral=0.2`。
  这是独立扣分，车辆停着且偏离中心时也生效。参考车道优先使用车辆当前所在的导航车道，
  因而不会把相邻同向车道的中心当作横向偏离。
  换道跨越中间线时 `|y|/lane_width` 约为 0.5，单步惩罚约 0.1。
  在 20 km/h、0.1 秒控制周期下，正常前进与速度奖励约 0.58，仍有正收益；
  该初始权重用于温和鼓励居中，不禁止合法超车，是否学会超车仍需实验验证。
- `R4` 在普通步骤为 0，到达目标为 `+10`，碰撞为 `-5`，驶离道路为 `-5`。
  事件优先级为成功、碰撞、驶离道路；同时发生碰撞和驶离道路只扣一次。
  事件项与前三项相加，不覆盖基础奖励。超时没有额外惩罚。

同向并行车道之间跨虚线换道原本就不会单独终止，真实模拟器测试已验证。
`on_broken_line_done=false`、`out_of_route_done=false` 保留此行为；
`on_continuous_line_done=true` 仍限制压黄/白实线、撞路缘，离开有效道路及碰撞仍终止。
这些开关位于 YAML 的 `termination` 中。

新奖励与此前使用的 MetaDrive 默认奖励不同，跨实验比较时应统一各方法的奖励和终止定义。

## 验证和短训练

```bash
python -m pip check
python -m unittest discover -s tests -v
python train.py --config-name metadrive \
  logdir=logdir/metadrive_smoke env.steps=256 \
  env.env_num=2 env.eval_episode_num=1 env.time_limit=64 \
  batch_size=4 batch_length=8 env.train_ratio=8 \
  buffer.max_size=512 model.compile=false \
  trainer.eval_every=128 trainer.update_log_every=64
```

短训练验证多进程采样、评估、CPU 回放、GPU 世界模型/策略更新和 `latest.pt` 保存，
不代表策略已学会通过环岛。正式配置默认开启 `torch.compile`，首次运行有编译开销；
可用 `model.compile=false` 关闭。

## 后续训练与监控

```bash
# 固定日志目录
python train.py --config-name metadrive logdir=logdir/roundabout_lidar_500k_ratio512_lateral

# 无交通的起步实验
python train.py --config-name metadrive env.traffic_density=0 logdir=logdir/roundabout_no_traffic

# 增加交通密度
python train.py --config-name metadrive env.traffic_density=0.2 logdir=logdir/roundabout_traffic02

tensorboard --logdir logdir
```

评估日志包含 `episode/eval_success`、`eval_crash`、`eval_out_of_road`、
`eval_route_completion`、`eval_cost` 和 `eval_score`。
另外记录 `eval_off_lane`、`eval_yellow_line`、`eval_white_line`、`eval_crash_sidewalk`，
用于细分失败原因；`eval_reward_progress/speed/lateral/terminal` 记录各奖励项的回合累计均值。
成功率、碰撞率、驶离道路率按回合统计；路线完成度取回合结束时的值；成本按回合求和。
每次评估后按以下顺序比较并保存日志目录下的 `best.pt`：成功率越高越好；
成功率相同则比较路线完成度；两者相同再比较平均回合奖励。指标完全相同不覆盖。
使用原子替换写入，检查点包含模型、优化器、评估指标和训练步数。
训练结束还会评估一次，纳入最佳模型选择；`latest.pt` 继续保存最终模型。
环岛配置 `trainer.save_best=true` 开启此功能。使用不同实验目录，避免覆盖旧实验模型。
目前没有恢复训练入口；这些检查点也不包含经验回放。

## 顺序对比 R2-Dreamer 与 DreamerV3

```bash
python scripts/train_metadrive_compare.py --logdir logdir/compare_roundabout_500k
```

先训练 `r2dreamer`，成功完成并保存检查点后自动训练 `dreamerv3`。
后者使用本仓库的 PyTorch DreamerV3 复现，配置为 `model.rep_loss=dreamer`。
两者都从随机初始化开始，使用相同 seed、奖励、固定评估场景、500k 步预算、
`train_ratio=512`、500k CPU 回放、8 个训练进程和 2 个评估进程；不继承前一个方法的参数。
运行前保存源码和配置快照，两个方法都从该快照执行，避免中途编辑影响对比。
中断或失败会停止队列，不启动下一个方法。

实验根目录的 `status.json` 记录运行阶段、进程 PID、命令和依赖版本；
两个方法分别在 `r2dreamer/` 和 `dreamerv3/` 中保存 TensorBoard、`best.pt`、`latest.pt`。
TensorBoard 指向实验根目录即可同时显示两条 run；尚未开始的方法会先显示排队状态文本。
要验证整个队列，可使用 `--smoke`，它会缩短预算并关闭编译。

仅用原 MetaDrive 奖励重训 DreamerV3：

```bash
python scripts/train_metadrive_compare.py \
  --methods dreamerv3 --reward-mode metadrive \
  --logdir logdir/dreamerv3_original_reward_500k
```

这项消融保留原 DreamerV3 模型损失、500k 预算、512 训练比例和固定评估场景，
从随机初始化开始，不继承此前检查点。两种奖励定义下的累计奖励不能直接用于算法排名，
应重点比较成功率、路线完成度和失败类型。

在当前基线完成后顺序验证两个独立因素：

```bash
python scripts/train_metadrive_compare.py \
  --ablations ratio256 buffer200k --reward-mode metadrive \
  --source-from BASELINE_DIR/source --after-status BASELINE_DIR/status.json \
  --logdir logdir/dreamerv3_controlled_ablations
```

`ratio256` 只将训练比例由 512 改为 256，回放保持 500k；
`buffer200k` 只将回放容量由 500k 改为 200k，训练比例保持 512。
两组世界模型损失及权重都与当前基线完全相同，不叠加这两个修改。
每项独立随机初始化，均为 500k steps、seed=0、原 MetaDrive 奖励，
其余源码/配置从基线快照复制。先等待基线成功结束，再依次训练两项，前项失败则停止。
最佳指标会写入队列 `status.json`。单种子结果用于初步诊断，不能替代多种子统计。

RSSM 的 `Deter.forward` 已将输入动作幅度约束到 [-1, 1]，真实和想象动态都经过此处，
因此不再把“原始采样动作超界”单独视为已确认的动态输入不一致，也不为此增加重复实验。

## 依赖与复现

本机复用 `/home/server/anaconda3/envs/r2dreamer`。
MetaDrive 已以 editable 方式安装自 `/home/server/projects/metadrive`，版本 `0.4.3`，
源码提交 `f0a18fdf55c720e6e3023e97fae8b76beb652846`；资产版本 `0.4.3`。
没有改动这份共享 MetaDrive 源码。
关键依赖：Python 3.11、PyTorch 2.8.0 + CUDA 12.8、NumPy 1.26.0、
TorchRL 0.9.2、TensorDict 0.9.1、Panda3D 1.10.16、Gymnasium 1.2.0、TensorBoard 2.21.0。

接入验证：4 项测试通过，覆盖真实 LiDAR/环岛/种子隔离、真实驶离道路终止、
动作重复中途截断和奖励/成本累加。256 步短训练完成，检查点参数全部有限，
优化器实际更新 52 次；日志保存在 `logdir/metadrive_smoke_extended`。
此前 4+4 进程、16×64 批次及 `torch.compile` 也完成了 320 步验证，
检查点参数全部有限；日志保存在 `logdir/metadrive_smoke_default`。

在已安装 MetaDrive 的环境中，可安装本项目及所需额外依赖：

```bash
python -m pip install -e '.[metadrive]'
```

在新机器上，优先按官方建议安装 MetaDrive 源码并固定提交，然后安装本项目。
`opencv-python<4.12` 避免新版 OpenCV 对 NumPy 2 的要求与本项目 NumPy 1.26 冲突。
第一次使用需要 MetaDrive 资产；本机已有。
官方安装说明：[Install MetaDrive](https://github.com/metadriverse/metadrive/blob/main/documentation/source/install.rst)。
