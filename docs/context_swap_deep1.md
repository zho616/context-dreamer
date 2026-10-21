# Deep 条件注入与交换权重1：准备记录

用户于2026-10-20要求停止0.1实验、改交换权重1、参考DALI Deep条件注入。
旧runner12276/训练12298均已退出，status为interrupted；最后日志47082步，
最后周期latest40002步。中断不会把最后内存权重自动另存；未声称47082步可恢复。

新配置：[metadrive_context_swap_deep1.yaml](../configs/metadrive_context_swap_deep1.yaml)。
准备快照：`logdir/context_swap_deep1_prepared_20261020/prepared_source`。
仅准备及测试，未重启环境训练。模型输入尺寸改变，不能strict续接旧recurrent检查点。

## 注入范围

依据 [DALI §4.3及Algorithm 3](https://arxiv.org/html/2508.20294v1#S4.SS3)：

- 循环：将detached context与随机隐变量拼接进入原block-GRU输入投影。
- prior/posterior：通过h间接条件化，不直接追加context。
- reward/continue：直接输入`[RSSM feature; detached context]`。
- Actor/Critic/slow Critic：直接输入同一步的上述条件特征。
- 观测预测：DreamerV3 decoder接收context；R2Dreamer无重建decoder，
  对应将context加入Barlow latent→embedding projector，作为本项目的适配。
- replay、真实动作选择、想象及终止判定均使用对应时刻context。
- 保留原动态想象context更新；DALI伪代码使用fixed rollout，这里不移植该设置。
- context编码器仅由原多步预测与交换损失训练。

这次采用Deep注入范围，不移植DALI跨模态损失或全部算法。
本地参考：`/home/server/projects/DALI/dreamerv3_compat/dreamerv3/nets.py`的
`img_step/obs_step`，以及`contextual_mbrl/dreamer/configs.yaml`的pgm_ctx配置。
代码中adv配置额外直接条件化prior/posterior，不属于论文§4.3的基本Deep定义，未采用。

## 损失

`L_context = 1 * L_aux + 1 * L_swap`，权重相等不代表两个数值或梯度相等。
默认通用配置`swap_weight=0`继续兼容旧路径，新实验配置显式设1。
配对规则仍为完整历史、同实际车型/质量/最大转角、跨真实episode。
当前batch无配对时L_swap为0，不能解释成交换预测完美。

原L_aux为最多5步、每步detach、固定锚点context的归一化L1：
`mean_valid(sum_channels(abs(pred-target) / [10,1,30,pi/3]))`。
0.09是无量纲加和均值，不是0.09m/s²、也不是所有状态9%的相对误差。

新增原单位MAE指标，记录最多5步合并及单步：

- `context/mae_accel_mps2`、`one_step_mae_accel_mps2`
- `context/mae_yaw_rate_radps`、`one_step_mae_yaw_rate_radps`
- `context/mae_speed_mps`、`one_step_mae_speed_mps`
- `context/mae_wheel_angle_rad`、`one_step_mae_wheel_angle_rad`

指标不参与损失、不修改归一化。旧日志没有这些通道误差，不能从一个总标量还原。

## 验证

42项测试通过：Deep在R2Dreamer和DreamerV3下完整梯度/更新，
输入尺寸、prior/posterior间接条件、context梯度隔离、交换损失整合、
预测头新context列首步readout更新后的梯度、物理MAE与归一化损失一致。
另进行CUDA FP16 autocast的Deep+swap1梯度/优化器更新检查。
未启动新500K实验。
