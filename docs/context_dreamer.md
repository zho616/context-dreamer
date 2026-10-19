# Context Dreamer：MetaDrive / PyTorch 迁移

来源：Adaptive-Context-Dreamer 的 `research/context-dreamer` 分支，commit
`8e9c19a97ae889df3aaeea0d5e9be19e27817e88`，`migration/context_dreamer_core`。
这是算法结构与训练语义的迁移，不加载旧 JAX 权重，也不声称跨框架数值等价。
只支持当前仓库的 R2Dreamer 与 DreamerV3 (`model.rep_loss=dreamer`)。

三车型整数质量/转角随机化见 [底盘随机化训练](chassis_randomization.md)。

## 启动

```bash
# 两种方法分别启用 context；原 metadrive 配置默认关闭。
python train.py --config-name metadrive_context model.rep_loss=r2dreamer
python train.py --config-name metadrive_context model.rep_loss=dreamer

# 同一份冻结源码依次运行两种基线及各自 context 版本。
python scripts/train_metadrive_compare.py --context-comparison --smoke --logdir /path/to/new-smoke
# 正式实验去掉 --smoke。使用新目录，避免覆盖既有结果。
```

`model.context.enabled=false` 保留基线 RSSM、参数形状和训练路径。
开启后需使用新 checkpoint；基线 checkpoint 与 context checkpoint 不可直接 strict 互载。
同一配置的 context checkpoint 使用原有保存/恢复机制。渲染时应读取训练时保存的配置。

## 数据与时间对齐

环境额外提供 `ctx_ego_state(4)`、`ctx_prev_action(2)`、`ctx_history(5,6)`、
`ctx_mask(5)`、`ctx_collision`，全部随 replay 存储，但排除于常规 encoder/decoder。
四维状态为 `[纵向加速度, 右转偏航角速度, 平面车速, 实际前轮转角]`，
单位依次是 m/s²、rad/s、m/s、rad。转角与横摆角速度均右正。
加速度由世界速度差分后投影至车头方向，避免直接差分旋转中的车体系速度。
角速度取 Bullet 世界 z 轴角速度的负值，适用于本轮环岛平面驾驶。
重置时加速度置零；车速取平面速度模长。前轮转角直接读取 Bullet 两个前轮
`getSteeringValue` 的均值，从左正角度转换为右正弧度，不读取最大转角等底盘标签。
车速、前轮转角在每一步均进入历史，并同步成为辅助预测目标，使 imagination 可以闭环更新。
`dt=physics_world_step_size * decision_repeat`，当前 0.1 秒；暂要求 action_repeat=1。

MetaDrive 原生动作是 `[左转向, 油门/制动]`；辅助模型动作是 `[油门/制动, 右转向]`。
在执行动作后采样：历史 token 为 `[x_t, a_(t-1)]`，重置历史为空，左侧零填充并用 mask 排除。
当前状态、历史、辅助预测标签、想象状态均使用上述四维；主 observation 的 state 不变。

Replay 中 `action` 已向前移位供 RSSM 使用。辅助监督独立使用目标端的
`ctx_prev_action[:, t+1]` 预测 `x_(t+1)`，避免再移位导致错配。
训练时用当前 E 重编码原始历史，不缓存过期 context。

## 网络与损失

- E：K5，每步输入6维；hidden64、4头、两层 Transformer、FF128；masked mean，输出确定性8维。
  固定尺度 `[10,1,30,π/3,1,1]` 仅归一化一次；空历史精确输出零。
- D：输入 context8、归一化状态4、动作2（总输入14维），两层128 SiLU，预测物理尺度残差4。
  下一车速限制为非负，下一前轮角限制在 MetaDrive 通用的 ±60°范围；训练展开与想象一致。
- 辅助损失：最多五步，固定起点 context，逐步 detach 预测状态，归一化 L1 坐标求和后
  按有效预测步平均；允许序列末尾的部分监督。reset、collision、terminal、truncation 不跨越。
- E/D 只接受 `context_aux` 梯度；世界模型使用 detached context。
  默认 `model.context.aux_weight=1.0`，共享原优化器和梯度缩放，但梯度图隔离。
- E 固定以 eager FP32 执行，避开 PyTorch 2.8 Inductor 在正式 batch 下将 attention
  错误降为 FP16 导致的投影类型不匹配；世界模型仍使用原 compile/AMP，辅助梯度仍正常回传。
- 默认 `injection=recurrent_concat`：将 detached context 与上一 stochastic latent 拼接，
  进入原 block-GRU 的输入投影，使共享 h 条件化，进而影响 prior 和 posterior。
  迁移 cRSSM / DALI 的循环输入条件化路线，保留原 GRU 门控、特征维数和 actor/critic。
- `injection=adarms_deter`：保留此前的 RMSNorm scale/shift/gain 调制作为可选消融，
  不再作为默认方案；该 RSSM 适配尚无本项目的驾驶性能证据。
- `injection=prior_posterior`：保留 prior/posterior 输入拼接作为对照。
- Context 确定性编码，不增加 VAE KL；原 RSSM 的 dyn/rep KL 保持，训练世界模型和调制参数，
  不反传 E/D。完整理由、公式及论文对应见 [设计说明](context_dynadreamer.md)。

不同 injection 的 checkpoint 参数形状不同。恢复已有实验应使用它保存的原配置；
新默认方案需新建实验，不能直接恢复旧 AdaRMS checkpoint。

四维输入仍非完整 Markov 车辆状态，未显式包含侧向速度与悬架状态。
本次只补齐已知工况，没有添加参数识别、跨工况一致性或对抗损失；
不能声称 context 已解耦车速/转角或已识别底盘参数。

新版编码器、辅助模型的权重和 replay 形状与此前两维状态版本不兼容，需要新实验。
`env.context_kinematics=false` 恢复旧 `state2/history(5,4)` 格式，模型根据 observation
自动选择相应网络维度，支持旧两维 context 权重的 strict 恢复；也可使用旧实验的冻结源码。
原基线的网络参数形状与训练路径不受影响。

## 想象与诊断

每步先用 D 和策略动作预测下一物理状态，更新 K5，E 生成下一 context，
再把它注入该步 RSSM prior。整个想象过程无梯度，不读取未来真实状态。
预测 continuation 的硬判定用于使历史失效；actor/value 原有的概率 continuation 权重保持不变。
E/D 在 `no_grad` 下读取当前参数，沿用本仓库串行采样/更新执行方式。

`model.context.rollout` 支持 `dynamic`（默认）、`fixed`、`zero`、`shuffled`。
后三者只改变想象时注入 RSSM 的 context；过滤与辅助训练仍正常，D/E 侧支仍推进。
它们只诊断想象时的 context 使用；切换注入位置使用 `model.context.injection`。

监控 `loss/context_aux`、`context/valid_targets`、`context/rms`。
辅助 loss 降低不等价于驾驶性能提升，仍需相同预算、地图划分及多 seed 正式对照。

## 验证

```bash
python -m unittest discover -s tests
```

测试包含物理坐标/符号、重置历史、mask、动作时序、episode 边界、逐步 detach、
想象历史更新、世界模型梯度隔离及两种方法开关 context 的反向传播/优化/动作输出。
迁移实施时另做修改前快照精确基线比较、真实 MetaDrive 采样、GPU AMP 更新和四组短训练。
短训练验证运行完整性，不构成性能结论。
