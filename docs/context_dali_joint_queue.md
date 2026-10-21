# DALI Deep 联合输入：后续500K实验

用户授权在当前Deep/交换权重1实验完成后追加实验，当前进程不修改。

## 条件输入

以DALI本地官方实现`dreamerv3_compat/dreamerv3/nets.py:352–369`为依据：

```text
concat(flatten(previous stochastic state), previous action, detached context)
 -> bias-free Linear
 -> LayerNorm(eps=1e-3)
 -> SiLU
 -> recurrent update
```

新版注入名为`dali_deep_joint`。删除旧版三路输入投影，改为上面的联合投影，
投影宽度仍使用R2Dreamer hidden=256。循环部分继续采用R2Dreamer的block-GRU：
联合输入广播到各块，与各块原始h拼接后经原块隐藏层和门控。
第一块隐藏投影的输入宽度相应调整，不保留无用的旧输入参数。

沿用Deep对reward/continue、Actor/Critic/slow Critic的直接context拼接。
R2Dreamer没有重建decoder，仍对其latent→embedding projector追加context。
prior/posterior无额外context列，通过h间接条件化。
context在主模型各入口detach，只由原多步辅助预测和交换损失训练。

这更接近DALI的条件输入方法，不是完整DALI复现：
保留block-GRU/RMSNorm门控骨干、R2表征损失、现有历史编码器和动态context rollout。
DALI使用其普通GRU及fixed-context想象；本轮不修改这些设置。
因此与当前实验比较的是整个循环输入路径，不能只归因于拼接顺序。

## 实验与队列

- 从头500K步，seed0，8环境，ratio512，CPU replay500K，batch16×64。
- 原多步辅助损失权重1，交换损失权重1，state4/token6/K5/context8。
- 车型/质量/转角分布及paper reward与前一组相同。
- 每10K验证20回合，每20K保存latest；结束后best在新地图/S/XL各100回合。
- 实验根目录：`logdir/context_swap_dali_joint_500k_20261020_run1`。
- 前置状态：`logdir/context_swap_deep1_500k_20261020/training/status.json`。
- 使用runner的`--after-status`，等待前置phase=completed（包含独立评估完成）。
  前置失败/中断/进程消失时停止队列，不启动训练。
- 本组源码提前冻结到prepared_source，然后由runner复制到training/source；
  后续工作区修改不会影响已排队源码。
- TensorBoard6014显示`10_500K_state4_DALI_joint_swap1`。

## 验证

43项测试通过，含两种算法的完整更新、Deep头/交换损失联合训练，
联合投影的逐列输入顺序和梯度、reset清零、prior无直接context依赖。
另通过联合路径CUDA FP16前反向与context梯度隔离检查。
本组没有在前置训练期间启动环境训练。
