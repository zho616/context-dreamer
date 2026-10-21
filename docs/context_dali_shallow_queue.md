# DALI Shallow参考注入：后续500K实验

用户于2026-10-21授权在当前实验后新增Shallow参考注入实验。
前置为`context_swap_dali_joint_500k_20261020_run1/training/status.json`，
等待它的训练和best独立评估全部完成，再从头启动本组。不修改运行中冻结源码。

## 注入定义

采用[DALI论文§4.3](https://arxiv.org/html/2508.20294v1#S4.SS3)的posterior-only Shallow定义：

`q(z_t | h_t, observation_embedding_t, detached c_t)`。

context追加到观测embedding后，与h一同进入posterior首层；只有该层增加8个输入维度。
循环网络仍为原R2Dreamer block-GRU，输入为原h/z/action，循环更新不直接输入context。
prior、reward、continue、R2 latent→embedding projector、Actor/Critic/slow Critic
均不直接输入context；它们可以经posterior随机状态间接利用历史context。

想象中没有新观测，因此context不直接进入想象转移；起始posterior已携带其影响。
沿用现有辅助状态递推/历史更新代码，但不将递推context额外注入任何想象预测头。
真实交互及replay训练的posterior均使用对应时刻context；reset时context清零。

本组隔离注入位置：编码器仍只由原辅助预测损失和交换损失训练，主模型入口detach。
不移植DALI的cross-modal损失或Shallow的额外context训练协议。
这是论文定义的注入位置参考，不能称完整DALI实现；本地官方源码不同ctx配置和
`dcontext/add_enc_ctx`开关组合还涉及额外循环路由，不盲目混用这些路径。

## 训练设置

- injection=dali_shallow；actor_critic=false。
- 500K环境步，从头训练；seed0，8环境，ratio512，batch16×64，CPU replay500K。
- state4/token6/K5/context8，原MLP残差预测器及动态辅助历史更新。
- 原多步预测损失权重1，单步同底盘跨真实episode交换损失权重1。
- 保持DEFAULT/M/L及75个车型/质量/最大转角组合、paper reward。
- 每10K验证20回合选best，每20K保存latest。
- 结束后best在新地图/S/XL各独立100回合。
- 实验根：`logdir/context_swap_dali_shallow_500k_20261021`。
- TensorBoard6014：`11_500K_state4_DALI_shallow_swap1`。

本组与Deep组的区别是注入范围，不能单独检验交换损失作用；
若需损失消融，须固定注入方式比较swap_weight0/1。

## 验证

44项测试通过，包括两个算法的Shallow完整更新、posterior直接context依赖、
同输入下循环和prior不依赖context、预测头/Actor/Critic无额外输入、
reset清零、主模型对context停止梯度以及posterior新增context列的梯度。
另执行Shallow+swap1的CUDA FP16前反向及优化器更新检查。
