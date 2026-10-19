# 两维 context 的 RSSM 条件化：cRSSM / DALI 路线

早期辅助分支参考：[DynaDreamer arXiv:2607.13410v2，2026-09-11](https://arxiv.org/html/2607.13410v2)。
核对范围为原文 IV-A、IV-B、IV-C；这里实现的是适配，不是论文全系统复现。
当前默认注入已改用下文 cRSSM / DALI 的循环条件化路线；DynaDreamer 的调制不再默认启用。
原文用物理参数识别和 ODE 辅助预测训练 context，并通过 detached context 对 Transformer
做 AdaLN 调制；世界模型保留 prior/posterior KL（式12），context 本身没有新增 VAE KL。

## 为什么不直接新增 KL

当前 E 是确定性的 `c=E(history)`，D 预测未来 `[a_long, yaw_rate]`，不是对输入历史的重建。
没有定义 `q(c|history)` 的概率密度、方差、重参数采样或 context prior，不能直接计算 VAE 式 KL。
对向量套 softmax 再算 KL，会人为改变潜变量语义；也不能把 context8 与 RSSM categorical z 直接算 KL。

如以后明确需要概率信息瓶颈，可另做 `q(c|H)=N(mu,diag(sigma²))`、重参数采样、
`beta*KL(q||N(0,I))` 的消融，同时校准预测似然和 beta。该正则会压缩信息，
并不自动使真实/想象 context 一致；较大权重可能压掉需要的动力学差异，因此本版不加。

保留的训练目标：原方法损失 + `aux_weight * L_aux`。
原 RSSM 中 `dyn=KL(sg(q)||p)`、`rep=KL(q||sg(p))` 保留 free-nats 及原权重。
在循环条件化下注入时，两者通过共享 h 更新 context 输入投影参数；context 已 detach，E/D 仍只接受 L_aux。
这已经在世界模型层面做分布协调，但不保证 context 预测误差自动消失。

## 注入选择与具体实现

默认 `model.context.injection=recurrent_concat`。依据：

- [cRSSM / Dreaming of Many Worlds §4.1.1](https://arxiv.org/html/2403.10967v1#S4.SS1.SSS1)：
  context 进入确定性递归 `h_t=f(h_(t-1),z_(t-1),a_(t-1),c)`。
- [DALI §4.3 Deep Integration](https://arxiv.org/html/2508.20294v3#S4.SS3)：
  学习到的 context 进入递归动力学；其全系统还直接条件化各预测头和 actor/critic。
- 官方源码均在 `RSSM.img_step` 将 `[prev_stoch, prev_action, dcontext]` 拼接后送入 GRU 输入层。
  核对版本：[cRSSM ee8293d](https://github.com/sai-prasanna/dreaming_of_many_worlds/blob/ee8293d0ea6fc120619403961b225e16006cb9fb/dreamerv3_compat/dreamerv3/nets.py#L146)、
  [DALI 5ca3934](https://github.com/frankroeder/DALI/blob/5ca393468defffe79e4e3537ba9766f6d31b510b/dreamerv3_compat/dreamerv3/nets.py#L352)。

当前仓库将 h、z、a 分别投影，再经 block-GRU，因此采用以下适配：

```text
c_t = stop_gradient(E(history_t))
u_h = activation(RMSNorm(W_h h_(t-1) + b_h))
u_zc = activation(RMSNorm(W_zc [flatten(z_(t-1)); c_t] + b_zc))
u_a = activation(RMSNorm(W_a normalized(a_(t-1)) + b_a))
h_t = BlockGRU(h_(t-1), [u_h; u_zc; u_a])
p(z_t | h_t), q(z_t | h_t, embedding(o_t))
policy = pi(a_t | h_t, z_t)
```

只扩展原 stochastic 输入投影的输入维度8；projection 和原 GRU 由世界模型损失训练。
context 在拼接前 detach，E/D 仍只由辅助动力学预测损失更新。
动作保持原幅值裁剪；context 不经过动作裁剪。空历史/episode reset 的 context 为零。
prior/posterior 通过条件化的 h 获取动力学信息，策略也通过 h/z 间接获取。

这是迁移已有工作的**循环输入条件化机制**，不是完整复现 cRSSM 或 DALI。
cRSSM 使用可观测的真实环境参数；这里使用历史学习的、随时间更新的 context。
DALI Deep 还显式条件化预测头和 actor/critic；这里暂保留这些接口，避免同时改变多处。
官方实现是普通 GRU，本仓库是分别投影的 block-GRU；投影布局并非逐层相同。
已有实验为选择这条机制提供依据，不能直接证明本项目中的收益或稳定性。

旧 `adarms_deter` 和 `prior_posterior` 保留为显式可选方案，旧配置仍可使用。
AdaRMS 是本项目对调制机制的 RSSM 适配，没有已验证的驾驶收益，不再默认启用。

时间约定沿用当前实现：过滤用端点 history_t 得到 c_t，在形成 h_t 时注入；
想象先预测 x_(t+1) 并重编码 c_(t+1)，再执行对应 latent transition。
这是本实现的端点时间约定，不声称逐行复现论文 Algorithm 2 的 rollout 顺序。
过滤用真实端点、想象用预测端点存在分布差异，应该用多步预测和驾驶实验检查。

## 本次明确保留的差异

- 用户指定两维状态；E 仍是 K5 两层 Transformer→8维，D 为纯残差 MLP→2维。
  未移植论文的 GRU context encoder、底盘参数头、轮胎模型或 bicycle ODE。
- 未增加参数识别损失，没有底盘真值监督，不能声称 context 已辨识物理参数。
- 想象终止后沿用前版的清空无效历史策略；论文采用冻结最后 context。未把它混入本次注入消融。
- 普通 observation/actor feature 和基线开关关闭时的结构不变；仅 context 数据契约改变。
- 同步减少监督维数后，L_aux 仍是按有效预测步平均的坐标L1和，scale=[10,1]，
  权重仍为1；不能直接将新旧 loss 数值当作预测质量变化。

## 复现实验

```bash
python train.py --config-name metadrive_context model.rep_loss=r2dreamer model.context.injection=recurrent_concat
python train.py --config-name metadrive_context model.rep_loss=dreamer model.context.injection=recurrent_concat
# 两种方法均可用以下覆盖切换为旧 head 拼接对照：
# model.context.injection=prior_posterior
```

使用新 logdir 与 replay；旧六维 context checkpoint 不能直接恢复到两维模型。
不同 injection 的 checkpoint 结构不同；恢复旧实验必须使用其原配置。
此次仅替换默认注入机制，是否改善底盘自适应仍需正式训练和同预算评估。
