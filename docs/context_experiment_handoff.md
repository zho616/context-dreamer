# Context Dreamer 实验交接

更新于 **2026-10-16 10:56，Asia/Shanghai**。这是当前交接快照；接手时应重新核对进程和文件，不要仅依据 status.json 判断训练是否仍在运行。

### 续训更新：2026-10-16 15:32，Asia/Shanghai

- **后续用户明确要求停止训练，已执行。** 恢复训练末日志301470步；最新已保存`recovery_latest.pt`为300011步，20回合验证成功率45%；best仍是原140005步、验证成功率70%。runner、训练及环境worker已停止，后续新地图/S/XL独立测试取消。拼接日志已同步到301470步，TensorBoard保留可查看；不要根据此前启动记录误判仍在训练。
- 用户已明确要求继续 Actor/Critic 组；确认原训练进程不存在，原记录末日志302629，但仅有140005步best。
- 从该best恢复模型和LaProp优化器（最大参数更新计数69709），学习率保持4e-5；旧文件未保存replay、环境、RNG、AMP scaler和精确scheduler计数，因此不是302K无缝续训。重新采集replay，按本段新增采样量判断更新就绪，不按全局140K直接采样空buffer。
- 正式恢复目录：`/home/server/projects/r2dreamer/logdir/context_injection_500k_20261016_020630/r2dreamer_recurrent_actor_critic/recovery_20261016_153114`，独立复制冻结source并仅修改恢复/计数/保存逻辑；原source、日志、best及AdaRMS均保留。
- 正式runner PID149073、训练PID149075（需重新核查）；从140005继续到500K，原8环境/ratio512/compile/AMP保持。继承原best选择阈值，每次验证额外保存`recovery_latest.pt`，完成后自动对best运行新地图/S/XL各100回合。状态在恢复目录`status.json`及原队列`status.json`中更新。
- 恢复短测通过：140005→140101，实际CUDA优化步增加、参数变化且全部有限、LR4e-5、检查点保存成功；短测缩小batch并关闭compile，仅验证恢复，正式训练从原始140005重新加载。详见`validation.json`。
- 续训日志是原TensorBoard目录下独立子run，不能将原140K→302K日志尾部与新恢复曲线拼作连续训练。当前不应声称已经完成500K或独立测试。
- 用户随后要求本次不重启、后续每20K保存latest：主仓库新增`trainer.save_every=20000`，跨20K环境步边界后原子覆盖`latest.pt`，训练完成再保存；包含模型、优化器、步数、更新数、best阈值以及scheduler/scaler状态，仍不包含replay/环境。保存与best是否改善无关。两项checkpoint测试通过。本次既有进程继续运行其冻结源码，仍每次验证保存`recovery_latest.pt`，不会热加载主仓库改动。
- 用户要求拼接日志后，创建独立实时视图`logdir/context_injection_500k_20261016_020630/tensorboard_recurrent_actor_critic_stitched`：原日志保留step≤140005，续训接step>140005，不混入原140K→302K丢失权重的分支或smoke；原文件不改。`scripts/stitch_recovery_logs.py`每5秒同步event和metrics.jsonl，PID182127（需核查），状态见视图内`stitch_status.json`。TensorBoard仅重启显示服务，PID182335，6010地址与四个显示名保持，Actor/Critic指向拼接视图；训练PID149075未重启。验证HTTP200且只显示四组run，拼接指标原段逐条一致，已接到180019步（会继续增长）。

## 1. 用户确认的研究目标——以此为准

**通过编码车辆历史状态—动作序列，提取与当前车辆类型及动力学参数相关的潜在车辆动力学条件特征，并将其作为条件注入世界模型，增强世界模型对当前车辆控制效应的理解，最终改善驾驶性能与底盘自适应能力。**

形式上：

\[
c_t=E_\phi(\{(x_\tau,a_{\tau-1})\}_{\tau=t-K+1}^{t}),\qquad
p_\theta(s_{t+1}\mid s_t,a_t,c_t).
\]

- 这里的 context 应包含历史反映出的车辆动作—响应规律，不要求直接还原真实质量、轴距或最大转角，也不要求等同于车型分类标签。
- **不要把“完全不随车速/转角变化”或“同底盘只有一个固定向量”写成研究目标的必要条件。** 跨工况替换是检验表征性质的一种诊断；context 可以同时包含工况信息。
- 需要区分三层证据：①含有车辆类型/参数相关动力学信息；②条件注入实际改善世界模型在相同测试数据上的控制效应预测；③最终改善驾驶与底盘适应。
- PCA 分团、车型分类、辅助预测改善，均不能单独替代 RSSM 预测改善和驾驶性能证据。

## 2. 工作范围和环境

- 仓库：`/home/server/projects/r2dreamer`。当前实现是 **PyTorch** R2Dreamer / DreamerV3，环境 MetaDrive。
- 原迁移参考是 CARLA/Cardreamer 上的 JAX Context Dreamer，私有仓库 `https://github.com/zho616/Adaptive-Context-Dreamer.git`，本地副本 `/home/server/projects/Adaptive-Context-Dreamer`。
- 用户只关心 R2Dreamer 和 DreamerV3，不扩展 DreamerPro 等其他方法。**当前正式队列只训练 R2Dreamer，DreamerV3 保持停止。**
- Python：`/home/server/anaconda3/envs/r2dreamer/bin/python`。
- 正确 MetaDrive checkout：`/home/server/projects/metadrive`，版本0.4.3。运行设置 `PYTHONPATH=/home/server/projects/metadrive`，需要直接导入仓库模块时再加入本仓库路径。
- 不要换成另一个 conda 环境的 MetaDrive：此前发现过 manager/runtime 行为不同，导致误判测试失败。
- 仓库有大量已有 tracked/untracked 改动，包含前期迁移实现；**不要 reset、清理或覆盖，也不要把所有未提交改动都当作本次新增。**

## 3. 当前 context 编码、预测和训练

关键实现：

| 文件 | 职责 |
|---|---|
| `/home/server/projects/r2dreamer/context.py` | E、D、辅助损失、想象历史更新 |
| `/home/server/projects/r2dreamer/context_rssm.py` | 三种 RSSM 条件注入 |
| `/home/server/projects/r2dreamer/dreamer.py` | 世界模型、Actor/Critic、context rollout |
| `/home/server/projects/r2dreamer/envs/ego_history.py` | 物理量测量、动作对齐、K5历史 |
| `/home/server/projects/r2dreamer/envs/chassis.py` | 车型与质量/转角随机化 |
| `/home/server/projects/r2dreamer/configs/model/_base_.yaml` | context 开关与注入配置 |

### 新版 state4

- 状态 `[a_long, yaw_rate, planar_speed, actual_front_wheel_angle]`，即纵向加速度、横摆角速度、平面速度大小、实际前轮转角。
- 角速度、转角和 canonical 转向均采用右正约定；转角使用弧度，直接测量实际前轮角，不只是动作指令。
- token = `[端点状态4, 到达该端点的前一已执行 canonical 动作2]`；canonical 动作为 `[throttle/brake, right_steering]`。
- K=5，输入 `[5,6]`。动作不与端点错移；辅助预测使用目标时刻存储的 preceding action。
- E：token投影到64维、可学习位置编码、两层 pre-LN Transformer、4头、FF128、masked mean pooling、线性输出 **确定性8维 context**。
- D：输入 context8＋状态4＋动作2，共14维；128→128 SiLU MLP，输出四维状态残差。
- 状态归一化尺度 `[10,1,30,π/3]`；辅助损失是归一化 L1，固定 anchor context，最多五步展开，每步 detach 预测状态，按有效预测步平均，不跨 reset/碰撞/终止等边界。
- 预测速度限制为非负、前轮角限制在全局±60°。想象中用 D 预测、滚动更新历史、再用 E 编码，不读取未来真值。
- E/D **只由辅助损失更新**；注入 RSSM/Actor/Critic 的 context 均 detach。没有 VAE KL、车型标签监督、参数识别、跨工况一致性或对抗解耦损失。
- E 保持 eager FP32，避开 Torch2.8 Inductor/AMP 注意力类型问题；主体仍使用原 compile/AMP。
- state4 不是完整车辆 Markov 状态，例如未显式包含侧向速度和悬架状态。

### 旧版 state2

- 状态仅 `[a_long,yaw_rate]`，token4，D预测两维；context仍8维。
- `env.context_kinematics=false` 支持旧格式，构造器按 observation 自动选择 state2/state4。
- 新旧版本同时改变了编码输入和辅助预测目标，**不是纯输入单因素消融**；aux总损失不能直接比较。

## 4. 注入机制和用户指定的两个实验

### 已有主对照：recurrent_concat

\[
h_t=f(h_{t-1},[z_{t-1};\operatorname{sg}(c_t)],a_{t-1}).
\]

本仓库分别投影 h、z、a，将 context 拼到 z 的输入投影；保留原 block-GRU 门控。context 经 h/z 间接影响先验、后验和策略。迁移的是 cRSSM/DALI 的循环条件化路线，不是完整复现。

### 用户明确要求依次验证

1. **AdaRMS 调制**：`model.context.injection=adarms_deter`，`model.context.actor_critic=false`。替代循环拼接；用 context 产生 scale/shift/gain，在 GRU 更新前处理隐藏特征。调制层零初始化，原门控不变。是参考 DynaDreamer 调制思想的本仓库 RSSM 适配，不能声称该具体适配已获论文证明。
2. **Actor/Critic 直接条件化**：`injection=recurrent_concat`，`actor_critic=true`。保留循环拼接，额外向 Actor、Critic、slow Critic 输入 `[rssm_feature; sg(c_t)]`。已覆盖真实推理、想象采样、想象 Actor/Value 损失和 replay value learning；同一步 latent/action/context 对齐，reset清空直接条件。

第二组参考 DALI Deep 的直接决策条件化，但**没有新增 reward/continuation/观测头的直接 context 输入，也不是完整 DALI Deep**。官方源码参考：
`https://github.com/frankroeder/DALI/blob/5ca393468defffe79e4e3537ba9766f6d31b510b/dreamerv3_compat/dreamerv3/agent.py`。

- `prior_posterior` 已实现，但用户澄清后**不在本次训练队列**。
- `model.context.rollout` 另有 `dynamic/fixed/zero/shuffled`。本轮均使用 dynamic；其他是想象使用方式诊断，不要与注入位置混为同一消融。
- 新 `actor_critic` 默认false，旧配置缺少该字段也按false处理，原 checkpoint 输入维度兼容。

## 5. 共同训练与独立测试设置

- 每组从头训练500K环境步，seed0，8训练环境，train ratio512，CPU replay容量500K。
- 相同 paper reward；每次训练验证20回合，保存验证集选出的 best。best高点不代表收敛。
- 训练车型 DEFAULT/M/L；质量与最大转角分别在 `[0.8,0.9,1.0,1.1,1.2]` 离散scale随机，实际物理参数取整数。
- 独立测试每组各100回合：新地图10020–10119、未见S车型10320–10419、未见XL车型10420–10519。
- 测试质量/转角scale仍使用训练档位，车型未见不等于scale域外。
- 每种方法只有一个训练seed；地图配对区间不代表跨训练seed不确定性。

## 6. 已完成驾驶结果

各自按训练验证集选择best，独立测试每列100回合，成功率如下：

| 方法 | best步数 | 新地图 | S | XL |
|---|---:|---:|---:|---:|
| R2Dreamer基线 | 440007 | 78% | 86% | 42% |
| 旧版state2 context，循环拼接 | 150006 | **93%** | 95% | **50%** |
| 新版state4 context，循环拼接 | 480006 | 69% | 80% | 21% |
| 新版state4 context，AdaRMS | 480007 | 87% | **96%** | 19% |
| 新版state4，循环拼接＋Actor/Critic | 未完成 | 未测试 | 未测试 | 未测试 |

目前旧版context的独立测试成功率整体最好。旧版平均驾驶速度明显更低：新地图约8.77km/h，新版循环拼接24.52km/h、AdaRMS24.77km/h。速度可能参与差异，但未做速度匹配干预，不能直接归因。

AdaRMS独立测试补充：

| 条件 | 碰撞率 | 出界率 | 奖励 | 路线完成率 |
|---|---:|---:|---:|---:|
| 新地图 | 9% | 4% | 146.37 | 89.80% |
| S | 3% | 1% | 166.54 | 95.12% |
| XL | 17% | 64% | 76.99 | 57.92% |

### 结果路径（均位于本仓库）

- 基线：`/home/server/projects/r2dreamer/logdir/chassis_baselines_500k_20261015_020330/r2dreamer`。
- 旧context：`/home/server/projects/r2dreamer/logdir/context_chassis_500k_20261015_103141/r2dreamer_context`。
- 新context循环拼接：`/home/server/projects/r2dreamer/logdir/context_kinematics_500k_20261015_180340/r2dreamer_context`。
- 三者对比报告/曲线/配对测试：`/home/server/projects/r2dreamer/logdir/context_kinematics_500k_20261015_180340/comparison_20261016_011017/RESULTS.md`，同目录含 `training_comparison.json`、`evaluation_comparison.json` 和图形。
- 新消融根目录：`/home/server/projects/r2dreamer/logdir/context_injection_500k_20261016_020630`。
- AdaRMS：根目录下 `r2dreamer_adarms_deter`，独立测试在 `evaluation/new_maps`、`evaluation/s`、`evaluation/xl`。
- Actor/Critic：根目录下 `r2dreamer_recurrent_actor_critic`。

## 7. 世界模型指标：尚无整体改善证据

训练末20条日志损失均值（不是末20回合、不是共同测试集）：

| 指标 | 基线 | 旧context | 新context循环拼接 |
|---|---:|---:|---:|
| Dynamics KL | 2.4156 | 2.5818 | 2.5366 |
| Barlow | 4.0996 | 4.1801 | 4.4261 |
| Reward | 0.5065 | 0.5189 | 0.5078 |
| Continuation | 0.01037 | 0.01216 | 0.01238 |

- 新版相对旧版只是部分下降；相对基线未整体降低。
- 各策略自己采集的 replay 分布不同，不能直接根据训练loss判断世界模型预测优劣；KL低也不等于预测一定更准。
- 后述辅助E+D误差改善不是RSSM改善。还没有完成所有世界模型在同一批测试轨迹上的统一预测对照。
- AdaRMS与直接Actor/Critic组的共享世界模型损失，尚待统一汇总比较，不要虚构结论。

## 8. 已做表征诊断与解释边界

### 同轨迹PCA与探针

目录：`/home/server/projects/r2dreamer/logdir/context_kinematics_500k_20261015_180340/analysis/vehicle_context_pca`。
脚本：`scripts/analyze_vehicle_context_associations.py`、`scripts/probe_front_wheel_angle.py`。

- 同一批1680窗口，5原生车型×2初速(5/10m/s)×7实际前轮角(-30,-15,-5,0,5,15,30°)×24动作profile。
- 固定原生参数，不混合质量/最大转角随机scale；无交通宽平路；训练车型和未见车型都包含。
- 新旧E编码同一物理历史，旧版取对应state2/action2列；按profile分组交叉验证，预处理仅训练fold拟合。
- 混合工况线性车型分类：旧37.1%，新59.8%；kNN5：旧97.7%，新99.8%。新版可线性读出的车型信息增加，但旧版已能非线性区分。
- 新版PC1与速度Pearson相关约0.946；PC2含较强车型分离。仍然存在强工况依赖。
- 固定工况下原始历史也可以区分车型，随机编码器也有较高分类能力，不能把聚类全部归功于训练。
- 固定相同实际前轮角需要不同车型使用不同归一化转向动作，角度/动作比值本身暴露转向增益；不是纯隐藏参数识别证据。
- 这里的车型探针包含所有5种车型标签训练，不是“未见车型零样本分类”。
- CaDM相关讨论只用来参考表征验证思路，当前采样/投影协议不是精确复现CaDM图。

### 冻结辅助E+D的跨工况/跨底盘干预

目录：`/home/server/projects/r2dreamer/logdir/context_kinematics_500k_20261015_180340/analysis/context_transfer_prediction`。
报告 `REPORT.md`，原始结果 `prediction_results.json`、`prediction_errors.npz`、`per_target_errors.csv`，有两组PNG/PDF/SVG图和QA记录。
脚本：`scripts/test_context_transfer_prediction.py`；采集器增加可选 `--future-steps 5`，默认采样行为保留。

- 1680条轨迹：先5步support，再5步future，dt0.1s；固定目标真实起点/未来动作，只替换donor context。
- 比较同/异底盘×原工况/换速度/换角度/两者都换；其他角度平均其余6档，其他底盘平均其余4类，每个donor分别预测后平均绝对误差，不先平均context。
- 两版只比较共同目标a_long和yaw_rate。误差定义 `|Δa|/10+|Δr|`，以下为五步平均。

| Context来源 | 旧版 | 新版 |
|---|---:|---:|
| 自身原工况 | 0.2207 | 0.0707 |
| 同底盘换车速 | 0.2278 | 0.1505 |
| 同底盘换转角 | 0.2495 | 0.0947 |
| 同底盘速度转角都换 | 0.2493 | 0.1644 |
| 其他底盘、工况匹配 | 0.2275 | 0.1125 |
| 保持当前加速度和角速度不变 | 0.0565 | 0.0565 |

- 新版原工况下换底盘误差+59.1%，84.8%的目标在对其他底盘求平均后变差，支持有底盘相关的预测线索。
- 同底盘换车速误差+113%，换角度+34%，都换+133%；两工况都不匹配时，异底盘相比同底盘仅再差1.7%，仅54.0%目标变差。
- 每步使用真实当前状态的单步诊断仍出现跨速度退化，不仅是自回归误差累积。
- 平滑油门、固定转向的探测中，persistence比两个模型都好；这些轨迹可能偏离on-policy分布。因此不能仅由替换退化声称准确辨识参数，或由跨工况退化断言context不能用于条件注入。
- 这些是**辅助E+D**测试，不是RSSM、actor或驾驶干预。profile bootstrap只表示固定模型/固定设计下的采样不确定性，不是训练seed稳定性。
- 此诊断独立重采集与上次PCA动作/角度一致，物理状态有很小数值差异，已记录；两版在本次使用完全相同的历史。

## 9. 当前运行状态与接手优先事项

### 状态核查：2026-10-16 10:56 Asia/Shanghai

1. AdaRMS已完成训练，末日志500002步，best480007步，三组各100回合测试均完成。
2. Actor/Critic直接条件化末日志302629步；**训练进程和原runner已不存在**。`status.json`仍写running，是陈旧状态；中断原因未知，不能声称仍在训练。
3. 第二组目录**没有 `latest.pt`**；现存 `best.pt` 实际步数 **140005**，验证成功率约70%。checkpoint keys为 `agent_state_dict,optims_state_dict,step,eval_metrics,selection_order`。
4. 不要把302629日志步数当作可恢复的checkpoint步数。继续500K实验前，先查恢复逻辑、replay/优化器/计数器可恢复程度、是否另有断点，并说明会损失哪些训练状态。不要直接重跑整个队列覆盖AdaRMS；不要把只加载best权重当作无缝续训。
5. 本次写交接文档时**没有擅自重启训练，也没有修写陈旧status**。

建议接手顺序：

- 先读本文件及 `/home/server/projects/r2dreamer/docs/context_injection_ablations.md`。
- 检查新消融目录的 `status.json`、`launch.json`、`queue.log`，和第二组 `launcher.log/console.log/.hydra/config.yaml/best.pt`。
- 检查 `/home/server/projects/r2dreamer/train.py`、`trainer.py` 的保存/恢复行为，再处理第二组中断。
- 完成直接Actor/Critic组后，用同样协议测试新地图/S/XL各100回合，对比新版循环拼接与AdaRMS；旧state2与基线可作为补充对照。
- 分开汇总驾驶表现、共享世界模型指标、辅助预测/表征证据，不混淆“context有信息”和“注入使世界模型/驾驶变好”。

## 10. 启动器、验证记录与TensorBoard

- 启动器：`/home/server/projects/r2dreamer/scripts/train_metadrive_compare.py`，支持 `--context-variants adarms_deter recurrent_actor_critic --methods r2dreamer --chassis-randomization --evaluate-best`。**这是新实验启动参数，不是恢复命令，不要直接用于覆盖已有目录。**
- `--evaluate-best` 在每组训练后用冻结代码执行新地图/S/XL各100回合，失败停止队列。
- 测试器：`/home/server/projects/r2dreamer/scripts/evaluate_metadrive.py`，逐局保存JSONL、summary、checkpoint hash、配置和注入模式。
- 正式消融根目录含 `EXPERIMENT.md`、冻结 `source/`、`source_sha256.json`、`startup_validation.json`、`validation_tests.log`、`validation_cuda.log`。
- 已通过34项测试，覆盖时间对齐/梯度隔离/reset/两种Dreamer兼容性；两组各96步真实MetaDrive短训练通过；R2Dreamer和DreamerV3×两注入方案CUDA AMP均得到有限优化更新。GradScaler初始溢出时跳过更新并降低scale，记录见validation_cuda.log，不是改用不同训练设置。
- Smoke目录：`/home/server/projects/r2dreamer/logdir/context_injection_smoke_20261016_0207`。
- TensorBoard6010此前退出，已重启，核查时PID7908在运行，监听0.0.0.0。地址 **http://192.168.118.78:6010/**；服务器localhost6010也可访问。
- `R2_recurrent_context`是**新版state4循环拼接**，不是旧state2。其余显示名：`R2_baseline`、`R2_AdaRMS`、`R2_recurrent_plus_actor_critic`。
- TensorBoard运行不代表训练仍在运行；进程PID会变化，接手时重新查。

## 11. 其他现有说明

- `/home/server/projects/r2dreamer/docs/context_dreamer.md`
- `/home/server/projects/r2dreamer/docs/context_dynadreamer.md`
- `/home/server/projects/r2dreamer/docs/context_injection_ablations.md`
- `/home/server/projects/r2dreamer/docs/context_kinematics_validation.md`
- `/home/server/projects/r2dreamer/docs/context_validation.md`
- `/home/server/projects/r2dreamer/docs/chassis_randomization.md`

早期说明可能基于旧state2或较早训练状态；研究目标优先遵循本文件第1节，实验状态以实际文件/进程核查为准。

## 12. 新增任务：两种方法的每类200回合重测（2026-10-16 21:33）

- 用户要求将 `R2_recurrent_context` 与 R2Dreamer基线分别按每种200回合重测；沿用新地图、S、XL三个测试池，共6组、1200回合。
- 使用现有各自best：基线440007步；新版state4循环context480006步。不是旧state2，也不是AdaRMS或直接Actor/Critic组；不训练或重新选best。
- 新地图种子10020–10219，S10320–10519，XL10420–10619。同测试池内两方法完全配对；测试池之间的部分地图种子可重复，不将它们当作600个独立地图。
- 本轮每类200是总量，包含此前100回合的种子范围，不与旧100相加为300；全部重新执行并独立保存。
- 输出：`/home/server/projects/r2dreamer/logdir/context_baseline_eval200_20261016_213331`。启动时runner PID323795；已核实基线新地图组开始输出逐局结果。后续状态请重新核查 `status.json` 和进程。
- 固定原评估脚本及各训练冻结source；复用相同推理RNG约定和底盘scale分布。每完成一对200回合自动核对实际底盘参数/种子，并更新 `comparison.json` 和 `RESULTS.md`，报告成功、碰撞、出界、奖励、路线完成、速度以及配对bootstrap区间。失败停止队列。
- 脚本：`/home/server/projects/r2dreamer/scripts/evaluate_context_baseline_200.py`。本任务没有恢复此前中断的Actor/Critic训练。

## 13. 200回合测试更新与旧版补测（2026-10-17 00:36）

第12节的6组评估已经全部完成（status=completed）。每类200回合结果：

| 条件 | 基线成功率 | 新版state4 context成功率 | 新版减基线95%配对区间 |
|---|---:|---:|---|
| 新地图 | 82.0% | 71.5% | -10.5pp，[-18.0,-2.5]pp |
| S | 82.0% | 78.5% | -3.5pp，[-10.0,+3.5]pp |
| XL | 37.5% | 20.0% | -17.5pp，[-24.5,-10.5]pp |

完整结果：`/home/server/projects/r2dreamer/logdir/context_baseline_eval200_20261016_213331/RESULTS.md`。

用户随后要求把之前效果较好的旧版state2 context也按每类200回合测试。已启动：

- 模型：`context_chassis_500k_20261015_103141/r2dreamer_context/best.pt`，严格加载其冻结source，实际checkpoint步数150006。
- 三个测试池及种子与第12节完全一致，新地图/S/XL各200，共600；原100回合结果保留，不计为额外样本。
- 输出：`/home/server/projects/r2dreamer/logdir/old_context_eval200_20261017_003604`。启动时runner PID428896，已确认模型加载成功并进入新地图评估；进展需重新核查。
- 复用已完成的基线/新版各200回合结果，完成后核对逐局种子/实际底盘参数，自动生成三者 `RESULTS.md`、`comparison.json`，含旧版减基线、旧版减新版的配对成功率区间。
- 启动器新增 `--old-context-reference <已完成200回合结果目录>`，只测试旧版，不重跑基线和新版。正式输出含冻结runner和评估脚本。

## 14. 旧版state2每类200回合最终结果（2026-10-17 00:58）

旧版600回合全部完成，status=completed。与相同种子/实际底盘参数的基线、新版state4对照：

| 条件 | 基线成功率 | 旧state2成功率 | 新state4成功率 | 旧版减基线95%配对区间 |
|---|---:|---:|---:|---|
| 新地图 | 82.0% | 92.5% | 71.5% | +10.5pp，[+4.0,+17.0]pp |
| S | 82.0% | 92.5% | 78.5% | +10.5pp，[+4.0,+17.0]pp |
| XL | 37.5% | 59.5% | 20.0% | +22.0pp，[+13.0,+31.0]pp |

旧版碰撞率新地图/S/XL为5.5%/6.0%/11.0%，基线14.0%/14.0%/15.5%；旧版路线完成92.2%/92.1%/78.8%，基线88.0%/87.6%/61.9%。旧版速度8.64/7.60/8.19km/h，基线20.11/20.98/16.49km/h。XL旧版超时5.5%，基线0.5%。因此驾驶成功率提升，但伴随明显更慢策略，不能直接解释为纯动力学表征收益。

完整报告：`/home/server/projects/r2dreamer/logdir/old_context_eval200_20261017_003604/RESULTS.md`。这里只使用单个训练seed，不代表跨seed稳定性。

已重新核对旧版冻结config/source：state2，K5 token4，context8，`injection=recurrent_concat`，`rollout=dynamic`；拼接context与上一随机隐变量进入原z输入投影，参与h更新，prior/posterior经h间接条件化；Actor/Critic没有直接context输入，原GRU门控不变。

## 15. 旧版best PCA展示（2026-10-17）

用户要求按“随机初始化vs训练后，旧state2/新state4”的2×2布局绘图，已严格复算导出：
`/home/server/projects/r2dreamer/logdir/old_context_eval200_20261017_003604/analysis/initial_vs_best/`。
主图`fixed_initial_vs_best.png/pdf/svg`：初速10m/s、实际轮角15°，每车型24个窗口；另有`mixed_initial_vs_best`（1680窗口）。右上面板即本次600回合测试的旧150006步best。随机seed20261015，是独立随机初始化对照，不是训练原始step0。各面板独立PCA，只能比较分布结构，不能把绝对坐标当作同一基底。
模型输出与既有缓存逐值核对，checkpoint hash一致。此为已有物理探测数据，非刚完成600回合驾驶历史；固定初速后自然演化，不锁定后续车速。原始8D、PCA坐标CSV和元数据齐全。脚本`scripts/plot_context_initial_vs_best.py`支持`--output`导出到独立目录。

## 16. 基线与旧 state2 Context 从头重训至 1M（2026-10-17 01:15 启动）

用户要求依次重训 R2Dreamer 基线、效果最好的旧版 Context，各 1,000,000 环境步，并新增 episode/sucess。

- 实验根目录：`/home/server/projects/r2dreamer/logdir/retrain_1m_20261017_0115`。
- 正式队列：`training/status.json`；启动日志：`training_launcher.log`；PID 及命令保存在 `launch.json`。开始时 runner PID 717010。
- 顺序：`training/r2dreamer` → `training/r2dreamer_context`，首组失败则不启动次组。均从随机初始化开始，没有加载旧 best 或 latest。
- 使用旧实验 `context_chassis_500k_20261015_103141/source` 的完整冻结模型、环境、配置；只覆盖当前 `trainer.py` 和 `train.py`，增加日志、每20K步原子保存 latest 和最终保存。未改旧版网络、辅助预测目标、环境或注入方式。正式源为 `training/source`，准备来源记录于 `provenance.json`。
- 旧版严格为 state2=[纵向加速度,右正横摆角速度]，token4，K5，context8，recurrent_concat，dynamic rollout，Actor/Critic无直接context。不是 state4/token6。
- 与旧500K实验相同：seed0、8训练环境、train_ratio512、CPU replay容量500K、paper reward；DEFAULT/M/L，质量与最大转角按原离散scale随机且最终为整数；每10K步评估20回合（2环境），按成功率/路线完成率/reward依次选best。仅预算增加至1M，批量步进可能略超阈值。
- 新训练日志 `episode/success`：每个已完成训练回合的成功0/1；`episode/sucess`为用户指定拼写的同值别名。按回合清零，最终一步终止的回合也会输出，不记录未完成回合。原 `episode/eval_success` 仍为20回合评估均值。
- 验证：36项单元测试通过，含成功→失败→成功、最后一步终止日志和20K检查点边界测试；同一冻结源的两组96步短跑均完成，产生best/latest、有限训练损失以及两个TensorBoard成功tag。短跑结果在 `smoke`。
- TensorBoard：`http://192.168.118.78:6011`（仅此正式队列）；HTTP本机检查200。6010既有服务保留。
- 启动时基线正在运行、context排队；此处不是完成记录。查询实际进度请读取status及各模型metrics.jsonl/launcher.log，并核验进程。

## 17. 1M训练完成与独立测试启动（2026-10-18）

两组均于10月17日完成1,000,006步、正常退出。基线耗时7.44h，best=540000；旧state2 Context耗时10.07h，best=940005。两组best验证均20/20成功；最终验证成功率为基线60%、Context80%。最后10次验证平均成功率62.5% vs 76.5%，奖励119.92 vs143.64，不能用best验证100%代替独立测试。

按用户“训练结果测试对比”启动原协议独立测试：
- 输出 `/home/server/projects/r2dreamer/logdir/retrain_1m_eval200_20261018_1908`，状态`status.json`，逐回合结果各子目录`episodes.jsonl`，聚合`RESULTS.md`/`comparison.json`。
- runner `scripts/evaluate_retrained_pair.py`，冻结runner/evaluator到输出目录；开始PID2293883，3进程并发，最多1200回合（2模型×3池×200）。GPU有足够显存；各进程独立环境/随机种子，严格加载同一训练冻结source和各自best。
- 新地图DEFAULT/M/L种子10020..10219；S10320..10519；XL10420..10619。参数scale与之前一致。actor mode，RSSM随机后验，逐回合seed+12345。报告生成时核对两模型实际底盘参数与地图种子完全一致，配对bootstrap10000次。
- 训练总结 `TRAINING.md`/`training_comparison.json` 已输出。最后20条世界模型损失记录未见Context整体降低，且数据来自各自策略replay，不能直接比较预测能力。
- 此节记录为测试启动状态；实际是否全部完成须读取测试status，勿把部分回合当最终结果。

## 18. 1M best配对测试全部完成（2026-10-18 19:16左右）

输出 `logdir/retrain_1m_eval200_20261018_1908` 的status=completed，6组各200，总1200回合。所有配对地图seed和实际底盘参数一致，strict checkpoint加载；模型best分别540000、940005。正式聚合见RESULTS.md/comparison.json，训练总结TRAINING.md，500K历史对照COMPARISON_500K_1M.md。

| 测试池 | 基线成功率 | 旧Context成功率 | 基线碰撞率 | Context碰撞率 | 基线奖励 | Context奖励 | 基线路线完成率 | Context路线完成率 | 基线速度km/h | Context速度km/h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 新地图DEFAULT/M/L |90.5%|85.5%|4.5%|10.0%|144.66|139.97|91.3%|86.8%|19.57|21.90|
| 未见S |89.5%|89.0%|6.5%|9.5%|149.75|154.39|91.3%|91.0%|19.64|21.96|
| 未见XL |59.0%|54.0%|12.5%|13.5%|108.94|114.13|73.2%|74.9%|15.77|18.38|

成功率Context减基线：新地图-5.0pp，配对bootstrap95%[-10.5,+0.5]；S -0.5pp[-5,+4]；XL -5pp[-13.5,+3.5]。三个区间均跨0，当前1M实验未显示明确成功率提升；负点估计亦不足以证明显著更差。Context更快约2.3–2.6km/h，碰撞率更高，S/XL奖励略高。先前500K旧Context best约8km/h、成功率92.5/92.5/59.5，这次1M best约18–22km/h、85.5/89/54；不能沿用先前“全面领先”的结论。两次独立训练/单seed/不同best阶段，勿归因于单一机制。

## 19. 共同轨迹世界模型与context干预验证（2026-10-19，已完成）

用户要求验证上一节提出的机制问题。输出 `logdir/world_model_context_validation_20261019_cpu`；status=completed。CUDA初始化失败（GPU仍能显示但PyTorch不可用），所有数据采集/推理改CPU FP32完成，未修改训练代码或权重。主脚本 `scripts/validate_world_model_context.py`，补充统计 `scripts/summarize_world_model_validation.py`；输出目录包含脚本snapshot、模型hash、采集NPZ和误差数组。

协议：1M best基线540000、context940005；10个新地图30000..30009×五标称车型×两行为策略=100条实际驾驶轨迹。每车型质量/最大转角固定nominal，隔离车型身份；这不是全随机参数驾驶复测。2153起点，1530起点（71.1%）同时有同车型及异车型donor；donor来自同策略、其他地图/回合，速度±0.75m/s、实际前轮角±3°、最近执行转向/油门各±0.15；核对同车型实际底盘参数一致。

共享观测过滤起点z/h，未来动作逐值相同；仅从当前history预测更新context，不输入未来真值context/观测。baseline、dynamic、zero、fixed_own/same/other/shuffle，另self-swap逐值检查，3次future RSSM采样共用seed。统一展开10步并mask真实终止后目标，不按预测continuation阈值门控；dynamic原样调用冻结refresh_context，但不是完整Actor想象循环。

结果：
- 辅助E+D匹配子集5步归一化L1（|Δacc|/10+|Δyaw|）：own0.04849，same0.05221，other0.11821，zero0.37206，persistence0.47377。other减same地图等权差+0.06620，10地图bootstrap95%[0.06427,0.06817]。有用的车型相关响应信息证据明确；并不证明纯参数/跨所有工况不变。
- 世界模型10步reward MAE：baseline0.63204，dynamic0.58783，zero0.70688，fixed_own0.56525。去掉context的reward误差明显增加，但固定context在本诊断优于动态更新；fixed减dynamic差CI[-0.03761,-0.01059]（地图等权点差-0.02417）。
- reward NLL（越低越好）：baseline0.93582，dynamic1.02470，zero1.70060，fixed0.98772。动态比baseline更差，不能宣称整体世界模型改善；10步固定vs动态NLL差区间跨0。
- continuation Brier：baseline0.04493，dynamic0.04175，zero0.03731；置零反而更低，说明任务影响混合。
- 匹配子集5步reward MAE同车型donor0.49717，错车型0.50237，车型错配对RSSM reward影响小于对辅助物理预测的影响。

结论/下一步候选：编码器有有用响应信息、RSSM也使用c，但使用收益混合；优先考虑fixed vs dynamic的训练及闭环消融，不能仅凭该离线诊断断言驾驶下降原因。本次未启动新训练。首要限制：单训练seed、10地图、近邻匹配并非全历史严格控制，正确prefix h可能已有车辆信息，且RSSM无加速度/横摆角速度重建头。奖励从四个obs log_reward_*分量恢复，冻结原环境返回scalar reward独立检查25步最大差1.74e-9。详见FINDINGS.md/REPORT.md/results.json/additional_checks.json。

## 20. mass/friction比例扫描PCA（2026-10-19）

用户要求分别绘制mass与friction比例变化的latent context PCA。已完成240个真实物理探测窗口：`logdir/mass_friction_context_pca_20261019`，独立mass_context_pca和friction_context_pca、组合mass_friction_context_pca均导出PNG/PDF/SVG，完整source_data.csv/representations.npz/metadata/QA/REPORT在同目录。脚本`scripts/plot_mass_friction_context_pca.py`。

冻结1M旧state2 best940005，M车型，K5/context8，初速10m/s、实际前轮15°、同一宽平直道无交通；每档24条相同配对油门序列。mass倍率0.8/.9/1/1.1/1.2→960/1080/1200/1320/1440kg；摩擦固定0.75。friction倍率同五档→Bullet轮胎friction slip .6/.675/.75/.825/.9，质量固定1200。其他参数不变。30步静置，速度仅初始化一次，不在窗口内钳制；实际窗口速度9.7449..9.9869m/s（初始10另计）。实际质量/四轮摩擦前后读回、动作配对、前轮角均检查。

注意MetaDrive的set_wheel_friction直接raise；本次必须用vehicle.system.getWheel(i).setFrictionSlip，再从同一BulletVehicle索引读回，不能依赖vehicle.wheels缓存对象句柄（曾读到异常值，已在无采样时拦截）。质量是整数设置，但body.getMass有float舍入，检查使用rtol1e-6/atol1e-4。

同一个8D标准化器拟合全部240点，每个面板再独立PCA。mass PC1/2=93.5/4.7%，friction=98.3/1.2%。点是窗口，X为组均值，灰线连接配对动作。质量档位有有序位移和部分重叠，摩擦档位分离较清楚。不能把PC方差比例当参数识别准确率，也不能单凭图证明纯参数编码。摩擦不是独立训练随机化变量；此图是额外单因素探测，不是驾驶成绩或参数辨识检验。

## 21. 混合速度/前轮角的mass-friction PCA（2026-10-19）

用户要求取消固定初速/前轮角后重新绘制。已完成 `logdir/mass_friction_context_pca_mixed_20261019`，模型仍为同一旧state2 best940005，hash与固定条件图完全一致。M车型；质量/摩擦各扫描0.8/.9/1/1.1/1.2×；初速5/10/15m/s×实际前轮角-30/-15/0/15/30°×24配对油门profile=每档360窗，2扫描共3600窗。窗口间速度和角度变化；每窗只初始化一次速度、不钳制，每窗内对应转向恒定。各参数档位完整覆盖同一工况网格，无选择性剔除。

主图 `mass_friction_context_pca.png/pdf/svg`，独立mass_context_pca、friction_context_pca，另 `operating_condition_colors` 在完全相同PCA坐标上分别按速度、轮角着色。CSV/NPZ/metadata/QA/REPORT齐全；240固定窗原图保留。`scripts/plot_mass_friction_context_pca.py`新增--speeds/--angles，默认仍10m/s/15°。

结果：不同比例在混合工况PCA中大量重叠，速度、轮角着色结构更清晰。8D共同标准化latent中，单因素组均值方差关联：质量扫描scale0.33%、speed22.68%、angle59.60%；摩擦扫描scale2.84%、speed18.95%、angle57.01%。这是平衡设计中的描述性主效应，不是参数信息占比/识别准确率；参数与工况交互不计入单因素主效应。PCA重叠也不能证明没有参数信息，只说明全局最大方差方向主要受工况影响。两个扫描各自PCA，跨图坐标不可当同一物理量。
