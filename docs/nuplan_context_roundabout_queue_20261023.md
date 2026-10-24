# nuPlan context及环岛补充方法顺序队列

更新：用户随后要求环岛统一改为质量/轮胎摩擦243配置，并补充R2Dreamer及R2Dreamer+context从头重训。原四项环岛任务尚未启动，现已由新队列替代；nuPlan训练进程保持不变并由新队列接管。最新协议与入口见[环岛质量/摩擦六方法队列](roundabout_mass_friction_queue_20261023.md)，本文件下文保留为历史安排。

用户授权：当前nuPlan R2Dreamer基线结束后，依次安排nuPlan R2Dreamer+context，以及环岛SAC、PPO、DreamerV3基线、DreamerV3+context。用户已确认“dreamer”指本仓库DreamerV3基线。

队列：`logdir/nuplan_context_roundabout_queue_20261023/`，监督进程PID2219628。
前置：`logdir/nuplan_pittsburgh_baseline_500k_20261022/status.json`。
当前基线的训练及既定自动测试成功完成后，队列自动开始；不修改正在训练的源码或权重。

| 顺序 | 场景 | 方法 | 训练步数 |
| --- | --- | --- | ---: |
| 1 | nuPlan Pittsburgh | R2Dreamer+Shallow11 context | 500000 |
| 2 | Roundabout | SAC | 500000 |
| 3 | Roundabout | PPO | 500000 |
| 4 | Roundabout | DreamerV3 reproduction baseline | 500000 |
| 5 | Roundabout | DreamerV3+Shallow11 context | 500000 |

每项从头训练，seed0。context版本为state4/K5/context8、posterior-only Shallow、aux1/swap1/wm_grad_scale0，Actor/Critic不直接输入context。
各项正式训练前自动完成独立小预算短训练；检查实际更新、有限权重、best/latest，以及context有效交换配对。前置或任一任务失败时停止后续任务，并写入status.json。

## 两套场景协议分别冻结

nuPlan：直接复制当前nuPlan基线的source，配置复制其实际Hydra配置，仅启用context及其辅助/交换损失，环境其余字段逐项核对不变。
保持271维LiDAR+原生轨迹导航、Pittsburgh8000/100/100、原paper奖励、8环境、ratio512、batch16×64、CPU replay500K、DEFAULT/M/L及质量/摩擦0.6–1.4、名义最大转角。
验证集选best，完成后按当前基线相同的三组推理随机种子在100个test场景测试。这是一个训练seed，不是三次独立训练。

环岛：直接复制`context_swap_dali_shallow_500k_20261021/training/source`，配置以该组已完成Shallow实验的实际Hydra配置为准。
保持259维向量、原paper奖励、traffic_density0.2、8环境、原地图/出生规则；DEFAULT/M/L、质量与最大转角倍率均0.8–1.2，共75种配置，摩擦固定为各车型名义中点。
没有把T路口或nuPlan的243种质量/摩擦随机化搬到环岛。
训练地图0–99，验证10000–10019；best在新地图10020–10119、S10320–10419、XL10420–10519各测100回合。

环岛DreamerV3和DreamerV3+context共用同一冻结实现及环境，区别为context开关及相应损失，均采用rep_loss=dreamer。
早期已有其他版本的DreamerV3基线，此处按本次统一冻结实现重新配对训练，原实验保留。
SAC/PPO共用同一旧环岛原生环境，通过现有SB3适配器仅暴露state，不向策略传入底盘标签或context；保留自然终止与超时截断的区别，不加观测/奖励归一化。
SAC/PPO使用已安装SB3 2.9.0及此前T路口对比所用的算法超参数；没有套用Dreamer的ratio512。

## 可核对的文件与验证

- `plan.json`：五项正式命令、短训练命令及测试命令。
- `configs/`：每项完整配置；三套Dreamer配置已通过Hydra实际解析。
- `nuplan_source/`、`roundabout_source/`、`controls/`：两套模型/环境源码及控制脚本，plan保存SHA256。
- 4项新增队列测试通过：context开启历史输入、环境公平性保护、用户指定顺序、前置失败时不启动任务。
- `status.json`：实时阶段、每项状态、PID和返回码；本文PID仅是启动时记录。

## TensorBoard

- nuPlan：<http://192.168.118.78:6016/>，当前R2基线和后续context。
- Roundabout：<http://192.168.118.78:6017/>，原R2基线/旧11参考及新增四项。

已验证HTTP访问。尚未开训的方法会在产生event文件后显示曲线；短训练日志不进入这两个视图。

```bash
PYTHONPATH=/home/server/projects/metadrive /home/server/anaconda3/envs/r2dreamer/bin/python scripts/run_nuplan_roundabout_queue.py --output logdir/nuplan_context_roundabout_NEW --after-status logdir/nuplan_pittsburgh_baseline_500k_20261022/status.json --nuplan-reference logdir/nuplan_pittsburgh_baseline_500k_20261022 --roundabout-reference logdir/context_swap_dali_shallow_500k_20261021/training
# 检查plan/config后启动已经准备好的目录：
PYTHONPATH=/home/server/projects/metadrive /home/server/anaconda3/envs/r2dreamer/bin/python logdir/nuplan_context_roundabout_NEW/controls/run_nuplan_roundabout_queue.py --output logdir/nuplan_context_roundabout_NEW --run
```
