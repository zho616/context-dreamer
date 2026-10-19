# 三车型 × 整数底盘参数随机化

训练池固定为 `default / m / l`，每次 reset 均匀选择一个车型。
质量 scale 与最大转向角 scale 从 `[0.8,0.9,1.0,1.1,1.2]` 独立均匀采样。
相对各车型标称参数缩放，用 `floor(value + 0.5)` 四舍五入为整数。
整局车型、质量和转向参数保持不变，K5 仍在每次 reset 清空。

| 车型 | 标称质量 kg | 允许质量 kg | 标称最大转角 ° | 允许最大转角 ° |
|---|---:|---|---:|---|
| default | 1100 | 880,990,1100,1210,1320 | 40 | 32,36,40,44,48 |
| m | 1200 | 960,1080,1200,1320,1440 | 45 | 36,41,45,50,54 |
| l | 1300 | 1040,1170,1300,1430,1560 | 40 | 32,36,40,44,48 |

共75个离散的车型/质量/转角组合，重复采样是正常行为。
M车型转角取整后，实际scale可能略偏离采样scale，例如41/45。
整数选择便于复现，不能替代物理稳定性检查。

## 物理生效与其他参数

通过原生车型类保留尺寸、轴距、轮胎及外形。在agent生成之后、首次物理步之前，
实际调用 `vehicle.body.setMass(integer_mass)`，并同步控制器的 `vehicle.max_steering`。
这是因为本地MetaDrive普通车型的MASS是类常量，仅改config不能保证实际质量改变。
每次reset也覆盖池中复用车辆的参数，并检查物理刚体读回质量。

原生车型自带的随机驱动力/制动力固定为各车型预设范围的中点；轮胎摩擦固定为车型预设值。
因此同一车型内本实验只随机质量和最大转向角，车型之间仍保留原生参数差异。
训练/评估均使用相同规则，不给背景交通车辆添加此随机化。

底盘采样使用独立随机流，不依赖地图/出生位置的采样结果。显式seed和评估seed
确定车型和scale，因此不同方法可以使用相同评估底盘集合。默认评估覆盖训练池的
一个确定性随机样本集，不保证每个组合都被评估，也不作为未见底盘测试。

## 配置与运行

```bash
# 随机底盘基线：不带context
python train.py --config-name metadrive_chassis model.rep_loss=r2dreamer
python train.py --config-name metadrive_chassis model.rep_loss=dreamer

# 随机底盘 + context
python train.py --config-name metadrive_context_chassis model.rep_loss=r2dreamer
python train.py --config-name metadrive_context_chassis model.rep_loss=dreamer

# 短训练对照：两种backbone，分别关闭/开启context，使用相同底盘分布。
python scripts/train_metadrive_compare.py --smoke --context-comparison --chassis-randomization --logdir /path/to/new-run
# 正式500K/组对照去掉 --smoke；每个实验使用新的目录。
```

`metadrive`、`metadrive_context` 的底盘随机化默认关闭。
可用Hydra覆盖离散列表，例如 `env.chassis.mass_scales=[0.9,1.0,1.1]`。
更改最大转角scale时会检查取整后的值在1..60°内，质量需为正整数。

未见车型评估设置：`env.chassis.eval_models=[s,xl]`。
可同时用 `env.chassis.eval_mass_scales=[1.0] env.chassis.eval_steering_scales=[1.0]`
固定其标称参数。该设置只改变评估池；训练池仍保持default/m/l。
正式零样本结论应在训练完成后用固定checkpoint、冻结网络单独评估，并按车型分别报告。
默认save_best仍依据训练池的验证结果选择checkpoint，不应依据未见车型的测试结果选模型。

## 记录与防止参数标签泄漏

`env.chassis_parameters` 和step info的`chassis`包含车型、标称值、整数实际参数、
采样scale与实际scale。Observation/replay额外存储数值`log_chassis_*`用于诊断：
model_id、mass、max_steering、mass_scale、steering_scale。
车型id映射：s=0, default=1, m=2, l=3, xl=4。
encoder与decoder都排除log_键；E仅读取四维状态-动作历史，模型不直接读取底盘标签。
评估统计对这些常量取每局终点值再跨局平均，不把质量按步累加。

## 验证

`python -m unittest discover -s tests`覆盖：离散采样合法性、三车型实际生成、
整数质量进入刚体、实际车轮角度、跨reset参数刷新、整局恒定、评估重现、
地图采样独立性、未见车型池以及质量/转角改变真实车辆响应。
这些是运行与物理生效验证，驾驶收益和底盘适应收益仍需要正式训练评估。
