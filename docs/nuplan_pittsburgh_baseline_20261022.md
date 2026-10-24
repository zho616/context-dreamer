# Pittsburgh真实场景：R2Dreamer基线500K

用户最终要求：只对齐VDD的Pittsburgh场景设置；输入、奖励和原有闭环训练保持一致，先训练R2Dreamer基线。没有改用BEV，没有启动Shallow11正式训练。

## 数据与公开设置的对应

[VDD V-A1](https://arxiv.org/html/2512.02417v1#S5.SS1)说明Pittsburgh交通数据经ScenarioNet导入MetaDrive，8000个训练场景、额外100个随机测试场景。其MetaDrive奖励统一为路线进度+0.1×speed/max_speed，终止到达+10、出界−10、碰撞−20，max_speed=50km/h；没有为nuPlan另定义奖励。我们保持原有paper reward实现。

使用官方`nuplan-v1.1_train_pittsburgh.zip`：30,620,248,893字节，1560个DB，204个采集session，原始DB解压共55,726,387,200字节。下载文件及manifest在`/home/server/datasets/nuplan/`，只下载日志和地图，无sensor_blobs。分片可续传，完整包通过multipart ETag、SHA256和ZIP CRC检查后才允许转换。

正式输出目标：`/home/server/datasets/nuplan/scenarionet_pittsburgh_8000/`：

- train：8000场景。
- val：额外100场景，仅用于训练期间验证和best checkpoint选择。
- test：100场景，仅训练结束后测试。

额外验证集是我们的约定，避免拿test选checkpoint；不是声称VDD公开了这样的验证集。
按采集session固定seed20261022划分70/15/15；抽样日志按session交错，日志内候选随机排列；token、DB、session集间均不可重叠。
ScenarioNet 10Hz、20秒窗口、起始时间至少间隔10秒、保留有效目标、ego位移至少10米，不限定场景类型。位移筛选沿用之前nuPlan接入规则，避免静止片段出生即到达。筛选与split属于复现约定；作者未公开token列表，不能声称逐场景完全一致。

## 模型与预算

配置：`configs/metadrive_nuplan_pittsburgh.yaml`，context=false、swap=0、R2Dreamer、500K环境步、seed0、8训练环境、batch16×64、train_ratio512、CPU replay500K、原size12M与优化器、compile=true。验证集100场景，每10K步评估；latest每20K步保存，best按成功率/路线完成率/回报选择。

沿用271维LiDAR+ScenarioEnv原生trajectory navigation向量。与原程序生成场景的259维差异来自原生导航接口，输入仍为向量，不引入BEV。
底盘沿用DEFAULT/M/L、mass/friction0.6–1.4、最大转角名义值；没有改成VDD的名义车辆/转角扰动协议。
自车仍是EnvInputPolicy+Bullet闭环动力学；周车按真实记录回放，场景记录结束按truncation处理。

## 自动运行

运行目录：`logdir/nuplan_pittsburgh_baseline_500k_20261022/`。
监督进程PID1535827；启动时为下载阶段，正式GPU训练尚未开始。`status.json`是后续阶段的实时依据，不应根据本文静态描述推定训练已开始或已完成。

流程：完整下载和校验→正式8000/100/100转换及隔离检查→真实物理检查→仅baseline的GPU短训练→500K正式训练→best在100个test场景上测试三组推理随机种子。三组推理测试不是三次独立训练，不等同于论文的多次训练试验。

任何阶段失败均写入status.json并停止后续步骤，不会以Mini、少量场景或失败数据替代正式训练集。源码已经冻结在source/，后续工作区修改不影响这个任务。

```bash
PYTHONPATH=/home/server/projects/metadrive /home/server/anaconda3/envs/r2dreamer/bin/python scripts/run_nuplan_pittsburgh_baseline.py --output logdir/nuplan_pittsburgh_baseline_NEW --workers 24 --steps 500000
```

## 已完成的预检

`logdir/nuplan_pittsburgh_preflight_20261022/`只用于链路验证，不是正式训练数据。
7项unittest通过；新的转换器实际生成4/2/2 Pittsburgh小集、隔离审计通过；物理检查通过；GPU基线短训练1027步、31次更新尝试、优化器实际23次更新，权重与训练指标有限。独立held-out场景评估入口也已实际跑通。预检权重不用于正式训练初始化。
