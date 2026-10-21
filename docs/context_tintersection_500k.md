# T-intersection：baseline 与 11 Shallow+swap1

已按用户后续要求停止，由`context_tintersection_mass_friction_500k.md`中的
质量/摩擦随机化实验替代；以下保留为原质量/最大转角实验记录。

实验目录：`logdir/tintersection_500k_baseline_shallow_swap1_20261021/`。

两组均从头初始化，不加载 roundabout 权重。沿用原 11 的冻结源码，模型、
优化器、训练器文件 SHA256 保持相同，仅为环境增加原生 `T` 地图块及配置入口。
原 roundabout 默认仍为 `O`；旧配置缺少 map_block 时保持 roundabout。

共同设置：seed0，各500K环境步，8训练环境，ratio512，batch16×64，
CPU replay500K，paper reward，2车道，traffic_density0.2，10Hz决策。
训练地图0–99，固定验证10000–10019；T路口三入口随机出生，选择不同可达出口。
车型default/M/L，质量及最大转角倍率各0.8/0.9/1.0/1.1/1.2，共75配置。

- baseline：context关闭，swap权重0。
- 11：state4、K5/context8、posterior-only Shallow、动态context机制、aux1、swap1；
  actor/critic不直接接收context。其他模型设计不变。

正式训练顺序为baseline→其best独立测试→11→其best独立测试。
每组独立测试为新地图、S、XL各100回合，均使用T路口；S/XL未参与训练。
每10K训练步固定验证20回合选择best，每20K保存latest。

验证：46项测试通过，含T路口三入口可达性、交通出生安全距离、重复种子一致性、
state4历史形状及旧地图兼容性。两组96步真实T路口smoke均完成，Shallow有效交换
配对最高24个；smoke专用固定底盘只用于执行损失路径，不代表正式75配置覆盖率。
初次smoke的任务名白名单问题已修复，失败记录保留于smoke/；通过版本在smoke_validated/。

启动命令与PID见实验launch.json；正式进度在training/status.json，启动输出runner.log。
模型源码快照在training/source；准备差异在source.patch及provenance.json。

TensorBoard沿用6014：

- `13_500K_Tintersection_baseline`
- `14_500K_Tintersection_DALI_shallow_swap1`

`scripts/train_metadrive_compare.py`支持`--map-block T --include-baseline
--context-variants dali_shallow --context-swap-weight 1 --chassis-randomization
--steps 500000 --evaluate-best`。完整启动命令以实验launch.json为准。
