# T-intersection Shallow：单向 WM → context，倍率 0.5

2026-10-22 从头启动 500K，不加载旧 checkpoint。

实验目录：`logdir/tintersection_shallow_wmgrad05_500k_20261022/`。
TensorBoard：原 T-intersection 6015 服务新增 `12_500K_state4_DALI_shallow_swap1_wmgrad05`。

相对上一组 T-intersection 11，仅启用 `model.context.wm_grad_scale=0.5`。
aux=1、swap=1；state4/K5/context8，DALI Shallow posterior 条件注入。
WM 正常更新自身；dyn、rep、barlow、rew、con 对 context 的梯度以 0.5 倍回传编码器。
repval 不回传编码器。context 两项辅助损失不更新 WM 参数。
默认配置仍为 wm_grad_scale=0，旧实验行为不变。

沿用 seed0、8 环境、batch16×64、ratio512、CPU replay500K、traffic_density=0.2。
训练车型 default/M/L，质量与 wheel_friction 分别随机倍率 0.6–1.4、间隔 0.1；最大前轮角保持车型名义值。
每 10K 评估 20 回合；训练结束后 runner 自动评估 best：新地图、S、XL，各 100 回合。

启动命令/PID 见 launch.json，冻结源码见 training/source，实时状态见 training/status.json。
训练过程指标包括 context/wm_input_grad_norm 与 context/wm_grad_scale；前者是 context 输出处的梯度范数，不是编码器参数梯度范数。
诊断数据位于上一组目录的 context_wm_gradient_audit_20261022，19 项相关测试已通过。
