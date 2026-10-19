# Context 每步车速与前轮转角输入（2026-10-15）

本轮改动：每步历史由 `[a_long, yaw_rate, throttle, right_steer]` 扩展为
`[a_long, yaw_rate, speed, front_wheel_angle, throttle, right_steer]`。
K=5、Transformer 结构与 context 输出 8 维保持。车速为平面速度模长，前轮角为
物理引擎左右前轮设定读数均值，转换为右正弧度。没有引入底盘参数标签。

为了使想象闭环一致，辅助预测器同步增加车速与前轮角残差预测，其归一化 L1 监督
也覆盖这两个通道。它们与原加速度、角速度通道一并使用最多五步的固定锚点 context
监督。不是参数识别或一致性损失；新增可测输入不保证 context 已解耦工况。

## 检查

- Context 单元测试 9 项通过：物理读数及坐标、mask、动作时序、episode 边界、逐步 detach、
  想象末端 token、世界模型梯度隔离、R2Dreamer/DreamerV3 三种注入更新、旧格式兼容。
- 两种方法关闭 context 时，两维/四维历史观测下初始化的全部模型参数逐项精确一致。
- 两种方法各完成一次 CUDA FP16 autocast 的反向传播和优化器更新，所有指标有限。
- 使用 `env.context_kinematics=false`，当前代码成功 strict 加载已有 step 150006 的
  R2Dreamer context best 检查点，编码器仍为原四维 token 输入。
- 完整测试 33 项中 31 项通过；真实 MetaDrive 底盘采样测试已核对车速、前轮角、历史形状与空间契约。
- 两项失败为 `test_random_entrances_and_fixed_evaluation_routes` 和
  `test_random_spawn_does_not_overlap_native_traffic`。在相同运行时下替换为已有实验的冻结
  MetaDrive 适配器后，两项仍以相同断言失败（未清空触发交通、首步碰撞/终止）。
  不归因于本轮 context 改动，也未为本轮改动修改交通或出生点逻辑。
- Python 语法编译与 git diff whitespace 检查通过。

本轮没有启动正式训练或改写旧实验。已有两维 context 检查点可使用
`env.context_kinematics=false` 配合旧 replay；新增六维 token 的实验需要新目录和新检查点。
也可继续通过已有实验的冻结源码运行旧版评估。

## 运行时说明

完整环境测试使用当前 Python 环境的依赖，并显式加载机器上可用的完整 MetaDrive
包；没有覆盖或修改模拟器包。本次交通测试结论限定于该运行时，不能宣称全套回归全通过。

### 正式训练前的运行时复核

原实验配置的 editable MetaDrive 运行时为 `/home/server/projects/metadrive`。
显式设置 `PYTHONPATH=/home/server/projects/metadrive` 后重新运行完整测试，
33/33 全部通过，包括上述两项交通测试。此前 31/33 是另一份机器上可用的模拟器包下的结果，
不作为本轮正式训练的运行时结果。新训练沿用原实验 editable 路径并记录此环境变量。
