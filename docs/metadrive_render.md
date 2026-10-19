# MetaDrive 模型渲染测试

在项目根目录使用已有 `r2dreamer` Conda 环境。脚本只执行推理，不启动训练、不建立 replay buffer、不修改模型权重。

## 一键启动脚本

编辑 `scripts/test_metadrive_render.sh` 开头的 `CHECKPOINT`（权重路径）、`POLICY`（模型接口）和 `RENDER`（显示方式），然后执行：

```bash
bash scripts/test_metadrive_render.sh
```

默认接口是 `scripts.metadrive_policy_template:load_policy`，在对应模板中修改模型即可。`CONFIG` 留空会读取权重旁边的 `.hydra/config.yaml`；找不到训练配置时脚本会报错，避免误用其他环境参数。

`EPISODES`、`SEED_START`、`DEVICE`、`MAX_STEPS` 默认留空，分别沿用训练的评估回合数、评估起始种子、设备和回合时限。道路、交通、雷达、奖励、动作重复和终止条件也沿用训练配置。渲染测试使用一个环境逐回合运行。JSON 默认输出到 `logdir/render_test/latest_test.json`，每次运行覆盖该测试结果；需要保留多次结果可修改 `OUTPUT`。

## 同时显示 3D 与 top-down

```bash
conda activate r2dreamer
cd /home/server/projects/r2dreamer
python scripts/test_metadrive_render.py \
  --checkpoint logdir/dreamerv3_original_reward_500k_20261014_005208/dreamerv3/best.pt \
  --render both --episodes 20 --seed-start 10000 --fps 10 \
  --output logdir/render_test/dreamerv3_best.json
```

`--render 3d` 只显示 Panda3D 窗口，`--render topdown` 只显示俯视窗口，`--render both` 同时显示两个窗口，`--render none` 关闭显示。窗口需要当前桌面的 DISPLAY。Ctrl+C 退出并保存已完成回合的数据；top-down 窗口也可用 Esc 退出。

每步显示场景 seed、步数、累计奖励和路线完成率。终止后打印成功、碰撞、驶出道路、压黄/白实线等标记；这些标记可能同时成立。默认回合时限来自训练配置。`--max-steps` 可额外限制测试的策略决策次数，提前结束会标记 `test_step_cap`，不能把这种短回合当成完整评估。`--fps 0` 不额外等待；默认 10 约对应当前 action_repeat=1、decision_repeat=5 的仿真时间。

模型输入仍是训练用的 `state` 雷达向量（当前 259 维），渲染图像不会作为模型输入。每个回合重置模型记忆；评估采用动作分布的 mode，RSSM 隐状态采样仍可能引入随机性。固定种子提高重复测试的一致性，但单环境渲染测试不保证与原来两环境并行评估逐回合完全一致。

## 配置与 checkpoint

脚本默认读取 checkpoint 同目录的 `.hydra/config.yaml`，保留模型结构、奖励模式、车道/交通设置与终止条件。若将权重复制到其他目录，请用 `--config /absolute/path/to/.hydra/config.yaml` 指明完整训练配置。当前默认使用工作区中的模型代码，便于修改代码后测试。需要兼容原 checkpoint 的网络结构；结构变化时请加载匹配权重。

可重复传入 `--override` 调整新模型配置，例如：

```bash
python scripts/test_metadrive_render.py --checkpoint /path/to/model.pt \
  --config /path/to/training_config.yaml \
  --override model.rep_loss=dreamer --render both
```

本仓库的 R2Dreamer 与 PyTorch DreamerV3 共用 `Dreamer` 类，脚本会通过训练配置选择相应结构。可换成 R2Dreamer 的 checkpoint 路径直接测试。默认 device 沿用训练配置，也可指定 `--device cpu`。

奖励默认与 checkpoint 的训练配置一致；需要对比时使用 `--reward-mode additive` 或 `--reward-mode metadrive`。修改奖励只改变评测分数，不会更新策略。

## 修改模型接口

模板：`scripts/metadrive_policy_template.py`。在 `MyPolicy` 中修改：

- `load_model(...)`：模型构造、权重加载和 eval 模式设置。
- `reset(seed)`：每个回合清空循环状态、历史或其他缓存。
- `act(obs)`：输入观测字典，返回 NumPy `(2,)` 动作 `[转向, 油门/刹车]`，范围 `[-1, 1]`。可通过 `obs['state']` 获取雷达状态。

```bash
python scripts/test_metadrive_render.py \
  --policy scripts.metadrive_policy_template:load_policy \
  --checkpoint /path/to/your_model.pt \
  --config /path/to/your_training_config.yaml \
  --render both --episodes 20 --output logdir/render_test/custom.json
```

也可以新建自己的模块，用 `--policy your_module:your_factory` 加载。factory 签名为 `(config, obs_space, action_space, checkpoint, device)`，返回具有 `reset(seed)` 和 `act(obs)` 方法的对象。可完全替换为其他框架的模型。环境负责动作的有效性检查和裁剪。

## 检查渲染与保存俯视图

不加载模型、先用随机动作检查窗口：

```bash
python scripts/test_metadrive_render.py --policy random --render both --episodes 1 --max-steps 30
```

无桌面显示时可生成俯视图：

```bash
python scripts/test_metadrive_render.py --policy random \
  --render topdown --no-window --episodes 1 --max-steps 30 --fps 0 \
  --save-topdown logdir/render_test/topdown
```

`--save-topdown` 保存每回合起始及结束的 PNG；也可以配合窗口模式。`--output` JSON 包含实际测试配置、每回合指标和平均指标。脚本运行异常会正常释放环境；未完成回合不会计入统计。
