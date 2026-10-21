# T-intersection：质量与轮胎摩擦随机化，baseline / 11各500K

实验：`logdir/tintersection_mass_friction_500k_20261021/`。

用户要求替换上一组质量/最大转角实验；旧队列已停止，目录和检查点保留，
不把两种随机化的曲线拼接。两组均从头训练，不从旧模型续训。

每回合独立抽取，回合内固定：

- 车型：default / M / L。
- 质量倍率：0.6、0.7、0.8、0.9、1.0、1.1、1.2、1.3、1.4；质量按原规则取整。
- wheel_friction倍率：同上9档，乘各车型名义摩擦值。
- 最大前轮角：各车型名义值，steering_scales=[1.0]，不再随机缩放。

共3×9×9=243个参数配置。default/M/L摩擦范围约0.54–1.26、0.45–1.05、
0.48–1.12。该值是Bullet轮胎FrictionSlip参数，并非直接标定的真实路面μ。
发动机、制动力仍固定为各车型参数范围中点。

物理实现：出生前写入车辆配置；每次reset后对四轮逐一setFrictionSlip，
读回getFrictionSlip核验，涵盖车辆池复用。实际质量仍写入Bullet车体。

11保持state4/K5/context8、posterior-only Shallow、aux权重1、swap权重1；
只将交换配对的物理标识扩展为车型、实际质量、实际最大转角、实际wheel_friction。
摩擦元数据只用于选供体，不作为context编码器、响应预测器或RSSM的输入。
baseline关闭context与swap。网络结构、优化器和训练预算不变。

共同设置：T路口、三入口随机出生和不同可达出口，seed0，8训练环境，
batch16×64，ratio512，CPU replay500K，paper reward；训练地图0–99，
固定验证10000–10019，每10K步评估20回合、每20K保存latest。
baseline训练完成并测试后，自动开始11；各自best在新地图/S/XL各测100回合，
测试质量/摩擦倍率与训练相同。

实现及来源见provenance.json/source.patch；launch.json为正式命令和PID，
training/status.json为进度。smoke固定default/质量1/摩擦1确保执行交换路径，
不能用来估计正式243配置的配对覆盖率。

TensorBoard已单独放到`http://192.168.118.78:6015/`，仅显示：

- `01_500K_baseline`
- `11_500K_state4_DALI_shallow_swap1`

原6014的T-intersection入口13–16均已移出展示；原13/14对应旧质量/转角实验，
停止状态、日志和检查点保留在原实验目录，不放入新TensorBoard。
