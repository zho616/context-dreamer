# 实际轨迹、控制曲线与稳定性评估

本次结果：11在三类场景的实际前轮角平均变化率分别比基线降低约23%、23%、21%，转向动作变化率也降低，且共同前10秒的配对比较保持同方向。09在三类场景的转向动作及实际轮角变化率均高于基线，不能宣称其控制更平滑。11纵向jerk在新地图点估计略低、S/XL略高，三类全段配对区间均跨0，尚无全面改善证据。

解释：11在本样本中平均车速并不低于基线；15–25km/h速度分层中转向变化率仍较低（XL样本区间跨0）。但速度分层不是相同交通状态的严格控制，不能独立归因于context。单训练seed、每类20回合，不能代替此前100回合驾驶成功率结果。

重现核验：180回合的成功/失败、步数、实际车辆参数和奖励与此前已保存的独立测试相同，见reproducibility_check.json。

冻结基线、09、11的best，严格加载各自训练源码；新地图/S/XL各20回合，每组60，共180回合。实际地图种子、底盘车型/质量/最大转角、初始位置和航向均逐回合核验一致。该样本是先前100回合评估的前20个种子，不能与旧结果叠加计算独立样本数。

## 驾驶表现

|测试池|方法|成功率|碰撞率|出界率|速度km/h|路线完成率|
|---|---|---:|---:|---:|---:|---:|
|new_maps|baseline|80%|20%|0%|21.03|89.5%|
|new_maps|09_deep|85%|10%|5%|20.03|89.9%|
|new_maps|11_shallow|80%|20%|0%|24.25|87.4%|
|s|baseline|85%|5%|10%|21.41|87.8%|
|s|09_deep|85%|5%|10%|19.74|88.7%|
|s|11_shallow|90%|10%|0%|23.02|90.3%|
|xl|baseline|55%|15%|30%|17.46|70.7%|
|xl|09_deep|35%|30%|35%|15.28|63.4%|
|xl|11_shallow|35%|30%|35%|17.95|64.8%|

## 每回合等权的稳定性指标

排除初始1秒、结束/碰撞帧；原始决策频率信号，不作平滑。转向/油门变化率为mean(|Δu|/Δt)；实际轮角变化率单位deg/s；jerk为纵向投影加速度差分，单位m/s³；加速度单位m/s²。轮角/横摆角速度均右正；转向动作保留MetaDrive左正，两者符号不同。更小的控制变化率不能单独代表更安全。

|测试池|方法|转向动作变化率|油门变化率|前轮角变化率|平均绝对jerk|平均绝对加速度|车道横偏RMS m|
|---|---|---:|---:|---:|---:|---:|---:|
|new_maps|baseline|1.183|4.159|50.540|29.247|2.756|1.019|
|new_maps|09_deep|2.172|4.076|91.141|29.043|2.789|1.115|
|new_maps|11_shallow|0.913|3.636|38.752|26.800|3.170|1.167|
|s|baseline|1.182|4.037|58.171|26.250|2.540|1.111|
|s|09_deep|1.856|4.144|91.824|28.091|2.666|1.124|
|s|11_shallow|0.910|3.982|45.072|27.602|2.972|1.213|
|xl|baseline|0.978|2.280|34.450|10.267|1.584|0.891|
|xl|09_deep|1.936|3.322|68.950|15.437|1.705|1.063|
|xl|11_shallow|0.769|2.618|27.384|12.497|1.829|0.889|

## 速度分层与相同时间窗配对差

以下差值为方法减基线，负值表示变化率/jerk更小。速度分层不是严格的同状态控制，子集回合数可少于20。common_prefix使用每个场景三方法共同可用的前≤10秒，排除初始1秒；样本不足5帧的回合不进入该指标。95%区间按配对地图种子bootstrap10000次；一个训练seed。

|测试池|范围|方法|指标|配对回合数|差值|95%区间|
|---|---|---|---|---:|---:|---|
|new_maps|all_safe|09_deep|steer_rate|20|+0.989|[+0.751, +1.274]|
|new_maps|all_safe|09_deep|throttle_rate|20|-0.083|[-0.547, +0.432]|
|new_maps|all_safe|09_deep|wheel_rate_deg|20|+40.601|[+31.634, +50.435]|
|new_maps|all_safe|09_deep|jerk_abs|20|-0.203|[-2.505, +2.688]|
|new_maps|all_safe|11_shallow|steer_rate|20|-0.269|[-0.405, -0.143]|
|new_maps|all_safe|11_shallow|throttle_rate|20|-0.523|[-1.081, -0.082]|
|new_maps|all_safe|11_shallow|wheel_rate_deg|20|-11.788|[-17.978, -6.603]|
|new_maps|all_safe|11_shallow|jerk_abs|20|-2.447|[-6.247, +0.142]|
|new_maps|common_prefix|09_deep|steer_rate|20|+0.573|[+0.340, +0.794]|
|new_maps|common_prefix|09_deep|throttle_rate|20|+0.306|[-0.488, +1.084]|
|new_maps|common_prefix|09_deep|wheel_rate_deg|20|+24.227|[+14.454, +33.813]|
|new_maps|common_prefix|09_deep|jerk_abs|20|+0.709|[-4.516, +5.512]|
|new_maps|common_prefix|11_shallow|steer_rate|20|-0.229|[-0.419, -0.064]|
|new_maps|common_prefix|11_shallow|throttle_rate|20|-0.460|[-1.237, +0.154]|
|new_maps|common_prefix|11_shallow|wheel_rate_deg|20|-10.102|[-18.943, -2.502]|
|new_maps|common_prefix|11_shallow|jerk_abs|20|-4.309|[-10.907, +0.085]|
|new_maps|speed_15_25|09_deep|steer_rate|20|+0.995|[+0.652, +1.371]|
|new_maps|speed_15_25|09_deep|throttle_rate|20|-0.080|[-0.865, +0.717]|
|new_maps|speed_15_25|09_deep|wheel_rate_deg|20|+39.008|[+25.822, +52.745]|
|new_maps|speed_15_25|09_deep|jerk_abs|20|-1.685|[-6.470, +3.218]|
|new_maps|speed_15_25|11_shallow|steer_rate|20|-0.227|[-0.403, -0.061]|
|new_maps|speed_15_25|11_shallow|throttle_rate|20|-0.368|[-0.970, +0.309]|
|new_maps|speed_15_25|11_shallow|wheel_rate_deg|20|-10.248|[-18.616, -3.088]|
|new_maps|speed_15_25|11_shallow|jerk_abs|20|-3.170|[-7.977, +1.716]|
|new_maps|speed_25_35|09_deep|steer_rate|20|+0.729|[+0.425, +1.047]|
|new_maps|speed_25_35|09_deep|throttle_rate|20|+0.031|[-0.530, +0.666]|
|new_maps|speed_25_35|09_deep|wheel_rate_deg|20|+29.274|[+16.871, +41.907]|
|new_maps|speed_25_35|09_deep|jerk_abs|20|-2.971|[-5.590, -0.084]|
|new_maps|speed_25_35|11_shallow|steer_rate|20|-0.310|[-0.452, -0.181]|
|new_maps|speed_25_35|11_shallow|throttle_rate|20|+0.345|[-0.061, +0.752]|
|new_maps|speed_25_35|11_shallow|wheel_rate_deg|20|-13.426|[-19.190, -8.009]|
|new_maps|speed_25_35|11_shallow|jerk_abs|20|+1.271|[-0.795, +3.348]|
|s|all_safe|09_deep|steer_rate|20|+0.674|[+0.379, +0.981]|
|s|all_safe|09_deep|throttle_rate|20|+0.107|[-0.271, +0.477]|
|s|all_safe|09_deep|wheel_rate_deg|20|+33.653|[+18.397, +50.400]|
|s|all_safe|09_deep|jerk_abs|20|+1.842|[+0.118, +3.707]|
|s|all_safe|11_shallow|steer_rate|20|-0.271|[-0.396, -0.150]|
|s|all_safe|11_shallow|throttle_rate|20|-0.055|[-0.502, +0.358]|
|s|all_safe|11_shallow|wheel_rate_deg|20|-13.099|[-19.349, -7.108]|
|s|all_safe|11_shallow|jerk_abs|20|+1.353|[-1.181, +3.938]|
|s|common_prefix|09_deep|steer_rate|20|+0.404|[+0.080, +0.744]|
|s|common_prefix|09_deep|throttle_rate|20|+0.519|[-0.069, +1.108]|
|s|common_prefix|09_deep|wheel_rate_deg|20|+21.146|[+4.288, +39.769]|
|s|common_prefix|09_deep|jerk_abs|20|+3.486|[+0.512, +6.475]|
|s|common_prefix|11_shallow|steer_rate|20|-0.182|[-0.382, -0.029]|
|s|common_prefix|11_shallow|throttle_rate|20|-0.430|[-1.059, +0.166]|
|s|common_prefix|11_shallow|wheel_rate_deg|20|-8.733|[-18.372, -1.533]|
|s|common_prefix|11_shallow|jerk_abs|20|-1.402|[-4.996, +2.384]|
|s|speed_15_25|09_deep|steer_rate|20|+0.724|[+0.309, +1.164]|
|s|speed_15_25|09_deep|throttle_rate|20|+0.559|[-0.196, +1.385]|
|s|speed_15_25|09_deep|wheel_rate_deg|20|+37.313|[+14.671, +61.377]|
|s|speed_15_25|09_deep|jerk_abs|20|+3.881|[-0.633, +8.674]|
|s|speed_15_25|11_shallow|steer_rate|20|-0.345|[-0.508, -0.187]|
|s|speed_15_25|11_shallow|throttle_rate|20|+0.461|[-0.216, +1.153]|
|s|speed_15_25|11_shallow|wheel_rate_deg|20|-17.163|[-25.796, -8.848]|
|s|speed_15_25|11_shallow|jerk_abs|20|+3.312|[-1.693, +7.990]|
|s|speed_25_35|09_deep|steer_rate|18|+0.465|[+0.161, +0.751]|
|s|speed_25_35|09_deep|throttle_rate|18|-0.240|[-0.648, +0.205]|
|s|speed_25_35|09_deep|wheel_rate_deg|18|+23.203|[+8.961, +37.394]|
|s|speed_25_35|09_deep|jerk_abs|18|-3.594|[-6.638, -0.676]|
|s|speed_25_35|11_shallow|steer_rate|19|-0.256|[-0.397, -0.132]|
|s|speed_25_35|11_shallow|throttle_rate|19|-0.458|[-0.982, +0.058]|
|s|speed_25_35|11_shallow|wheel_rate_deg|19|-11.677|[-17.828, -6.010]|
|s|speed_25_35|11_shallow|jerk_abs|19|-2.895|[-6.063, +0.174]|
|xl|all_safe|09_deep|steer_rate|20|+0.957|[+0.650, +1.296]|
|xl|all_safe|09_deep|throttle_rate|20|+1.042|[+0.407, +1.722]|
|xl|all_safe|09_deep|wheel_rate_deg|20|+34.500|[+22.900, +47.867]|
|xl|all_safe|09_deep|jerk_abs|20|+5.170|[+2.872, +7.691]|
|xl|all_safe|11_shallow|steer_rate|20|-0.210|[-0.365, -0.058]|
|xl|all_safe|11_shallow|throttle_rate|20|+0.338|[-0.260, +0.990]|
|xl|all_safe|11_shallow|wheel_rate_deg|20|-7.066|[-13.295, -0.960]|
|xl|all_safe|11_shallow|jerk_abs|20|+2.230|[-0.554, +5.275]|
|xl|common_prefix|09_deep|steer_rate|20|+0.271|[+0.034, +0.516]|
|xl|common_prefix|09_deep|throttle_rate|20|+1.149|[+0.478, +1.831]|
|xl|common_prefix|09_deep|wheel_rate_deg|20|+10.359|[+1.788, +19.374]|
|xl|common_prefix|09_deep|jerk_abs|20|+5.366|[+2.896, +7.796]|
|xl|common_prefix|11_shallow|steer_rate|20|-0.227|[-0.391, -0.076]|
|xl|common_prefix|11_shallow|throttle_rate|20|+0.044|[-0.283, +0.374]|
|xl|common_prefix|11_shallow|wheel_rate_deg|20|-8.159|[-14.672, -2.353]|
|xl|common_prefix|11_shallow|jerk_abs|20|+0.460|[-0.869, +1.902]|
|xl|speed_15_25|09_deep|steer_rate|19|+1.016|[+0.552, +1.501]|
|xl|speed_15_25|09_deep|throttle_rate|19|+0.962|[+0.130, +1.764]|
|xl|speed_15_25|09_deep|wheel_rate_deg|19|+36.016|[+19.546, +54.834]|
|xl|speed_15_25|09_deep|jerk_abs|19|+4.961|[+1.765, +8.128]|
|xl|speed_15_25|11_shallow|steer_rate|17|-0.177|[-0.398, +0.026]|
|xl|speed_15_25|11_shallow|throttle_rate|17|+0.512|[-0.506, +1.496]|
|xl|speed_15_25|11_shallow|wheel_rate_deg|17|-5.938|[-13.956, +1.578]|
|xl|speed_15_25|11_shallow|jerk_abs|17|+3.165|[-0.743, +6.909]|
|xl|speed_25_35|09_deep|steer_rate|9|+0.886|[+0.359, +1.441]|
|xl|speed_25_35|09_deep|throttle_rate|9|+0.569|[-0.070, +1.166]|
|xl|speed_25_35|09_deep|wheel_rate_deg|9|+29.164|[+13.432, +44.197]|
|xl|speed_25_35|09_deep|jerk_abs|9|+0.412|[-2.130, +2.818]|
|xl|speed_25_35|11_shallow|steer_rate|12|-0.381|[-0.694, -0.151]|
|xl|speed_25_35|11_shallow|throttle_rate|12|-0.030|[-0.830, +0.819]|
|xl|speed_25_35|11_shallow|wheel_rate_deg|12|-14.165|[-27.330, -4.863]|
|xl|speed_25_35|11_shallow|jerk_abs|12|-0.139|[-3.046, +2.940]|

## 图与解释边界

- stability_overview.png/pdf/svg：全部回合分布、均值和bootstrap区间。
- examples/new_maps_10020、s_10320、xl_10420：预先固定的首个种子，均导出PNG/PDF/SVG；examples内还有全部60个种子的PNG，未按优势挑选案例。
- 每张案例图包含实际位置轨迹和原始控制、轮角、速度、加速度、jerk曲线，碰撞/出界结束点也保留用于展示。统计时事件帧被排除。静态道路背景来自实际生成的同一地图；不是世界模型生成。交通车辆省略，策略交互后其他车辆可出现不同响应。
- 当前lane的横偏会受车道切换影响，不能单独作为稳定性证据。异常帧和结束造成的生存偏差需结合common_prefix、速度分层和失败率阅读。
- 这些图不涉及预测解码器，也不能证明RSSM未来重建更准，结论仅限闭环驾驶与控制。
