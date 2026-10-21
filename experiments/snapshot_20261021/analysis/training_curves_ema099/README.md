# 对齐TensorBoard的EMA0.99训练曲线

原始51个固定20回合验证点，与TensorBoard逐点核验一致。沿时间顺序按TensorBoard偏差修正EMA算法计算：acc_i=0.99*acc_(i-1)+0.01*y_i，初始acc=0；smooth_i=acc_i/(1-0.99^(i+1))。两组和四项指标使用相同权重。百分比面板乘100显示。没有额外高斯平滑、插值、标准差阴影或原始细线。原始数据及平滑数据保留在CSV。单训练seed。
