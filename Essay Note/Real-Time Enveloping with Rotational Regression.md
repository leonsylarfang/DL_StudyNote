Real-Time Enveloping with Rotational Regression
====
[Wang et al.](https://dl.acm.org/doi/10.1145/1276377.1276468)
> 相较线性混合蒙皮更真实且一样快

## Overview
![](/Essay%20Note/images/RTERR_Overview.png)
如上图所示，本文的主题思想是把包裹问题变为一个 mapping 问题，将骨骼姿态（skeletal pose）$\mathbf{q}$ 与其对应的 mesh $\mathbf{y(q)}$ 做 mapping。该 mapping 分为两步：

1. 根据 skeletal pose 去预测 mesh 的形变梯度（deformation gradients）。
2. 通过将上述预测带入求解 Poisson equation 来重建点的位置。

## 形变梯度预测


