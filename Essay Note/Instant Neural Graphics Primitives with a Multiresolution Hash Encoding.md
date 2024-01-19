Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
====
[Muller et al.](https://nvlabs.github.io/instant-ngp/)
> 新的通用性输入编码

## 多分辨率哈希编码
该编码技术的主要优势在于：快速训练(rapid training)、高质量(high quality)和易用性(simplicity)。
本笔记重点旨在介绍**多分辨率哈希编码** **(Multiresolution Hash Encoding)** 在 [NeRF](https://arxiv.org/abs/2003.08934) 上的应用。

首先是快速渲染效果图：
![](/Essay%20Note/images/instantNGP_1.png)

### 概念原理
给定一个全连接神经网络 $m(\mathbf{y};\Phi)$，其输入 $\mathbf{y}=\mathrm{enc}(\mathbf{x};\theta)$ 的编码 (encoding) 对预测结果的质量和训练速度至关重要。本文提出的网络同时包含可训练的网络权值参数 $\Phi$ 和 编码参数 $\theta$， 这些参数被分配到 $L$ 层网络中，每层包含 $T$ 个维度为 $F$ 的特征向量，其取值范围如下表：

| 参数 | 符号 | 取值 |
| ---- | ---- | ----|
| 层数 | $L$ | $16$ |
| 每层最多元素（哈希表大小） | $T$ | $2^{14}$ 到 $2^{24}$ |
| 每个元素的特征维度 | $F$ | $2$ |
| 粗分辨率 | $N_{min}$ | $16$ |
| 精分辨率 | $N_{max}$ | $512$ 到 $524288$ |

那么 2D 哈希编码可以如下图分为五步（其中蓝色和红色代表不同的分辨率下独立的操作）：

![](/Essay%20Note/images/instantNGP_2.png)

(1) 对于给定的输入坐标 $\mathbf{x}$，找到其周围 $L$ 个分辨率级别下的体素，然后通过散列其整数坐标来将索引分配给体素的8个顶点。
(2) 在哈希编码表 $\theta_l$ 中 找到8个顶点的索引对应的 $F$ 维特征向量。
(3) 根据 $\mathbf{x}$ 在相应的第 $l$ 个体素内的相对位置来进行线性插值。
(4) 将每个分辨率下的结果，以及辅助输入 $\xi \in \mathbb{R}^E$ 拼接作为编码输入 $y\in \mathbb{R}^{LF+E}$。 
(5) 将编码输入 $y$ 送入 MLP 中训练。

每一层的分辨率都会在区间 $[N_{min},N_{max}]$ 内按增长因子 $b$ 几何变化，那么第 $l$ 层的分辨率为：
$$
N_l:= \left \lfloor N_{min} \cdot b^l \right \rfloor 
$$
$$
b:=\exp\frac{\ln N_{max}-\ln N_{min}}{L-1}
$$

只考虑单层情况，第 $l$ 层的输入坐标 $\mathbf{x}\in \mathbb{R}^d$ 在上下舍入之前会被该层网格分辨率缩放，即 $\left \lfloor \mathbf{x}_l \right \rfloor := \left \lfloor \mathbf{x}\cdot N_l \right \rfloor, \left \lceil \mathbf{x}_l \right \rceil := \left \lceil  \mathbf{x}\cdot N_l  \right \rceil $。
$\left \lfloor \mathbf{x}_l \right \rfloor$ 到 $\left \lceil \mathbf{x}_l \right \rceil$ 跨越了1个包含 $2^d$ 个 $\mathbb{Z}^d$ 范围内整点的体素。


