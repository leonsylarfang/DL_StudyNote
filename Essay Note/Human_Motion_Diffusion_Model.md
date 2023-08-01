Human Motion Diffusion Model
====
[Tevet et al.](https://arxiv.org/abs/2209.14916)
> 结合diffusion来生成动作

## 结构
首先我会根据框架来介绍本文的模型结构：
![](/Essay%20Note/images/MDM_1.jpg)

文中提出了一个基于运动的扩散模型（Motion Diffusion Model, MDM）。该模型由一个只有编码器结构的transformer模型构成。这个结构是时间感知的，可以学习任意长度的运动。通过给定的CLIP文本提示 $c$ ，并给MLP喂入噪声步长 $t$ 和对应动作序列 $x_t^{1:N}$ 来得到 $z_{tk}$ 。每一帧的噪声输入 $x_t$ 都被线性投影到transformer维度并与标准位置嵌入求和，再跟 $z_{tk}$ 一起输入编码器中经过编码得到预测 $\hat x_0=G(x_t,t,c)$ 。

**扩散采样**框架如下：
![](/Essay%20Note/images/MDM_2.jpg)

Diffusion模型是一个向原始数据分布 $x_0$ 不断添加噪声的过程。对于每一个时间 $t$ ，总有

$$
q(x^{1:N}_t|x^{1:N}_{t-1})=\mathcal{N}(\sqrt{\alpha _t}x^{1:N}_{t-1},(1-\alpha_t)I)
$$
其中 $\alpha _t$ 是一个常超参。当 $\alpha _t$ 足够小时，$x_T^{1:N}$ 可以被近似看做服从标准正态分布 $\mathcal{N}(0,I)$ 。

条件运动合成将 $p(x_0|c)$ 建模为逐渐给 $x_T$ 去噪的反向扩散过程。在每一步中都把自身值即 $\hat{x}_0=G(x_t,t,c)$ 作为预测值输入扩散网络中去噪得到 $x_{t-1}$ 知道最终得到 $x_0$，其简单损失方式为：
$$
L_{simple}=\mathbb{E}_{x_0\sim q(x_0|c),t\sim [1,T]}[\left\|x_0-G(x_t,t,c) \right\|^2_2]
$$

---
## 损失函数
[Petrovich et al.](https://arxiv.org/abs/2104.05670) 和 [Shi et al.](https://arxiv.org/abs/2006.12075) 都曾提出利用几何损失来正则化生成网络可以使生成的动作具有物理属性并能更自然。文中试验了以下三种几何损失：
$$
\begin{aligned}
L_{position}&=\frac{1}{N}\sum^{N}_{i=1}\left\|FK(x_0^i)-FK(\hat{x}_0^i) \right\|^2_2 \\
L_{foot\;contact}&=\frac{1}{N-1}\sum^{N-1}_{i=1}\left\|\big (FK(\hat x_0^{i+1})-FK(\hat{x}_0^i) \big ) \cdot f_i \right\|^2_2 \\
L_{velocity}&=\frac{1}{N-1}\sum^{N-1}_{i=1}\left\|(\hat x_0^{i+1} - x_0^i)-(\hat{x}_0^{i+1}-\hat x_0^i) \right\|^2_2
\end{aligned}
$$

其中 $FK(\cdot)$ 表示将关节旋转转换为关节位置的正运动学函数(否则表示恒等函数)，$f_i \in \{0,1 \}^J$ 表示每一帧的二进制掩码，代表是否触碰到地面。

由此可得训练损失函数为：
$$
L=L_{simple}+\lambda_{pos}L_{pos}+\lambda_{vel}L_{vel}+\lambda_{foot}L_{foot}
$$

## MDM详解

MDM的目标是根据给定的任意条件 $c$ 生成长度为 $N$ 的**人体运动**（**human motion**） $x^{1:N}$。

该条件 $c$ 可以使现实世界中任何可以影响生成的信号，包括音频、自然语言（文字）和离散类（行为-动作）。若把该条件设为空集即可实现无条件动作生成。所以文中把随机10%样本的 $c$ 设成空集来拟合 $p(x_0)$ ，从而使得模型 $G$ 同时学习条件分布和无条件分布。这样当采样G时，我们可以通过使用s内插或外推两个变体来平衡多样性和精准度：
$$
G_s(x_t,t,c)=G(x_t,t,\emptyset)+s\cdot \big(G(x_t,t,c)-G(x_t,t,\emptyset) \big)
$$

而该生成的动作 $x^{1:N}=\{x^i\}^N_{i=1}$ 是一组由关节的旋转或位置 $x^i \in \mathbb{R}^{J\times D}$ 来表征的人体姿势序列，其中 $J$ 是关节个数， $D$ 是关节表征的维度。

---

整个采样扩散过程会分成时域和空间域来看待。时域中，每一次循环都会用根据运动输入重新计算出 $\hat x_0$ ，从而使得生成结果不会与原始输入偏差过大。而空间域中，身体的局部可以在不影响其他部分的情况下完成修改编辑。

## 实验
本文共进行了三种不同的实验：
- 文本到动作：通过输入文本提示生成动作
- 行为到动作：通过输入行为组来生成动作
- 无条件生成