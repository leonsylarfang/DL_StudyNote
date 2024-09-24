Kalman-Inspired Feature Propagation for Video Face Super-Resolution
=====
[Feng et al.](https://arxiv.org/abs/2408.05205#)

> 保证帧一致性的人脸复原

## 算法介绍
文中将基于图像的 [CodeFormer](https://arxiv.org/abs/2206.11253) 用于 VFSR (Video Face Super Resolution)，并提出 KEEP (Kalman-inspired Feature Propagation) ，其能在给定含噪声且不准确预测的情况下，根据时间在隐空间中预测稳定的人脸先验，从而实现时序相关恢复。
### 1. 公式部分
#### 状态空间模型
首先定义视频序列：
- 长度为 $T$ 的低质量（LQ）视频序列被记为 $X=\{{\boldsymbol{x}_t}\}^T_{t=1}$，其中 $\boldsymbol{x}_t\in\mathbb{R}^{H\times W\times 3}$。
- 长度为 $T$ 的高质量（HQ）视频序列被记为 $Y=\{{\boldsymbol{y}_t}\}^T_{t=1}$，其中 $\boldsymbol{y}_t\in\mathbb{R}^{H\times W\times 3}$。

根据 Kalman filter，线性动态系统的特征是一个由高斯噪声驱动的空间状态模型：$$\boldsymbol{y}_t=\boldsymbol{F}_t\boldsymbol{y}_{t-1}+\boldsymbol{q}_t$$ 其中 $\boldsymbol{F}_t$ 是转移矩阵，$\boldsymbol{q}_t$ 代表从高斯噪声中采样得到的处理噪声。
观察值 $\boldsymbol{x}_t=\boldsymbol{H}\boldsymbol{y}_t+\boldsymbol{r}_t$，$\boldsymbol{H}$ 是测量矩阵，$\boldsymbol{r}_t$ 代表测量噪声。

为了在真实世界的复杂场景中表示，需要引入非线性 Kalman filter，即：$$\boldsymbol{y}_t=d(\boldsymbol{y}_{t-1},\boldsymbol{q}_t)\\\boldsymbol{x}_t=h(\boldsymbol{y}_t)+\boldsymbol{r}_t$$   其中 $d(\cdot)$ 和 $h(\cdot)$ 是非线性转移和测量模型。

由 VQ-GAN 和 Stable Diffusion 启发，潜在隐式表征 $Z=\{{\boldsymbol{z}_t}\}^T_{t=1}$ 可以通过一个生成模型 $g_\theta$ 得到 $\boldsymbol{y}_t$，即 $\boldsymbol{y}_t=g_\theta(\boldsymbol{z}_t)$。

#### Kalman filter 模型
<div align="center">
<img src="/Essay%20Note/images/KEEP_Kalman_filter.jpg" width=500 height=250 />
<br>图1：Kalman filter 步骤
</div>

如图1所示，Kalman filter 可以被分为两步：状态预测和状态更新。其中 $\boldsymbol{x}_t$ 代表人脸图片，$\boldsymbol{z}_t$ 代表状态。

1. 在**状态预测**步骤，根据之前状态和动态模型的后验估计 $\hat{\boldsymbol{z}}^+_{t-1}$ 来得到当前状态 $\boldsymbol{z}_t$ 的先验估计 $\hat{\boldsymbol{z}}^-_t$，即 $$\hat{\boldsymbol{z}}^-_t=f(\hat{\boldsymbol{z}}^+_{t-1})\\\hat{\boldsymbol{x}}^-_t=h(g_\theta(\hat{\boldsymbol{z}}^-_t))\tag{1}$$ 其中 $+$ 和 $-$ 分别代表后验和先验。$f$ 定义了隐状态 $\boldsymbol{Z}$ 随着时间的变化，并融入了任何对当前状态产生影响的控制输入。
2. 在**状态更新**步骤，根据先验估计 $\hat{\boldsymbol{z}}^-_t$ 和新的观测值 $\boldsymbol{x}_t$ 来计算后验状态估计 $\hat{\boldsymbol{z}}^+_t$，即 $$\hat{\boldsymbol{z}}^+_t=\hat{\boldsymbol{z}}^-_t+\mathcal{K}_t\Delta\boldsymbol{z}_t\tag{2}$$ 其中 $\mathcal{K}_t$ 是 Kalman Gain ；$\Delta\boldsymbol{z}_t=\hat{\boldsymbol{z}}^-_t-\tilde{\boldsymbol{z}}_t$ 代表先验估计 $\hat{\boldsymbol{z}}^-_t$ 和由 $\boldsymbol{x}_t$ 得到的当前状态估计 $\tilde{\boldsymbol{z}}_t=e(\boldsymbol{x}_t)$ 之间的残差。

这样就得到了最终的预测值 $\hat{\boldsymbol{y}}_t=g_\theta(\hat{\boldsymbol{z}}^+_t)$。

### 2. 参数化模型
图2是KEEP算法总览：
<div align="center">
<img src="/Essay%20Note/images/KEEP_overview.jpg" width=800 height=320 />
<br>图2：KEEP总览
</div>

#### 2.1 生成模型
生成模型 $g_\theta$ 由一个 LQ 编码器 $\mathcal{E}_L$，一个 HQ 编码器 $\mathcal{E}_H$，一个结合了码本查找（codebook lookup） Transformer 跟量化层 $T_Q$ 的解码器 $\mathcal{D}_Q$ 组成。观测状态 $\tilde{\boldsymbol{z}}_t$ 可以被估计为 $\tilde{\boldsymbol{z}}_t=e(\boldsymbol{x}_t)=\mathcal{E}_L(\boldsymbol{x}_t)$。

#### 2.2 状态动态系统
当前时刻状态 $\tilde{\boldsymbol{z}}_t$ 的预测值可以由前一状态的后验估计 $\hat{\boldsymbol{z}}^+_{t-1}$ 推断得到，即 $$\hat{\boldsymbol{z}}^-_t=f(\hat{\boldsymbol{z}}^+_{t-1})=\mathcal{E}_H(\omega(\mathcal{D}_Q(\hat{\boldsymbol{z}}^+_{t-1}),\mathit\Phi_{t-1\to t}))$$ 其中 $\mathit\Phi_{t-1\to t}$ 代表从 LQ 的 $\boldsymbol{x}_{t-1}$ 到 $\boldsymbol{x}_{t}$ 的流型（flow）预测，$\omega$ 是空间包装模块（spatial warping module）。
先将前一帧的预测代码 $\hat{\boldsymbol{z}}^+_{t-1}$ 解码得到预测值 $\hat{\boldsymbol{y}}_{t-1}=\mathcal{D}_Q(\hat{\boldsymbol{z}}^+_{t-1})$ ，再将其包装到当前帧，从而可以从新编码回阴空间来得到当前状态的预测值 $\hat{\boldsymbol{z}}^-_t$。

#### 2.3 Kalman filter 系统
滤波系统旨在促进时间信息传播并保持稳定的隐代码先验，其递归地融合了动态系统的近似观测状态 $\tilde{\boldsymbol{z}}_t$ 和先验估计 $\hat{\boldsymbol{z}}^-_t$，形成了对当前状态更准确的后验估计 $\hat{\boldsymbol{z}}^+_t$，即**状态更新**。公式(2)中的 Kalman Gain $\mathcal{K}_t$ 衡量了预测状态比较近似观测状态的准确度，从而更新状态并介绍不确定度

### 3. 本地时序一致性
在解码器应用跨帧注意力（cross-frame attention, CFA）层，将前一帧 $v_{t-1}$ 和当前帧 $v_t$ 投影到 embedding 层，输出特征 $v'_i=\text{Atten}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d}})\cdot V$，其中 $Q=W_Q\cdot v_t,\; K=W_K\cdot v_{t-1},\; V=W_V\cdot v_{t-1}$。直观地说，跨帧注意力模块是从前一帧中搜索匹配相似的 patch 进行相应的融合，促进了解码器中时间信息的传播。
















