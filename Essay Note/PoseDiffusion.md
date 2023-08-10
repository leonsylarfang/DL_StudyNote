PoseDiffusion: Solving Pose Estimation via Diffusion-aided Bundle Adjustment
===
[Wang et al.](https://arxiv.org/pdf/2306.15667.pdf)
> 通过图片来预测相机参数

## 结构
首先我会根据框架来介绍本文的结构：

![](/Essay%20Note/images/PoseDiffusion_1.png)


1. 通过输入相机参数 $x$ 和图片 $\mathrm{I}$ 来训练扩散模型 $D_\theta$ 。
2. 利用降噪器 $D_\theta$ 的采样预测结果 $p(x|\mathrm{I})$ 来得到相机参数的预测值 $x$ 。
3. 在采样中，将Sampson极线差的相关梯度加入到降噪器中来得到 $p(\mathrm{I}|x)$，从而输出所有相机姿态。


## PoseDiffusion详解
本文旨在解决，通过给定一组单场景的图片，来预测相机的内在和外在参数的问题。

给定 $N\in \mathbb{N}$ 张图片 $I^i\in \mathbb{R}^{3\times H\times W}$ 的元组 $\mathrm{I}=(I^i)^N_{i=1}$ ，其中 $i\in[1,N]$ 表示图片序号， $H$ 和 $W$ 分别表示图片的高度和宽度。

还原其对应的相机参数 $x^i=(K^i,g^i)$ 的元组 $x=(x^i)^N_{i=1}$ ，其中 $K^i\in \mathbb{R}^{3\times 3}$ 表示内在参数， $g^i\in \mathbb{SE}(3)$ 表示外在参数。外在参数 $g^i$ 表示一个世界坐标3D点 $\mathbf{p}_w\in \mathbb{R}^3$ 到相机坐标3D点 $\mathbf{p}_c\in \mathbb{R}^3$ 的映射，即 $\mathbf{p}_c=g^i(\mathbf{p}_w)$ 。内在参数 $K^i$ 则将这个相机点 $\mathbf{p}_c$ 投影到屏幕坐标2D点 $\mathbf{p}_s\in \mathbb{R}^2$ ，且 $K^i\mathbf{p}_c \sim \lambda[\mathbf{p}_s;1], \lambda \in R$， 其中 $\sim$ 表示齐次等价。

### 扩散辅助的BA（Diffusion-aided Bundle Adjustment）
不同于训练扩散模型的加噪过程中，每一步的转变都相对独立：
$$
q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta _t}x_{t-1},\beta_t \mathbb{I})
$$

降噪过程中的每一步都要受到 Ground Truth 图片 $\mathrm{I}$ 的约束：
$$
p_\theta(x_{t-1}|x_t,\mathrm{I})=\mathcal{N}(x_{t-1};\sqrt{\alpha _t}\mathcal{D}_\theta(x_t,t,\mathrm{I}),(1-\alpha_t)\mathbb{I})
$$

文中将降噪器 $\mathcal{D}_\theta$ 应用成一个 Transformer ：
$$
\mathcal{D}_\theta(x_t,t,\mathrm{I})=\mathrm{Trans}\left[\left(cat\left(x^i_t,t,\psi(I^i)\right)^N_{i=1}\right) \right]=\mu_{t-1}
$$

其中 $x^i_t$ 表示噪声姿态元组， $t$ 表示扩散时间步长， $\psi(I^i)\in \mathbb{R}^{D_\psi}$ 表示输入图片 $I^i$ 的内嵌特征，由一个预训练 **DINO** (self-distillation with no labels) 权值的视觉transformer模型得到。 $\mu_{t-1}=(\mu^i_{t-1})^N_{i=1}$ 表示 $\mathcal{D}_\theta$ 输出的对应降噪相机参数。

在训练时，$\mathcal{D}_\theta$ 是由以下降噪损失控制的：
$$
\mathcal{L}_{\mathrm{diff}}=\mathbb{E}_{t\sim [1,T],x_t\sim q(x_t|x_0,\mathrm{I})}\left \|  \mathcal{D}_\theta(x_t,t,\mathrm{I})-x_0 \right \|^2
$$

其中该期望是所有扩散步长 $t$ 的总期望，其对应的采样 $x_t\sim q(x_t|x_0,\mathrm{I})$ 是基于所有场景对应的图像 $\mathrm{I}_j$ 和 相机参数 $x_{0,j}$ 组成的训练集 $\mathcal{T}=\{(x_j,\mathrm{I}_j\}^S_{j=1}, S\in \mathbb{N}$ 来训练的。

训练得到的降噪器 $\mathcal{D}_\theta$ 会以类似DDPM的方式，从随机的相机 $x_T\sim \mathcal{N}(0,\mathbb{I})$ 中采样出 $x_{t-1}$：
$$
x_{t-1}\sim \mathcal{N}(x_{t-1};\sqrt{\bar{\alpha}_{t-1}}\mathcal{D}_\theta(x_t,t,\mathrm{I}),(1-\bar{\alpha}_{t-1}\mathbb{I}))
$$

### 几何制约采样（Geometry-Guided sampling）
为了提高 PoseDiffusion 在回归精度，文中引入了**双视角几何约束** (**two-view geometry constraints**)。

令 $P^{i,j}={(\mathbf{p}^i_k,\mathbf{p}^j_k)}^{N_{P^{i,j}}}_{k=1}$ 代表一对场景图片 $(I^i,I^j)$ 同一个像素点 $p_k\in \mathbb{R}^2$ 的二维对应关系；$(x^i,x^j)$ 代表对应的相机姿态。那么可以得到相机及其二维对应关系的极差 $e^{i,j}\in \mathbb{R}$ ：
$$
e^{i,j}(x^i,x^j,P^{i,j})=\sum_{k=1}^{|P^{i,j}|}\left[ \frac{\tilde {\mathbf{p}}_k^{j\mathrm{T}}F^{i,j}\tilde{\mathbf{p}}^i_k}{(F^{i,j}\bar{\mathbf{p}}^i_k)^2_1+(F^{i,j}\bar{\mathbf{p}}^i_k)^2_2+(F^{i,j\mathrm{T}}\bar{\mathbf{p}}^i_k)^2_1+(F^{i,j\mathrm{T}}\bar{\mathbf{p}}^i_k)^2_2} \right]_\epsilon
$$

其中 $\bar{\mathbf{p}}=[\mathbf{p};1]$ 表示 $\mathbf{p}$ 的齐次坐标，$[z]_\epsilon=min(z,\epsilon)$ 是鲁棒限制方程，$F^{i,j}\in \mathbb{R}^{3\times 3}$ 是图片 $I^i$ 和 $I^j$ 之间从点 $\mathbf{p}^i_k$ 到行的映射关系。

此外，为了使采样结果满足图片之间的极坐标限制，分类器会加入一个由 $x_t$ 限制的分布 $p(\mathrm{I}|x_t)$ 的梯度来扰动预测平均值 $\mu_{t-1}=\mathcal{D}_\theta(x_t,t,\mathrm{I})$:
$$
\hat{\mathcal{D}}_\theta(x_t,t,\mathrm{I})=\mathcal{D}_\theta(x_t,t,\mathrm{I})+s\nabla_{x_t}\mathrm{log}p(\mathrm{I}|x_t)
$$

其中，$s\in\mathbb{R}$ 控制了扰动强度。

通过把 $p(\mathrm{I}|x_t)$ 建模成独立指数分布，与成对Sampson误差 $e^{i,j}$ 的乘积，可以求得相机 $x$ 的均匀先验分布：
$$
p(\mathrm{I}|x_t)=\prod_{i,j}p(I^i,I^j|x^i_t,x^j_t)\propto\prod_{i,j}\mathrm{exp}(-e^{i,j})
$$

### 内外参数表征
外参 $g^i$ 被表示为一个二元组 $(\mathbf{q}^i,\mathbf{t}^i)$，其组成为一个由 旋转矩阵 $R^i\in\mathbb{SO}(3)$ 和相机位移向量 $\mathbf{t}^i\in\mathbb{R}^3$ 组成的四元组  $\mathbf{q}^i\in \mathbb{H}$ 。其代表了世界-相机的线性变换 $\mathbf{p}_c=g^i(\mathbf{p}_w)=R^i\mathbf{p}_w+t^i$。

内参 $K^i$ 则被表示为一个相机标定矩阵 $[f^i,0,p_x;0,f^i,p_y;0,0,1]\in\mathbb{R}^{3\times 3}$。其中 $f^i\in \mathbb{R}^+$ 表示焦距，为了保证焦距为正，$f^i=\mathrm{exp}(\hat{f}^i)$，$\hat{f}^i\in\mathbb{R}$ 由降噪器 $\mathcal{D}_\theta$ 预测。

所以transformer降噪器 $\mathcal{D}_\theta$ 输出的相机元组即为 $x=\left((\hat{f}^i,\mathrm{q}^i,\mathrm{t}^i) \right)^N_{i=1}=\left((K^i,g^i) \right)^N_{i=1}$。

### 其他优化
- 为了防止过拟合到场景特定的训练坐标帧，我们在传递给去噪器之前对外参 $g_j=(\hat{g}^1_j,...,\hat{g}^N_j)$ ，作为相关相机姿态对一个随机选择的主摄像机 $\hat{g}_j^*$ 进行了归一化。
- 为了规范化 $s$，将输入摄像机的平移值除以轴向标准化平移的规范的中位数。
- 通过将二进制标志 $p^i_{pivot}\in\{0,1\}$ 附加到图片特征 $\psi(I^i)$ 上来将轴枢相机添加到降噪器中。

## 总结
PosePosition 将深度学习与基于对应的约束优雅地结合在一起，无论在稀疏视图还是密集视图下都能高精度地重建摄像机位置，从而很好地预测相机姿态信息。




