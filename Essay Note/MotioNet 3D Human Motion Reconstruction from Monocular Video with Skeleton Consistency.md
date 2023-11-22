MotioNet: 3D Human Motion Reconstruction from Monocular Video with Skeleton Consistency
====
[Shi et al.](https://arxiv.org/abs/2006.12075)
> 单相机视角视频生成motion


## 目标和参数定义
本文的目标是将3D joint positions 的投影 $C(\mathbf{P}_{s,q,r};c)$ 解构出以下四个属性:
- $\mathbf{q}\in \mathbb{R}^{T\times QJ}$ 代表motion的动态旋转信息
- $\mathbf{s}\in \mathbb{R}^L$ 代表单一且一致的骨骼
- $\mathbf{r}\in \mathbb{R}^{T\times 3}$ 代表在各个时刻的root位置
- $\mathbf{f}\in \{0,1\}^{T\times 2}$ 代表每一帧双脚是否接触地面

投影 $C(\mathbf{P}_{s,q,r};c)$ 是由3D joint positions $\mathbf{P}_{s,q,r}$ 通过一个透视投影算子 $C$ 和相机参数 $c\in C$得到，其对应参数如下：
- $P_{s,q,r}\in \mathbb{R}^{T\times 3J}$ : 由骨骼 $s\in \mathbb{R}^L$ 生成的一组3D节点位置时序列。
- $s\in \mathbb{R}^L$ : 由 joint rotations $\mathbf{q}\in \mathbb{R}^{T\times QJ}$ 和 global root positions $\mathbf{r}\in \mathbb{R}^{T\times 3}$ 得到。
- $L$ : 四肢数量
- $T$ : 运动序列的时序长度
- $J$ : joint个数
- $Q$ : rotation vector的size

其中，前三个属性 $\widetilde{\mathbf{q}}, \mathbf{s}, \mathbf{r}$ 可以通过 FK 整合来从关节位置来预测 3D pose 序列 $\widetilde{P}_{\widetilde{s},\widetilde{q},\widetilde{r}}\in \mathbb{R}^{T\times 3J}$ 。


## 置信度和增广
每个数据样本 $\mathbf{P}_{s,q,r}\in \mathbb{R}^{T\times 3J}$ 先通过透视投影算子 $C(\cdot;c)$ 转化到 2D 图像空间，而后引入 confidence value 和 augmentation 来消除训练时和测试时输入分布的差别。
### Confidence Value
为了应对遮挡问题，给每个关节添加置信度属性 $c_n\in[0,1]$ ，完全不可见的关节置信度设为 $0$ 。所以向量 $c\in\mathbb{R}^{J\times T}$ 在训练和测试时都被连结到输入中。测试时置信度由 OpenPose 库中得到，训练时则是基于经验确定的分布随机产生。第 $j$ 个关节的该分布由两部分组成：
- 一个在零点附近的单位脉冲函数（概率为 $\delta_j$）
- 一个在 $1$ 附近高斯分布 $(\mu_j,\sigma_j)$ （概率为 $1-\delta_j$）

每当出现置信度 $c_n>0$ 时，一个空间扰动会作为噪声加入到 2D 坐标里，这个扰动的方向在 $[0^\circ,360^\circ]$ 内均匀取得，大小在 $[0,\beta(1-c_n)](\beta=6)$ 里取得。对 $c_n=0$ 则将 2D 坐标设为 $(0,0)$。

### Augmentation
为了扩充样本，采用以下三种方式增广：

1. Clip length：每次迭代从 60 到 120 中随机选取一个整数作为 batch 中样本的时序长度 $T$。这样可以增强静态参数的解缠性和序列的时间长度。 
2. Camera Augmentation: 通过修改 3D 角色的深度（$z$ 轴上的 global translation）和 方向（root 的global rotations）来使不同2D scales 的相似 pose 与相同（local）的 3D 参数 （rotations 和 bones length）匹配。
3. Flipping：在投影前左右翻转相机坐标空间下的 3D 节点位置来增广。
$$
\widetilde{P}^r_j=\left(-(P^l_j)_x,(P^l_j)_y,(P^l_j)_z\right), \qquad 
\widetilde{P}^l_j=\left(-(P^r_j)_x,(P^r_j)_y,(P^r_j)_z\right)
$$
其中 $\widetilde{P}^r_j$ 和 $\widetilde{P}^l_j$ 分别代表对称的第 $j$ 个节点左右位置，而没有对称的 joint 则采用 $\widetilde{P}_j=\left(-(P_j)_x,(P_j)_y,(P_j)_z\right)$。

## 网络结构
如下图所示，训练时每个数据样本 $C(\mathbf{P}_{s,q,r};c)\in \mathbb{R}^{T\times 2J}$ 先通过添加 positional noise 加入置信度属性，再被并行喂入两个网络 $E_S$ 和 $E_Q$ 。$E_Q$ 输出的 $\mathbf{\widetilde{f}}$, $\mathbf{\widetilde{r}}$ 和 $\mathbf{\widetilde{q}}$ 会被喂入判别器 $D$ 中来判断角速度是否合理。此外这些参数信息还会连同$E_S$ 输出的骨骼$\mathbf{\widetilde{s}}$（bones length）和 T-pose 模型一起输入 FK 层中算出 3D poses。
![](/Essay%20Note/images/MotioNet_1.png)

$E_S$ 是用来预测单一骨骼 $\mathbf{\widetilde{s}}$；$E_Q$ 与时序关联，是用来预测包括 rotations，global positions 和 foot contact label 这些参数，即：
$$
\begin{aligned}
\mathbf{\widetilde{s}}&=E_S(C(\mathbf{P}_{s,q,r};c)) \\
\mathbf{\widetilde{q},\widetilde{r},\widetilde{f}}&=E_Q(C(\mathbf{P}_{s,q,r};c))
\end{aligned}
$$

![](/Essay%20Note/images/MotioNet_2.png)

### Forward Kinematics (FK)
如下图所示，T-pose模型通过一系列旋转得到特定的模型。因为其包含一系列不同的算子，故可以作为FK层引入神经网络来反向传播。
![](/Essay%20Note/images/MotioNet_3.png)

在每个时刻 $t$，会根据关节旋转四元组 $\widetilde{\mathbf{q}}^t \in \mathbb{R}^{4J}$ 叠加一个新的FK层到初始T-pose骨骼 $\mathbf{\widetilde{s}_{init}}\in \mathbb{R}^{3J}$ 上：
$$
\mathbf{\widetilde{P}}^t_{\widetilde{s},\widetilde{q},\widetilde{r}}=FK(\mathbf{\widetilde{s}_{init}},\widetilde{\mathbf{q}}^t)+\widetilde{\mathbf{r}}^t
$$

其中，$\mathbf{\widetilde{P}}^t_{\widetilde{s},\widetilde{q},\widetilde{r}}$ 是在 $t$ 时刻的 local 3D pose；旋转按父子关系 $\mathbf{\widetilde{P}}^t_n=\mathbf{\widetilde{P}}^t_{parent(n)}+R^t_n\mathbf{\widetilde{s}}_n$ 从root节点应用到每一个链末节点，$\mathbf{\widetilde{P}}^t_n\in \mathbb{R}^3$ 代表第 $n$ 个节点在 $t$ 时刻的位置，$\mathbf{\widetilde{P}}^t_{parent(n)}$ 代表其父节点的位置信息，$R^t_n\in \mathbb{R}^{3\times 3}$ 代表第 $n$ 个节点 的旋转矩阵，$\mathbf{\widetilde{s}}_n\in \mathbb{R}^3$ 代表第 $n$ 个节点和其父节点的3D offset; $\widetilde{\mathbf{r}}^t$ 代表了root位置。

## 训练和损失函数
### 损失函数
文中主要使用4种损失函数：
#### 1. skeleton loss $\mathcal{L}_\mathrm{S}$
若用 $\mathbf{P}_{s,q,r}\in \mathcal{P}$ 代表数据集 $\mathcal{P}$ 的一组3D motion序列， $C(\mathbf{P}_{s,q,r};c_i)$ 代表该序列根据相机 $c_i\in C$ 得到的2D投影。那么 $\mathcal{L}_\mathrm{S}$ 使得 encoder 网络 $E_S$ 能准确地提取出skeleton(骨骼平均长度被归一化)：
$$
\mathcal{L}_\mathrm{S}=\mathbb{E}_{\mathbf{P}_{s,q,r}\sim \mathcal{P},c_i\sim C}\left [  \left\| E_S(C(\mathbf{P}_{s,q,r},c_i))-\mathbf{s} \right\|^2 \right ]
$$

#### 2. joint position loss $\mathcal{L}_\mathrm{P}$
尽管 $E_S$ 和 $E_Q$ 已经输出motion信息，但为了防止节点误差在每条关节链的累积，joint position loss 被引入来确保每个节点的3D位置与GT匹配：
$$
\mathcal{L}_\mathrm{P}=\mathbb{E}_{\mathbf{P}_{s,q,r}\sim \mathcal{P},c_i\sim C}\left [  \left\| FK(\mathbf{\widetilde{s}_{init}},\widetilde{\mathbf{q}})-\mathbf{P}_{s,q,r=0} \right\|^2 \right ]
$$

另外，为了抑制 end-effectors 部分的失真，带有权重 $\lambda_{P_{EE}}$ 的损失 $\mathcal{L}_{\mathrm{P}_{EE}}$ 被特别引入。

#### 3. rotations GAN loss $\mathcal{L}_\mathrm{Q\_GAN}$
使用一个GAN网络来训练自然速度分布得到的rotations，从而关注joint rotations的时序差异而非绝对值来避免将相似的motion共享一个T-pose。在这个GAN网络中，$J$ 个判别器 $D_j$ 被加入到网络中，来区分选定joint的输出rotations和GT的rotation之间的时序差：
$$
\mathcal{L}_\mathrm{Q\_GAN}=\mathbb{E}_{q\sim \mathcal{Q}}\left [  \left\|D_j(\Delta_tq_j) \right\|^2 \right ]+\mathbb{E}_{\mathbf{P}_{s,q,r}\sim \mathcal{P},c_i\sim C}\left [  \left\| 1-D_j\Delta_tE_Q(C(\mathbf{P}_{s,q,r},c_i))_{q_j} \right\|^2 \right ]
$$

其中，$E_Q(\cdot)_{q_j}$ 代表 $E_Q$ 网络输出的第 $j$ 个节点的 rotations ，$\mathcal{Q}$ 代表从数据集 $\mathcal{P}$ 得到的节点角，$\Delta_t$ 代表时序差。

此外，为了得到自然连贯的动画效果，预测的 joint rorations 需要进行**平滑**（**smooth**）处理。

#### 4. global root position loss $\mathcal{L}_\mathrm{R}$
因为rotations是由velocities指导的，所以其绝对值并不一定准确。引入每个角色的 **reference T-pose** 来训练网络，从而矫正rotations。

为了排除在重建在 $t$ 帧 global root position $\mathbf{r}^t=(X^t_r,Y^t_r,Z^t_r)$ 时的非必要信息影响训练，训练时只预测深度参数 $Z^t_r$，然后用其推导出global root position：
$$
(X^t_r,Y^t_r,Z^t_r)=(\frac{Z^t_r}{f}x^t_r,\frac{Z^t_r}{f}y^t_r,Z^t_r)
$$

其中，$(x_r,y_r)$ 是2D投影序列的root position，$f=1$ 是相机模型的假定焦距。

Global Root Position可以由下得到：
$$
\mathcal{L}_\mathrm{R}=\mathbb{E}_{\mathbf{P}_{s,q,r}\sim \mathcal{P},c_i\sim C}\left [  \left\| E_Q(C(\mathbf{P}_{s,q,r},c_i))_r-Z_r \right\|^2 \right ]
$$

其中，$E_Q(\cdot)_r$ 代表 $E_Q$ 网络输出的 root position。

#### Foot Contact Loss $\mathcal{L}_\mathrm{F}$
除了以上4种主要损失函数之外，还有 $\mathcal{L}_\mathrm{F}$ 被引入来处理与地面接触时的脚步滑动问题:
$$
\mathcal{L}_\mathrm{F}=\mathbb{E}_{\mathbf{P}_{s,q,r}\sim \mathcal{P},c_i\sim C}\left [  \left\| E_Q(C(\mathbf{P}_{s,q,r},c_i))_f-\mathbf{f} \right\|^2 \right ]
$$

其中，$E_Q(\cdot)_f$ 代表了 $E_Q$ 网络输出的 foot contact label $\mathbf{\widetilde{f}}\in\{0,1\}^{T\times 2}$。 

基于GT二元向量 $\mathbf{f}$，引入损失函数 $\mathcal{L}_\mathrm{FC}$ 来使得脚接触地面时速度为0:
$$
\mathcal{L}_\mathrm{FC}=\mathbb{E}_{\mathbf{P}_{s,q,r}\sim \mathcal{P},c_i\sim C}\left [  \left\| \mathbf{f}_i\sum_j\Delta_tFK(\mathbf{\widetilde{s}_{init}},\widetilde{\mathbf{q}})_{f_i} \right\|^2 \right ]
$$

其中，$FK(\cdot,\cdot)_{f_i}\in\mathbb{R}^{T\times 3}$ 和 $\mathbf{f}_i(i\in left,right)$ 代表 positions 和其中一只脚的 contact label，$\sum_j$ 是沿着坐标轴的累加。

#### 总loss $\mathcal{L}_\mathrm{tot}$
把上述所有项加起来乐意得到总损失函数：
$$
\mathcal{L}_\mathrm{tot}=\mathcal{L}_\mathrm{P}+\lambda_\mathrm{S}\mathcal{L}_\mathrm{S}+\lambda_\mathrm{Q}\sum_{j\in J}\mathcal{L}_\mathrm{Q\_GAN}+\lambda_\mathrm{R}\mathcal{L}_\mathrm{R}+\lambda_\mathrm{F}\mathcal{L}_\mathrm{P_F}+\lambda_{FC}\mathcal{L}_\mathrm{P_{FC}}
$$

在本文实验中，这些参数被设定为 $\lambda_\mathrm{S}=0.1,\lambda_\mathrm{Q}=1,\lambda_\mathrm{R}=1.3,\lambda_\mathrm{F}=0.5,\lambda_\mathrm{FC}=0.5$ 。











---

