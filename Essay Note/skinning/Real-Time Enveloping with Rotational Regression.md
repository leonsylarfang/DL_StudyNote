Real-Time Enveloping with Rotational Regression
====
[Wang et al.](https://dl.acm.org/doi/10.1145/1276377.1276468)
> 相较线性混合蒙皮更真实且一样快

## Overview
![](/Essay%20Note/images/RTERR_Overview.png)
如上图所示，本文的主题思想是把包裹问题变为一个 mapping 问题，将骨骼姿态（skeletal pose）$\mathbf{q}$ 与其对应的 mesh $\mathbf{y(q)}$ 做 mapping。该 mapping 分为两步：

1. 根据 skeletal pose 去预测 mesh 的形变梯度（deformation gradients）。
2. 通过将上述预测带入求解 Poisson equation 来重建点的位置。

## 1. 形变梯度预测
通过学习来建立 **bone tranformations** 与**形变梯度**的映射关系。

* 形变梯度 $D$ 代表一个变形网格表面 $\mathbf{y}$ 相对于一个静止（或未变形）表面 $\hat{\mathbf{y}}$ 的关于方向、缩放和剪切变化信息。对于一个三角面片 mesh， 对每个三角面片的形变梯度是一个简单的 $3\times 3$ 矩阵。$\mathbf{q}$ 代表每个骨骼节段的 transformation。
* 形变梯度预测 $D(\mathbf{q})$ 则被用来捕捉如扭曲和肌肉凸起所带来的额外形变。

    ![](/Essay%20Note/images/RTERR_1.jpg)
    如上图所示，通过极坐标分解，形变梯度 $D$ 可以被分解为旋转分量 $R$ 和 缩放/剪切分量 $S$。在用两个不同的回归模型预测出两个分量 $R(\mathbf{q})$ 和 $S(\mathbf{q})$ 来组合出 $D(\mathbf{q})$。其中 $W,u$ 是旋转回归模型中的参数，$H$ 是缩放/剪切模型的参数。

* 从 skeletal poses 样本中提取出谷歌旋转序列 $\mathbf{q}^i$，再从其对应的 mesh 中提取出形变梯度 $D^i$ 来完成上述回归模型。

### 1.1 符号注释
* 每一个 skeletal pose $\mathbf{q}$ 都记做由 $J$ 个 bone transformations（**bone tranformations 是跟静止姿态的变化而非上一帧姿态**） 组成的 vector $[\mathbf{vec}(Q_1)^T,d_1^T,...,\mathbf{vec}(Q_J)^T,d_J^T]\in \mathbb{R}^{12J\times 1}$。
* 每个 mesh $\mathbf{y}\in\mathbb{R}^{3V\times 1}$ 被记做由 $V$ 个点位置组成的 vector。
* $\theta$ 和 $\rho$ 分别指代 bone rotations $Q$ 和 mesh rotations $R$ 的轴角形式（axis-angle）。轴角量表示为由大小编码的角度和由方向编码的轴组成的三维向量。

### 1.2 Rotational Regression
SSD (skeletal subspace deformation) 的基本思想是 mesh 上顶点的变换受对应骨骼的控制。

但如下图，当肌肉凸起时，mesh 的某些部分沿着与关节旋转相反或沿着不同旋转轴旋转。为此，文中提出了一个新的泛用回归模型来关联关节旋转跟 mesh 的旋转。
<div align=center>  <img src="/Essay%20Note/images/RTERR_2.jpg" width=30%>
</div>

* 训练模型    
    对于skeleton pose $\mathbf{q}$，其三角面片旋转（triangle rotations）$\tilde{\rho}$ 和 骨骼旋转（bone rotations）$\tilde{\theta}$ 之间的关系可用如下线性方程来表示：
    $$
        \tilde{\rho}(\mathbf{q})=uW\tilde{\theta}_b(\mathbf{q})
    $$

    其中，骨骼旋转即为关节旋转（joint rotations），$u$ 是 triangle rotation 的角度相对关节角度（joint angle）的缩放比例，$W$ 是 triangle rotation 的轴相对关节轴（joint axis）的旋转偏移量。$b$ 代表当前提取旋转所用骨骼。
    <div align=center>  <img src="/Essay%20Note/images/RTERR_3.jpg" width=80%>
    </div>

    对于每个三角面片，作者设置4个参数来控制$(W\in\mathbb{SO}(3),u\in\mathbb{R})$。($\mathbb{SO}(3)$ 是三维特殊正交群)
* 训练过程
    给定骨骼 $\mathbf{q}^i$ 和 三角面片 $\rho^i$ 的旋转序列，将其变换到关节框架，可得到 $\tilde{\theta}_b(\mathbf{q}^i)$ 和 $\tilde{\rho}^i$。那么最优的 $W$ 和 $u$ 可由如下式子得到：
    $$
    \underset{W,u}{\argmin}\sum_{i\in 1,...,N}||uW\tilde{\theta}_b(\mathbf{q})-\tilde{\rho}||^2
    $$

    可用坐标下降法来交替求解 $W$ 和 $u$ 的最优值，即互相给定一个值来求另一个值 closed-form solution。

    * 作者首先独立出 $W$ 单独初始化 $u$：
        $$
        \underset{u}{\argmin}\sum_{i\in 1,...,N}(u||\tilde{\theta}_b(\mathbf{q})||-||\tilde{\rho}||)^2
        $$
    * 针对每个关节都建立一个模型，再比较每个模型与旋转形变的吻合度，来挑选出每个三角面片旋转序列对应的最合适的旋转关节，从而将任意表面旋转序列与关节旋转序列关联起来。
    * 针对具有多个关节依赖关系的区域，可以对附加骨骼添加残余旋转（residual rotations）。

### 1.3 Scale/Shear Regression
根据两个joint的轴角表征，作者线性地预测缩放/剪切矩阵的每一个值：
    $$
    \mathbf{vec}(S(\mathbf{q}))=H\tilde{\theta}_{b_1,b_2}(\mathbf{q})
    $$

这两个 joint 分别是旋转回归预测器中找到的关联 joint $\theta_{b_1}$ 与其父 joint $\theta_{b_2}$。将这两个旋转和一个偏差项拼接后组成：
    $$
    \tilde{\theta}_{b_1,b_2}(\mathbf{q})=[\tilde{\theta}_{b_1}(\mathbf{q})^T \tilde{\theta}_{b_2}(\mathbf{q})^T 1]^T\in\mathbb{R}^{7\times 1}
    $$

给定缩放/剪切序列 $S^i$ 和骨骼旋转序列 $\mathbf{q}^i$，我们通过最小二乘法来优化下列预测器，从而得到参数 $H\in\mathbb{R}^{9\times7}$：
    $$
    \underset{H}{\argmin}\sum_{i\in{1,...,N}}||H\tilde{\theta}_{b_1,b_2}(\mathbf{q}^i)-\mathbf{vec}(S^i)||^2
    $$

## 2. Mesh 重建
在得到形变梯度预测值后，作者通过解一个 Poisson 方程来将其映射回顶点位置：
* 当在每一个三角面片上只有形变梯度预测时，作者通过修改公式来整合一组 near-rigid 顶点的全局位置。
* 这些 near-rigid 顶点可以通过 SSD 很简单的预测出来，并通过将全局平移量引入 Poisson 问题来提高其精准度。
* 最后通过三角形面片和顶点的一致性与协调性，来制定 mesh 重建优化的简化形式。
### 2.1 Poisson Mesh 重构
下列 Poisson 方程公式通过边（edges）将形变梯度与顶点关联起来：
$$
\underset{\mathbf{y}}{\argmin}\sum_{k\in1,...,T}\sum_{j=2,3}||D_k(\mathbf{q})\hat{\mathbf{v}}_{k,j}-\mathbf{v}_{k,j}||^2    \tag{1}
$$

其中，$\mathbf{v}_{k,j}=\mathbf{y}_{k,j}-\mathbf{y}_{k,1}$ 代表当前正在处理的 pose $\mathbf{q}$ 的第 $k$ 个表面的第 $j$ 条边，$\hat{\mathbf{v}}_{k,j}$ 代表静止姿态下的同一条边。

由此，给定当前姿态每个三角面片的形变梯度预测 $D$，即可通过反向替换获得顶点位置。
### 2.2 Near-Rigid/SSD 顶点约束






