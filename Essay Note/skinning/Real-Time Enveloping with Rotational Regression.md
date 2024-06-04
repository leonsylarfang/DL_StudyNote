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

* 从 skeletal poses 样本中提取出骨骼旋转序列 $\mathbf{q}^i$，再从其对应的 mesh 中提取出形变梯度 $D^i$ 来完成上述回归模型。

### 1.1 符号注释
* 每一个 skeletal pose $\mathbf{q}$ 都记做由 $J$ 个 bone transformations（**bone tranformations 是跟静止姿态的变化而非上一帧姿态**） 组成的 vector $[\mathbf{vec}(Q_1)^T,d_1^T,...,\mathbf{vec}(Q_J)^T,d_J^T]\in \mathbb{R}^{12J\times 1}$。
* 每个 mesh $\mathbf{y}\in\mathbb{R}^{3V\times 1}$ 被记做由 $V$ 个点位置组成的 vector。
* $\theta$ 和 $\rho$ 分别指代 bone rotations $Q$ 和 mesh rotations $R$ 的轴角形式（axis-angle）。轴角量表示为由大小编码的角度和由方向编码的轴组成的三维向量。

### 1.2 Rotational Regression
SSD (skeletal subspace deformation) 的基本思想是 mesh 上顶点的变换受对应骨骼的控制。

但如下图，当肌肉凸起时，mesh 的某些部分沿着与关节旋转相反或沿着不同旋转轴旋转。为此，文中提出了一个新的泛用回归模型来关联关节旋转跟 mesh 的旋转。
<div align=center>  <img src="/Essay Note/images/RTERR_2.jpg" width=30%>
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

给定缩放/剪切序列 $S^i$ 和骨骼旋转序列 $\mathbf{q}^i$，我们通过最小二乘法来优化下列预测器，从而得到参数 $H\in\mathbb{R}^{9\times 7}$：
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
* 由于骨架平动分量的缺失，Poisson 优化不能检测或补偿全局的平动问题，这导致低频误差的累积，角色的末端可能与节点配置（joint configuration）不符。而通过识别出一组 near-rigid 点，并将其修正到 SSD 预测值，就可以解决这一问题，因为 SSD 模型中每个顶点都依赖于骨骼的平移分量，其中包含了每段骨骼的全局位置信息。
<div align=center>  <img src="/Essay%20Note/images/RTERR_4.jpg" width=80%><br>(a)通过预测边来用 Poisson 公式重构顶点位置；(b)累积的低频误差导致 mesh 末端和 joint configuration 不匹配；(c)通过将 near-rigid 点修正到 SSD预测位置（红点）来解决问题。
</div>

* 通过每个顶点在训练集和阈值上的误差，选择出 SSD 预测的最佳点集 $F$，将 $F$ 内的顶点固定为目标函数中的 SSD 对应预测值。
    
    * 定义线性映射 $\Psi_a$ 使得 $\Psi_a\mathbf{q}$ 等价于 SSD 对顶点 $a$ 在 pose $\mathbf{q}$ 时的预测 $\sum_b^Jw_{a,b}T_b(\mathbf{q})\hat{\mathbf{y}}_a$
    * 在通过非负最小二乘法得到 SSD 权重 $w_{a,b}$ 后，将 $F$ 中所有点 $\mathbf{y}_a=\Psi_a\mathbf{q}$ 代入 Eq.1 可以得到：
    $$
    \underset{\mathbf{y}}{\argmin}\sum_{k\in1,...,T}\sum_{j=2,3}||D_k(\mathbf{q})\hat{\mathbf{v}}_{k,j}-\mathbf{v}_{k,j}||^2
    $$
    其中，
    $$
    \mathbf{v}_{k,j}= \left\{\begin{array}{ll}
    {\mathbf{y}_{k,j}-\mathbf{y}_{k,1}}& {\text{if} \; \mathbf{y}_{k,j} \notin F \;\mathrm{and}\; \mathbf{y}_{k,1} \notin F}\\
    {\mathbf{y}_{k,j}-\Psi_{k,1}\mathbf{q}}& {\mathrm{if\; only} \; \mathbf{y}_{k,1} \in F } \\
    {\Psi_{k,j}\mathbf{q}-\mathbf{y}_{k,1}}&{ \mathrm{if\; only} \; \mathbf{y}_{k,j} \in F }\\
    \end{array}\right. \tag{2}
    $$
    如果一条边的两个顶点都是固定的，那么可以把该边的误差项从目标函数中去除。
    * 通过对对应线性系统中的左侧进行预因式分解，再反向替换来预测新的姿态，从而与 Eq.1 类似的方法来求解 Eq.2。
### 2.3 简化 Mesh 重建
如下图，从一组 skeleton-mesh 配对数据中提取出三角面片形变序列 $D_k^i$。经过预测器简化环节后可以得到一组关键形变梯度序列 $D^i_l$ 来训练关键形变梯度预测器。文中模型则由这些预测器跟 mesh 重建矩阵 $C_1$ 和 $C_2$ 组成。
<div align=center>  <img src="/Essay%20Note/images/RTERR_5.jpg" width=100%><br></div>

* 将每个三角面片的形变梯度预测器表示为 $P$ 个关键形变梯度预测器的线性组合：
    $$
    D_k(\mathbf{q})=\sum_{l\in 1,...,P}\beta_{k,l}D_l(\mathbf{q}) \tag{3}
    $$

    每个从一组 proxy-bones 获得的顶点可以表达为类 SSD 形式：
    $$
    \mathbf{y}_a(\mathbf{t})=\sum_{l\in 1,...,P}\alpha_{a,b}T_b(\mathbf{t})\hat{\mathbf{y}}_a=\Phi_\alpha \mathbf{t} \tag{4}
    $$
    
    其中，$\Phi_\alpha$ 被定义成类似于映射 $\Psi$, $\mathbf{t}$ 则以类似 $\mathbf{q}$ 的形式包含了 proxy-bone 的变换。
* 将 Eq.3 和 Eq.4 代入 Eq.2 可以得到 proxy-bone 变换 $\mathbf{t}$：
    $$
    \mathbf{t(q)}=\underset{\mathbf{t}}{\argmin}\sum_{k\in1,...,T}\sum_{j=2,3}||\sum_{l\in1,...,P}\beta_{k,l}D_l(\mathbf{q})\hat{\mathbf{v}}_{k,j}-\mathbf{v}_{k,j}||^2
    $$

    其中，
    $$
    \mathbf{v}_{k,j}= \left\{\begin{array}{ll}
    {\mathbf{\Phi}_{k,j}\mathbf{t}-\mathbf{\Phi}_{k,j}\mathbf{t}}& {\text{if} \; \mathbf{y}_{k,j} \notin F \;\mathrm{and}\; \mathbf{y}_{k,1} \notin F}\\
    {\mathbf{\Phi}_{k,j}\mathbf{t}-\mathbf{\Psi}_{k,1}\mathbf{q}}& {\mathrm{if\; only} \; \mathbf{y}_{k,1} \in F } \\
    {\mathbf{\Psi}_{k,j}\mathbf{q}}-\mathbf{\Phi}_{k,1}\mathbf{t}&{ \mathrm{if\; only} \; \mathbf{y}_{k,j} \in F }\\
    \end{array}\right. \tag{5}
    $$
* 因为预测器和顶点重构都是遵循线性模型，所以 $\mathbf{t(q)}$ 与 形变梯度预测器 $D_l(\mathbf{q})$ 跟 骨旋转 $\mathbf{q}$ 也都是线性关系，所以可用以下式子来表达：
    $$
    \mathbf{t(q)}=C_1\mathbf{d(q)} + C_2\mathbf{q} \tag{6}
    $$

    其中，$\mathbf{d(q)}=[\mathbf{vec}(D_1(\mathbf{q}))^T\dots\mathbf{vec}(D_P(\mathbf{q}))^T]^T$。
    求解常数项 $C_1,C_2$ 需要先设定：
    $$
    \;(k,j)\in\left\{\begin{array}{ll}
    {F_0}& {\text{where both} \; \mathbf{y}_{k,j} \;\text{and}\;\mathbf{y}_{k,1} \text{are not fixed}}\\
    {F_1}& {\text{where only} \;\mathbf{y}_{k,1} \text{is fixed}}\\
    {F_2}& {\text{where only} \;\mathbf{y}_{k,j} \text{is fixed}}\\
    \end{array}\right. 
    $$

    那么可以得到：
    $$
    \begin{aligned}
    A\in\mathbb{R}^{12P\times 12P}&=\sum_{(k,j)\in F_0}(\Phi_{k,j}-\Phi_{k,1})^T(\Phi_{k,j}-\Phi_{k,1})\\ &+\sum_{(k,j)\in F_1}\Phi_{k,j}^T\Phi_{k,j}+\sum_{(k,j)\in F_2}\Phi_{k,1}^T\Phi_{k,1}
    \end{aligned}
    $$

    $$
    \begin{aligned}
    B\in\mathbb{R}^{12P\times 9}&=\sum_{(k,j)\in F_0}(\Phi_{k,j}-\Phi_{k,1})^T\beta_{kl}(\hat{\mathbf{v}}_{k,j}^T\otimes I_{3\times 3})\\ &+\sum_{(k,j)\in F_1}\Phi_{k,j}^T\beta_{kl}(\hat{\mathbf{v}}_{k,j}^T\otimes I_{3\times 3})\\&+\sum_{(k,j)\in F_2}(-\Phi_{k,1})^T\beta_{kl}(\hat{\mathbf{v}}_{k,j}^T\otimes I_{3\times 3})
    \end{aligned}
    $$

    其中 $\otimes$ 代表克罗内克积（Kronecker product）。
    从而可以得到：
    $$
    \begin{aligned}
    &B_1\in\mathbb{R}^{12P\times 9P}=[B_{11}\dots B_{1P}] \\
    &B_2\in\mathbb{R}^{12P\times 12J}=\sum_{(k,j)\in F_1}\Phi_{k,j}^T\Psi_{k,1}+\sum_{(k,j)\in F_2}\Phi_{k,1}^T\Psi_{k,j} \\
    \Rightarrow\; &C_1\in \mathbb{R}^{12P\times 9P}=A^{-1}B_1\\
    &C_2\in\mathbb{R}^{12P\times 12J}=A^{-1}B_2
    \end{aligned}
    $$

    这样一来，整个 Poisson mesh 重建步骤被简化成 matrix-vector 乘法（Eq.6）和 matrix-palette skinning （Eq.4）两步可以在GPU上执行的操作。

## 3. 降维（Dimensionality Reduction）
用聚类（clustering）的方法来求简化参数：
1. 用于顶点简化的 SSD 权重 $\alpha$
2. 用于简化预测器的混合权重 $\beta$
3. 关键形变梯度预测器 $D_l(\mathbf{q})$

### 3.1 顶点简化
通过在一组训练 mesh $\mathbf{y}^i$ 上，基于 SSD 的 proxy-bone 预测和 GT 顶点位置之间的 $L^2$ 差值来测量顶点简化的误差 $E(T^i_b,\alpha_{a,b})=\sum_i^N\sum_a^V||\mathbf{y}_a^i-\sum_b^P\alpha_{a,b}T_b^i\hat{\mathbf{y}}_a||^2$。为了找到给定最大误差阈值 $\epsilon$ 时， proxy-bone 数量 $P$ 的最小值，即
$$
\underset{T_b^i,\alpha_{a,b}}{\min}P \qquad \text{subject to}\; E(T_b^i,\alpha_{a,b})<\epsilon
$$

$\alpha_{a,b}$ 可以通过非负最小二乘法由 $P$ 和 $T_b^i$ 求得，而 $T_b^i$ 也同样可以通过非负最小二乘由 $P$ 和 $\alpha_{a,b}$ 解得。为了同时解得两者最小值，采用以下方法：
* 定义连接 proxy-bone $A$ 到 proxy-bone $B$ 的误差 $E_{A\to B}=\sum_i^N\sum_{\alpha\in G_A}||\mathbf{y}_a^i-T_b^i\hat{\mathbf{y}}_a||^2$。该误差即连接 $G_A$ 组顶点到 $G_B$ 组顶点真实近似误差的上限。 
* 如下图，将相邻组之间所有的可能连接添加到一个优先级队列中，迭代执行最低误差的连接直到误差达到阈值：
1. 先把每个三角面片 $k$ 初始化到变换矩阵 $T_k^i$，该矩阵映射了从静置姿态到每个姿态 $i$ 的所有顶点。初始化对应组 $G_k$ 以包含三角面片 $k$ 的所有顶点。
2. 选择具有最小误差 $E_{A\to B}$ 的连接 $A\to B$，把 $A$ 组的点都加入到 $B$组。
3. 由当前变化矩阵 $T_b^i$ 解得权重 $\alpha_{a,b}$ 。
4. 由当前权重 $\alpha_{a,b}$ 解得变换矩阵 $T_b^i$。
5. 如果 $E(T^i_b,\alpha_{a,b})<\epsilon$ 则重回步骤 2 。
<div align=center>  <img src="/Essay%20Note/images/RTERR_6.jpg" width=80%><br>连续的迭代将协调的顶点合并到越来越少的 proxy-bones 里。
</div>

### 3.2 预测器简化
要获得关键形变梯度预测器 $D_l(\mathbf{q})$，需要先从三角面片的形变梯度序列 $D_k^i$ 中 找到关键形变梯度序列 $D_l^i$。然后根据这些序列训练预测器。接着用 Eq.2 中目标函数做如下替换来找到最佳关键序列作为误差度量：
$$
D_k^i=\sum_{l\in 1\dots P}\beta_{k,l}D_l^i
$$

其中，$\beta_{k,l}$ 是混合权重：
$$
\underset{\beta_{k,l},D_l^i}{\argmin}\sum_{i\in 1\dots N}\sum_{k\in 1\dots T}\sum_{j=2,3}||\sum_{l\in 1\dots P}\beta_{k,l}D_l^i\hat{\mathbf{v}}_{k,j}-\mathbf{v}_{k,j}^i||^2
$$

用坐标下降法求解上述优化问题，交替求解 $\beta_{k,l}$ 和 $D_l^i$。这里可以把 $D_l^i$ 初始化为顶点聚类中 $T_b^i$ 左上角的 $3\times 3$ 矩阵

## GPU应用







