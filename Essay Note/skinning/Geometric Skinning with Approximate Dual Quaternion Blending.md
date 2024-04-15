Geometric Skinning with Approximate Dual Quaternion Blending
====
[Kavan et al.](https://dl.acm.org/doi/abs/10.1145/1409625.1409627)
> 更快更简单且更准确的蒙皮技术，基于对偶四元数的混合

## Overview


## 1. Basic Conception
### geometric skinning
针对一个具有 $p$ 个关节的模型，每个关节的相较于其rest pose的变换矩阵为 $C_j\in SE(3)$。顶点 $\mathbf{v}$ 依据权重 $\mathbf{w}=(w_1,w_2,...,w_n)$ 附着于关节 $j_1,j_2,...,j_n$。$n$ 代表作用于该顶点的关节数量。权重 $\mathbf{w}$ 通常被视为凸，即 $\sum_{i=1}^nw_i=1, w_i\geqslant 0$，代表了每个关节对该顶点的影响。 那么每个形变后的顶点 $\mathbf{v}'$ 可以由下公式计算得到：
$$
\mathbf{v}'=\sum_{i=1}^nw_iC_{j_i}\mathbf{v}_i \tag{1}
$$

这其实就是对每个影响的关节的形变进行一个加权平均。但这样可能会导致出现 **candy-wrapper** 现象出现，即预测点与关节点重合，导致部分皮肤坍缩到同一个点。

为了解决这一问题，可以用 matrix-verctor 乘法的分配性改写 Eq.(1)，先计算变换矩阵 $C_j$， 再将其应用于顶点 $\mathbf{v}$：
$$
\sum_{i=1}^nw_iC_{j_i}\mathbf{v}_i=\left(\sum_{i=1}^nw_iC_{j_i}\right)\mathbf{v}_i
$$

但混合矩阵 $\sum_{i=1}^nw_iC_{j_i}$ 并不一定是刚体变换，所以线性混合蒙皮会被非预期的缩放所影响。

### spherical blend skinning
几何蒙皮还涉及到每一个顶点对应的旋转关节问题，老的方法是单纯让顶点由最近的关节作为旋转中心，但并不总是准确的。为了解决这一问题，球形混合蒙皮技术被提了出来。其原理是将平移和旋转独立进行。其定义每个顶点的旋转中心 $\mathbf{r}$ 的方法为:
$$
\argmin \sum_{1\leqslant a<b \leqslant n}\left \| C_{j_a}\mathbf{r}-C_{j_b}\mathbf{r} \right \|
$$

但因为最小二乘用 SVD 求解每个点的旋转中心消耗太高，无法满足实时需求。

### logmatrix blengding
为了给每个点都快速找到其对应的旋转中心，可以用对数矩阵混合。该方法线性组合了矩阵对数而非矩阵本身。矩阵 $M\in SE(3)$ 的对数矩阵可以被写作：
$$
\log M=\begin{pmatrix}
0 & -\theta a_3 & \theta a_2 & m_1\\
\theta a_3 & 0 & -\theta a_1 & m_2 \\
-\theta a_2 & \theta a_1 & 0 & m_3\\
0 & 0 & 0 & 0 \\
\end{pmatrix}
$$ 

其中，$\theta$ 是旋转角度；$\mathbf{a}=(a_1,a_2,a_3)$，$\|\mathbf{a}\|=1$ 代表旋转轴的方向；$\mathbf{m}=(m_1,m_2,m_3)=\theta\mathbf{r}\times\mathbf{a}+d\mathbf{a}$，其中 $\mathbf{r}$ 是旋转中心，$\times$ 代表外积，$d$ 代表了强度，是平移与旋转速度的比值。


## 2. Rigid Transformation Blending
#### 引理
如果给定一个平移为 $\mathbf{t}\in R^2$，旋转角度为 $\alpha$ 的 2D 变换，其旋转中心 $\mathbf{r}$ 为：
$$
\mathbf{r}=\frac{1}{2}\left(\mathbf{t+z\times t}\cot\frac{\alpha}{2} \right)\tag{2}
$$

其中，$\mathbf{z}=(0,0,1)$ 代表 z 轴。
#### 证明
假定 $\mathbf{t}$ 是一个复数， 那么旋转中心 $\mathbf{r}\in C$ 即为刚体旋转的一个静止点，其有：
$$
\mathbf{r}=\mathbf{t}+e^{i\alpha}\mathbf{r}
$$

由此可以得到：
$$
\mathbf{r}=\frac{\mathbf{t}}{1-e^{i\alpha}}\cdot\frac{1-e^{-i\alpha}}{1-e^{-i\alpha}}=\frac{1-e^{-i\alpha}}{2(1-\cos\alpha)}\mathbf{t}=\frac{1-\cos\alpha+i\sin\alpha}{2(1-\cos\alpha)}\mathbf{t} 
$$

再代入三角恒等式
$$
\frac{\sin\alpha}{1-\cos\alpha}=\cot\frac{\alpha}{2}
$$

即可得到 Eq.2。将 z 轴替换成任意旋转轴即可将该公式推广到 3D 场景。

### 2.1 2D 例子
给定两个 2D 刚体变换 $M_1, M_2\in SE(2)$，分别由旋转角 $\alpha_1,\alpha_2$ 和平移向量 $(2\cos\alpha_1,2\sin\alpha_1),(2\cos\alpha_2,2\sin\alpha_2)$ 组成。

所有的旋转中心其实都可以从下述族中找到：
$$
\begin{pmatrix}
\cos\alpha & -\sin\alpha & 2\cos\alpha \\
\sin\alpha & \cos\alpha & 2\sin\alpha \\
0 & 0 & 1 \\
\end{pmatrix},\quad \alpha\in\left[ 0,\frac{\pi}{2} \right]
$$

如果将 $\mathbf{t}=(2\cos\alpha,2\sin\alpha)$ 代入 Eq.2，可以得到：
$$
\mathbf{r}=\left( -1,\frac{\sin\alpha}{1-\cos\alpha} \right)=\left( -1,\cot\frac{\alpha}{2} \right) \tag{3}
$$

那么 $M_i$ 对应的旋转中心为：
$$
\mathbf{r}_i=(r_{i,x},r_{i,y})=\left( -1,\cot\frac{\alpha_i}{2}\right),\quad i=1,2
$$

为了求得插值矩阵 $M(t)$，我们指定 $\mathbf{p}=(cos\frac{\alpha_i}{2},\sin\frac{\alpha_i}{2})$，那么插值角 $\alpha(t)$ 有：
$$
\left( cos\frac{\alpha(t)}{2},\sin\frac{\alpha(t)}{2} \right)=\frac{(1-t)\mathbf{p}_1+t\mathbf{p}_2}{\left\| (1-t)\mathbf{p}_1+t\mathbf{p}_2 \right\|}
$$

根据半角公式，




