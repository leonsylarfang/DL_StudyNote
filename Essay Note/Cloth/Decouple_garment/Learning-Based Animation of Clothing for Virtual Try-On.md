Learning-Based Animation of Clothing for Virtual Try-On
=========
[Santesteban et al.](https://arxiv.org/abs/1903.07190)
> 用循环神经网络来回归衣服褶皱
## 布料模型
### 人体定义
$M_b$：可形变人体网格，由形状参数 $\beta$ 和姿态参数 $\theta$ 决定。
$M_c$：可形变衣物网格，穿戴在人体网格上。
$S_c(\beta,\theta)$：布料网格，由 $M_c$ 穿戴在 $M_b$ 上物理仿真的结果。

人体网格由如下公式(1)定义:
$$
M_b(\beta,\theta)=W(T_b(\beta,\theta),\beta,\theta,\mathcal{W}_b)
\tag{1}
$$

其中，$T_b(\beta,\theta)\in\mathbb{R}^{3\times V_b}$ 是一个包含 $V_b$ 个顶点的无姿态人体网格，由模板人体网格 $\mathbf{\bar{T}}_b$ 根据人体形状和姿态校正形变得到；$W(\cdot)$ 是一个蒙皮方程，将无姿态人体网格 $T_b$ 根据以下条件形变：
- 形状参数 $\beta$：骨骼的关节位置
- 姿态参数 $\theta$：根据蒙皮权重矩阵 $\mathcal{W}_b$ 得到的连结网格的关节角度。

### 布料蒙皮流程
1. 将包含 $V_c$ 个顶点的模板布料网格 $\mathbf{\bar{T}}_c\in\mathbb{R}^{3\times V_c}$ 形变，计算得到无姿态布料网格 $T_c(\beta,\theta)$。
2. 用蒙皮方程 $W(\cdot)$ 来产生所有的布料形变。

一个关键点是将**体型相关的服装合身度**与**形状-姿态相关服装褶皱**作为模板布料网格的纠正位移（corrective displacement）来计算，从而产生无姿态布料网格：
$$
T_c(\beta,\theta)=\mathbf{\bar{T}}_c+R_G(\beta)+R_L(\beta,\theta)
\tag{2}
$$

其中，$R_G()$ 和 $R_L()$代表了两个非线性回归量。
所以最终的布料可被表达成公式(3)：
$$
M_c(\beta,\theta)=W(T_c(\beta,\theta),\beta,\theta,\mathcal{W}_c)
\tag{3}
$$
