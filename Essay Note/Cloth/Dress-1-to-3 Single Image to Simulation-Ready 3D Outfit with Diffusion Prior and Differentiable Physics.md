Dress-1-to-3: Single Image to Simulation-Ready 3D Outfit with Diffusion Prior and Differentiable Physics
=====
[Li et al.](https://export.arxiv.org/abs/2502.03449)

> 从自然场景图片中重建含服装样板的服装和人体


## 1. 可微分服装仿真
### 1.1 前向仿真
作者采用**共维增量势接触法**（**Codimensional Incremental Potential Contact, CIPC**）作为仿真方法，其通过基于距离的对数势垒能量（distance-based log barrier energy）和连续碰撞检测（continuous collision detection, CCD）来保证无穿。

模拟的共维表面被离散成由点 $\textbf{\emph{V}}$ 和面 $\textbf{\emph{F}}$，$\textbf{\emph{X}}$ 指代未形变状态下顶点位置，$x^n$ 和 $v^n$ 则代表顶点在 $t^n$ 时刻的位置和速度。CIPC采用基于优化的时间积分器来实现从 $t^n$ 时刻到 $t^{n+1}=t^n+h$ 时刻的状态转换，即最小化下列能量：
$$
x^{n+1}=\argmin_x E(x)=\frac{1}{2}\left \| x-\tilde x \right \|^2_\textbf{\emph{M}}+\Psi(x;\textbf{\emph{X}})+B(x)
\tag{1}
$$

其中，$\tilde x=x^n+v^nh+gh^2$ 代表后向欧拉积分下的预测位置；$\left\|\cdot\right\|_\textbf{\emph{M}}$ 指代顶点质量 $\textbf{\emph{M}}_{ii}$ 的加权 $L_2$ 范数；$\Psi(x;\textbf{\emph{X}})$ 是弹性势能，包括拉伸能（stretching）和弯曲能（bending）；$B(x)$ 是IPC引入的对数势垒能，定义在所有接触顶点-三角形和边-边对上。当间隔从阈值 $\hat d$ 减小到 $0$ 时，每对primitive的势垒能从零增加到无穷大，提供了足够的斥力来阻止穿透。

使用线搜索牛顿法来解决优化问题，每次迭代都需要梯度和能量Hessian矩阵的解析计算。如果 $x^n$ 初始不相交，每一行搜索的步长上界由CCD限制来确保所有中间状态没有相交。最后根据位置计算得到新的速度 $v^{n+1}=(x^{n+1}-x^n)/h$。

### 1.2 可微分CIPC
CIPC仿真的控制方程可以被表示为一个由Eq.(1)的最小值的一阶最优条件推导得到的隐式非线性方程组：
$$
G(x^*;x^n,v^n,\varsigma^n)=\nabla E(x^*;x^n,v^n,\varsigma^n)=0
\tag{2}
$$

$$
x^{n+1}=x^*,v^{n+1}=\frac{1}{h}(x^*-x^n)
\tag{3}
$$

其中，$x^*$ 是系统能量 $E$ 的最小值；$x^n,v^n$ 代表最后的系统状态；$\varsigma^n$ 指代该隐式方程所有连续参数的集合，包括形状参数 $\textbf{\emph{X}}$，质量矩阵 $\textbf{\emph{M}}$，弹性模量等。尽管 $\varsigma^n$ 可能共享相同的值，仍假设其是独立的。这样能允许模拟器作为一个以 $x^n,v^n,\varsigma^n$ 作为输入，$x^{n+1},v^{n+1}$ 作为输出的可微层，从而能用 PyTorch 之类的自动可微框架处理。对于给定的训练损失函数 $\mathcal{L}$，后向算子计算 $\frac{d\mathcal{L}}{dx^n}，\frac{d\mathcal{L}}{dv^n}$ 和给定 $\frac{d\mathcal{L}}{dx^{n+1}}$。

在Eq.(2)的两边对 $x^n,v^n,\varsigma^n$ 进行全导数后可得：
$$
\frac{\partial G}{\partial x^*}[\frac{dx^*}{dx^n},\frac{dx^*}{dv^n},\frac{dx^*}{d\varsigma^n}]+[\frac{\partial x^*}{\partial x^n},\frac{\partial x^*}{\partial v^n},\frac{\partial x^*}{\partial \varsigma^n}]=0
\tag{4}
$$

进一步可得：
$$
[\frac{dx^*}{dx^n},\frac{dx^*}{dv^n},\frac{dx^*}{d\varsigma^n}]=-[\frac{\partial G}{\partial x^*}]^{-1}[\frac{\partial G}{\partial x^n},\frac{\partial G}{\partial v^n},\frac{\partial G}{\partial \varsigma^n}]
\tag{5}
$$

根据链式法则可以得到：
$$
[\frac{d\mathcal{L}}{dx^n},\frac{d\mathcal{L}}{dv^n},\frac{d\mathcal{L}}{d\varsigma^n}]=\frac{d\mathcal{L}}{dx^{n+1}}[\frac{dx^{n+1}}{dx^n},\frac{dx^{n+1}}{dv^n},\frac{dx^{n+1}}{d\varsigma^n}]+\frac{d\mathcal{L}}{dv^{n+1}}[\frac{dv^{n+1}}{dx^n},\frac{dv^{n+1}}{dv^n},\frac{dv^{n+1}}{d\varsigma^n}]
\tag{6}
$$

为了保证维度一致性，$\frac{d\mathcal{L}}{dx^n},\frac{d\mathcal{L}}{dv^n},\frac{d\mathcal{L}}{d\varsigma^n}$ 必须是行向量。这样从Eq.(3)可以得到：
$$
dx^{n+1}=dx^*,\qquad dv^{n+1}=\frac{1}{h}(dx^*-dx^n)
\tag{7}
$$

将Eq.(7)代入Eq.(6)，可以得到：
$$
[\frac{d\mathcal{L}}{dx^n},\frac{d\mathcal{L}}{dv^n},\frac{d\mathcal{L}}{d\varsigma^n}]=\frac{d\mathcal{L}}{dx^{n+1}}[\frac{dx^*}{dx^n},\frac{dx^*}{dv^n},\frac{dx^*}{d\varsigma^n}]+\frac{1}{h}\frac{d\mathcal{L}}{dv^{n+1}}[\frac{dx^*}{dx^n}-\mathbf{\emph{I}},\frac{dx^*}{dv^n},\frac{dx^*}{d\varsigma^n}]
\tag{8}
$$

重整理后可以得到：
$$
\begin{aligned}
\frac{d\mathcal{L}}{dx^n}&=[\frac{d\mathcal{L}}{dx^{n+1}}+\frac{1}{h}\frac{d\mathcal{L}}{dv^{n+1}}]\frac{dx^*}{dx^n}-\frac{1}{h}\frac{d\mathcal{L}}{dv^{n+1}} \\
[\frac{d\mathcal{L}}{dv^n},\frac{d\mathcal{L}}{d\varsigma^n}]&=[\frac{d\mathcal{L}}{dx^{n+1}}+\frac{1}{h}\frac{d\mathcal{L}}{dv^{n+1}}][\frac{dx^*}{dv^n},\frac{dx^*}{d\varsigma^n}]
\end{aligned}\tag{9}
$$

令 $\mathcal{A}=[\frac{d\mathcal{L}}{dx^{n+1}}+\frac{1}{h}\frac{d\mathcal{L}}{dv^{n+1}}][\frac{\partial G}{\partial x^*}]$，那么由Eq.(5)可以得到：
$$
\frac{d\mathcal{L}}{dx^n}=-\mathcal{A}\frac{\partial G}{\partial x^n}-\frac{1}{h}\frac{d\mathcal{L}}{dv^{n+1}}
\tag{10}
$$

$$
[\frac{d\mathcal{L}}{dv^n},{\frac{d\mathcal{L}}{d\varsigma^n}}]=-\mathcal{A}[\frac{\partial G}{\partial v^n},\frac{\partial G}{\partial \varsigma^n}]
\tag{11}
$$

其中，$\mathcal{A}$ 是通过求解一个系数矩阵为$\frac{\partial G}{\partial x^n}$的线性系统得到的，这个系数矩阵是系统能量 $E$ 的Hessian矩阵。项 $\mathcal{A}[\frac{\partial G}{\partial x^n},\frac{\partial G}{\partial v^n},\frac{\partial G}{\partial \varsigma^n}]$ 通过 $G$ 将 $\mathcal{A}$ 的微分分别反向传播到 $x^n,v^n$ 和 $\varsigma^n$。这个过程可以通过将 $G$ 视为一个支持自动微分的可微分层来实现。使用AutoDiff，我们消除了手动推导 $\frac{\partial G}{\partial v^n}$ 和 $\frac{\partial G}{\partial \varsigma^n}$ 的解析表达的需求。

## 2.算法总览
如图1，首先从单视角图片来预测初始的服装样板（sewing pattern），然后生成连续的多视角RGB图片，再据此预测的人体姿态。3D服装初始化的方式是将2D样板拼接和悬垂到预测的人体模型上。使用一个可微分CIPC模拟器来模拟衣服与人体的互动，这样我们就可以去由之前生成的多视角RGB图片、法线图和分割信息来优化物理参数和服装样板的形状。优化后的状态可以产生一个模拟就绪的场景，包含一个穿着与输入对齐的3D服装的人类模型，其服装材质使用视觉-语言模型和图片扩散模型来自动生成。最后，通过应用CIPC模拟器，可以模拟出预测人体模型穿着优化服装同时执行复杂动作序列的动态场景。
<div align="center">
<img src="/Essay%20Note/images/dress1-2-3_method_overview.png" width=1000 height=350 />
<br> 图1：文章框架
</div>

## 3.预优化步骤
### 3.1 可模拟服装样板的生成
首先使用[SewFormer](https://arxiv.org/abs/2311.04218)对单视角图像生成带拼接信息的初始服装样板，由一组2D平面上的二次贝塞尔曲线组成的一组不相连的版片所表征。每块版片的曲线会连接成一段环路。参数化表征则为：一组曲线 $\mathcal{E}$ 由一组曲线顶点 $\mathcal{P}={P_i}$ 和曲线上每条边 $e\in\mathcal{E}$ 的一组控制点 $\mathcal{K}={K^e}$ 来表征。为了实现服装模拟，每个版片会被离散化为三角形面片。

然后使用弧长参数化来实现沿着版片边缘的均匀采样。对每条拼接版片边，保证采样点数量相同，这使得服装模拟中可以应用顶点到顶点的拼接约束，从而简化缝纫过程。接着对每个板块的内部独立执行[狄洛尼三角剖分](https://www.cs.cmu.edu/~quake/triangle.html)。

### 3.1.1 版片参数化









fewfwef

