Cloth Simulation for Computer Graphics
=====
[Tuur Stuyck](/PDFs/cloth-simulation-for-computer-graphics.pdf)

# 1. 引言
工程界为了有助于建模和预测现实世界场景的模拟，仿真更侧重于物理精度。相较于工程界，计算机图形领域更侧重于最终结果和外观，而物理精度是起到对视觉物理可信目标的辅助作用。

布料动态模拟的计算机实现使其能快速迭代设计，并可视化服装的悬垂，以及由于材料和缝制样式（sewing pattern）而产生的褶皱和折叠等效果。这样能节省材料和制造成本，并加快设计过程。
## 1.1 布料模拟的应用
- 离线模拟（offline simulations）：用多个不同的设置进行多个模拟，经过计算、调整和后处理再呈现到屏幕上。该类方法通常具有较高的可信度和可控性，但不具备交互性。
- 实时模拟（real-time simulations）：允许用户输入和虚拟环境变化对模拟结果实时产生影响，通常在GPU上实现。这类方法通常要求快速和稳定。

## 1.2 布料模拟的发展历史
### 纯几何
- [Weil(1986)](https://dl.acm.org/doi/10.1145/15922.15891) 开发了第一个计算机图形模型来模拟在约束点处的织物悬垂。
### 基于物理
#### 基于连续表征
- [Feynman(1986)](https://dspace.mit.edu/handle/1721.1/14924) 用一种连续表征将服装建模成一个弹性模片（elastic sheet），连续动力学用连续的质量表征而非离散粒子来模拟物理性质和运动。
- [Terzopoulos et al.(1987)](https://dl.acm.org/doi/10.1145/37402.37427) 提出了更通用的布料弹性建模。其依赖于一个假定，即所讨论的材料均可用连续动力学精准建模。该假定对于结构能在只在微观或分子层面可视的塑料或橡胶等材料是合理的。
- [Carignan et al.(1992)](https://dl.acm.org/doi/10.1145/142920.134017) 总结了90年代的针对连续表征的主要后续研究。
#### 基于粒子系统
- [Haumann(1987)](https://ohiostate.pressbooks.pub/graphicshistory/chapter/19-3-physical-based-modeling/) 和 [Breen et al.(1992)](https://link.springer.com/article/10.1007/BF01897114) 研究了基于粒子的布料模拟，离散粒子模型使用连接粒子的机械系统来模拟宏观动态。布料行为并不能由连续模型有效描述，而应由纱线之间的机械作用来描述，当布料移动时，纱线会发生碰撞、弯曲和滑动，从而产生摩擦。
- [Provot(1995)](https://graphics.stanford.edu/courses/cs468-02-winter/Papers/Rigidcloth.pdf) 使用弹簧连接的质点来模拟布料的弹性行为。
- [Baraff and Witkin(1998)](https://dl.acm.org/doi/abs/10.1145/3596711.3596792) 引入使用隐式积分的基于三角形的布料模拟，快速模拟相对复杂的服装，成为主流方法的基础。
- [Volino and Magnenat-Thalmann(2000)](https://link.springer.com/book/10.1007/978-3-642-57278-4) 、[House and Breen(2000)](https://www.taylorfrancis.com/books/edit/10.1201/9781439863947/cloth-modeling-animation-donald-house-david-breen) 和 [Thalmann et al.(2004)](https://diglib.eg.org/items/4668a00b-0130-4c14-a7e4-e8a22740b389) 都是很好的相关介绍。

# 2. 布料表征
## 2.1 三角形
在计算机上表示几何图形的一种方法是使用三角形（trangles）。如图2.1所示，左图衣服模型是由右图线框模型的许多的小三角形面片构成的。归功于基于物理的布料模拟，这种服装在虚拟角色的身体上有自然褶皱，会对重力或风等外力做出反应，并随着角色与身体的碰撞而移动。
<div align="center">
<img src="/Essay%20Note/images/CSCG_triangle_model.png" width=368 height=400
 />
<br> 图2.1：三角面片构成服装
</div>

如图2.2，三角形由三个<font color=red>定点</font>或者<font color=red>粒子</font>组成，由边连接。虽然布料是一种连续材料，但在计算机图形学中，我们会使用离散的粒子来表征，因为这样更方便计算。注意过大的时间步长，即过度的离散化会导致系统不稳定。
<div align="center">
<img src="/Essay%20Note/images/CSCG_triangle_intro.png" width=368 height=250
 />
<br> 图2.2：三角形实例
</div> 

## 2.2 粒子
一个粒子 $i$ 由一个3D位置 $\mathbf{x}_i\in\mathbb{R}^3$ 和速度 $\mathbf{v}_i\in\mathbb{R}^3$ 定义，而两者又可组合成*粒子状态* $\mathbf{q}_i=\left <\mathbf{x}_i,\mathbf{v}_i\right >$。因为例子的位置和速度会随着时间变化，所以必须遵循描述材料属性的物理定律。对于布料而言，不能过度拉伸、存在切变和弯曲、产生褶皱。

将包含 $N$ 个粒子的粒子系统用单个长向量 $\mathbf{x}\in\mathbb{R}^{3N}$ 和 $\mathbf{v}\in\mathbb{R}^{3N}$来表示：
$$
\mathbf{x}=\begin{bmatrix}
 x_{0_x}\\
 x_{0_y}\\
 x_{0_z}\\
 \vdots\\
 x_{{N-1}_x}\\
 x_{{N-1}_y}\\
 x_{{N-1}_z}\\
\end{bmatrix},\qquad \mathbf{v}=\begin{bmatrix}
 v_{0_x}\\
 v_{0_y}\\
 v_{0_z}\\
 \vdots\\
 v_{{N-1}_x}\\
 v_{{N-1}_y}\\
 v_{{N-1}_z}\\
\end{bmatrix}
\tag{2.1}
$$

有不同的数据结构可用来存储三角面片，最直接的方法是将粒子位置和速度存储在独立的数组中，再用独立且唯一的指针 $i$ 来索引。[Botsch et al.(2010)](http://dx.doi.org/10.1201/b10688) 使用了更为复杂的方法来编码三角形连接。

## 2.3 力
布料会受到如风、重力或碰撞等外力的影响，但内力才是让布料具有特有表现的原因。这些作用在粒子上的拉伸、切变和弯曲力使其表现出纺织品的特性。我们使用数值积分来计算这些作用在三角形顶点上的内力，从而随时间推进模拟。计算这些模拟的方法通常是计算离散时间步长的粒子状态，比如从 $t_0$ 时刻开始，经过长度为 $h$ 的步长后，更新 $t_0+h$ 和 $t_0+2h$ 等时刻的粒子状态，从而更新模拟的结果。

## 2.4 帧与步骤
在图形学中，通过每秒向观看者显示多张图像来产生连续运动的错觉。每秒显示图像的数量被称为每秒帧数或帧率，通常使用的帧率为每秒24、30甚至是60帧。为了获得稳定的结果，每帧需要执行多个模拟步骤，所以模拟步骤的数量要远高于帧率。
总结起来，布料的计算模型有两种离散化方法：
- **空间离散化**：连续布料由有限数量的三角形组成，而三角形又由具有位置和速度的粒子组成。
- **时间离散化**：连续时间被离散为持续时长为 $h$ 的离散时间步长。

# 3. 显式积分
## 3.1 积分形式
粒子 $i$ 的状态由其位置 $\mathbf{x}_i=[x_{i_x},x_{i_y},x_{i_z}]\in\mathbb{R}^3$ 和速度 $\mathbf{v}_i=[v_{i_x},v_{i_y},v_{i_z}]\in\mathbb{R}^3$ 来定义。

由运动定律可以得到：
$$
\begin{aligned}
    \frac{d\mathbf{x}(t)}{dt}&=\mathbf{v}(t)\\
    \frac{d\mathbf{v}(t)}{dt}&=\mathbf{a}(t)
\end{aligned}
\tag{3.1}
$$

将公式 3.1 简单离散化：
$$
\begin{aligned}
    \frac{\mathbf{x}(t+h)-\mathbf{x}(t)}{h}&\approx\mathbf{v}(t)\\
    \frac{\mathbf{v}(t+h)-\mathbf{v}(t)}{h}&\approx\mathbf{a}(t)
\end{aligned}
\tag{3.2}
$$

其中 $h$ 指代时间步长。公式 3.2 可被改写为：
$$
\begin{aligned}
    \mathbf{x}(t+h)&\approx\mathbf{x}(t)+h\mathbf{v}(t)\\
    \mathbf{v}(t+h)&\approx\mathbf{v}(t)+h\mathbf{a}(t)
\end{aligned}
\tag{3.3}
$$

从牛顿第二定律可知，力 $\mathbf{f}\in\mathbb{R}^{3N}$ 可以以加速度的形式作用于系统：
$$
\mathbf{f(x,v},t)=\mathbf{M\cdot a}(t)
\tag{3.4}
$$

其中矩阵 $\mathbf{M}$ 表征了整个系统的质量矩阵，包含了所有的粒子质量 $m_i$。

所有力的结合最终构成了粒子的加速度，其可分为内力和外力：
- **内力**是由布料模型产生的力，其对布料的内部形变做出反应，包括拉伸、压缩、切变和弯曲。
- **外力**不是由布料本身产生，包括重力、碰撞或如风一类的空气动力学效应，具体可参见 [Ling(2000)](https://dl.acm.org/doi/10.5555/350448.351319)

将公式 3.3 和 3.4 改写成时间符号离散形式：
$$
\begin{aligned}
    \mathbf{x}_{n+1}&=\mathbf{x}_n+h\mathbf{v}_n\\
    \mathbf{v}_{n+1}&=\mathbf{v}_n+h\mathbf{M}^{-1}\mathbf{f}_n
\end{aligned}
\tag{3.5}
$$

其中 $n+1$ 代表下一时刻，$n$ 代表当前时刻。通过计算出力对应的加速度，即可得到下一时刻的粒子速度，从而计算出其下一时刻新的位置。

图 3.1 展示了时间关于位置的导数。这种离散化方法是最简单但并不是最好的，其被称为**正向欧拉积分**或**显式积分**。离散化可以被解释为在当前时刻沿函数的切线行进一个时间步长，是一个近似值，从图中可以看到，代表 $t_1=t_0+h$ 时刻的粒子新位置已经不在曲线 $\mathbf{x}(t)$ 上了。因为存在误差，故步长 $h$ 值的选择会对累计误差产生影响。
<div align="center">
<img src="/Essay%20Note/images/CSCG_integration.png" width=500 height=250
 />
<br> 图3.1：经过一个步长 h，速度的有限差分近似更新
</div> 

## 3.2 稳定性分析
如 [3.1](#31-积分形式) 中所述，$h$ 值的选择过大会导致累计误差过大，故显示方法只在相对较小的时间步长 $h$ 内是稳定的，所以我们需要跟进一步的量化误差。

### 3.2.1 测试方程
这个限制问题可以先从分析下述简单初值问题入手：
$$
\begin{aligned}
    \frac{dy(t)}{dt}&=f(t,y(t))\\
    y(0)&=y_0
\end{aligned}
\tag{3.6}
$$

在时间中，积分器的稳定性不是在实际系统方程上测试的，而是在高度简化的测试方程上。我们这里要使用的测试方程是大林方程（Dahlquists equation）：
$$
\frac{dy(t)}{dt}=\lambda y(t)
\tag{3.7}
$$

其中系数 $\lambda=\alpha+i\beta\in\mathbb{C}$ 是一个和时间无关的复数。该方程的解析解为：
$$
y(t)=y_0e^{\lambda t}
\tag{3.8}
$$

将公式 3.8 代入 3.7，这个微分方程的值可由以下方程得到：
$$
|y(t)|=|y_0|\cdot\left |e^{(\alpha+i\beta)t}\right |
\tag{3.9}
$$

这个微分方程的解只有在 $\alpha$ 为非正时才会随时间衰减，从而使方程稳定。当 $\alpha$ 是正值时，解的值会随时间不短累加。所以在接下来，我们假设 $\lambda$ 都满足这个非正的条件。

### 3.2.2 显式欧拉分析
用显式欧拉方法离散化公式 3.6 可得:
$$
\begin{aligned}
    y_{k+1}&=y_k+hf(t_k.y_k)\\
    &=y_k+h\lambda y_k\\
    &=(1+h\lambda)y_k
\end{aligned}
\tag{3.10}
$$

基于初始状态 $y_0$，方程在时刻 $t_k=hk$ 的解为：
$$
y_k=(1+h\lambda)^ky_0
\tag{3.11}
$$

为了使方程稳定，即当 $k$ 趋于无穷大时，这个离散解会收敛，必须保证 $|1+h\lambda|<1$。满足这一条件的 $h\lambda$ 取值集合被称为稳定趋于，在数学上被表示为
$$
S=\{h\lambda\in\mathbb{C}:|1+h\lambda|<1\}
\tag{3.12}
$$

图 3.2 更直观地展现了稳定区域。因为 $\lambda$ 是由方程本身决定的，我们只关注 $h$ 的取值。通常而言， $h$ 都会选的比较小，从而使 $h\lambda$ 保持在稳定区域。但这种对于时间步长的限制会导致布料模拟的的方程很僵硬，即 $\alpha$ 为一个极大的负值。
<div align="center">
<img src="/Essay%20Note/images/CSCG_stable_region.png" width=500 height=350
 />
<br> 图3.2：显示欧拉方法的稳定区域（蓝色圆盘）。Re 和 Im 分别代表 hλ 的实部和虚部。
</div> 

#### 离散化导致的问题
使用显式欧拉方法解一个解为同心圆的方程 $y(t)=-\omega^2\frac{d^2y(t)}{dt}$ 时，真实解应该永远环绕在圆上开始。然而因为离散化，误差总会导致解向外螺旋。这就需要引入阻尼来减缓扩散现象。

## 3.3 自适应时间步长
在采用显示积分法时，需要实时监控误差，从而改变步长以减少系统不稳定的风险。前面提到，不同于工业界更关注精度的问题，图形学更侧重于稳定性，仅需要视觉上合理即可。因为相较于抗弯曲和抗切变，布料的抗拉伸力更强。基于这一点，[Baraff and Witkin(1998)](https://dl.acm.org/doi/abs/10.1145/3596711.3596792) 提出了一种检测拉伸量并相应调整步长的方法。

利用当前的时间步长和预测的下一时刻位置 $\Delta\mathbf{x}$ 来计算布料模型的拉伸力。当拉伸量超过特定阈值时，预测位置将被舍弃，并将步长减半后重新计算预测位置。这个阈值可以宽松的选择，因为不稳定的系统会快速导致过量的位置更新。

当模拟还未开始不稳定时，则希望增加步长来提高计算效率，所以当模拟系统能用一个特定步长执行多个步骤时，只要不超过设置的最大值，步长就会翻倍。

# 4. 质量-弹簧模型
质量-弹簧模型（mass-spring model）即由弹簧连接的质点，这些点不仅有位置和速度等属性，还有质量。确定质量的方法是确定材料的表面密度 $\rho$。

## 4.1 计算质量
我们使用没有厚度的 2D 三角形元素来建模布料。越重的材料会有更高的密度。通过遍历所有的三角形，计算三角形表面的质量和密度的乘积来得到每个三角形质量，而每个质点的质量则等于累加其所属每个三角形质量的 $1/3$。这样得到的参考配置被称为无形变配置，即几何形状尚未因受力而产生形变。

因为衣服的边缘会存在更多的折叠和缝合的情况，通常会在这部分粒子增加一些额外的质量，其不一定和三角形面积成正比。这样就已经可以更好地模拟出较重的双层布料效果，而非需要实际表现边缘的几何折叠。

为了方便表达运动方程，一个维度为 $\mathbb{R}^{3N\times 3N}$ 的质量矩阵 $\mathbf{M}$ 会被引入（见公式 4.1）。而在实际中，因为该质量矩阵为一个对角矩阵，我们只需要存储一个长度为 $N$ 的数组即可。
$$
\mathbf{M}=\begin{bmatrix}
m_0 & 0 & 0 & 0 & \cdots & 0 \\
0 & m_0 & 0 & 0 & \cdots & 0 \\
0 & 0 & m_0 & 0 & \cdots & 0 \\
0 & 0 & 0 & m_1 & \cdots & 0 \\
\cdots & \cdots & \cdots & \cdots & \ddots & \cdots \\
0 & 0 & 0 & 0 & cdots & m_{N-1}
\end{bmatrix}
\tag{4.1}
$$

## 4.2 计算力
以日常经验来看，布料不会拉伸或切变太多，但另一方面，布料很容易弯曲从而产生褶皱和堆叠。为了模拟形变阻力，可以在每一组相邻粒子之间构建一根弹簧。在哪些粒子之间用弹簧连接会直接影响到布料的表现。

图 4.2 是一个包含 $9$ 个粒子的简单质量-弹簧系统，其中绿线代表拉伸弹簧，抵抗网格拉伸；紫线代表切变弹簧抵消切变力；黄线代表弯曲弹簧，抵消弯曲力。相邻质点之间由拉伸和切变弹簧连接，而两环相邻（2-ring neighbor）的粒子则越过相邻粒子由弯曲弹簧连接。一般而言，布料模型的拉伸弹簧会设置很大的弹簧系数，而切变和弯曲弹簧则会设置较小的数值。显然这三者之间并没有完全的分离，互相之间会产生影响。
<div align="center">
<img src="/Essay%20Note/images/CSCG_mass_spring.png" width=400 height=360
 />
<br> 图4.1：一个简单的质量-弹簧系统
</div> 

### 4.2.1 能量最小化
物理系统总是试图达到最小的能量状态，所以我们可以通过定义能量并将其最小化来建模物理系统。当所有的力达到平衡且使系统处于能量较低的状态，这就是热力学第二定律。

为了达到低能量状态，那*保守力*应该为能量方程 $E(\mathbf{x})\in\mathbb{R}$的负梯度：
$$
\mathbf{f(x)}=-\frac{\partial E(\mathbf{x})}{\partial\mathbf{x}}
\tag{4.2}
$$

类似碰撞和摩擦力等力，其做工并不仅由粒子的初位置和终位置决定，而保守力的做工与路径无关，可以由势能来定义。以最简单的保守力———重力来举例，重力只沿 $\mathit{z}$ 轴作用，质点 $i$ 在重力作用下的势能为 $E_g(\mathbf{x}_i)=m_igx_{iz}$，$x_{iz}$ 代表粒子沿 $z$ 轴的位置分量。那么根据公式 4.2，其合力为：

$$
\begin{aligned}
    \mathbf{f}_i(\mathbf{x})&=-\frac{\partial E_g(\mathbf{x})}{\partial\mathbf{x}_i}\\
    &=-\left[\frac{\partial E_g}{\partial x_{ix}},\frac{\partial E_g}{\partial x_{iy}},\frac{\partial E_g}{\partial x_{iz}}\right] \\
    &=-\begin{bmatrix}
    0 \\
    0 \\
    m_ig \\
    \end{bmatrix} 
\end{aligned}
\tag{4.3}
$$

### 4.2.2 弹簧势能和力
由胡克定律可以得到，连接着粒子 $i$ 和 $j$ 的，静置长度为 $L$，弹簧刚度系数为 $k$ 的弹簧，其势能为：
$$
E_{i,j}(\mathbf{x})=\frac{1}{2}k(||\mathbf{x}_i-\mathbf{x}_j||-L)^2
\tag{4.4}
$$

其中，$||\cdot||$ 代表欧氏距离。从而我们可以得到弹簧施加在两个粒子上的力：
$$
\begin{aligned}
\mathbf{f}_i(\mathbf{x})&=-\frac{\partial E_{i,j}(\mathbf{x})}{\partial\mathbf{x}_i}\\
&=-k(||\mathbf{x}_i-\mathbf{x}_j||-L)\frac{\mathbf{x}_i-\mathbf{x}_j}{||\mathbf{x}_i-\mathbf{x}_j||}\\
\end{aligned}
\tag{4.5}
$$

$$
\begin{aligned}
\mathbf{f}_j(\mathbf{x})&=-\frac{\partial E_{i,j}(\mathbf{x})}{\partial\mathbf{x}_j}\\
&=k(||\mathbf{x}_i-\mathbf{x}_j||-L)\frac{\mathbf{x}_i-\mathbf{x}_j}{||\mathbf{x}_i-\mathbf{x}_j||}\\
\end{aligned}
\tag{4.6}
$$

如图4.2，我们可以很直观地理解 $\mathbf{f}_i=-\mathbf{f}_j$，即连接两个粒子的弹簧将沿着相同的轴以同样大小的力在相反的方向上拉动或推动粒子。这个力同样为保守力，其与重力一样做功无关于路径，只关乎粒子的始末位置。所产生力与拉伸或压缩量呈线性比例的弹簧被称为线性弹簧或胡克弹簧。
<div align="center">
<img src="/Essay%20Note/images/CSCG_spring_force.png" width=520 height=100
 />
<br> 图4.2：连接两个粒子的弹簧所施加的力
</div> 

### 4.2.3 弹簧阻力
如 [3.2.2](#离散化导致的问题) 中提到的，保持模拟系统的稳定需要添加阻力。而添加这个阻力最简单的方式就是加一个与运动方向相反的力，对于与粒子 $j$ 相连接的粒子 $i$，其阻力为：
$$
\begin{aligned}
\mathbf{d}_i(\mathbf{x})&=-k_d(\mathbf{v}_i-\mathbf{v}_j)\\
&=-\mathbf{d}_j(\mathbf{x})
\end{aligned}
\tag{4.7}
$$

其中，$k_d$ 代表阻尼系数。这样就简单地模拟了现实世界的能量耗散表现。

## 4.3 整合
根据公式 3.5 离散牛顿定律，我们可以得到如下系统：
$$
\begin{aligned}
\Delta\mathbf{x}&=h\mathbf{v}_n\\
\Delta\mathbf{v}&=h\left(\mathbf{M}^{-1}\mathbf{f}(\mathbf{x}_n,\mathbf{v}_n)\right)
\end{aligned}
\tag{4.8}
$$

其中，$\Delta\mathbf{x}=\mathbf{x}_{n+1}-\mathbf{x}_n$ 和 $\Delta\mathbf{v}=\mathbf{v}_{n+1}-\mathbf{v}_n$ 分别代表了 $n+1$ 和 $n$ 时刻位置差和速度差。

对于系统中的每一个粒子，内力通过计算负的能量方程梯度得到，再把外力加到内力上，这样就可以获得速度更新 $\Delta\mathbf{v}$，所以 $n+1$ 时刻的状态为：
$$
\begin{aligned}
\mathbf{x}_{n+1}&=\mathbf{x}_n+\Delta\mathbf{x}\\
\mathbf{v}_{n+1}&=\mathbf{v}_n+\Delta\mathbf{v}
\end{aligned}
\tag{4.9}
$$

## 4.4 可撕布料
通常情况下，布料会在没有太多抵抗下进行小幅度拉伸。但当拉伸超过一定量时，会导致极大的力来抵抗这种形变。本节讨论的就是拉建模这种情形。

针对这种布料表现，一种处理方法是将其建模成可撕布料，即当质点间弹簧被拉伸超过静置长度的一定比例时，布料会断裂和撕裂。要实现撕裂效果只要把弹簧从质量-弹簧网络中去除即可，而 [Metaaphanon et al.(2009)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2009.01561.x) 提出了一种更为先进的办法。


# 5. 隐式积分
虽然在图形学中，只需要视觉可信的结果，但过大的时间步长还是会过度降低模拟结果的准确度，甚至导致结果完全无法使用。为了获得更稳定的模拟结果，[Baraff and Witkin(1998)](https://dl.acm.org/doi/abs/10.1145/3596711.3596792) 将隐式积分方案应用于布料模拟。虽然隐式积分单步计算相较于显示欧拉方法更为复杂，但其能使得在大时间步长时仍然获得稳定的结果，故我们可以采取更大的步长来加快模拟，而不用担心稳定性问题。

[这篇笔记](https://liaohuming.com/2017/10/19/20171019-%E6%98%BE%E5%BC%8F%E6%97%B6%E9%97%B4%E7%A7%AF%E5%88%86%E4%B8%8E%E9%9A%90%E5%BC%8F%E6%97%B6%E9%97%B4%E7%A7%AF%E5%88%86/)很好地介绍了显示时间积分和隐式时间积分的区别。隐式方法只在步长结束时计算力，即用当前时刻的结果和下一时刻预测结果来进行反复迭代得到下一时刻的结果。必须通过迭代得到，无条件收敛，是能量平衡的结果，通常用于静力分析。其最大的优点在于无条件稳定性，即时间步长任意大。

而显式方法是用上一时刻结果和当前时刻的结果来计算下一时刻的结果，有条件收敛，且要求步长较小，通常用于动力分析。其最大的优点在于不需要迭代，计算简单。

## 5.1 后向欧拉
从时刻 $t_n$ 开始，每一步位置更新 $\Delta\mathbf{x}=\mathbf{x}_{n+1}-\mathbf{x}_{n}$ 和速度更新 $\Delta\mathbf{v}=\mathbf{v}_{n+1}-\mathbf{v}_{n}$ 可按如下隐式积分（又称后向欧拉）计算：
$$
\begin{aligned}
    \Delta\mathbf{x}&=h(\mathbf{v}_n+\Delta\mathbf{v})\\
    \Delta\mathbf{v}&=h\left(\mathbf{M^{-1}f(x}_n+\Delta\mathbf{x,v}_n+\Delta\mathbf{v})\right)
\end{aligned}
\tag{5.1}
$$

将公式 5.1 改写得到：
$$
\begin{aligned}
    \mathbf{x}_{n+1}&=\mathbf{x}_n+h\mathbf{v}_{n+1}\\
    \mathbf{v}_{n+1}&=\mathbf{v}+h\mathbf{M}^{-1}\mathbf{f}_{n+1}
\end{aligned}
\tag{5.2}
$$

其中，$\mathbf{f}_{n+1}=\mathbf{f(x}_{n+1},\mathbf{v}_{n+1})=\mathbf{f(x}_n+\Delta\mathbf{x,v}_n+\Delta\mathbf{v})$

### 5.1.1 线性化
公式 5.2 是非线性的，可以采用 Newton-Raphson 方法精确地求解。然而在计算机图形学中，我们会采用更快但精确度较低的线性系统来近似非线性系统，来得到后向欧拉公式的近似解。

这个线性化是通过一阶泰勒近似替代非线性力项来实现的：
$$
\mathbf{f(x}_n+\Delta\mathbf{x,v}_n+\Delta\mathbf{v})\approx\mathbf{f}_n+\frac{\partial\mathbf{f}}{\partial\mathbf{x}}\Delta\mathbf{x}+\frac{\partial\mathbf{f}}{\partial\mathbf{v}}\Delta\mathbf{v}
\tag{5.3}
$$

其中，$\mathbf{f}_n=\mathbf{f(x}_n,\mathbf{v}_n)$，$\mathbf{f,x}\in\mathbb{R}^{3N}$，所以 $\frac{\partial\mathbf{f}}{\partial\mathbf{x}},\frac{\partial\mathbf{f}}{\partial\mathbf{v}}\in\mathbb{R}^{3N\times 3N}$。将方程 5.3 代入非线性方程 5.1 即可得到近似的线性系统，再将公式 5.1 合并消除 $\Delta\mathbf{x}$ 可以得到速度的更新：
$$
\Delta\mathbf{v}=h\mathbf{M}^{-1}\left(\mathbf{f}_n+\frac{\partial\mathbf{f}}{\partial\mathbf{x}}h(\mathbf{v}_n+\Delta\mathbf{v})+\frac{\partial\mathbf{f}}{\partial\mathbf{v}}\Delta\mathbf{v}\right)
\tag{5.4}
$$

这种方法被称为隐式方法，可以看到未知的速度更新 $\Delta\mathbf{v}$ 同时出现在等式的两边，我们不能通过简单的移项来解，而是需要一个线性系统。将方程 5.4 重新排序可得：
$$
\left(\mathbf{I}-h\mathbf{M}^{-1}\frac{\partial\mathbf{f}}{\partial\mathbf{v}}-h^2\mathbf{M}^{-1}\frac{\partial\mathbf{f}}{\partial\mathbf{x}}\Delta\mathbf{v}\right)=h\mathbf{M}^{-1}\left(\mathbf{f}_n+h\frac{\partial\mathbf{f}}{\partial\mathbf{x}}\mathbf{v}_n\right)
\tag{5.5}
$$

其中，$\mathbf{I}\in\mathbb{R}^{3N\times 3N}$ 是一个单位矩阵，方程 5.5 是一个形式为 $\mathbf{A}\Delta\mathbf{v=b}$ 的线性方程组,其中 $\mathbf{A}$ 是一个[稀疏矩阵](https://zhuanlan.zhihu.com/p/620446480)，具有特定的块结构。。在构造这些矩阵时，我们会计算内力的 $\frac{\partial\mathbf{f}}{\partial\mathbf{x}}$ 和 $\frac{\partial\mathbf{f}}{\partial\mathbf{v}}$，但所有的外力都会被组合到 $\mathbf{f}_n$ 中，且除非知道如何计算，不需要考虑对系统的导数。


## 5.2 稳定性分析
类似 [3.2](#32-稳定性分析)，将连续时间方程离散化，但采用后向隐式欧拉的方法，可以得到离散化后的测试方程：
$$
\begin{aligned}
y_{k+1}&=y_k+hf(t_{k+1},y_{k+1})\\
&=y_k+h\lambda y_{k+1}
\end{aligned}
\tag{5.6}
$$

合并同类项后，我们可以得到下一时刻的值 $y_{k+1}$：
$$
y_{k+1}=\frac{1}{1-h\lambda}y_k
\tag{5.7}
$$

从而可以逐步推导得到：
$$
y_k=(\frac{1}{1-h\lambda})^ky_0
\tag{5.8}
$$

若假设，当时间 $t$ 趋于无穷大时，方程的精确解是有界的，边界条件为 $\mathrm{Re}(h\lambda)$ 非正。因为时间步长 $h$ 为正，这个边界条件就变成了 $\mathrm{Re}(\lambda)$ 非正。从而我们可以得到离散解的边界条件为：
$$
\left |\frac{1}{1-h\lambda}\right |<1,\; \mathrm{Re}(\lambda)<0
\tag{5.9}
$$

对于任意的时间步长 $h$ 和实数 $\lambda$ 





 



ewr