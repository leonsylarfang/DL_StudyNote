#整理近期NeRF相关文章

---

NeRF的优点：
1. 广应用，包括：robotics, urban mapping, autonomous navigation, VR/AR. 
2. 自监督。只需要images和poses来学习场景，不需要3D/depth supervision。
3. 效果逼真。

---

## [NeRF](https://arxiv.org/abs/2003.08934)原理
### 参数概念
NeRF用辐射场（Radiance Field）来表示三维场景中每个点在每一个相机视角下的体密度（volume density）和颜色（color），记作：
$$
F(\mathbf{x},\theta,\phi)\to (\mathbf{c},\sigma)
$$
其中，$\mathbf{x}=(x,y,z)$ 代表场景内坐标，$(\theta,\phi)$ 代表方位角（azimuthal angle）和极视角（polar viewing angle），$\mathbf{c}=(r,g,b)$ 代表颜色，$\sigma$ 代表体密度。一个或多个 MLP $F_\Theta$ 被用来近似（approximate）这5维数据。此外，这两个视角参数 $(\theta,\phi)$ 常用一个3维笛卡尔单位向量 $\mathbf{d}=(d_x,d_y,d_z)$ 来表示。

点颜色 $\mathbf{c}$ 同时依赖于视角和坐标，但体密度 $\sigma$ 与视角无关，故限制对其预测，可以保持该神经网络的多视图一致性。在基准NeRF中，MLP被设计成两个分支$\sigma$-MLP 和 $\mathbf{c}$-MLP通过两步来达成这一目标，这两个MLP是同一个网络的两个不同分支，而非不同的MLP：
1. 以 $\mathbf{x}$ 为 $\sigma$-MLP 的输入，输出 $\sigma$ 和一个高维（256）特征向量。
2. 将特征向量与视角 $\mathbf{d}$ 拼接后传入$\mathbf{c}$-MLP，输出颜色 $\mathbf{c}$。

### 训练和计算过程
NeRF的训练过程可按下图分为三个步骤：
![](/DL_StudyNote/Generative%20Content/NeRF/images/NeRF_1.png)
(1) 对图片上的每个像素，发送相机光线穿过场景，并生成一组采样点。
(2) 对每个采样点，通过送入视角与采样位置到NeRF MLP来计算出局部颜色和密度。
(3) 根据计算出的颜色和密度，由[体渲染](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html)的方法生成图像，然后与GT计算loss。

**体渲染**计算图片像素颜色的方法为：给定体密度方程 $\sigma$ ，点颜色方程 $\mathbf{c}$，相机位置 $\mathbf{o}$，视角 $\mathbf{d}$，那么在近点 $t_n$ 到远点 $t_f$ 区间，由任意相机光线 $\mathbf{r}$ 上所有的点 $\mathbf{r}(t)=\mathbf{o}+t\mathbf{d}$ 的渲染出的像素颜色 $C(\mathbf{r})$ 可以这样得到：
$$
C(\mathbf{r})=\int_{t_n}^{t_f}T(t)\cdot\sigma(\mathbf{r}(t))\cdot\mathbf{c}(\mathbf{r}(t),\mathbf{d})\cdot dt
$$

其中，$dt$ 表示光线在每个积分步骤中传播的微分距离，$\sigma(\mathbf{r}(t))$ 和 $\mathbf{c}(\mathbf{r}(t),\mathbf{d})$ 表示光线 $\mathbf{r}$ 在视角 $\mathbf{d}$ 下 $t$ 点的体密度和颜色。$T(t)=\exp(-\int_{t_n}^{t}\sigma(\mathbf{r}(u))\cdot du)$ 表示从近点 $t_n$ 到点 $t$ 在无遮挡情况下的累计透明度。

在实际计算过程中，会采用分级体素抽样法 (hierarchical volume sampling)，将光线分为N个等距的空间栈，并从每个栈中均匀采样，从而图像像素颜色可以被近似为：

$$
\hat{C}(\mathbf{r})=\sum_{i=1}^N\alpha_iT_i\mathbf{c}_i \qquad T_i=\exp(-\sum_{j=1}^{i-1}\sigma_j\delta_j)
$$

其中，$\delta_i$ 表示采样点 $i$ 到采样点 $i+1$ 的距离，$(\sigma_i,\mathbf{c}_i)$ 表示光线上采样点 $i$ 的由NeRF MLP计算出的体密度和颜色，$\alpha_i=1-\exp(-\sigma_i\delta_i)$ 表示采样点的透明度。

那么光线的预期深度则可由累计透明度计算得到，并同上离散方法可以得到近似值：

$$
d(\mathbf{r})=\int_{t_n}^{t_f}T(t)\cdot\sigma(\mathbf{r}(t))\cdot t\cdot dt
\qquad
\hat{D}(\mathbf{r})=\sum_{i=1}^N\alpha_it_iT_i
$$

对于每一个像素点，都可以用光度损失（photometric loss）来优化 MLP 参数，所以可以得到全图loss为：
$$
L=\sum_{\mathbf{r}\in R}||\hat{C}(\mathbf{r})-C_{gt}(\mathbf{r})||_2^2
$$

其中 $C_{gt}(\mathbf{r})$ 代表光线 $\mathbf{r}$ 对应的 GT 像素颜色，$R$ 代表渲染出合成图像对应得到所有的光线集合。

### 优化方法
#### 位置编码（Positional Encoding）
对场景坐标 $\mathbf{x}$（归一化到 $[-1,1]$）和 视角单位向量 $\mathbf{d}$ 添加如下位置编码 $\gamma$ 可以极大提升重构精度：

$$
\gamma(v)=( sin(2^0\pi v),cos(2^0\pi v),sin(2^1\pi v),cos(2^1\pi v),...,sin(x^{N-1}\pi v),cos(x^{N-1}\pi v))
$$

其中 $N$ 是一个用户自定义参数，原文为 $\mathbf{x}$ 和 $\mathbf{d}$ 分别设置了 $N=10$ 和 $N=4$

#### 深度处理
通过用光线的预期深度来将体密度限制成类delta函数或强制平滑深度，可以提高生成精度。

#### 分级体素抽样 (hierarchical volume sampling)
因为**自由空间**（**free space**）和**遮挡区域**（**occluded region**）这样对于计算颜色值毫无贡献但仍被重复采样的区域存在，拖慢了NeRF的训练效率，NeRF采用了分级表征的方式同时优化两个网络：**粗糙**（**coarse**）和**精细**（**fine**）网络。

先通过分级采样得到 $N_c$ 个点，通过粗糙网络的渲染方程计算得到：

$$
\hat C_c(\mathbf{r})=\sum^N_{i=1}w_ic_i   \qquad w_i=T_i(1-exp(\sigma_i\delta_i))
$$

然后对 $w_i$ 进行归一化 $\hat w_i=\frac{w_i}{\sum^{N_c}_{j=1}w_j}$ 来产生分段常数概率密度函数，然后通过逆变换采样获得 $N_f$ 个点，并添加至 $N_c$ 个点中用于精细网络的渲染(共计 $N_f+N_c$个点)。通过二次采样的方式，可以使采样点更多采用对于计算颜色有贡献的体素进行计算。

最后通过同时优化两个网络的最小残差即可得到损失方程：
$$
L=\sum_{r\in R}\big[||\hat{C}_c(\mathbf{r})-C(\mathbf{r})||^2_2+||\hat{C}_f(\mathbf{r})-C(\mathbf{r})||^2_2 \big]
$$

最小化该损失方程即可使NeRF达到优化条件。

---

## 数据集
#### [LIFF](https://arxiv.org/abs/1905.00889)
该数据集由手机拍摄的24个现实生活场景组成，每个场景包含20~30张图像，其视角都是朝向中心物体，并用 [COLMAP](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf) 包来计算图像位姿。这个数据集**经过良好的基准测试**，可以与已知方法做对比。

#### [DTU](http://roboimagedata.compute.dtu.dk)
该数据集由80个场景组成，每个场景都在中心物体半径50厘米的球体范围内采样了49张视图。其中21个场景额外提供了半径65厘米范围内的15个不同相机视角下的视图。还有44个附加场景，每个场景以90度为间隔旋转和扫描4次。在7种不同的照明条件下保持 $1600\times1200$ 的**高分辨率**。其提供了**高精度的相机参数和位姿**，以及参考密集点云来提供参考三维几何形状。但因自遮挡问题，部分区域的扫描并不完整。

#### [ScanNet](https://arxiv.org/abs/1702.04405)
该数据集是一个大规模多模态数据集，包含超过250万个室内场景视图，已经带注释的相机位姿、参考三维表面、语意标签和CAD模型。深度帧以 $640\times480$ 个像素捕获，RGB图像则以 $1296\times968$ 个像素捕获。其丰富的**语意信息**使其非常适用于满足场景编辑、场景分割、语义视图合成等需求。

#### [ShapeNet](https://arxiv.org/abs/1512.03012)
该数据集是一个简单的大规模合成 3D 数据集，由分为 3135 个类别的 3D CAD 模型组成。最常用的是 12 个常见对象类别子数据集。当**基于对象的语义标签是特定 NeRF 模型的重要组成部分**时，有时会使用此数据集。在 ShapeNet CAD 模型中，Blender 等软件通常用于渲染已知姿势的训练视图。

### 建筑规模数据集（Building-scale Dataset）
#### [Tanks and Temples](https://dl.acm.org/doi/10.1145/3072959.3073599)
该数据集是视频 3D 重建数据集。它由14个场景组成，包括“坦克”、“火车”等个体物体，以及“礼堂”、“博物馆”等**大型室内场景**。使用高质量工业激光扫描仪捕获三维GT数据。GT点云用于通过对应点的最小二乘优化来估计相机位姿。该数据集包含大规模场景，其中一些场景是室外的，适合**无边界背景的NeRF模型**训练。它的GT点云也可用于某些数据融合方法，或测试深度重建。

#### [Matterport-3D](https://arxiv.org/abs/1709.06158)
该数据集是一个现实生活数据集，由 90 个建筑规模场景的 194400 个 RGB-D 全局注册图像的 10800 个全景视图组成。提供**深度、语义和实例注释**。以 $1280\times1024$ 分辨率提供彩色和深度图像，每张全景图片有 18 个视点。 90 栋建筑每栋平均面积为 2437 平方米。总共提供了 50811 个对象实例标签，映射到 40 个对象类别。

#### [Replica](https://arxiv.org/abs/1906.05797)
该数据集是是一个真实的室内数据，由使用带有红外投影仪的定制 RGB-D 设备捕获的 18 个场景和 35 个室内房间组成。某些 3D 特征是手动固定的（**精细尺度的网格细节**，例如小孔），并且手动分配反射表面。**语义注释**（88 个类别）分两步执行，一次在 2D 中，一次在 3D 中。基于类和基于实例的语义标签都可用。

### 大规模城市数据集（Large-scale Urban Dataset）
#### [KITTI](https://www.cvlibs.net/datasets/kitti/)
该数据集是一个著名的城市规模 2D-3D 计算机视觉数据集套件，专为**自动驾驶视觉算法**的训练和**基准测试**而创建。该套件包含用于立体 3D 语义 + 2D 语义分割、流量、里程计、2D-3D 对象检测、跟踪、车道检测和深度预测/完成的标记数据集，包含超过 93000 个深度图以及相应的 RGB 图像和原始 LiDAR 扫描。然而，该数据集与 NeRF 特定数据集相比，其相机覆盖范围相对**稀疏**，因此在设计模型时需要考虑稀疏视图。

#### [Waymo](https://arxiv.org/abs/1912.04838)
该数据集是 KITTI 的替代品。该数据集覆盖 72 平方公里，是根据点云和视频数据创建的，还包含**用于 2D 和 3D 对象检测和跟踪的带注释标签**。该数据集包含 1150 个独立场景（相对于 KITTI 的 22 个场景），并且具有更高的 LiDAR 和相机分辨率，此外它也包含了两个数量级（80K vs 12M）的对象注释。

### 人脸/头像数据集（Human Avatar/Face Dataset）
#### [Nerfies](https://arxiv.org/abs/2011.12948) & [HyperNerf](https://arxiv.org/abs/2106.13228)
这两个数据集是专注于**人脸**的单相机数据集，通过相对于主体移动连接到杆子的两个相机来生成运动。前者包含五个**保持静止**的人类主体，以及另外四个包含移动人类主体、一只狗和两个移动物体的场景。后者侧重于**拓扑变化**，包括人类主体打开和关闭眼睛和嘴巴、剥香蕉、3D打印小鸡玩具和变形扫帚等场景。

#### [ZJU-MOCap LightStage](https://arxiv.org/abs/2012.15838)
该数据集是一个多视图（20+摄像机）运动捕捉数据集，由9个**动态人体序列**组成，使用 21 个同步摄像机拍摄的，序列长度在 60 到 300 帧之间。

#### [NeuMan](https://arxiv.org/abs/2203.12575)
该数据集由 6 个视频组成，每个视频时长为 10 到 20 秒，由手机摄像头拍摄，跟踪行走的人类受试者执行其他**简单动作**，例如旋转或挥手。

#### [CMU Panoptic](https://arxiv.org/abs/1612.03153)
该数据集是一个大型多视角 5 多主题数据集，由参与社交互动的人群组成。该数据集包含 65 个序列和 150 万个标记骨架。该传感器系统由 480 个 VGA 视图 ($640\times480$)、30+HD ($1920\times1080$) 视图和 10 个 RGB-D 传感器组成。场景标有**个体主题和社会群体语义、3D 身体姿势、3D 面部标志以及文字记录 + 说话者 ID** 的标签。

---

## 质量评估指标
### PSNR↑（Peak Signal to Noise Ratio）
PSNR是一个无参考质量指标：
$$
PSNR(I)=10\cdot\log_{10}(\frac{MAX(I)^2}{MSE(I)})
$$

其中，$MAX(I)$ 表示图像中最大像素可能值（8位整型则是255），$MSE(I)$ 是所有颜色通道上的像素均方误差。

### [SSIM](https://ieeexplore.ieee.org/document/1284395)↑（Structural Similarity Index Measure）
SSIM是一个全参考质量评估指标：
$$
SSIM(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
$$

其中，$C_i=(K_iL)^2$，$L$ 代表像素变化范围（8位整型是255），$K_1=0.01$，$K_2=0.03$；局部统计量 $\mu$，$\sigma$ 是在 $11\times11$ 圆形对称高斯加权窗口内计算得到，权重 $w_i$ 的标准差为1.5，归一化到1。

### [LPIPS](https://arxiv.org/abs/1801.03924)↓（Learned Perceptual Image Patch Similarity）
LPIPS是一个完整的参考质量评估指标:
$$
LPIPS(x,y)=\sum_l^L\frac{1}{H_lW_l}\sum_{h,w}^{H_l,W_l}||w_l\odot(x_{hw}^l-y_{hw}^l)||_2^2
$$

其中，$x_{hw}^l$ 和 $y_{hw}^l$ 当像素宽度为 $w$，像素高度为 $h$ 时在第 $l$ 层的参考和评估图像特征；$H_l$ 和 $W_l$ 在 $l$ 层的特征图高度和宽度。

---

## 文章
### 1. Survey
#### 2021
[Neural Fields in Visual Computing and Beyond](https://arxiv.org/abs/2111.11426)   (CVPR 2022) : 神经场 (Neural Fields) 在计算机视觉中的应用 

[Multimodal image synthesis and editing: The Generative AI Era](https://arxiv.org/abs/2112.13592) ： 重点介绍多模态 NeRF。

#### 2022
[Advances in Neural Rendering](https://arxiv.org/abs/2111.05849) : 全面且详细地介绍了神经渲染（Neural Rendering），但只介绍NeRF的模型发展。


#### 2023 
[NeRF: Neural Radiance Field in 3D Vision, A Comprehensive Review](https://arxiv.org/abs/2210.00379) ：全面而详细的介绍自2020 ECCV NeRF 发布以来的，关于NeRF更方面的优化，更侧重于针对NeRF的发展做出详尽的介绍。

### 2. 优化
关于NeRF的优化点其实可以根据NeRF的特点分为下图的几类：

1. 生成图像质量优化。
2. 训练/推理速度。
3. 稀疏视图。
4. 结合其他方法的联合优化。
5. NeRF结构优化。
6. 位姿估计。

针对每个优化点，已经有如下的研究工作。

### 2.1 基于几何的生成视图质量优化
与NeRF童年的一篇[文章](https://arxiv.org/abs/2006.10739)通过引入傅里叶特征（Fourier Features）来让神经网络学习低维域中的高频函数，从而提高合成图像的质量，使其保留更多的细节特征。
#### （1）更好的视图合成效果
##### · [Mip-NeRF](https://arxiv.org/abs/2103.13415)  [2021]
Mip-NeRF通过引入集成位置编码 IPE（Integrated Positional Encoding）来用锥体追踪（cone tracing）取代普通光线追踪。
![](/DL_StudyNote/Generative%20Content/NeRF/images/Mip-NeRF.png)
如上图所示，对每个像素，都从相机中心沿着观察方向投射出由多元高斯近似出的圆锥体，来产生 IPE。

相较于基准NeRF，Mip-NeRF 的**性能更优**，在**低分辨率**下更明显，此外还扩展到了**无界场景**。其主要优化包括：
- 提出了用 NeRF MLP 而非图像监督的 MLP，该 MLP 仅用来预测体密度来选择合适的采样间隔。
- 为高斯构建了新的的场景参数化方式。
- 引入新的正则化方法来防止浮动几何现象和背景塌缩问题。

##### · [Ref-NeRF](https://arxiv.org/abs/2112.03907) [2021]
Ref-NeRF是基于Mip-NeRF构建的，能精准建模**镜面反射**和**高光效果**。其利用基于观察方向的局部法向量的反射来参数化 NeRF 辐射率。

##### · [RapNeRF](https://arxiv.org/abs/2205.05922) [2022]
相较于基准NeRF更适用于视图插值，RapNeRF更适合外推（**extrapolation**）。其提出的随机光线投射算法 RRC（Random Ray Casting）和射线图谱 RA （Ray Atlas）可以适应其他 NeRF 框架，并带来更好的视图合成质量。

#### （2）其他几何优化
##### · [SNeS](https://arxiv.org/abs/2206.06340) [2022]
SNeS通过对几何和材料属性的软对称约束，学习部分对称和部分隐藏场景中物体的可能对称性，从而改进了几何构造，产生更真实的合成效果。

##### · [S$^3$-NeRF](https://arxiv.org/abs/2210.08936) [2022]
S$^3$-NeRF 使用阴影和着色来推断场景几何形状，并实现了**单图像** NeRF 训练，重点是几何形状恢复。该方法从合成数据集和现实数据集的单个图像中实现了出色的**深度图**和**表面法线重建**。

### 2.2 关于训练速度和推理速度的优化
在基准NeRF中，采用了分级渲染，先渲染粗网格，再将其输出作为精网格的采样点来大量减少采样时间。而在后续关于NeRF训练和推理速度优化的方向大致分为：

- 烘焙模型（Baked Models）：该类模型训练、预计算并将 NeRF MLP 的评估结果存储到更易于访问的数据结构中，这只会加快推理速度。
- 非烘焙模型（Non-baked Models）：这类模型通常（但并非总是）尝试从训练好的 MLP 参数中学习分散的场景特征。这允许使用更小的 MLP ，以内存为代价提高训练和推理速度，通常用混合场景表示。

##### · [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf) [2020]
原始NeRF基于[JAX](https://github.com/google/jax)的实现，**速度稍快**，更适合**分布式**计算。

#### （1）烘焙模型
##### · [SNeRG](https://arxiv.org/abs/2103.14645) [2021]
在“烘焙”的过程中将预先计算的漫反射颜色、密度和特征向量存储在稀疏体素网格上。在评估期间，MLP 用于产生镜面反射颜色，它与沿光线合成的镜面反射颜色的 alpha 相结合，产生最终的像素颜色。该方法比原始NeRF实现快 3000 倍。

##### · [PlenOctree](https://arxiv.org/abs/2103.14024) [2021]
训练了一个球谐函数 NeRF（NeRF-SH），它预测颜色函数的球谐系数，而不是直接预测颜色函数。他们构建了 MLP 颜色的预先计算的球谐 (SH) 系数的八叉树。 PlenOctrees 可以使用初始训练图像进一步优化，实现速度与 SNeRG 相当。

##### · [FastNeRF](https://arxiv.org/abs/2103.10380) [2021]
将颜色函数 $\mathbf{c}$ 分解为位置相关 MLP 的输出（也生成密度 $σ$）和方向相关 MLP 的输出的内积。这使得 Fast-NeRF 能够轻松地在场景的密集网格中缓存颜色和密度评估，从而将推理时间大大缩短了 3000 倍以上。

##### · [KiloNeRF](https://arxiv.org/abs/2103.13744) [2021]
将场景分成数千个单元，并训练独立的 MLP 以对每个单元进行颜色和密度预测。这数千个小型 MLP 是使用大型预训练教师 MLP 的知识蒸馏进行训练的，这与“烘焙”密切相关。他们还采用了早期光线终止和空白空间跳跃。仅这两种方法就将基线 NeRF 的渲染时间缩短了 71 倍。将基线 NeRF 的 MLP 分成数千个较小的 MLP 进一步将渲染时间缩短了 36 倍，从而使渲染时间总共加快了 2000 倍。

#### （2）非烘焙模型
##### · [NSVF](https://arxiv.org/abs/2007.11571) [2020]
将场景建模为一组由体素界定的辐射场。特征表示是通过对存储在体素顶点的可学习特征进行插值获得的，然后由计算 $σ$ 和 $\mathbf{c}$ 的共享 MLP 进行处理。 NSVF 对射线使用基于稀疏体素相交的点采样，这比密集采样或原始 NeRF 的分层两步方法要高效，但更加占用内存。

##### · [Instant-NGP](https://arxiv.org/abs/2201.05989) [2022]
提出了一种学习参数多分辨率哈希编码，该编码与 NERF 模型 MLP 同时训练。他们还采用了先进的射线行进技术，包括指数步进、空白空间跳跃、样本压缩。这种新的位置编码和相关的 MLP 优化实现极大地提高了训练和推理速度，以及最终 NERF 模型的场景重建精度。在训练的几秒钟内，他们就取得了与之前 NERF 模型中数小时的训练相似的结果。

#### （3）深度监督 & 点云
除了优化模型，还有研究者通过对从 LiDAR 或 SfM 获取的**点云对预期深度进行监督**，使模型收敛速度更快，且质量更高，并且需要更少的训练视图，适用于 few-shot/稀疏视图的 NeRF 场景。
##### · [DS-NeRF](https://arxiv.org/abs/2107.02791) [2021]
除了通过体积渲染和光度损失进行颜色监督之外，DS-NeRF 还使用 COLMAP 的方法从训练图像中提取的稀疏点云对深度进行监督，使其被建模为围绕由稀疏点云记录的深度的正态分布，再添加 KL 散度项以最小化光线分布和噪声深度分布的散度。

##### · [NerfingMVS](https://arxiv.org/abs/2109.01129) [2021]
先用 COLMAP 从点云提取出稀疏先验，送入预训练好的单相机深度网络中提取出深度先验图。最后在体渲染过程中用来**限制光线边界**。

##### · [PointNeRF](https://arxiv.org/abs/2201.08845) [2022]
PointNeRF计算出**特征点云**作为体渲染的中间步骤来拟合局部密度和颜色，用于体渲染。此外特征点云的使用还可以领模型跳过空白空间，使得其训练速度是基准NeRF的三倍。

#### （3） MLP优化方法
部分研究者提出的优化神经渲染的方法也加快了训练速度。

##### [Plenoxels](https://arxiv.org/abs/2112.05131) [2021]
快速无 MLP 体积渲染基于 Plenoctree，对场景进行体素化并存储密度标量和球谐系数方向相关的颜色。Plenoxel 完全**跳过了 MLP 训练**，而是直接将这些特征拟合到体素网格上，却保持了与 NeRF ++ 和 JaxNeRF **相当的结果**，**训练时间缩短了数百倍**。这些结果表明 NeRF 模型的主要贡献是给定密度和颜色的新视图的体渲染，而不是密度和颜色 MLP 本身。

##### · [DVGO](https://arxiv.org/abs/2111.11215) [2021]
作者使用了类似于原始 NeRF 论文的粗细采样的采样策略，首先训练粗体素网格，然后基于粗网格的几何形状训练细体素网格。该模型被命名为直接体素网格优化 (DVGO)，它在 SyntheticNeRF 数据集上仅进行 15 分钟的训练，其性能优于基准 NeRF（1-2 天）训练。

##### · [TensoRF](https://arxiv.org/abs/2203.09517) [2022]
将标量密度和向量特征（可以使用 SH 系数或通过 MLP 解码的特征）存储为分解张量。他们的 VM 分解在视觉质量方面表现更好，尽管需要牺牲内存。训练速度与 Pleoxels 相当，并且比基准 NeRF 模型快得多。

##### · [IBRNet](https://arxiv.org/abs/2102.13090) [2021]
作为视图合成的 NeRF 相邻方法，广泛应用于基准测试中。对于目标视图，IBRNet 从训练中的 16 个视图集中选择了 N 个视图，这些视图的观看方向最相似。使用 CNN 从这些图像中提取特征。这是针对每个查询点完成的，并且（沿射线的所有查询点的）结果被连接在一起并输入到预测密度的射线变换器中。


### 2.3 Few-shot / 稀疏训练视图 NeRF
##### · [PixelNeRF](https://arxiv.org/abs/2012.02190) [2020]
使用卷积神经网络的预训练层（和双线性插值）来提取图像特征。 NeRF 中使用的摄像机光线随后被投影到图像平面上，并提取每个查询点的图像特征。然后将特征、视图方向和查询点传递到 NeRF 网络，生成密度和颜色。

##### · [MVSNeRF](https://arxiv.org/abs/2103.15595) [2021]
用预训练的 CNN 提取 2D 图像特征。然后使用平面扫描和基于方差的成本将这些 2D 特征映射到 3D 体素。使用预训练的 3D CNN 来提取 3D 神经编码体积，该编码体积用于使用插值生成每点潜在代码。当执行体渲染的点采样时，NeRF MLP 使用这些潜在特征、点坐标和观察方向作为输入来生成点密度和颜色。训练过程涉及 3D 特征量和 NeRF MLP 的联合优化。在 DTU 数据集上进行评估时，训练后 15 分钟内，MVNeRF 可以达到与基线 NeRF 训练数小时相似的结果。

##### · [DietNeRF](https://arxiv.org/abs/2104.00677) [2021]
额外引入了基于 [Clip-ViT](https://arxiv.org/abs/2103.00020) 提取的图像特征的语义一致性损失 $L_{sc}$，减少为余弦相似度损失归一化特征向量。使用 DietNeRF 的语义一致性损失进行微调的 [PixelNeRF](https://arxiv.org/abs/2012.02190) 模型可以生成很好的单视角生成图像。

##### · [RegNeRF](https://arxiv.org/abs/2112.00724) [2021]
旨在解决稀疏输入视图的 NeRF 训练问题。RegNeRF 采用了额外的深度和颜色正则化。该模型在 [DTU](http://roboimagedata.compute.dtu.dk) 和 [LIFF](https://arxiv.org/abs/1905.00889) 数据集上进行了测试，优于 [PixelNeRF](https://arxiv.org/abs/2012.02190)、[MVSNeRF](https://arxiv.org/abs/2103.15595) 等模型。 RegNeRF 不需要预训练，其性能与这些在 [DTU](http://roboimagedata.compute.dtu.dk) 上预训练并按场景进行微调的模型相当。它的性能优于 [DietNeRF](https://arxiv.org/abs/2104.00677)。

#### （1）GAN-based NeRF
##### · [GRAF](https://arxiv.org/abs/2007.02442) [2020]
首个对抗训练的 NeRF 模型。基于NeRF的生成器 $G$ 会被隐式的 appearance code $\mathbf{z}_\mathbf{a}$ 和 shape code $\mathbf{z}_\mathbf{s}$ 限制：
$$
G(\gamma(\mathbf{x}),\gamma(\mathbf{d}),\mathbf{z}_\mathbf{s},\mathbf{z}_\mathbf{a}) \to (\sigma,\mathbf{c})
$$

在实践中，shape code，调节场景密度会跟 embeded 位置进行拼接，作为无关方向 MLP 的输入。appearance code，调节场景辐射会跟 embeded 视角方向进行拼接，作为方向相关 MLP 的输入。最后 NeRF 体采样生成的图像会使用判别器 CNN 来进行对抗训练。


##### · [$\pi$-GAN](https://arxiv.org/abs/2012.00926) [2020]
生成器 [SIREN-based NeRF](https://arxiv.org/abs/2006.09661) 体积渲染器，用正弦激活代替密度和颜色 MLP 中的标准 ReLU 激活。 π-GAN 在 Celeb-A 、CARLA 和 Cats 等标准 GAN 数据集上的表现优于 [GRAF](https://arxiv.org/abs/2007.02442)。

##### · [EG3D](https://arxiv.org/abs/2112.07945) [2021]
使用新颖的混合三平面表示，其特征存储在三个轴对齐的平面上，并使用小型解码器 MLP 在 GAN 框架中进行神经渲染。 GAN 框架由用于三平面的姿势条件 StyleGAN2 特征图生成器、将三平面特征转换为低分辨率图像的 NeRF 渲染模块以及超分辨率模块组成。然后将超分辨率图像输入 StyleGAN2 鉴别器。

##### · [StyleNeRF](https://arxiv.org/abs/2110.08985) [2022]
是一项极具影响力的工作，专注于 2D 图像合成，通过使用 NeRF 将 3D 感知引入 [StyleGAN](https://arxiv.org/abs/1812.04948) 图像合成框架。 StyleNeRF 使用带有上采样模块的风格代码条件 NeRF 作为生成器和 [StyleGAN2](https://arxiv.org/abs/1912.04958) 判别器，并为 StyleGAN 优化目标引入了一个新的路径正则化项。

#### （2）联合优化的隐式模型
这些模型使用 latent code 作为视图合成的关键方面，但与场景模型联合优化 latent code，可以被理解为**无判别器的 GAN**。
##### · [CLIP-NeRF](https://arxiv.org/abs/2112.05139) [2021]
CLIP-NeRF 的神经辐射场对 [Edit-NeRF](https://arxiv.org/abs/2105.06466) 进行了创新，基于标准潜在条件 NeRF，即以形状和外观潜在代码为条件的 NeRF 模型。然而，通过使用CLIP，CLIP-NeRF 可以使用形状和外观映射器网络从用户输入文本或图像中提取引起的潜在空间位移。然后，这些位移可根据这些输入文本或图像修改场景的 NeRF 表示。此步骤允许跳过 Edit-NeRF 中使用的每次编辑潜在代码优化，从而根据任务将速度提高约 8-60 倍。

#### （3）Diffusion & NeRF
##### · [DreamFusion](https://arxiv.org/abs/2209.14988) [2022]
是一个 text-to-3D 的 diffusion NeRF 模型。DreamFusion 中的 NeRF 模型使用 2D 扩散模型的图像进行训练。对于每个要生成的对象或场景，文本提示输入到扩散模型中，并从头开始训练基于 [Mip-NeRF](https://arxiv.org/abs/2103.13415) 的 NeRF 模型。文本提示允许在扩散图像生成阶段控制主体的视图（有些提示使用诸如“俯视图”、“前视图”、“后视图”等关键字）。 NeRF 训练的一个关键修改是表面颜色的参数化由 MLP 而不是辐射率控制。但生成的 NeRF 模型缺乏生成更精细细节的能力。

##### · [Magic3D](https://arxiv.org/abs/2211.10440) [2022]
Magic3D 以 DreamFusion 为基础，针对低分辨率扩散图像引起的问题。使用了两阶段粗细方法。在粗略阶段，Magic3D 使用 [Instant-NGP](https://arxiv.org/abs/2201.05989) 作为 NeRF 模型，该模型使用图像扩散模型 eDiff-I [140] 根据文本提示生成的图像进行训练。然后将从 [Instant-NGP](https://arxiv.org/abs/2201.05989) 中提取的粗略几何形状放置在网格上，该网格使用潜在扩散模型生成的图像在精细阶段进行了优化。该方法允许基于提示的场景编辑、通过对主题图像进行调节的个性化文本到 3D 生成以及风格引导的文本到 3D 生成。实验了超过 397 个提示生成的对象，每个对象由 3 个用户评分，这也表明 Magic3D 优于 DreamFusion。

### 2.4 无界场景和场景合成
当尝试将 NeRF 模型应用于室外场景时，需要分离前景和背景。这些户外场景还对图像间的照明和外观变化提出了额外的挑战。
##### · [NeRF-W](https://arxiv.org/abs/2008.02268) [2020]
解决了基线 NeRF 模型的两个关键问题。同一场景的真实照片可能包含由于光照条件而导致的每张图像的外观变化，以及每张图像中不同的瞬态物体。场景中所有图像的密度 MLP 保持固定。然而，NeRF-W 将其彩色 MLP 调节为每个图像的外观 embedding。此外，另一个以每图像瞬态 embedding 为条件的 MLP 预测了瞬态对象的颜色和密度函数。

##### · [NeRF++](https://arxiv.org/abs/2010.07492) [2020]
该模型适用于通过使用球体分隔场景来为无界场景生成视图。球体内部包含所有前景物体和所有虚拟摄像机视图，而背景则位于球体外部。然后使用倒置球体空间对球体的外部进行重新参数化。训练了两个独立的 NeRF 模型，一个用于球体内部，一个用于球体外部。相机光线积分也分两部分进行评估。

##### · [GIRAFFE](https://arxiv.org/abs/2011.12100) [2020]
采用与 NeRF-W 类似的方法构建，使用生成潜在代码并分离背景和前景 MLP 以进行场景合成。 GIRAFFE 基于 [GRAF](https://arxiv.org/abs/2007.02442) [2020]，为场景中的每个对象分配一个 MLP，产生标量密度和深度特征向量（替换颜色）。这些具有共享架构和权重的 MLP 将形状和外观潜在向量以及输入姿势作为输入。然后使用特征的密度加权和来构建场景。然后使用体积渲染根据该 3D 特征场创建一个小型 2D 特征图，并将其输入到上采样 CNN 中以生成图像。 GIRAFFE 使用该合成图像和 2D CNN 判别器进行对抗训练。由此产生的模型具有解开的潜在空间，允许对场景生成进行精细控制。

##### · [Learning Object-Compositional Neural Radiance Field for Editable Scene Rendering](https://arxiv.org/abs/2109.01847) [2021]
创建了可以编辑场景内对象的合成模型。他们使用基于体素的方法创建了一个体素特征网格，并与 MLP 参数联合优化。他们使用了两种不同的 NeRF，一种用于对象，一种用于场景，两者都以插值体素特征为条件。对象 NeRF 进一步以一组对象激活潜在代码为条件。他们的方法在 [ScanNet](https://arxiv.org/abs/1702.04405) 以及带有实例分割标签的内部 ToyDesk 数据集上进行了训练和评估。他们将分割标签与识别每个场景内对象的掩模损失项结合起来。

### 2.5 位姿预测
NeRF 模型需要输入图像和相机姿势进行训练。在 2020 年的原始论文中，未知姿态是通过 COLMAP 获取的，该库也经常在后续许多 NeRF 模型中未提供相机姿态时使用。通常，使用 NeRF 构建执行姿态估计和隐式场景表示的模型被公式化为运动离线结构 (SfM) 问题。
##### · [iNeRF](https://arxiv.org/abs/2012.05877) [2020]
将姿势重建表述为逆问题，优化了位姿而不是网络参数。作者使用兴趣点检测器，并执行基于兴趣区域的采样。作者还进行了半监督实验，他们在未摆姿势的训练图像上使用 iNeRF 位姿估计来增强 NeRF 训练集，并进一步训练前向 NeRF。作者证明这种半监督可以将前向 NeRF 的摆拍照片要求减少 25%。

##### · [NeRF--](https://arxiv.org/abs/2102.07064) [2021]
NeRF-- 联合估计了 NeRF 模型参数和相机参数。这使得模型能够以端到端的方式构建辐射场并合成新颖的仅图像视图。 NeRF-– 在两个视图合成方面总体上取得了与使用 COLMAP 和 2020 NeRF 模型相当的结果。然而，由于姿势初始化的限制，NeRF-- 最适合**前置场景**，并且在旋转运动和对象跟踪运动方面遇到困难。

##### · [BARF](https://arxiv.org/abs/2104.06405) [2021]
在神经辐射场的训练的同时联合估计姿势，还通过自适应屏蔽位置编码来使用从粗到细的配准。总体而言，BARF 结果在相机姿势未知的 [LIFF](https://arxiv.org/abs/1905.00889) 前向场景数据集上超过了 NeRF（8 个场景的平均值）1.49 PNSR，并且比 COLMAP 注册基准 NeRF 好 0.45 PNSR。为了简单起见，BARF 和 NeRF 都使用了朴素的密集射线采样。



### 3. NeRF应用
### 3.1 城市重建
##### · [Urban Radiance Fields](https://openaccess.thecvf.com/content/CVPR2022/papers/Rematas_Urban_Radiance_Fields_CVPR_2022_paper.pdf) [2021]
旨在使用稀疏多视图图像并辅以 LiDAR 数据，将基于 NeRF 的视图合成和 3D 重建应用于城市环境。除了标准光度损失之外，他们还使用基于 LiDAR 的深度损失 $L_{depth}$ 和视力损失 $L_{sight}$，以及基于天空盒的分割损失 $L_{seg}$。

##### · [Mega-NeRF](https://arxiv.org/abs/2112.10703) [2021]
Mega-NeRF 通过使用更适合空中视角的椭球体扩展了 [NeRF++](https://arxiv.org/abs/2010.07492) [2020] 逆球体参数化来将前景与背景分开，还将 [NeRF-W](https://arxiv.org/abs/2008.02268) 的每图像外观嵌入代码合并到他们的模型中。他们将大型城市场景划分为多个单元，每个单元由自己的 NeRF 模块表示，并仅在具有潜在相关像素的图像上训练每个模块。对于渲染，该方法还将密度和颜色的粗略渲染缓存到八叉树中。

##### · [Block-NeRFs](https://arxiv.org/abs/2202.05263) [2022]
从 280 万张街道图像中进行了基于 NeRF 的城市规模重建。如此大规模的室外数据集带来了瞬态外观和物体等问题。每个单独的 BlockNeRF 都是通过使用其 IPE 和 [NeRF-W](https://arxiv.org/abs/2008.02268) 通过使用其外观潜在代码优化而构建在 [Mip-NeRF](https://arxiv.org/abs/2103.13415) 上。

### 3.2 人脸、头像和建筑物体重建
##### · [Nerfies](https://arxiv.org/abs/2011.12948) [2020]
这是一种使用变形场构建的 NeRF 模型，该模型在场景中存在非刚性变换（例如动态场景）时极大地提高了模型的性能。通过引入额外的 MLP，将输入观察帧坐标映射到变形的规范坐标，并通过自适应掩蔽位置编码添加弹性正则化、背景正则化和从粗到细的变形正则化，他们能够准确地重建某些**非静态场景**。

##### · [HyperNeRF](https://arxiv.org/abs/2106.13228) [2021]
建立在 Nerfies 的基础上，将规范空间扩展到更高的维度，并添加一个额外的切片 MLP，它描述了如何使用环境空间坐标返回到 3D 表示。然后使用规范坐标和环境空间坐标来调节基线 NeRF 模型的常用密度和颜色 MLP。 HyperNeRF 在合成具有拓扑变化的场景中的视图方面取得了很好的成果，例如人类张开嘴或香蕉被剥皮。

##### · [Neural Body](https://openaccess.thecvf.com/content/CVPR2021/papers/Peng_Neural_Body_Implicit_Neural_Representations_With_Structured_Latent_Codes_for_CVPR_2021_paper.pdf) [2020]
应用 NeRF 体积渲染来渲染视频中移动姿势的人体。作者首先使用输入视频来锚定基于顶点的可变形人体模型（[SMPL](https://dl.acm.org/doi/10.1145/2816795.2818013)）。在每个顶点上，作者附加了一个 16 维潜在代码 Z。然后使用人体姿势参数 S（最初在训练期间根据视频估计，可以在推理过程中输入）来使人体模型变形。标准 NeRF 方法难以处理移动的物体，而 Neural Body 的网格变形方法能够在帧之间和姿势之间进行插值。

### 3.3 图像处理
##### · [RawNeRF](https://arxiv.org/abs/2111.13679) [2021]
将 [Mip-NeRF](https://arxiv.org/abs/2103.13415) 改编为高动态范围（HDR）图像视图合成和去噪。 RawNeRF 使用原始线性图像作为训练数据在线性颜色空间中进行渲染。这允许改变曝光和色调映射曲线，本质上是在 NeRF 渲染之后应用后处理，而不是直接使用后处理图像作为训练数据。 RawNeRF 使用可变曝光图像进行监督，NeRF 模型的“曝光”根据训练图像的快门速度以及每个通道学习的缩放因子进行缩放。它在夜间和弱光场景渲染和去噪方面取得了令人印象深刻的结果。 RawNeRF 特别适合**低光照场景**。

##### · [HDR-NeRF](https://arxiv.org/abs/2111.14451) [2021]
HDR-NeRF 通过使用具有可变曝光时间的低动态范围训练图像（而不是 RawNeRF 中的原始线性图像）来实现 HDR 视图合成，并使用三个 MLP 相机响应函数（每个颜色通道一个） $f$ 将辐射亮度映射到颜色 $\mathbf{c}$ 。 HDR-NeRF 在低动态范围（LDR）重建方面远远优于基准 NeRF 和 [NeRF-W](https://arxiv.org/abs/2008.02268)，并在 HDR 重建方面取得了较高的视觉评估分数。

#### （1）语义 NeRF 模型
##### · [Semantic-NeRF](https://arxiv.org/abs/2103.15875) [2021]
是一种能够为新颖视图合成语义标签的 NeRF 模型。这是通过使用附加的方向无关 MLP（分支）来完成的，该 MLP 将位置和密度 MLP 特征作为输入并生成逐点语义标签 $s$，再使用分类交叉熵损失来监督语义标签。该方法能够使用稀疏语义标签数据（10% 已标记）进行训练，以及从像素级噪声和区域/实例级噪声中恢复语义标签。该方法还实现了良好的标签超分辨率结果和稀疏逐点注释的标签传播。

##### · [Panoptic Neural Fields](https://arxiv.org/abs/2205.04334) [2022]
该模型在 [KITTI](https://www.cvlibs.net/datasets/kitti/) 和 [KITTI 360](https://arxiv.org/abs/2109.13410) 上进行了训练和测试。除了新颖视图合成和深度预测合成之外，该模型还能够通过操作特定于对象的 MLP 进行语义分割合成、实例分割合成和场景编辑。

##### · [Decomposing NeRF for Editing via Feature Field Distillation](https://arxiv.org/abs/2205.15585) [2022]
将现成的 2D 特征提取器的知识提炼为 3D 特征字段，并结合场景内的辐射场对其进行优化，以生成具有语义理解的 NeRF 模型，该模型允许用于场景编辑。基于 CLIP 的特征提取器的精炼允许从一组开放的文本标签或查询中进行零样本分割。

#### （2）表面重建
NeRF 模型的场景几何是隐式的，然而对于某些应用，需要更明确的表示，例如 3D 网格。对于基准 NeRF，可以通过评估和阈值化密度 MLP 来提取粗略的几何形状。
##### · [UNISURF](https://arxiv.org/abs/2104.10078) [2021]
通过用离散占用函数 $o(\mathbf{x}) = 1$ 替换离散体渲染方程中使用的第 $i$ 个采样点处的 alpha 值 $\alpha_i$ 来重建场景表面占用空间，自由空间中 $o(\mathbf{x}) = 0$。该占用函数也是由 MLP 计算的，并且基本上取代了体积密度。然后通过沿着射线寻根来检索表面。 UNISURF 的性能优于基准方法，包括在基准 NeRF 模型中使用密度阈值以及 [IDR](https://arxiv.org/abs/2003.09852)。占用 MLP 可用于定义场景的显式表面几何形状。 

##### · [NeuS](https://arxiv.org/abs/2106.10689) [2021]
NeuS 像基准 NeRF 模型一样执行体积渲染。然而，它使用有符号距离函数来定义场景几何形状。它将 MLP 的密度输出部分替换为输出 SDF(Signed distance functions) 值的 MLP。该模型为基于 SDF 的场景密度的实现提供了理论和实验证明。[HF-NeuS](https://arxiv.org/abs/2206.07850) [2022] 通过将低频细节分离到基于 SDF 中，将高频细节分离到位移函数中，对 NeuS 进行了改进，极大地提高了重建质量。同时，[Geo-NeuS](https://arxiv.org/abs/2205.15848) [2022] 引入了一种新的多视图约束，其形式为稀疏点云监督的 SDF 的多视图几何约束和多视图光度一致性约束。 [SparseNeus](https://arxiv.org/abs/2206.05737) [2022] 则通过使用具有可学习图像特征的几何编码体作为混合表示方法，重点关注稀疏视图 SDF 重建，对 NeuS 进行了改进。

##### · [Neural RGB-D Surface Reconstruction](https://arxiv.org/abs/2104.04532) [2021]
将密度 MLP 替换为 truncated SDF MLP。相反，他们将像素颜色计算为采样颜色的加权和，它会截断距离各个表面太远的任何 SDF 值。从而提高表面重建的速率和精度。




















