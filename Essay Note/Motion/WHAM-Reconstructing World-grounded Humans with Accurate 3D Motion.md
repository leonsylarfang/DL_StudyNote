WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion
=====
[Shin et al.](https://openaccess.thecvf.com/content/CVPR2024/html/Shin_WHAM_Reconstructing_World-grounded_Humans_with_Accurate_3D_Motion_CVPR_2024_paper.html)

> 从移动相机的单目视频中提取出人体姿态

## Overview
如下图，输入为动作的未知原始视频数据 $\{ I^{(t)}\}^T_{t=0}$；最终目标输出为对应序列的 SMPL 模型参数 $\{\Theta^{(t)}\}^T_{t=0}$，根关节在世界坐标系中的方向 $\{\Gamma^{(t)}\}^T_{t=0}$ 与位移 $\{\tau^{(t)}\}^T_{t=0}$。其结构大致为：
![](/Essay%20Note/images/WHAM_overview.jpg)

* 首先用 [ViTPose](https://arxiv.org/abs/2204.12484) 来检测 2D 关键点 $\{x_{2D}^{(t)}\}^T_{t=0}$，并用动作编码器（motion encoder）从中得到动作特征 $\{\phi_m^{(t)}\}^T_{t=0}$。
* 另外用一个预训练的图片编码器（image encoder）来提取出静态图片特征 $\{\phi_i^{(t)}\}^T_{t=0}$，并将其与动作特征组合为带细粒度的（fine-grained）动作特征 $\{\hat\phi_m^{(t)}\}^T_{t=0}$，从而在世界坐标系中回归 3D 人体动作。

### 网络架构
#### Uni-directional Motion Encoder and Decoder
没有使用固定时间区间的窗口，本文的动作编码和解码器使用单向（uni-directional）循环网络（RNN），使得 WHAM 拥有在线推理的能力。

动作编码器 $E_M$ 的作用是从当前与过去的 2D 关键点序列和初始隐藏状态 $h_E^{(0)}$ 中提取动作特征 $\phi_m^{(t)}$：
$$
\phi_m^{(t)}=E_M\left( x_{2D}^{(0)},x_{2D}^{(1)},\dots,x_{2D}^{(t)}|h_E^{(0)} \right)
$$

将关键点归一化到一个在人体周边的边界框（bounding box）中，然后把框中心桥接起来，并缩放到关键点。

动作解码器 $D_M$ 的作用是从动作特征历史（motion fearture history）中还原 SMPL 参数 $(\theta,\beta)$，弱透视相机位移 $c$ 以及脚地接触概率 $p$：
$$
\left( \theta^{(t)},\beta^{(t)},c^{(t)},p^{(t)} \right)=D_M\left( \hat\phi_m^{(0)},\hat\phi_m^{(1)},\dots,\hat\phi_m^{(t)}|h_D^{(0)} \right)
$$

其中，$\hat\phi_m^{(t)}$ 包含了图像特征 $\phi_i^{(t)}$。在对合成数据进行预训练时，图像特征不可用，所以设置 $\hat\phi_m^{(t)}=\phi_i^{(t)}$。

由于编码器和解码器是用来从稀疏的 2D 输入信号 $x_{2D}$ 中重建密集的 3D 表征 $\Theta$，所以文中设计了一个中间任务来预测 3D 关键点 $x_{3D}$ 作为中间动作表征。这种级联方法引导 $\phi_m$ 来表征动作的隐藏内容和人体的三维空间结构。

另外，本文使用了神经初始化（Neural Initilization），用 MLP 来初始化动作编/解码器 $(h_E^{(0)},h_D^{(0)})$。

#### Motion and Visual Feature Integrator
本文使用 AMASS 数据集，将 3D SMPL关节投影到具有不同相机运动得到图像上来合成 2D 序列，从而能提供远超现有包含 GT 3D 姿态和形状的视频数据类型的训练数据。

为了更好的将 2D 关键点转换为 3D 网格，需要用能帮助减轻 3D 姿态歧义的图像线索来增强 2D 关键点信息。本文使用了在人体网格重建任务中预训练的图像编码器来提取图像特征 $\phi_i^{(t)}$，其中包含了与 3D 人体姿态和形状相关的密集视觉全局信息（contexual information）。

然后训练一个特征积分器网络 $F_I$ 来整合动作特征 $\phi_m$ 和视觉特征 $\phi_i$，其使用了一个简单高效的残差连接：
$$
\hat\phi_m^{(t)}=\phi_m^{(t)}+F_I\left( \text{concat}( \phi_m^{(t)},\phi_i^{(t)}) \right)
$$

#### Global Trajectory Decoder
本文额外设计了一个解码器 $D_T$ 从动作特征 $\phi_m^{(t)}$ 来预测粗糙全局根关节方向 $\Gamma_0^{(t)}$ 和速度 $v_0^{(t)}$。因为 $\phi_m$ 是从相机坐标系下的输入信号得到的，所以很难从中将人体和相机动作解耦。

为了解决这一问题，文中在动作特征 $\phi_m^{(t)}$ 中增添了相机的角速度 $\omega^{(t)}$ 来创建一个相机无关的运动内容。这使得 WHAM 兼容现成的 [SLAM](https://arxiv.org/abs/2208.04726) 算法和现代数码相机中广泛使用的陀螺仪测量。使用一个单向 RNN 来递归预测全局根关节方向 $\Gamma_0^{(t)}$：
$$
(\Gamma_0^{(t)},v_0^{(t)})=D_T(\phi_m^{(0)},\omega^{(0)},\phi_m^{(1)},\omega^{(1)},\dots,\phi_m^{(t)},\omega^{(t)})
$$

#### Contact Aware Trajectory Refinement
为了使 WHAM 预测出的 3D 运动能做到在非平坦地面的脚地准确接触，章提出了一个新的轨迹优化器（trajectory refiner），其分为两步：
* 首先根据从运动解码器 $D_M$ 估计的脚地接触概率 $p^{(t)}$，将根关节速度调整为 $\tilde v^{(t)}$ 以最小化脚步滑动：
$$
\tilde v^{(t)}=v_0^{(t)}-(\Gamma_0^{(t)})^{-1}\bar v_f^{(t)}
$$
其中，$\bar v_f^{(t)}$ 是当脚趾和脚跟与地面接触概率高于阈值时，两者在世界坐标系中的平均速度。然而，当接触和姿态估计不准确时，这样的速度调整会引入噪声平移。
* 所以文章提出了一种简单的学习机制，包含一个轨迹精炼网络（trajectory refining network）$R_T$ 来更新根关节方向和速度。然后再通过一个转出（roll-out）操作来计算全局位移：
$$
\begin{aligned}
(\Gamma^{(t)},v^{(t)})&=R_T(\phi_m^{(0)},\Gamma_0^{(0)},\tilde v^{(0)},\phi_m^{(1)},\Gamma_0^{(1)},\tilde v^{(1)},\dots,\phi_m^{(t)},\Gamma_0^{(t)},\tilde v^{(t)}) \\
\tau^{(t)}&=\sum^{t-1}_{i=0}\Gamma^{(i)}v^{(i)}
\end{aligned}
$$



### 训练
![](/Essay%20Note/images/WHAM_training.jpg)
如上图，这一训练过程包含两步：
* 预训练合成数据
* 根据真实数据微调

#### Pretraining on AMASS
在预训练过程中，我们创建了虚拟相机，并将从 GT 网格导出的 3D 关键点投影到这些虚拟相机上。这样就能从 AMASS 数据集中生成合成的 2D 关键点序列来训练运动编码器和解码器。从而让动作编码器学习从 2D 关键点序列中提取动作特征，并使动作和轨迹解码器将此动作特征映射到相应的 3D 运动和全局轨迹空间，即将编码提升到 3D。

文中引入了旋转和位移运动来对关键点投影应用动态相机，这一做法有两个好处。一个是其解释了静态和动态相机设置中补货的人体动作的固有差异；另一个是它允许学习和相机无关的运动表征，从而使轨迹解码器重建全局轨迹。此外还用噪声和掩蔽增强了 2D 数据。

#### Fine-tuning on Video Datasets
根据如下四个数据集来微调 WHAM ：[3DPW](https://openaccess.thecvf.com/content_ECCV_2018/html/Timo_von_Marcard_Recovering_Accurate_3D_ECCV_2018_paper.html), [Human3.6M](http://vision.imar.ro/human3.6m/description.php), [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) 和 [InstaVariety](https://arxiv.org/abs/1812.01601)。但这些微调数据量比预训练的数据集要小得多。

其中在人体网格复原任务中，使用了 AMASS 和 3DPW 数据集的 SMPL 参数， Human3.6M 和 MPI-INF-3DHP 的 3D 关键点 以及 InstaVariety 的 2D 关键点。在全局轨迹预测任务中，使用 AMASS， Human3.6M 和 MPI-INF-3DHP 数据集。

微调步骤的两个目标为：
1. 将网络在真实的2D关键点上，而不是仅仅在合成数据上进行训练。
2. 训练特征积分器网络来聚合动作和图像特征。

为了达到这两个目标，作者在视频数据集上联合训练整个网络，同时在预训练模块上设置较小的学习率。使用固定权重的预训练图像编码器和关键点检测器(❆)来提取图像特征和 2D 关键点。








