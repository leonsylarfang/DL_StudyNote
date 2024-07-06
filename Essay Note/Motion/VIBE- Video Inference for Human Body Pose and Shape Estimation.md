VIBE: Video Inference for Human Body Pose and Shape Estimation
=====
[Kocabas et al.](https://openaccess.thecvf.com/content_CVPR_2020/html/Kocabas_VIBE_Video_Inference_for_Human_Body_Pose_and_Shape_Estimation_CVPR_2020_paper.html)

> 从单目视频中提取出人体姿态

## 结构
首先我会根据框架来介绍本文的 GAN 结构：
![](/Essay%20Note/images/VIBE_architecture.jpg)

1. 给定长度 $T$ 的单人视频输入 $V=\{I_t\}^T_{t=1}$，用一个预训练的 CNN 来提取出每一帧 $I_t$ 的特征。
2. 训练一个由双向门控循环单元（GRU）组成的 temporal encoder 来输出包含过去和未来帧信息的隐空间变量。
3. 用这些特征输入生成器 $\mathcal{G}$ 来得到每个时刻的人体模型回归 SMPL 参数 $\Theta_{fake}$。
4. 将 $\Theta_{fake}$ 以及从 AMASS 数据集中采样得到的 $\Theta_{real}$ 输入到 motion 判别器 $\mathcal{D}_M$ 来微分化真假样本。

SMPL参数 $\Theta$ 代表了人体姿态和形状，由姿态参数 $\theta\in\mathbb{R}^{72}$ 和形状参数 $\beta\in\mathbb{R}^{10}$ 组成。其中姿态参数 $\theta$ 包含全局人体旋转（global body rotation）和23个关节轴角形式的相对旋转（relative rotation）；形状参数 $\beta$ 是一个 [PCA](https://papers.nips.cc/paper_files/paper/2000/hash/ad4cc1fb9b068faecfb70914acc63395-Abstract.html) 形状空间的前$10$个参数。通过这两个参数，SMPL模型变成一个输出含姿态三维网格的可微方程 $\mathcal{M}(\theta,\beta)\in\mathbb{R}^{6890\times 3}$。

时序生成器（temporal generator）$\mathcal{G}$ 从给定视频序列计算 SMPL 参数 $\hat \Theta =[(\hat\theta_1,\hat\theta_2,\dots,\hat\theta_T),\hat\beta]$，其中 $\hat\theta_t$  是 $t$ 时刻的姿态参数，$\hat\beta$ 是对该序列的单人体形状预测，但会对每一帧都进行预测，然后应用平均池化（average pooling）得到整个输入视频的平均身体形状 $\hat\beta$。    

### 1. Temporal Encoder
#### 结构
TE 使用了循环架构，使得未来帧可以从过去视频姿态信息中收益。因为过去信息可以解决和约束姿态估计，对模糊或遮挡人体很有效。TE 即是生成器 $\mathcal{G}$，给定帧序列 $I_1,I_2,\dots,I_T$，输出每一帧对应的姿态和形状参数。
- 先将 $T$ 帧视频序列喂入一个起到特征提取作用的卷积神经网络 $f$，对每帧输出一个向量 $f_i\in\mathbb{R}^{2048}\;(i=I_1,I_2,\dots,I_T)$。
- $f_i$ 随后被送入一个 GRU 层来生成基于之前帧的隐式特征向量 $g_i\;(i=f_1,f_2,\dots,f_T)$。
- 计算 $T$ 个具有迭代反馈的回归量（regressor），先用平均姿态 $\bar\Theta$ 初始化，然后将每次迭代k中的当前参数 $\Theta_k$ 和特征 $g_i$ 作为输入。这里用 6D 连续旋转表征（continuous rotation representation）取代轴角。

#### loss
TE 的 loss 由 2D loss $L_{2D}$，3D loss $L_{3D}$，SMPL loss $L_{SMPL}$（包含姿态$\theta$ 和形状 $\beta$）和对抗判别器 loss $L_{adv}$ 组成：
$$
\begin{aligned}
L_{\mathcal G}&=L_{2D}+L_{3D}+L_{SMPL}+L_{adv}\\
L_{3D}&=\sum^T_{t=1}\left\| X_t-\hat X_t \right\|_2\\
L_{2D}&=\sum^T_{t=1}\left\| x_t-\hat x_t \right\|_2\\
L_{SMPL}&=\left\| \beta-\hat\beta \right\|_2+\sum^T_{t=1}\left\| \theta_t-\hat \theta_t \right\|_2
\end{aligned}
$$

为了计算 2D 关键点 loss，需要使用一个包含尺寸 $s$ 和位移 $t\in\mathbb{R}^2$ 的弱透视相机模型，再通过预训练的线性回归量 $W$ 计算身体点来得到 SMPL 3D 关键位置 $\hat X(\Theta)=W\mathcal{M}(\theta,\beta)$。这样 3D 关节 $\hat X$ 的 2D 投影 $\hat x\in\mathbb{R}^{j\times 2}$ 可计算得到：$\hat x=s\Pi(R\hat X(\Theta))+t$，其中 $R\in\mathbb{R}^3$ 是全局旋转矩阵， $\Pi$ 代表正交投影。


### 2. Motion Dscriminator
动作判别器 $\mathcal{D}_M$ 的作用是区分生成序列和真实序列匹配与否。其结构如下图所示：
![Alt](/Essay%20Note/images/VIBE_DM.jpg#pic_center)

如上图，将生成器的输出 $\hat \Theta$ 输入到多层 GRU 模型 $f_M$ 中，来预测 $i$ 时刻的隐式代码 $h_i=f_m(\hat \Theta_i)$ 。然后用一个自注意力机制（self attention）来将其聚合成 $[h_1,h_2,\dots,h_T]$。最后用一个线性全连接层 来预测 $[0,1]$ 的一个值，代表属于可信的人类运动流形的置信度。反传到生成器 $\mathcal{G}$ 的对抗 loss 为：
$$
L_{adv}=\mathbb{E}_{\hat\Theta\sim p_G}\left[ (\mathcal{D}_M(\hat\Theta)-1)^2 \right]
$$

判别器 loss 为：
$$
L_{\mathcal{D}_M}=\mathbb{E}_{\Theta\sim p_R}\left[ (\mathcal{D}_M(\Theta)-1)^2 \right]+\mathbb{E}_{\hat\Theta\sim p_G}\left[ (\mathcal{D}_M(\hat\Theta))^2 \right]
$$

其中 $p_G$ 是生成的动作序列，$p_R$ 是从 AMASS 中得到的真实动作序列。由于 $\mathcal{D}_M$ 是在真实姿势上训练的，它也学习了可信的身体姿势配置，因此减轻了对单独的单帧判别器的需求。

#### Motion Prior (MPoser)
MPoser 将变分人体姿态先验模型 [VPoser](https://arxiv.org/abs/1904.05866) 拓展到时序序列。首先在 AMASS 数据集上将 MPoser 训练成一个序列 VAE 来学习可信人体动作的隐式表征。然后将其作为一个正则器（regularizer）来去除不可信序列。MPoser 的编码器和解码器都由对 $i$ 帧输出隐式向量 $z_i\in\mathbb{R}^{32}$的 GRU 层组成。当应用 MPoser 时，$\mathcal{D}_M$ 被禁用，并且向 $L_\mathcal{G}$ 添加一项额外的 loss $L_{MPoser}=\left\| z \right\|_2$。

#### 自注意力机制
循环网络会在处理输入时依次更新其隐藏状态，从而使得最终的隐藏状态包含所有的序列信息。使用自注意力机制放大最终表征中重要帧的贡献，而不是使用最终隐藏状态 $h_t$ 或整个序列隐藏状态特征空间的硬选择池化结果。

这一机制能使 $\mathcal{D}_M$ 的输入序列 $\hat \Theta$ 表征为经过学习得到凸隐藏状态组合的 $r$。权重 $a_i$ 从线性 MLP 层 $\phi$ 中学习得到，并用一个 softmax 归一化成一个概率分布：
$$
\begin{aligned}
\phi_i&=\phi(h_i) \\
a_i&=\frac{e^{\phi_i}}{\sum^N_{t=1}e^{\phi_i}}\\
r&=\sum^N_{i=1}a_ih_i
\end{aligned}
$$

代表每一帧隐藏状态的特征 $h_i$会被平均和最大池化，由起得到的 $r_{avg}$ 和 $r_{max}$ 会被桥接成最终的静态向量 $r$。  

















