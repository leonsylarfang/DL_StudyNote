DressCode: Autoregressively Sewing and Generating Garments from Text Guidance
=====
[He et al.](https://arxiv.org/abs/2401.16465)

> text生成服装

## 文章整体结构





## 1. 缝制样式生成
借助大语言生成模型，作者提出了 **SewingGPT**，这是一个基于 GPT（Generative Pre-trained）的用于根据文字提示来生成缝制样式（sewing pattern）的自回归模型。这一模型首先将缝制样式参数转化到一组的量化符号（tokens）序列，并训练一个结合了嵌入文本条件的交叉注意力机制（cross-attention）的掩码变换器（Transformer）解码器。在训练完成后，该模型能基于用户条件产生自回归地生成 token 序列，之后被去量化来重建缝制样式。

### 1.1 缝制样式的量化
#### 样式表征
[Korosteleva & Lee 2022](https://arxiv.org/abs/2201.13063) 提出的缝制样式模板涵盖了大量服装形状。

- 每种缝制样式包括了 $N_P$ 个 panel $\{P_i\}^{N_P}_{i=1}$ 和拼接信息 $S$。
    - 每个 panel $P_i$ 组成一个包含 $N_i$ 条边 $\{E_{i,j}\}^{N_i}_{j=1}$ 的闭合 2D 多边形。
        - 每条边 $E_{i,j}$ 由4个参数 $(v_x,v_y,c_x,c_y)$ 控制，其中 $(v_x,v_y)$ 代表边的起点，$(c_x,c_y)$ 代表 Bézier 曲线的控制点（因为这些 panel 组成的是封闭多边形，所以不需要存储每条边的终点）。
        - 每个 panel 的 3D 位置是由旋转四元数 $R_i\in \mathrm{SO}(3)$ 和位移向量 $T_i\in\mathbb{R}^3$ 表示。
    - 从拼接信息 $S$ 中得到 panel $P_{i.j}$ 中边 $E_{i,j}$ 的拼接标签（stitch tag）$\{S_{i,j}\}^{N_i}_{j=1}$ 和 拼接标记（stitch flags） $\{U_{i,j}\}^{N_i}_{j=1}$。
        - 拼接 tags $S_{i,j}\in \mathbb{R}^3$ 由对应边的 3D 位置得到。
        - 拼接 flags $U_{i,j}=\{0,1\}$ 是一个二进制标记，指示该边是否存在拼接。

文章利用和 [Korosteleva & Lee 2022](https://arxiv.org/abs/2201.13063) 相似的方法将拼接 tags $S_{i,j}$ 之间的欧氏距离作为相似度。为了从拼接 tags 和拼接 flags 中还原拼接信息，文章过滤了包含拼接 flags 的自由边和连接边（free and connected edges），然后比较所有连接边对（pairs of connected edges）的拼接 tags。

#### 量化过程
文章首先使用了 [Korosteleva & Lee 2022](https://arxiv.org/abs/2201.13063) 类似的数据预处理方法，





 