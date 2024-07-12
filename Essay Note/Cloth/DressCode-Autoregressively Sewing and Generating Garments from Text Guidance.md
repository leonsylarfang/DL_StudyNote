DressCode: Autoregressively Sewing and Generating Garments from Text Guidance
=====
[He et al.](https://arxiv.org/abs/2401.16465)

> text生成多件服装+材质

## 1. 缝制样式生成
借助大语言生成模型，作者提出了 **SewingGPT**，这是一个基于 GPT（Generative Pre-trained）的用于根据文字提示来生成缝制样式（sewing pattern）的自回归模型。这一模型首先将缝制样式参数转化到一组的量化符号（tokens）序列，并训练一个结合了嵌入文本条件的交叉注意力机制（cross-attention）的掩码变换器（Transformer）解码器。在训练完成后，该模型能基于用户条件产生自回归地生成 token 序列，之后被去量化来重建缝制样式。

SewingGPT 的流程框架如下图：
<div align="center">
<img src="/Essay%20Note/images/DressCode_overview.jpg" width=1200 height=320 />
</div>

文章首先将缝制样式量化为 token 序列，再应用基于 GPT 的结构来自回归地生成 token。SewingGPT 允许用户仅通过文本提示来生成高多样性的高质量缝制样式。

### 1.1 缝制样式的量化
#### 样式表征
[Korosteleva & Lee 2022](https://arxiv.org/abs/2201.13063) 提出的缝制样式模板涵盖了大量服装形状。

- 每种缝制样式包括了 $N_P$ 个片（panel） $\{P_i\}^{N_P}_{i=1}$ 和拼接信息 $S$。
    - 所有片 $P_i$ 组成一个包含 $N_i$ 条边 $\{E_{i,j}\}^{N_i}_{j=1}$ 的闭合 2D 多边形。
        - 每条边 $E_{i,j}$ 由4个参数 $(v_x,v_y,c_x,c_y)$ 控制，其中 $(v_x,v_y)$ 代表边的起点，$(c_x,c_y)$ 代表 Bézier 曲线的控制点（因为这些片组成的是封闭多边形，所以不需要存储每条边的终点）。
        - 每个片的 3D 位置是由旋转四元数 $R_i\in \mathrm{SO}(3)$ 和位移向量 $T_i\in\mathbb{R}^3$ 表示。
    - 从拼接信息 $S$ 中得到片 $P_{i.j}$ 中边 $E_{i,j}$ 的拼接标签（stitch tag）$\{S_{i,j}\}^{N_i}_{j=1}$ 和 拼接标记（stitch flags） $\{U_{i,j}\}^{N_i}_{j=1}$。
        - 拼接 tags $S_{i,j}\in \mathbb{R}^3$ 由对应边的 3D 位置得到。
        - 拼接 flags $U_{i,j}=\{0,1\}$ 是一个二进制标记，指示该边是否存在拼接。

文章利用和 [Korosteleva & Lee 2022](https://arxiv.org/abs/2201.13063) 相似的方法将拼接 tags $S_{i,j}$ 之间的欧氏距离作为相似度。为了从拼接 tags 和拼接 flags 中还原拼接信息，文章过滤了包含拼接 flags 的自由边和连接边（free and connected edges），然后比较所有连接边对（pairs of connected edges）的拼接 tags。

#### 量化过程
<div align="center">
<img src="/Essay%20Note/images/DressCode_quantization.jpg" width=700 height=700 />
</div>
如上图所示，量化过程可以分为如下几个步骤：

1. 首先使用了 [Korosteleva & Lee 2022](https://arxiv.org/abs/2201.13063) 类似的数据预处理方法，对每个片，标准化所有边向量（edge vector）和控制点来使数据维持标准正态分布，并将其 3D 位置归一化到 $[0,1]$。
2. 量化所有参数并将其转换成 tokens。具体而言，对于片 $P_i$，通过用边向量、旋转、位移以及拼接特征（stitching feature）乘以其对应的预定义常数 $C_E,C_R,C_T,C_S$ 来将所有参数离散化，并保留其拼接 flags（0或1）。这些常数需要能保持缝制样式和管理词汇量表（vlocabulary size）之间的平衡。
3. 将 $N_i$ 条边、一个旋转四元数、一个位移向量、$N_i$ 个拼接向量以及 $N_i$ 个拼接 flags 打平（flatten）并桥接成一组 token 序列。

最后整个量化过程可以为表征为：
$$
\mathcal{T}(P_i,S)=C_E\{E_{i,j}\}^{N_i}_{j=1}\oplus C_RR_i\oplus C_TT_i\oplus C_S\{S_{i,j}\}^{N_i}_{j=1}\oplus\{U_{i,j}\}^{N_i}_{j=1}
$$

其中，$\mathcal{T}$ 指代量化方程；$\oplus$ 指代 token 的线性桥接，其后会被组成线性序列。

然后所有片都会被打平并合并为单个序列，从起始 token 开始到末尾 token 结束。每个片的边数会被限定在最大值 $K$，所以为了保持各个片之间统一的 token 数量，对 token 数量 $N_i<K$ 的片会进行零填充（zero-padding），从而避免在片**之间**插入填充 token。

如上图所示，由 $L_t=(8K+7)N_p$ 个 token $f_n\;(n=1,2,\dots,L_t)$ 组成的结果序列 $\mathcal{F}^{\mathrm{seq}}$ 可以被表示为：
$$
\mathcal{F}^\mathrm{seq}=\{\mathcal{T}(P_i,S)+C\}^{N_P}_{i=1}
$$

其中，$C$ 是一个常数，保证所有的 token 非负。


### 1.2 根据自回归模型生成
利用基于 GPT 的结构，作者应用只有解码器的 Transformer 来生成缝制样式的 token 序列。受到 [PolyGen](https://arxiv.org/abs/2002.10880) 的启发，作者为每个输入 token 设计了 3 层嵌入：
1. 位置嵌入（positional embedding）。指代 token 归属的片。
2. 参数嵌入（parameter embedding）。将 token 分类成边坐标、旋转、位移或者拼接特征向量。
3. 数值嵌入（value embedding）。对量化后的缝制样式进行数值嵌入。

这些源 token 会被输入到 Transformer 的解码器中来预测下一个时刻 token 的概率分布。因此该结构的目标是最小化训练序列的负对数似然：
$$
\mathcal{L}=-\prod^{L_t}_{i=1}p(f_i|f_{<i};\theta)
$$

通过优化目标，SewingGPT 能学习到每个片的形状、位置和拼接信息之间的复杂关系。在推理阶段，目标 token 序列会从起始 token 开始，
然后从预测分布 $p(f_i|f_{<i};\theta)$ 中递归采样到末尾 token。完成自回归 token 序列生成后，将其反向量化，并将生成的数据转换为原始的缝制样式表征形式。

#### 根据文本提示的条件生成
文章引入包含文本条件嵌入 $\mathbf{h}$ 的交叉注意力机制（cross-attention）。
- 首先应用 CLIP 模型来从文本提示中获得 CLIP 嵌入。
- 然后通过一个小多层感知器（MLP）将 CLIP 嵌入投影到一个特征嵌入上来压缩维度，从而与 Transformer 的维度匹配。这种方法可以提高内存效率和推理速度。
- 最后 Transformer 解码器与特征嵌入进行交叉注意力机制，参考 [Li et al. 2022](https://arxiv.org/abs/2201.12086) 的方法，

作者使用**成对**数据训练模型来促进生成特定条件的 token。

### 1.3 实施细节
#### 数据集
作者使用了 [Korosteleva & Lee 2022](https://arxiv.org/abs/2201.13063) 提供的大量缝制样式数据集，在 11 个基本类别的约 19264 个样本。

该数据集以其全面的服装缝制样式和风格而闻名，包括衬衫、兜帽、夹克、连衣裙、裤子、裙子、连衣裙、背心等。每件衣服都包含一个缝制样式文件、一个悬垂在 T 型人体模型上的 3D 服装 mesh 和一张渲染图像。

<div align="center">
<img src="/Essay%20Note/images/DressCode_GPT4V.jpg" width=700 height=300 />
</div>

如上图，对每件衣服，实验首先利用 [GPT-4V](https://arxiv.org/abs/2303.08774) 从渲染的前后视图中生成服装的通用名称（如：兜帽、T恤、衬衫）和特定的几何特征（如：长袖、宽衣、深领）。将这两部分结合起来即得到了每件服装的描述。

在实验中，作者使用数据集预定义的片顺序进行训练，并将其拆分为 9:1 来进行训练和验证。

#### 训练
该实验在单卡 A6000 上训练了 30 小时。各超参数配置如下：
* 每个片的最大边数 $K=14$
* 每个 token 的最大长度为 $1500$
* 仅包含解码器的 Transformer 有24层
    * 位置嵌入维度 $d_{pos}=512$
    * 参数嵌入维度 $d_{para}=512$
    * 数值嵌入维度 $d_{val}=512$
    * 文本特征嵌入维度 $d_f=512$
* 常数 $C_E=50$，$C_R=1000$，$C_T=1000$，$C_S=10000$，$C=1000$
* CLIP 嵌入维度 $d_{CLIP}=1024$
* 压缩特征嵌入的维度 $d_{feature}=512$
* Adam 优化器的学习率 $\mathrm{lr}=10^{-4}$，$\mathrm{batch\_size}=4$

## 2. 自定义服装生成
使用 **SewingGPT**，可以直接从文本提示生成各种缝制样式。在 CG 管线中，外观识别是很重要的一环，文章通过 **PBR 材质生成器** 来生成每种样式对应的基于物理（PBR）的渲染材质，从而与服装设计流程保持一致。

文章 DressCode 结构包含了上述两者，进一步利用大语言模型，通过自然语言交互为用户创建定制服装。其自定义服装生成流程如下图：
- 利用大语言模型从自然语言交互指导中获取形状和材质提示。
- 利用 SewingGPT 和一个微调 Stable Diffusion 来生成高质量且利于 CG 生成的服装。
<div align="center">
<img src="/Essay%20Note/images/DressCode_customized_pipeline.jpg" width=1000 height=280 />
</div>

### 2.1 PBR 材质生成
在常用的设计软件中，设计师通常在完成样式设计之后才创建对应材质，而在服装领域，设计师通常采用“瓷砖”和基于物理的材质，如颜色、粗糙和法线贴图，来增强织物的外观真实度。因此，为了生成自定义服装，作者预训练了一个 Stable Diffusion 模型，并采用渐进式训练来生成文本引导的 PBR 材质。

#### 隐式扩散模型微调
隐式扩散模型（LDM）因其强大的泛化能力，带动了文本生成图片的发展，但原始的 LDM 是在自然图像上训练的，需要微调来生成基于“瓷砖”的图像。为了实现这一点，同时保留其泛化能力，作者收集了一个带有说明的 PBR 数据集，并对基于该数据集的预训练 LDM 进行 U-net 去噪器微调，而不改变原始的编码器 $\mathcal{E}$ 和解码器 $\mathcal{D}$。这样在推理过程中，微调 LDM 可以利用文本提示生成基于“瓷砖”的颜色贴图 $U_d$。

#### VAE微调
为了从 $U_d$ 中生成法线贴图 $U_n$ 和粗糙贴图 $U_r$，作者还额外微调了两个特定的解码器 $\mathcal{D}_n$ 和 $\mathcal{D}_r$。对于文本输入的去噪材质隐码 $z$（可通过 $D$ 解码为颜色贴图），作者分别利用 $\mathcal{D}_n$ 和 $\mathcal{D}_r$ 来将 $z$ 解码为法线贴图和粗糙贴图。

这三种材质之间的关系可以用下图简单概括：
<div align="center">
<img src="/Essay%20Note/images/DressCode_texture.jpg" width=500 height=250 />
</div>


### 2.2 通过用户友好交互的自定义生成
#### 自然语言指导
完成从文本提示生成缝制样式和材质之后，文中提出的网络结构还能在实际场景中让设计师与生成器使用自然语言交互，而不是依赖于类似数据集格式化的提示。作者应用 GPT-4 的内容学习来解释用户的自然语言输入，随之产生形状和材质提示，然后这些提示被分别喂入 SewingGPT 和 PBR 材质生成器。缝制样式在被产生之后，会被拼接到一个 T 型人体模型上。然后生成的服装会和其对应的 PBR 材质无缝地继承到工业软件中，允许与人体模型动画和各种光照下的渲染，来产生生动逼真的结果。

#### 多层服装悬垂问题
实际的生产过程中通常需要同时生成多件服装（如：裤子、T恤、夹克）。过去的基于 mesh 或者隐式场的 3D 内容生成无法有效在目标人体模型上实现多层衣服分层悬垂（layered draping）效果。而采用缝制样式的表征形式可以分别生成多种服装并自然悬垂在人体模型上。

文中使用 [Qualoth 仿真器](https://dl.acm.org/doi/pdf/10.1145/1198555.1198571?casa_token=zUysXmQzdqkAAAAA:vDk-W4hcGPrRJhLtezWLJLL_NeBwcTtkoKYUEJOG0kvz8uDUdaowMX-Ixi_xW0d16qrG0zOBDFfMec36) 作为物理仿真仿真工具，并使用了和 [Korosteleva & Lee 2021](https://arxiv.org/abs/2109.05633) 相同的材料参数和 3D 人体模型。

如下图所示，在悬垂多层衣服的过程中，采用自动顺序多衣悬垂技术，即将一组衣服从里到外悬垂到人体模型上。每件服装仿真之后，仿真服装的 mesh 会和人体模型结合，下一件衣服在之前组合模型的基础上进行下一次仿真。
<div align="center">
<img src="/Essay%20Note/images/DressCode_multi_draping.jpg" width=600 height=250 />
</div>

#### 样式补全
受益于自回归模型，可以通过模型提供的概率预测来补全完整的缝制样式。另外，输入文本提示可以指导这一补完过程。

如下图所示，对于一个给定的未完成样式，如图中的袖子，模型可以根据不同的提示补全不同的缝制样式。这使得用户可以只手动设计部分样式，然后让 SewingGPT 在文本指导下补全服装。
<div align="center">
<img src="/Essay%20Note/images/DressCode_pattern_completion.jpg" width=600 height=300 />
</div>

#### 材质编辑
近期的大多数 3D 生成任务，特别是服装生成，都无法生成结构化的 UV 图。文中方法利用缝制样式表征，可以对每个片创建独特且有结构化的 UV 映射。这有助于在特定位置进行便捷的材质编辑，允许对材质进行有效的后处理。如下图所示，在米色T恤的颜色贴图上额外绘制 SIGGRAPH 图标，并通文本提示在裤子的颜色贴图上无缝融合了一只卡通鸭子。
<div align="center">
<img src="/Essay%20Note/images/DressCode_texture_editing.jpg" width=600 height=250 />
</div>


























