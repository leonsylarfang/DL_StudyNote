Attention Is All You Need
=====
[Vaswani et al.](https://arxiv.org/abs/1706.03762)

> Transformer


# 1. 模型结构
大多数神经序列转导模型都是**编码器** - **解码器**（**Encoder** - **Decoder**）的结构。

**编码器**做了符号（symbol）表征 $(x_1,\dots,x_n)$ 的输入序列到连续表征 $\mathbf{z}=(z_1,\dots.z_n)$的映射。而**解码器**的作用则是在给定 $\mathbf{z}$ 的基础上一次产生一个元素 $z_i\;(i\in{1,2,\dots,n})$ 对应的输出序列。

在每一时刻，模型都是自回归的，即之前生成的符号都会在生成下一时刻内容时作为额外的输入。

**变换器**（**Transformer**）的框架如图1所示，解码器和编码器都使用了堆叠的自注意力（self-attention）层和点（point-wise）全连接层（fully connected layer）。
<div align="center">
<img src="/Essay%20Note/images/Transformer_architechture.jpg" width=360 height=500 />
<br> 图1：Transformer整体结构
</div>

## 1.1 编码器和解码器堆
### 编码器
编码器由 $N=6$ 个完全相同的层堆叠得到，每一层都有两个子层。如图1左边部分，第一个子层是一个多头部自注意力机制（multi-head self-attention mechanism）层，第二个子层是一个简单的位置（position-wise）全连接前馈（feed-forward）网络。在每个子层的最后都进行了残差连接（residual connection）和层正则化（layer normalization）。LN 的对应的公式为 $$\hat{x}_j=\frac{x_j-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}$$   其中，$\hat{x}_j$ 代表 LN 后的数据；$x_j$ 代表 LN 前的数据；$\mu_j$ 和 $\sigma_j^2$ 分别代表第 $j$ 个样本分布的均值和方差；$\epsilon$ 增强数据稳定性，防止分母为 $0$，$\hat{x}_j$ 过大。具体而言，每个子层的输出是 $\text{LayerNorm}(x+\text{Sublayer}(x))$，其中 $\text{Sublayer}(x)$ 是子层本身实现的函数。为了方便残差连接，包括嵌入层在内的模型中所有子层的输出维度都是 $d_{model}=512$。

### 解码器
解码器也是由 $N=6$ 个完全相同的层堆叠得到，但除了两个子层之外，解码器还插入了第三个子层，来对编码器的堆叠输出执行多头注意力，并正则化该层。此外在解码器的自注意力子层上做了遮挡（mask），来防止位置对后续位置的影响。这种遮挡方法，加上输出嵌入（embedding）了一个位置的偏移，保证了对于位置 $i$ 的预测只和位置小于 $i$ 的已知输出相关。

## 1.2 注意力机制
注意力函数可以被描述为从查询（query）和一组键值对（key-value pairs）到输出的映射，其中查询、键、指和输出都是向量。输出是作为值的加权和（weighted sum）计算的，其中分配给每个值的权重是由查询与相应键的兼容性函数计算得到的。

### 1.2.1 缩放的点积注意力
如图2所示，缩放点积注意力由维度为 $d_k$ 的查询（$Q$）与键（$K$），以及维度为 $d_v$ 的值（$V$）组成。计算查询与所有键的点积，除以 $\sqrt{d_k}$，再应用一个 softmax 函数来获得值的权重。 而在实践中，会在一组被打包成矩阵 $Q$ 的查询上同时计算注意力函数，而键和值会被打包成矩阵 $K$ 和 $V$，这样注意力函数的输出为：$$\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$  
<div align="center">
<img src="/Essay%20Note/images/Transformer_sdp_attention.jpg" width=250 height=270 />
<br> 图2：缩放点积注意力
</div>

两种最常见的注意力函数是**加性注意力**（**additive attention**）和**点积注意力**（**dot-product**）。
- 点积注意力即上述方法，除了没有使用缩放因子 $\frac{1}{\sqrt{d_k}}$。
- 加性注意力使用了一个单隐藏层的前馈网络来计算兼容性函数。

虽然两者在理论复杂性上相似，但在实践中，点积注意力更**快**，更**节省空间**因为它可以使用高度优化的矩阵乘法代码来实现。

但当 $d_k$ 值较小时，两种机制的表现相近；当 $d_k$ 值较大且没有缩放时，加性注意力表现优于点积注意力。因为 $d_k$ 较大时，点积的级数会相应增大，从而导致 softmax 函数输出的梯度很小。

### 1.2.2 多头注意力
如图3所示，多头注意力将查询、键和值分别以不同的、学习过的线性投影（projection）$h$ 次线性地投影（project）到 $d_q$、$d_k$ 和 $d_v$ 维度，然后在每一个查询、键和值的投影版本上并行实行注意力函数，生成 $d_v$ 维的输出。把各次的输出桥接起来并再次投影得到最终值。
<div align="center">
<img src="/Essay%20Note/images/Transformer_MH_attention.jpg" width=250 height=300 />
<br> 图3：多头注意力
</div>

多头注意力允许模型在不同位置共同注意来自不同表征子空间的信息，而对于单一注意力头（attention head），平均会抑制这一点。
$$
\begin{aligned}
\text{MultiHead}(Q,K,V)&=\text{Concat}(\text{head}_1,\dots,\text{head}_\mathrm{h})W^O \\
\text{where head}_\mathrm{i}&=\text{Attention}(QW^Q_i,KW^K_i,VW^V_i)
\end{aligned}
$$ 其中，投影是参数矩阵 $W^Q_i\in\mathbb{R}^{d_{model}\times d_k}$，$W^K_i\in\mathbb{R}^{d_{model}\times d_k}$，$W^V_i\in\mathbb{R}^{d_{model}\times d_v}$ 和 $W^O\in\mathbb{R}^{hd_v0\times d_{model}}$。

在本实验中，作者应用 $h=8$ 层并行注意力层或者头。对其中每个，都采用参数 $d_k=d_v=d_{model}/h=64$。由于每个头部的维数降低，总计算成本与完整维度的单头不注意力接近。

### 1.2.3 在模型中注意力机制的应用
- 在“编码器-解码器注意力”层中，$Q$ 来自于之前的解码器层，$K$ 和 $V$ 则来自于编码器的输出。这允许解码器中的每个位置都注意到输入序列的所有位置。
- 编码器包含了自注意力层。在一个自注意力层中，所有的 $K$，$V$ 和 $Q$ 都从同一个地方得到，在本模型中是来自于编码器中前一层的输出。编码器中的每一个位置都可以注意到解码器前一层的所有位置。
- 解码器中的注意力层允许解码器的每个位置关注解码器中的所有位置，直到并包含该位置。为了保持模型的自回归属性，需要规避解码器中的左向（leftward）信息流，这可以通过屏蔽（设置为 $-\infty$）softmax 输入中对应于非法连接的所有值来实现缩放点积注意力。

## 1.3 位置前馈网络
除了注意力子层之外，编码器和解码器中的每一层都包含一个全连接的前馈网络，该网络分别被相同地应用于每一个位置。这个前馈网络（FFN）由两个中间包含一个 ReLU 激活的线性变换组成：$$\text{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2$$  虽然线性变换在不同位置上是相同的，但它们在每一层使用的参数是不同的。

另一种描述该网络的方法是两个核大小为 $1$ 的卷积，输入和输出的维度都是 $d_{model}=512$，内层的维度 $d_{ff}=2048$。

## 1.4 嵌入和 softmax
与其他序列转导模型（sequence transduction model）类似，文中的模型使用学习嵌入（learned embedding）来将输入 token 和输出 token 转换成 $d_{model}$ 维的向量，并使用通常学习线性变换和 softmax 函数来将解码器输出转换成预测的下一个 token 的概率。此外，作者在两个嵌入层和 pre-softmax 线性变换之间共享相同的权重矩阵，这些权重在嵌入层中会乘以 $\sqrt{d_{model}}$。

## 1.5 位置编码
由于该模型不包含递归和卷积，为了使模型利用序列的熟悉，作者注入了一些关于序列中 token 的相对或绝对位置的信息。为此还在编码器和解码器堆底部的输入嵌入中加入了“**位置编码**（**positional encoding**）”。该编码有着和嵌入相同的维度 $d_{model}$，所以可以对两者进行求和，由不同频率的 $\sin$ 和 $\cos$ 函数组成：
$$
\begin{aligned}
PE_{(pos,2i)}&=\sin(pos/10000^{2i/d_{model}})\\
PE_{(pos,2i+1)}&=\cos(pos/10000^{2i/d_{model}})
\end{aligned}
$$ 其中，$pos$ 是位置，$i$ 是维度。位置编码的每一个维度对应一个正弦波，这些波的波长形成一个从 $2\pi$ 到 $10000\cdot2\pi$ 的一个几何级数。该函数可以让模型很容易学习注意到相对位置，因为对于任何固定的偏移量 $k$，$PE_{pos+k}$ 都可以被表征为 $PE_{pos}$的线性函数。

# 2. 自注意力机制
将自注意力层的各个方面与循环神经网络（RNN）和卷积神经网络（CNN）中的层做比较。例如典型的序列转导编码器或解码器中的隐藏层，后两者通常用于映射一个可变长度的符号表征序列 $(x_1,\dots,x_n)$ 到另一组等长的序列 $(z_1,\dots,z_n)$ ，其中 $x_i,z_i\in\mathbb{R}^d$。

使用自注意力机制的主要是因为三点：
1. 每层的总计算复杂度
2. 通过所需最小顺序操作的数量来衡量可以并行的计算量。
3. 网络中远程依赖关系之间的路径长度。学习远程依赖关系是序列转到任务的重点，而影响这个学习的一个关键因素是网络前向和后向信号穿过路径的长度。输入和输出序列中任意位置组合之间的路径越短，学习远程依赖关系就越容易。

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>
<div class="center">

| 层类型 | 每层复杂度 | 顺序操作 | 最大路径长度 |
| :----: | :----: | :----: | :----: |
| 自注意力 | $O(n^2\cdot d)$ | $O(1)$ | $O(1)$ |
| 循环 | $O(n\cdot d^2)$ | $O(n)$ | $O(n)$ |
| 卷积 | $O(k\cdot n\cdot d)$ | $O(1)$ | $O(\log_k(n))$ |
| 自注意力(带限制) | $O(r\cdot n\cdot d)$ | $O(1)$ | $O(n/r)$ |
</div>
<p align="center"><font face="微软雅黑" size=2.>表1：各层比较</font></p>

如表1所示，一个自关注层用恒定数量的顺序执行操作（sequentially executed operation）连接所有位置，而一个循环层需要 $O(n)$ 次顺序操作。在计算复杂度方面，当序列长度 $n$ 小于表征纬度 $d$ 时，自注意力层要快于循环层。为了提高涉及很长序列的任务的计算表现，自注意力层会被限制为只考虑以输出位置为中心的输入序列中大小为 $r$ 的领域，而这会使最大路径长度增加到 $O(n/r)$。

一个核宽 $k<n$ 的单卷积层不能连接所有输入和输出位置对，在邻近核的情况下，这样做需要 $O(n/k)$ 个卷积层的堆叠，而在扩展卷积的情况下，需要 $O(\log_k(n))$ 层，这样增加啊了 网络中任意两个位置之间最长路径的长度。卷积层总体而言比循环层的计算成本高 $k$ 倍。

分离卷积（separable convolution）显著降低了复杂度到 $O(k\cdot n\cdot d+n\cdot d^2)$。然而即使 $k=n$，分离卷积的复杂度也等于自注意力层和点前馈层之和，即文章模型所用的方法。
































