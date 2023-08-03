StyleAvatar3D: Leveraging Image-Text Diffusion Models for High-Fidelity 3D Avatar Generation
=====
[Zhang et al.](https://arxiv.org/abs/2305.19012)

> 根据输入给定图片生成*同风格的*且符合*语言描述*的3D虚拟头像

## 结构
首先我会根据框架来介绍本文的结构：
![](/Essay%20Note/images/StyleAvatar3D_1.jpg)
该框架主要由三部分组成：
- 生成器（Generator）：通过多角度图片来训练3D生成器
- 判别器（Discriminator）：解决数据集中图像和姿态不一致问题的粗糙-精细化判别器（coarse-to-fine discriminator）
- 隐式扩散模型（Latent Diffusion Model）：支持有条件生成图像的扩散模型

先通过无条件的3D GAN训练得到训练结果，再通过训练有条件的隐式扩散模型来取代GAN中的映射网络来最终生成有条件的头像。

### 优化Generator
![](/Essay%20Note/images/StyleAvatar3D_2.jpg)
该generator主要是基于 [ControlNet](https://arxiv.org/abs/2302.05543) 来生成带pose guidance的多角度图片。其生成过程可总结为这个公式：
$$
I_s=\mathcal{C}_\theta(I_p,T)
$$
其中$I_s$表示风格化图片（stylized image），$\mathcal{C}_\theta$表示ControlNet，$I_p$表示姿态图片（pose image），$T=(T_{pos},T_{neg})$表示有效/无效文本提示（text prompt），指代合成风格化图像时需要/不需要的特征。

为了优化生成器，文中提出把$T_{pos}$拓展为$T_{pos}=\{T_{style},T_{view},T_{att}\}$，其中$T_{style}$表示的形态相关提示。通过把视角相关的提示$T_{view}$引入$T_{pos}$得到$T_{view}$来提升训练的准确率，并把不可见的面部特征与$T_{neg}$联系起来，最后再引入额外的属性相关提示$T_{att}$来提升结果的多样性。这样拓展后的$T_{pos}$能使generator生成的风格化图片具有多样而准确的风格化结果。

### 优化Discriminator
因为ControlNet对于头像的正面生成远比侧面或远视角的图像生成更准确，文中提出了粗糙-精细化判别器来优化图像与姿态的不对齐问题。
首先把每张图片关联上两种完全不同的姿态解释：只代表基本四个视角方向的粗糙姿态解释$c_{coarse}$和更准确的精细姿态解释$c_{fine}$。通过偏航值（yaw）和俯仰值（pitch）把所有渲染视角分为$N_{group}$组，并给每一组都指定唯一的独热偏航表征与独热俯仰表征，再把这些表征串联起来，我们可以得到$c=c_{fine}||c_{coarse}$个姿态标签。在训练的过程中，我们把“置信视角”（confident views），也即产生高对位精度结果的视角指定一个高采样精度$p_h$给精细姿态解释；产生低对位精度结果的视角指定一个低采样精度$p_l$给粗糙姿态解释。这样我们就可以使discriminator判断出靠近正面的置信视角，因为它们更容易与根据经验观察产生的图像准确对齐。

### 引入隐式扩散模型
为了预测风格化头像的姿态，从而产生更好的渲染效果，本文引入了一个运行在[StyleGAN](https://arxiv.org/abs/1912.04958)的隐式风格有空间 $\mathcal{W}$ 的条件扩散模型。通过随机采样训练好的generator得到图像和风格矢量（style vector）来训练该扩散模型，使它在的正面渲染图像的控制下，获得从噪声中逐步提取出风格矢量的能力。

该扩散模型 $\epsilon_\theta$ 通过输入一个噪声风格矢量 $w$ 和正面图像的CLIP编码嵌入 $y$ 来预测噪声 $\epsilon$ 。在训练过程中，采用[无分类器扩散指导](https://arxiv.org/abs/2207.12598)的方法使条件嵌入以$p_{drop}$ 的概率随机归零，该指导可根据调整参数 $\lambda$ 来根据我们提供的条件生成3D头像：
$$
\epsilon_\theta(y,z)=\lambda \epsilon_\theta(w,y)+(1-\lambda)\epsilon_\theta(w)
$$

训练完成后的扩散模型替换了原GAN模型中的风格映射网络，使得整个模型能以图像输入作为条件来生成风格化的3D虚拟头像。

---

该方法不需要预测输入图像的姿态，提高了生成风格化头像的准确性。
