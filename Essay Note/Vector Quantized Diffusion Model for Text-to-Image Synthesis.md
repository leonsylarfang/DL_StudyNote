Vector Quantized Diffusion Model for Text-to-Image Synthesis
===
[Gu et al.](https://arxiv.org/abs/2111.14822)
> 通


### 结构
整体分为两层结构：
1. VQ-VAE。把图像低分辨率的离散code
![](/Essay%20Note/images/VQDiffusion_1.png)
2. VQ-Diffusion。 使用 Mask and Replace 策略在离散空间中加噪，即两种加噪方式，一种是mask一组随机的code，一种是replace一组随机的code。这样在T时刻可以得到纯由随机的code或mask code组成的向量。之后再由一个transformer网络通过一个带噪声的code已经input文本信息，可以回复原来的code。

![](/Essay%20Note/images/VQDiffusion_2.png)

### contribution：
1. 和直接在像素空间中操作的方法相比，对图像的分辨率进行了压缩，所以可以用transformer网络进行建模
2. 可以直接用pair信息进行AdaLN的训练


### Mask-and-Replace 扩散策略 
概率转移矩阵
$$
q(x_t|x_{t-1}):=Q_tx_{t-1}
$$

$$
Q_t= \begin{bmatrix}
 \alpha_t+\beta_t & \beta_t & \beta_t & \cdots & 0 \\
 \beta_t & \alpha_t+\beta_t & \beta_t & \cdots & 0 \\
 \beta_t & \beta_t & \alpha_t+\beta_t & \cdots & 0 \\
 \vdots & \vdots & \vdots & \ddots & \vdots \\
 \gamma_t & \gamma_t & \gamma_t & \cdots & 1 \\
\end{bmatrix}
$$

$$
\alpha_t+K*\beta_t+\gamma_t=1
$$

$\alpha$ 表示code保持不变的概率，$\beta$ 表示replace的概率，$\gamma$ 表示code被mask的概率。

这个概率转移矩阵可以对code进行加噪。再引入贝叶斯公式，由先验概率分布 $p(x_T)=\left [ \bar\beta_T,\bar\beta_T,\cdots,\bar\beta_T,\bar\gamma_T \right ]^T$得到：
$$
q(x_{t-1}|x_t,x_0)=\frac{(v^T(x_t)Q_tv(x_{t-1}))(v^T(x_{t-1})\bar Q_{t-1}v(x_0))}{v^T(x_t)\bar Q_tv(x_0)}
$$

最后Diffusion的loss function 为：
$$
\begin{aligned}
L_{vlb}&:=L_0+L_1+\cdots+L_{T-1}+L_T \\
L_0&:=-\mathrm{log}p_\theta(x_0|x_1,y) \\
L_T&:=D_{KL}(q(x_T|x_0)\parallel p(x_T))\\
L_{t-1}&:=D_{KL}\left (q(x_{t-1}|x_t.x_0)\parallel p_\theta(x_{t-1}|x_t,y) \right )
\end{aligned}
$$

### Reparameterization trick
1. AutoRegressive模型的迭代次数和batch的数量是挂钩的，而VQ-Diffsuion的加噪步骤是人工设定的，和图片分辨率无关，所以总共只需要迭代 $T$ 次。
2. 训练时按照T步进行训练，但inference可以跳过部分timestep，做到 **fast inference**：
$$
p_\theta(x_{t-\Delta_t}|x_t,y):=\sum^{K}_{\tilde x_0=1}q(x_{t-\Delta_t}|x_t,\tilde x_0)p_\theta(\tilde x_0|x_t,y)
$$
这是一个精度和速度的平衡问题。
![](/Essay%20Note/images/VQDiffusion_3.png)

