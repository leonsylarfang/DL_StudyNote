Facial Blendshape Weight
====

参考文献：
- [Practice and Theory of Blendshape Facial Models](https://diglib.eg.org/handle/10.2312/egst.20141042.199-218)


## 概念
一个blendshape模型生成一个由不同面部表情线性组合而成的**面部姿态**(**facial pose**)，即blendshape的“目标”。通过改变线性组合的权重值，可以以很少的计算量得到一系列的面部表情。角色的表情范围可以通过拓展形状集而改善。

## blendshapes的定义和代数计算
blendshapes是一组线性面部模型，其中**单个基向量**（**individual basis vectors**）不是正交的而是代表了**单个面部表情**（**individual facial expressions**）。独立及向量被称为*blendshape target* 和 *morph target*，甚至直接被称为*shapes* 或 *blendshapes*。而相关的权重值由于其在用户界面往往以滑块的形式出现，经常被称为 *sliders*。

### delta blendshape 
我们可以向**中性脸**（**neutral face**）添加blendshape值来得到新的表情（resulting face） $\mathrm{\mathbf{f}}$ ，其可以被表示为
$$
\mathrm{\mathbf{f}}=\mathrm{\mathbf{b}}_0+\mathrm{\mathbf{B}}\mathrm{\mathbf{w}}
$$ 

其中，$\mathrm{\mathbf{B}}$ 矩阵的每一列代表单个面部blendshapes $\mathrm{\mathbf{b}}_k$， $\mathrm{\mathbf{b}}_0$ 表示 neutral face， $\mathrm{\mathbf{w}}$ 代表权重。

<div align=center>
<img src="/Essay Note/images/Practice and Theory of Blendshape Facial Models_1.jpg">
</div>
<center>

基本的delta blendshape 方案可以被可视化为将目标定位在**超立方体**（**hypercube**）的顶点上，这些顶点与位于原点的 neutral face 共享一条边。
</center>

### combination blendshapes
通过带权重地叠加多个blendshape，我们可以得到类似矢量和的组合结果。
$$
\mathrm{\mathbf{f}}=\mathrm{\mathbf{f}}_0+\mathrm{\mathbf{b}}_1w_1+\mathrm{\mathbf{b}}_2w_2+\mathrm{\mathbf{b}}_3w_3+\cdot \cdot \cdot+
$$

<div align=center>
<img src="/Essay Note/images/Practice and Theory of Blendshape Facial Models_2.jpg">
</div>
<center>

通过组合上方权重为 $w_1$ 和右方权重为 $w_2$ 的blendshape得到右上权重为 $w_1w_2$ 的修正形状 (correction shape)。组合目标位于blendshape超立方体的对角线上。
</center>

