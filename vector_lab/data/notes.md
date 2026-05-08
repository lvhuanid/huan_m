# Transformer 架构学习笔记

## 1. 自注意力机制 (Self-Attention)

自注意力允许模型在处理每个词时，关注输入序列中的所有位置，从而动态计算上下文相关的表示。

给定输入序列 \(X = [x_1, x_2, ..., x_n]\)，每个 \(x_i\) 被投影到三个向量：
- **Query (Q)**: 当前词想要查询的信息。
- **Key (K)**: 当前词所拥有的标签信息。
- **Value (V)**: 当前词的实际内容。

注意力权重的计算公式为：
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]
其中 \(d_k\) 是 Key 向量的维度，除以 \(\sqrt{d_k}\) 是为了防止点积过大导致 softmax 梯度消失。

### 1.1 多头注意力 (Multi-Head Attention)

多头注意力将 Q、K、V 线性投影到多个低维子空间，并行计算注意力，然后将结果拼接起来。这允许模型从不同表示子空间联合关注信息。

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
\]
每个 head 的计算：
\[
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\]

## 2. 位置编码 (Positional Encoding)

由于 Transformer 没有循环或卷积结构，本身无法捕捉序列顺序。位置编码通过向输入嵌入中添加固定或可学习的向量来注入位置信息。

原始论文使用了正弦和余弦固定编码：
\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]
\[
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]
这种设计使得模型可以轻松关注相对位置，因为对于任意固定偏移 k，\(PE_{pos+k}\) 可以表示为 \(PE_{pos}\) 的线性函数。

## 3. 前馈网络 (Feed-Forward Network)

每个 Transformer 层在注意力子层之后，包含一个全连接前馈网络，由两个线性变换和一个非线性激活函数（通常是 ReLU 或 GELU）组成：

\[
FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2
\]
前馈网络对每个位置独立且相同地操作，可以看作是对各位置特征的非线性变换。

## 4. 残差连接与层归一化

每个子层（自注意力和前馈网络）均使用残差连接，然后进行层归一化（LayerNorm）。原始论文是先加残差后归一化（Post-LN），后续研究发现先归一化后加残差（Pre-LN）训练更稳定。

## 5. 训练技巧与优化器

Transformer 通常使用 Adam 优化器，并采用自定义学习率调度：
\[
lr = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})
\]
标签平滑（Label Smoothing）和 dropout 被广泛用于正则化，防止过拟合。

## 6. 视觉 Transformer (ViT)

Transformer 已被成功应用于计算机视觉。ViT 将图像分割为固定大小的图块（Patches），将图块展平并线性投影到嵌入空间，然后作为序列输入标准 Transformer 编码器。在足够数据预训练后，ViT 可以媲美甚至超越卷积神经网络。

## 7. 大语言模型中的 Transformer 变体

- **仅编码器架构**: BERT 使用双向自注意力，适用于理解任务。
- **仅解码器架构**: GPT 系列使用因果（单向）自注意力，适合文本生成。
- **编码器-解码器架构**: T5 保持原始结构，用于序列到序列任务如翻译。

## 8. 推理与效率优化

自注意力的复杂度为 \(O(n^2)\)，针对长序列有多种优化方法：
- **稀疏注意力**: 限制每个位置只关注局部窗口或部分全局标记。
- **FlashAttention**: 通过分块计算和减少 HBM 读写，显著提高注意力计算速度与显存效率。
- **KV 缓存**: 推理时缓存历史的 Key 和 Value，避免重复计算，加速自回归生成。