欢迎Star我的Machine Learning Blog:[https://github.com/purepisces/Wenqing-Machine_Learning_Blog](https://github.com/purepisces/Wenqing-Machine_Learning_Blog)。
# Softmax
Softmax激活函数是一种向量激活函数，主要应用于神经网络的末端，将向量或原始输出转换为概率分布，其中输出元素总和为1。但是，它也可以像前面讨论过的其他激活函数一样应用在神经网络的中间。

## Softmax前向方程

给定一个$C$维输入向量$Z$，其第$m$个元素表示为$z_m$，$softmax.forward(Z)$将给出一个向量$A$，其第$m$个元素$a_m$由以下方程给出：

$$a_m = \frac{\exp(z_m)}{\sum\limits_{k=1}^{C} \exp(z_k)}$$

这里$Z$是一个单一向量。类似的计算可以用于大小为$N$的向量批次。

对于向量批次，输出矩阵$A$中每个元素的公式（大小也为$N \times C$）可以表示为：

$$A_{ij} = \frac{\exp(Z_{ij})}{\sum\limits_{k=1}^{C} \exp(Z_{ik})}$$

这里，$A_{ij}$表示批次中第$i$个输入向量对应的Softmax输出的第$j$个元素。$Z_{ij}$是批次中第$i$个向量的第$j$个元素。分母是对第$i$个输入向量中的所有元素求和，确保对于批次中的每个输入向量，Softmax输出的总和为1。

## Softmax反向方程

如前面章节中对向量激活函数的反向方法的描述所讨论的，反向传播导数的第一步是为批次中的每个向量计算Jacobian矩阵。让我们以输入向量$Z$（输入数据矩阵的一行）和相应输出向量$A$（通过softmax.forward计算得到的输出矩阵的一行）为例。Jacobian矩阵$J$是一个$C \times C$矩阵。其第$m$行第$n$列的元素由以下公式给出：

$$J_{mn} = 
\begin{cases} 
a_m(1 - a_m) & \text{如果} m = n \\
-a_m a_n & \text{如果} m \neq n 
\end{cases}$$

这里$a_m$指的是向量$A$的第$m$个元素。

现在，相对于该输入向量的损失的导数，即$dLdZ$是一个$1 \times C$向量，计算如下：

$$\frac{dL}{dZ} = \frac{dL}{dA} \cdot J$$

类似的导数计算可以应用于批次中的所有$N$个向量，并将结果向量垂直堆叠以得到最终的$N \times C$导数矩阵。

> 注意：$J_{mn}$表示Softmax函数的Jacobian矩阵的元素，对应于第$m$个输出$a_m$相对于第$n$个输入$z_n$的偏导数。Jacobian矩阵$J$由这些偏导数组成，捕捉每个输入维度$z_n$如何影响每个输出维度$a_m$。$J$的元素定义如下：
> $$J_{mn} = \frac{\partial a_m}{\partial z_n}$$
> 对于Softmax函数，这个微分有两种形式，取决于是否$m = n$或$m \neq n$：
> 
> 当$m=n$时：$J_{mn} = \frac{\partial a_m}{\partial z_m} = a_m (1 - a_m)$
> 
> 当$m \neq n$时：$J_{mn} = \frac{\partial a_m}{\partial z_n} = -a_m a_n$

```python
class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        exp_z = np.exp(Z)
        self.A = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.A

    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N, C = dLdA.shape

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros_like(dLdA)

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C))

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m, n] = self.A[i, m] * (1 - self.A[i, m])
                    else:
                        J[m, n] = -self.A[i, m] * self.A[i, n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i, :] = np.dot(dLdA[i, :], J)

        return dLdZ
```

## 在神经网络中使用Softmax激活函数的示例

考虑一个神经网络层，其中批次中有2个样本（$N=2$），每个样本有3个特征（$C=3$）。这个设置可以代表分类问题中3个类别的logits。

我们的输入向量批次$Z$给定如下：

$$Z = \begin{pmatrix} Z^{(1)} & Z^{(2)} \end{pmatrix}
= \begin{pmatrix}
1 & 2 & 3 \\
2 & 2 & 1
\end{pmatrix}$$

### 应用Softmax激活

Softmax被应用于每个样本$Z^{(i)}$，以产生输出向量$A^{(i)}$。向量$Z^{(i)}$中的元素$z_j$的Softmax函数定义如下：

$$\text{Softmax}(z_j) = \frac{e^{z_j}}{\sum\limits_{k=1}^{C} e^{z_k}}$$ 

应用Softmax到$Z$的每一行后，我们得到：

$$A = \begin{pmatrix} A^{(1)} & A^{(2)} \end{pmatrix}
= \begin{pmatrix}
0.09 & 0.24 & 0.67 \\
0.42 & 0.42 & 0.16
\end{pmatrix}$$

### Softmax的Jacobian矩阵

对于Softmax，单个样本$A^{(i)}$的Jacobian矩阵$J^{(i)}$的元素为：

$$J^{(i)}_{jk} = \begin{cases}
    A^{(i)}_j (1 - A^{(i)}_j) & \text{如果} j = k \\
    -A^{(i)}_j A^{(i)}_k & \text{否则}
\end{cases}$$

对于我们的第一个样本$A^{(1)}$，Jacobian矩阵$J^{(1)}$是一个3x3矩阵，其中每个元素都使用上述规则计算。

计算$J^{(1)}$

对于第一个样本，$A^{(1)} = [0.09, 0.24, 0.67]$。  

使用Softmax导数公式：

$$ 当 \ \ j = k: J^{(1)}_{jj} = A^{(1)}_j (1 - A^{(1)}_j)$$

$$ 当 \ \ j \neq k: J^{(1)}_{jk} = -A^{(1)}_j A^{(1)}_k$$

因此，我们有：

$$J^{(1)}_{11} = 0.09 \times (1 - 0.09) = 0.0819$$

$$J^{(1)}_{22} = 0.24 \times (1 - 0.24) = 0.1824$$

$$J^{(1)}_{33} = 0.67 \times (1 - 0.67) = 0.2211$$


以及对于 

$$j \neq k$$


$$J^{(1)}_{12} = J^{(1)}{21} = -0.09 \times 0.24 = -0.0216$$

$$J^{(1)}_{13} = J^{(1)}{31} = -0.09 \times 0.67 = -0.0603$$

$$J^{(1)}_{23} = J^{(1)}{32} = -0.24 \times 0.67 = -0.1608$$


所以，$J^{(1)}$是

$$J^{(1)} = \begin{pmatrix}
0.0819 &-0.0216 & -0.0603 \\
-0.0216 &0.1824 & -0.1608 \\
-0.0603 &-0.1608 &0.2211
\end{pmatrix}$$

类似地

$$J^{(2)} = \begin{pmatrix}
0.2436 & -0.1764 & -0.0672 \\
-0.1764 & 0.2436 & -0.0672 \\
-0.0672 & -0.0672 & 0.1344
\end{pmatrix}$$

计算梯度 $dLdZ^{(i)}$
假设我们有相对于批次的激活输出$dLdA$的损失梯度如下：

$$dLdA= \begin{pmatrix}
0.1 & -0.2 & 0.1\\
-0.1 & 0.3 & -0.2
\end{pmatrix}$$

每个样本的梯度$dLdZ^{(i)}$通过将相应的$dLdA$的行与$J^{(i)}$相乘来计算：

$$dLdZ^{(i)} =dLdA^{(i)} \cdot J^{(i)}$$

这个操作将对每个样本执行，并将得到的向量$dLdZ^{(1)}$和$dLdZ^{(2)}$垂直堆叠以形成整个批次的最终梯度矩阵$dLdZ$。

### 具体计算

由于Softmax导数的复杂性，为简洁起见，这里省略了对$J^{(1)}$和$J^{(2)}$每个元素的详细计算。然而，一般的过程涉及：

使用$A^{(1)}$和$A^{(2)}$计算$J^{(1)}$和$J^{(2)}$。

将$dLdA^{(1)} = [0.1, -0.2, 0.1]$乘以$J^{(1)}$以得到$dLdZ^{(1)}$。

将$dLdA^{(2)} = [-0.1, 0.3, -0.2]$乘以$J^{(2)}$以得到$dLdZ^{(2)}$。

垂直堆叠$dLdZ^{(1)}$和$dLdZ^{(2)}$以形成$dLdZ$


这个例子说明了使用向量激活函数计算层输入的损失梯度的过程，其中输入在产生输出时的相互依赖性要求为每个样本计算完整的Jacobian矩阵。


> 注意：在反向传播中使用$dLdZ$，因为它直接将损失与我们想要优化的参数（权重和偏置）通过$Z$联系起来，因为$Z = W \cdot A_{prev} + b$，接着是$A = f(Z)$，其中$f$是激活函数。
> 对于标量激活，$dLdZ$的计算如下：
> $$dLdZ = dLdA \odot \frac{\partial A}{\partial Z}$$
> 对于向量激活函数，$dLdZ$的计算如下：对于大小为$1 \times C$的每个输入向量$Z^{(i)}$及其对应的输出向量$A^{(i)}$（也是批次中的$1 \times C$），必须分别计算Jacobian矩阵$J^{(i)}$。该矩阵具有维度$C \times C$。因此，批次中每个样本的梯度$dLdZ^{(i)}$由以下方式确定：
> $$dLdZ^{(i)} = dLdA^{(i)} \cdot J^{(i)}$$



## 当$m = n$时，$J_{mn}$的推导，$\frac{\partial a_m}{\partial z_m} = a_m (1 - a_m)$

为了推导在Softmax函数的背景下$m=n$的情况下表达式$\frac{\partial a_m}{\partial z_m} = a_m (1 - a_m)$，我们将从Softmax函数对特定输出$a_m$的定义开始，然后应用链式法则来对其相应的输入$z_m$进行微分。对于输出$a_m$的Softmax函数的定义如下：

$$
a_m = \frac{e^{z_m}}{\sum\limits_{k=1}^{C} e^{z_k}}
$$

### 步骤1：应用商规则
由于$a_m$是一个分数，我们将使用商规则进行微分，即：

$$
\left( \frac{f}{g} \right)' = \frac{f'g - fg'}{g^2}
$$

这里，$f=e^{z_m}$，$g=\\sum\limits_{k=1}^{C} e^{z_k}$。因此，$f' = e^{z_m}$，因为对$z_m$微分的$e^{z_m}$的导数是$e^{z_m}$，$g' = e^{z_m}$，因为在对求和进行微分时，求和中只有一个非零导数的项$e^{z_m}$。

### 步骤2：代入和简化
将$f$，$g$，$f'$和$g'$代入商规则得到：

$$
\frac{\partial a_m}{\partial z_m} = \frac{e^{z_m} \sum\limits_{k=1}^{C} e^{z_k} - e^{z_m} e^{z_m}}{\left( \sum\limits_{k=1}^{C} e^{z_k} \right)^2}
$$

简化分子，我们得到：

$$
e^{z_m} \sum\limits_{k=1}^{C} e^{z_k} - e^{z_m} e^{z_m} = e^{z_m} \left( \sum\limits_{k=1}^{C} e^{z_k} - e^{z_m} \right)
$$

### 步骤3：因式分解和重新排列
我们可以因式分解$e^{z_m}$，并将括号中的项识别为Softmax函数的分母减去第$m$个项，这给出了：

$$
\frac{\partial a_m}{\partial z_m} = \frac{e^{z_m} \left( \sum\limits_{k=1}^{C} e^{z_k} - e^{z_m} \right)}{\left( \sum\limits_{k=1}^{C} e^{z_k} \right)^2}
$$

现在，观察到$\frac{e^{z_m}}{\sum\limits_{k=1}^{C} e^{z_k}} = a_m$，以及$\frac{\sum\limits_{k=1}^{C} e^{z_k} - e^{z_m}}{\sum_{k=1}^{C} e^{z_k}} = 1 - a_m$，因为从分母的总和中减去$e^{z_m}$，然后除以同样的总和给出除$a_m$外所有其他$e^{z_k}$的比例，这个比例补充到了1。

### 步骤4：最终表达式
将所有这些放在一起，我们得到：

$$
\frac{\partial a_m}{\partial z_m} = a_m (1 - a_m)
$$

这个表达式显示了$a_m$相对于$z_m$的变化速率取决于$a_m$本身以及$a_m$相对于所有$e^{z_k}$总和的比例，这反映了增加$z_m$不仅直接增加$a_m$，而且间接影响了所有类别概率分布的方式。


## 当$m \neq n$时，$J_{mn}$的推导，$\frac{\partial a_m}{\partial z_n} = -a_m a_n$

为了在Softmax函数的背景下推导表达式$\frac{\partial a_m}{\partial z_n} = -a_m a_n$，我们将从Softmax函数对特定输出$a_m$的定义开始，分析改变输入$z_n$对不同输出$a_m$的影响。对于输出$a_m$的Softmax函数的定义如下：

$$
a_m = \frac{e^{z_m}}{\sum\limits_{k=1}^{C} e^{z_k}}
$$

### 步骤1：分析$z_n$对$a_m$的影响

当对$a_m$相对于$z_n$（$m \neq n$）进行微分时，Softmax函数的分子$e^{z_m}$不依赖于$z_n$。因此，$z_n$对$a_m$的唯一影响来自分母，导致微分如下：

$$
\frac{\partial a_m}{\partial z_n} = -\frac{e^{z_m} e^{z_n}}{\left( \sum\limits_{k=1}^{C} e^{z_k} \right)^2}
$$

这个导数的出现是因为分母相对于$z_n$的导数引入了一个负号（由于链式法则），并且包括$e^{z_n}$，反映了$z_n$对总和的影响。

### 步骤2：简化表达式

识别到$a_m = \frac{e^{z_m}}{\sum\limits_{k=1}^{C} e^{z_k}}$和$a_n = \frac{e^{z_n}}{\sum\limits_{k=1}^{C} e^{z_k}}$，我们将它们代入步骤1的表达式中，得到：

$$
\frac{\partial a_m}{\partial z_n} = -\frac{e^{z_m}}{\sum\limits_{k=1}^{C} e^{z_k}} \cdot \frac{e^{z_n}}{\sum\limits_{k=1}^{C} e^{z_k}} = -a_m a_n
$$

### 结论

当$m \neq n$时，导数$\frac{\partial a_m}{\partial z_n} = -a_m a_n$体现了Softmax函数的竞争性质，其中一个输入$z_n$的增加导致不相关的输出$a_m$按比例减少。这种反向关系是由于总概率的守恒（必须总和为1），并且表示随着$z_n$的增加，导致$a_n$增加，必须对$a_m$进行补偿性减少，因此导数中出现了负号。

## 参考资料:
- CMU_11785_Introduction_To_Deep_Learning
