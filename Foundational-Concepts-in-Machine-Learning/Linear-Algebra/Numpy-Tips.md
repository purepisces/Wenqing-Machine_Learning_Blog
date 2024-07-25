# Numpy Tips:

- **Element-wise Multiplication**: Use $A * B$ for element-wise multiplication, denoted as $A \odot B$.
  
  **Example**: Given two matrices

$$A = \left(\begin{array}{cc} 
1 & 2\\ 
3 & 4
\end{array}\right)
$$ 

$$B = \left(\begin{array}{cc} 
5 & 6\\ 
7 & 8
\end{array}\right)
$$ 

The element-wise multiplication $A \odot B$ is 

$$\left(\begin{array}{cc} 
1\times5 & 2\times6\\ 
3\times7 & 4\times8
\end{array}\right)
$$ 

$$\left(\begin{array}{cc} 
5 & 12\\ 
21 & 32
\end{array}\right)
$$ 

> Note: In element-wise multiplication, the two matrix must be the same shape.
>

- **Matrix Multiplication**: Use $A @ B$ or $np.dot(A, B)$ for matrix multiplication, denoted as $A \cdot B$.
  

  **Example**: Given the same two matrices

$$A = \left(\begin{array}{cc} 
1 & 2\\ 
3 & 4
\end{array}\right)
$$ 

$$B = \left(\begin{array}{cc} 
5 & 6\\ 
7 & 8
\end{array}\right)
$$ 

  the matrix multiplication $A \cdot B$ is 

  $$\left(\begin{array}{cc} 
1\times5 + 2\times7 & 1\times6+2\times8\\ 
3\times5+4\times7 & 3\times6+4\times8
\end{array}\right)
$$ 

=

  $$\left(\begin{array}{cc} 
19 & 22\\ 
43 & 50
\end{array}\right)
$$ 


- **Element-wise Division**: Use $A / B$ for element-wise division, denoted as $A \oslash B$.
   
  **Example**: Given two matrices

$$A = \left(\begin{array}{cc} 
1 & 2\\ 
3 & 4
\end{array}\right)
$$ 

$$B = \left(\begin{array}{cc} 
5 & 6\\ 
7 & 8
\end{array}\right)
$$ 

The element-wise division $A \oslash B$ is calculated as:

  $$\left(\begin{array}{cc} 
\frac{1}{5} & \frac{2}{6}\\ 
\frac{3}{7} & \frac{4}{8}
\end{array}\right)
$$ 

> Note: In element-wise division, the two matrix must be the same shape.
>

# Vector Representations in NumPy
In the context of NumPy and linear algebra, a 1-dimensional array does not explicitly represent either a row vector or a column vector. Instead, it is simply a sequence of numbers. The distinction between a row vector and a column vector becomes meaningful when dealing with 2-dimensional arrays (matrices).

### 1-Dimensional Array in NumPy

A 1-dimensional array in NumPy:

- **Shape**: $(n,)$
- **Example**: `[1, 2, 3]`

### Row Vector vs. Column Vector

**Row Vector**: A 2-dimensional array with a single row.

- **Shape**: $(1, n)$
- **Example**:

$$
\begin{pmatrix}
1&2&3
\end{pmatrix}
$$

**Column Vector**: A 2-dimensional array with a single column.

- **Shape**: $(n, 1)$
- **Example**:

$$
\begin{pmatrix}
1\\
2\\
3
\end{pmatrix}
$$

--------------------------------------

## Using `keepdims=True` in `np.sum`

The `keepdims=True` parameter in the `np.sum` function ensures that the dimensions of the output are the same as the input, except for the reduced dimensions which are kept as singleton dimensions (dimensions with size 1). This helps in maintaining the original number of dimensions in the array, making it easier to perform element-wise operations with broadcasting.

### Explanation with Example

Let's revisit the example with logits and go through the step-by-step calculation with `keepdims=True` and without it.

#### Example Input

```python
logits = np.array([
    [1.0, 2.0, 3.0],
    [1.0, 3.0, 2.0]
])
```

#### Step-by-Step Calculation

**Exponentiation:**

```python
exp_logits = np.exp(logits)
# exp_logits will be:
# [[ 2.71828183  7.3890561  20.08553692]
#  [ 2.71828183 20.08553692  7.3890561 ]]
```
**Sum of Exponentiated Logits with `keepdims=False` (default behavior):**
```python
sum_exp_logits = np.sum(exp_logits, axis=1)
# sum_exp_logits will be:
# [30.19287485 30.19287485]
# shape: (2,)
```
Here, the dimension along axis 1 is removed, and the result is a 1D array with shape `(2,)`.

**Sum of Exponentiated Logits with `keepdims=True`:**
```python
sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)
# sum_exp_logits will be:
# [[30.19287485]
#  [30.19287485]]
# shape: (2, 1)
```
In this case, the dimension along axis 1 is kept as a singleton dimension, so the result is a 2D array with shape `(2, 1)`. This retention of dimensions helps maintain compatibility for subsequent operations that involve broadcasting.

### Why Use `keepdims=True`?

Using `keepdims=True` is particularly useful for maintaining the correct shape for broadcasting. When performing operations like element-wise division, the shapes of the operands need to be compatible. By keeping the dimensions, we ensure that the resulting array can be broadcast correctly with the original array.

Example:  `exp_logits` (shape `(2, 3)`) divided by `sum_exp_logits` (shape `(2, 1)`) performs element-wise division with broadcasting, resulting in `probabilities` with shape `(2, 3)` due to keepdims=True.

```python
import numpy as np

# Example logits with shape (2, 3)
logits = np.array([
    [1.0, 2.0, 3.0],
    [1.0, 3.0, 2.0]
])

# Compute the exponentials of the logits
exp_logits = np.exp(logits)
print("exp_logits:\n", exp_logits)
print("Shape of exp_logits:", exp_logits.shape)

# Sum the exponentials along axis 1 with keepdims=True
sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)
print("\nsum_exp_logits:\n", sum_exp_logits)
print("Shape of sum_exp_logits:", sum_exp_logits.shape)

# Compute the probabilities by dividing exp_logits by sum_exp_logits
probabilities = exp_logits / sum_exp_logits
print("\nprobabilities:\n", probabilities)
print("Shape of probabilities:", probabilities.shape)
```
Output
```
exp_logits:
 [[ 2.71828183  7.3890561  20.08553692]
 [ 2.71828183 20.08553692  7.3890561 ]]
Shape of exp_logits: (2, 3)

sum_exp_logits:
 [[30.19287485]
 [30.19287485]]
Shape of sum_exp_logits: (2, 1)

probabilities:
 [[0.09003057 0.24472847 0.66524096]
 [0.09003057 0.66524096 0.24472847]]
Shape of probabilities: (2, 3)
```
--------------------------------------
`np.random.seed(0)` is a function call that sets the seed for NumPy's random number generator. Setting the seed ensures that the sequence of random numbers generated by NumPy's random functions (such as `np.random.randn`, `np.random.rand`, etc.) is deterministic and reproducible.

Code
```python
import numpy as np

# Without setting the seed
print("Without setting the seed:")
print(np.random.randn(5))

# Setting the seed
np.random.seed(0)
print("\nWith setting the seed to 0:")
print(np.random.randn(5))

# Resetting the seed to 0
np.random.seed(0)
print("\nWith setting the seed to 0 again:")
print(np.random.randn(5))
```
Output
```python
Without setting the seed:
[ 0.12667243 -0.86820841  2.6421595   0.28067869 -1.27068428]

With setting the seed to 0:
[1.76405235 0.40015721 0.97873798 2.2408932  1.86755799]

With setting the seed to 0 again:
[1.76405235 0.40015721 0.97873798 2.2408932  1.86755799]
```
> Uniform Distribution (np.random.rand): Suitable for generating values with equal probability, useful for tasks requiring random sampling from a specific range.
>
> Normal Distribution (np.random.randn): Suitable for tasks requiring values that follow a bell curve, especially beneficial for initializing neural network weights due to its properties that help in better training dynamics. Symmetry Breaking: If all weights are initialized to the same value (or from a uniform distribution that doesn't spread well), neurons in the same layer might learn the same features, making the network less expressive. Normally distributed weights help break symmetry and ensure that neurons learn diverse features.
-----------------------------------
### Understanding Axes in NumPy
In NumPy, the term "axis" refers to the different dimensions of an array. Understanding axes is crucial for performing operations like summing, slicing, and reshaping arrays.

> A shape of `(3, 2, 3)` represents a 3D array. The three numbers in the shape tuple indicate the dimensions of the array:
> -   The first dimension (axis 0) has a size of 3.
> -   The second dimension (axis 1) has a size of 2.
> -   The third dimension (axis 2) has a size of 3.

#### 1D Arrays

For a 1D array, there is only one axis:

-   **Axis 0**: The only axis, representing the elements of the array.
```python
a = np.array([1, 2, 3, 4])
#a.shape: (4,)
# Axis 0: [1, 2, 3, 4]
```
#### 2D Arrays

For a 2D array (matrix), there are two axes:

-   **Axis 0**: The axis that runs vertically downwards (along the rows).
-   **Axis 1**: The axis that runs horizontally across (along the columns).
```python
a = np.array([[1, 2, 3], 
              [4, 5, 6]])
# a.shape: (2, 3)
# Axis 0 (rows):
# [1, 2, 3]
# [4, 5, 6]

# Axis 1 (columns):
# [1, 4]
# [2, 5]
# [3, 6]
```
#### 3D Arrays

For a 3D array (tensor), there are three axes:

-   **Axis 0**: The axis that runs along the depth (slices).
-   **Axis 1**: The axis that runs along the height (rows).
-   **Axis 2**: The axis that runs along the width (columns).

Example:
```python
a = np.array([[[1, 2, 3], 
               [4, 5, 6]],

              [[7, 8, 9], 
               [10, 11, 12]]])
# a.shape: (2, 2, 3)
# Axis 0 (depth):
# Slices along axis 0 will give us the 2D arrays (layers) at each depth level:
# [[1, 2, 3], [4, 5, 6]]
# [[7, 8, 9], [10, 11, 12]]

# Axis 1 (Rows):
# Slices along axis 1 will give us rows from each 2D array (layer):
# [[1, 2, 3], [7, 8, 9]]
# [[4, 5, 6], [10, 11, 12]]

# Axis 2 (Columns):
# Slices along axis 2 will give us columns from each row in each 2D array (layer):
# [[1, 4], [7, 10]]
# [[2, 5], [8, 11]]
# [[3, 6], [9, 12]]
```
-----------------------------------
### `PowerScalar`: raise input to an integer (scalar) power
**Example**

If you have the following `ndarray` and scalar:

-   **Ndarray**: `np.array([2, 3, 4])`
-   **Scalar**: `3`

The element-wise power would result in:

-   **Result**: `np.array([2**3, 3**3, 4**3])` which is `np.array([8, 27, 64])`

```python
class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION
```
### `EWiseDiv`: true division of the inputs, element-wise (2 inputs)

**Example**

If you have the following `ndarrays`:

- **Ndarray `a`**: `np.array([10, 20, 30])`
- **Ndarray `b`**: `np.array([2, 4, 6])`

The element-wise division would result in:

- **Result**: `np.array([10/2, 20/4, 30/6])` which is `np.array([5, 5, 5])`

```python
class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION
```
### `DivScalar`: true division of the input by a scalar, element-wise (1 input, `scalar` - number)

**Example**

If you have the following `ndarray` and scalar:

- **Ndarray**: `np.array([10, 20, 30])`
- **Scalar**: `2`

The element-wise division would result in:

- **Result**: `np.array([10/2, 20/2, 30/2])` which is `np.array([5, 10, 15])`

```python
class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION
```
### `MatMul`: matrix multiplication of the inputs (2 inputs)

**Example**

If you have the following `ndarrays`:

- **Ndarray `a`**: `np.array([[1, 2], [3, 4]])`
- **Ndarray `b`**: `np.array([[5, 6], [7, 8]])`

The matrix multiplication would result in:

- **Result**: `np.array([[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]])` which is `np.array([[19, 22], [43, 50]])`

```python
class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION
```
> If return a*b, then it become np.array([[5, 12], [21, 32]]), incorrect result.

### `Summation`: sum of array elements over given axes (1 input, `axes` - tuple)

**Example**

If you have the following `ndarray`:

- **Ndarray `a`**: `np.array([[1, 2, 3], [4, 5, 6]])`
- **Axes**: `(0,)`

The summation over the specified axes would result in:

- **Result**: `np.array([1+4, 2+5, 3+6])` which is `np.array([5, 7, 9])`

```python
class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)
```

### `BroadcastTo`: broadcast an array to a new shape (1 input, `shape` - tuple)

**Example**

If you have the following `ndarray`:

- **Ndarray `a`**: `np.array([1, 2, 3])`
- **Shape**: `(3, 3)`

The broadcasting to the specified shape would result in:

- **Result**: `np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])`

```python
class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)
```

### `Reshape`: Gives a new shape to an array without changing its data (1 input, `shape` - tuple)

**Example**
If you have the following `ndarray`:

- **Ndarray `a`**: `np.array([[1, 2, 3], [4, 5, 6]])`
- **Shape**: `(3, 2)`

The reshaping to the specified shape would result in:

- **Result**: `np.array([[1, 2], [3, 4], [5, 6]])`

```python
class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)
```
>```python 
>import numpy as np
>a = np.array([[1, 2, 3], [4, 5, 6]])
>print(a.shape) #(2, 3)
>new_shape = (3,2)
> print(np.reshape(a,new_shape)) 
> # [[1 2],[3 4],[5 6]]
>```
>  **When the `np.reshape` function is used to change the shape of an array, it rearranges the elements of the array in a specific order. By default, `np.reshape` fills the elements of the new array in a row-major (C-style) order, which means that it reads and writes elements row by row.**

### `Negate`: Numerical negative, element-wise (1 input)

**Example**

If you have the following `ndarray`:

- **Ndarray `a`**: `np.array([1, -2, 3])`

The negation would result in:

- **Result**: `np.array([-1, 2, -3])`

```python
class Negate(TensorOp):
    def compute(self, a):
        return -a
```

### `Transpose`: reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, `axes` - tuple)

**Example1**

If you have the following `ndarray`:

- **Ndarray `a`**: `np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])`
- **Axes**: `(0, 2)`

The transposition would result in:

- **Result**: `np.array([[[1, 5], [3, 7]], [[2, 6], [4, 8]]])`

**Example2**

If you have the following `ndarray`:

-   **Ndarray `a`**: `np.array([[1, 2, 3], [4, 5, 6]])`
-   **Axes**: `None` (defaults to the last two axes)

The transposition would result in:

-   **Result**: `np.array([[1, 4], [2, 5], [3, 6]])`
  
```python
class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
	### BEGIN YOUR SOLUTION
        # reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, `axes` - tuple)
        if self.axes is None:
            # Default to swapping the last two axes
            return array_api.swapaxes(a, -1, -2)
        else:
            # Swap the specified axes
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        ### END YOUR SOLUTION
```

> If we use `array_api.arange(a.ndim)`, it is creating an array of indices representing the axes of the input array `a`.
> -   If `a` is a 2D array (`a.ndim` is `2`), `self.axis` will be `array([0, 1])`.
> -   If `a` is a 3D array (`a.ndim` is `3`), `self.axis` will be `array([0, 1, 2])`.
>```python
> def compute(self, a):
> 	### BEGIN YOUR SOLUTION
> 	# reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, `axes` - tuple)
>	self.axis = array_api.arange(a.ndim)
>	if self.axes is None:
>		self.axis[-1], self.axis[-2] = self.axis[-2], self.axis[-1]
>	else:
>		self.axis[self.axes[0]], self.axis[self.axes[1]] = self.axes[1], self.axes[0]
>	return array_api.transpose(a, self.axis)
>	### END YOUR SOLUTION
>```
> 
> Given the requirement that `Transpose` should reverse the order of two axes (axis1, axis2), defaulting to the last two axes, the implementation can be simplified to handle exactly this case. The `axes` parameter should either be `None` (default case) or a tuple of two integers specifying which axes to swap.
> 

>A shape of `(3, 2, 3)` represents a 3D array. The three numbers in the shape tuple indicate the dimensions of the array:
>
>-   The first dimension (axis 0) has a size of 3.
>-   The second dimension (axis 1) has a size of 2.
>-   The third dimension (axis 2) has a size of 3.

### Detailed Broadcasting Rules
1.  **Right Alignment of Shapes**:
    -   The shapes of the arrays are compared element-wise from the trailing (rightmost) dimension to the leading (leftmost) dimension.
2.  **Compatibility**:
    -   Two dimensions are compatible if they are equal or if one of them is 1.
3.  **Expansion**:
    -   If a dimension of one array is 1, it can be expanded to match the dimension of the other array.
  
-   **Trailing Dimensions**: The dimensions at the end of the shape tuple. These are compared first when determining broadcasting compatibility.
-   **Leading Dimensions**: The dimensions at the beginning of the shape tuple. These are compared after the trailing dimensions when determining broadcasting compatibility.
**Example1: Can Broadcasting**
```python
import numpy as np
c = np.array([1, 2, 3])
print(c.shape) #(3,)
new_shape = (2,3)
print(np.broadcast_to(c, new_shape))
```

1.  **Original Shape**: `(3,)`
    
    -   This shape has only one dimension: `3`. For broadcasting purposes, it can be treated as `(1, 3)`.
2.  **Target Shape**: `(2, 3)`

**Comparing the Trailing Dimensions**:

-   The rightmost dimensions (trailing dimensions) of both shapes are `3` and `3`, which are compatible because they are equal.

**Comparing the Leading Dimensions**:

-   The next dimensions to the left (leading dimensions) are `1` (from the original shape treated as `(1, 3)`) and `2` (from the target shape `(2, 3)`).
-   These dimensions are compatible because `1` can be broadcasted to `2`.

**Example2: Can't Broadcasting**
```python
import numpy as np
c = np.array([1, 2, 3])
print(c.shape) #(3,)
new_shape = (2,3)
print(np.broadcast_to(c, new_shape))
```
**Comparison**

-   **Trailing Dimensions**: `3` (original) vs. `2` (target) – these are not compatible because they are not equal and neither is 1.
-   **Leading Dimensions**: `1` (original) vs. `3` (target) – these are compatible because 1 can be expanded to 3.

Since the trailing dimensions do not match and are not compatible, broadcasting cannot proceed.
