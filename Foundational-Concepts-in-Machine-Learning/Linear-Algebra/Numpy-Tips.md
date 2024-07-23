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
