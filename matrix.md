
# Rank 1 Matrix Explanation

## Rank of a Matrix:
The rank of a matrix refers to the maximum number of linearly independent rows or columns in the matrix. If a matrix has rank 1, this means that all rows (or all columns) can be written as multiples of a single row or column.

## Example of Rank 1 Matrix:
Consider the following matrix \(A\):

\[
A = \begin{bmatrix} 
1 & 2 & 3 \\
2 & 4 & 6 \\
3 & 6 & 9
\end{bmatrix}
\]

This is a rank 1 matrix because each row is a scalar multiple of the first row:

- The second row is \(2 \times\) the first row: 
  \[
  [2, 4, 6] = 2 \times [1, 2, 3]
  \]
- The third row is \(3 \times\) the first row: 
  \[
  [3, 6, 9] = 3 \times [1, 2, 3]
  \]

In this case, the matrix \(A\) can be decomposed into the product of a column vector and a row vector.

## Outer Product Decomposition:
The matrix \(A\) can be written as:

\[
A = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \times \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}
\]

So we have:

\[
A = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \times \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} = 
\begin{bmatrix}
1 \times 1 & 1 \times 2 & 1 \times 3 \\
2 \times 1 & 2 \times 2 & 2 \times 3 \\
3 \times 1 & 3 \times 2 & 3 \times 3
\end{bmatrix} = 
\begin{bmatrix} 
1 & 2 & 3 \\
2 & 4 & 6 \\
3 & 6 & 9
\end{bmatrix}
\]

This shows that \(A\) is an outer product of two vectors: a column vector \(v = [1, 2, 3]^T\) and a row vector \(w = [1, 2, 3]\).

## Why is this Rank 1?
The rank of a matrix refers to the number of independent rows or columns.
Since each row in \(A\) is a scalar multiple of the first row, the matrix only has one linearly independent row (or column). Hence, the matrix has rank 1.
In general, any rank 1 matrix can be expressed as the outer product of two vectors. This means all the rows (or columns) are scalar multiples of one another.

## Another Example:
Here is another example of a rank 1 matrix:

\[
B = \begin{bmatrix}
2 & 4 \\
6 & 12
\end{bmatrix}
\]

In this matrix, the second row is \(3 \times\) the first row:
\[
[6, 12] = 3 \times [2, 4]
\]
Therefore, the matrix \(B\) has rank 1.

It can also be written as an outer product:

\[
B = \begin{bmatrix} 2 \\ 6 \end{bmatrix} \times \begin{bmatrix} 1 & 2 \end{bmatrix}
\]

This gives:

\[
B = \begin{bmatrix} 2 \times 1 & 2 \times 2 \\ 6 \times 1 & 6 \times 2 \end{bmatrix} = \begin{bmatrix} 2 & 4 \\ 6 & 12 \end{bmatrix}
\]

Again, all rows (or columns) are multiples of one another, so \(B\) is a rank 1 matrix.

## Summary:
A rank 1 matrix is a matrix in which all rows (or all columns) are scalar multiples of each other. This allows the matrix to be represented as the outer product of a column vector and a row vector. The key point is that only one row or column is independent, which is why the matrix has rank 1.
___
