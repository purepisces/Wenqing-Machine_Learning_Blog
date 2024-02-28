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


