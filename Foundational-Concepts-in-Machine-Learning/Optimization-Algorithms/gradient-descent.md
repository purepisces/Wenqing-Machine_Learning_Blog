# Derivative vs. Partial Derivative

## Overview


### Derivative
- **Definition**: The derivative is applicable to single-variable functions.
- **Geometric Interpretation**: The derivative at a specific point on a curve represents the slope of the tangent line at that point.
- **Application**: It measures the rate of change of the function with respect to its variable.
- **Example**: For the function $f(x) = x^2$, the derivative with respect to $x$ is $f'(x) = 2x$. 
  - At $x = 1$, the derivative is $2 \times 1 = 2$. This means that at $x = 1$, for a very small increase in $x$, $y$ increases at twice that rate. If $x$ increases by 0.01, $y$ will increase approximately by 0.02 around that point.
  - At $x = 2$, the derivative is $2 \times 2 = 4$. Here, if $x$ increases by a small amount, say 0.01, $y$ will increase approximately by 0.04 around $x = 2$. The rate of change is higher here compared to $x = 1$.
  - At $x = 3$, the derivative is $6$, indicating the slope of the tangent line to the curve at that point.


### Partial Derivative
- **Definition**: Partial derivatives are used for functions of multiple variables.
- **Process**: When calculating a partial derivative, the value of one variable is changed while keeping others constant.
- **Geometric Interpretation**: Similar to the derivative but in the context of multivariable functions.
- **Example**: Consider the function $f(x, y) = x^2 + y^2$. The partial derivative with respect to $x$ is $\frac{\partial f}{\partial x} = 2x$, and the partial derivative with respect to $y$ is $\frac{\partial f}{\partial y} = 2y$. At a point $(x, y) = (1, 2)$, the partial derivative with respect to $x$ is $2$ and with respect to $y$ is $4$. These values represent the rate of change of $f$ along the $x$ and $y$ directions, respectively, at this point.


### Partial Derivative
- **Definition**: Partial derivatives are used for functions of multiple variables.
- **Process**: When calculating a partial derivative, the value of one variable is changed while keeping others constant.
- **Geometric Interpretation**: Similar to the derivative but in the context of multivariable functions.
- **Example**: Consider the function $f(x, y) = x^2 + y^2$. The partial derivative with respect to $x$ is $\frac{\partial f}{\partial x} = 2x$, and the partial derivative with respect to $y$ is $\frac{\partial f}{\partial y} = 2y$. At a point $(x, y) = (1, 2)$, the partial derivative with respect to $x$ is 2 and with respect to $y$ is 4. These values represent the rate of change of $f$ along the $x$ and $y$ directions, respectively, at this point.
  - Rate of Change with Respect to $x$: At $(1, 2)$, this becomes $2 \times 1 = 2$. This means that if you make a very small change in $x$ while keeping $y$ constant, the change in the function $f$ will be approximately 2 times the change in $x$. For example, if $x$ increases by 0.01 (and $y$ remains constant at 2), the function $f$ will increase by approximately $2 \times 0.01 = 0.02$.
  - Rate of Change with Respect to $y$: Similarly, the partial derivative with respect to $y$ is $2 \times 2 = 4$ at $(1, 2)$. This indicates that if you make a very small change in $y$ while keeping $x$ constant, the change in $f$ will be approximately 4 times the change in $y$. For instance, if $y$ increases by 0.01 (and $x$ remains constant at 1), the function $f$ will increase by approximately $4 \times 0.01 = 0.04$.



## Gradient and Directional Derivative

### Gradient Descent
- **Gradient**: It is a vector pointing in the direction of the steepest ascent.
- **Gradient Descent**: Involves taking steps proportional to the negative of the gradient at the current point.
- **Note**: The gradient vector does not indicate the direction of steepest descent, which is opposite to the gradient.

### Directional Derivative
- **Expression**: The directional derivative, represented as $\( D_{\mathbf{d}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{d} \)$, approximates the change in function value $\( \Delta f \)$ when moving a small distance from $\( \mathbf{x} \)$ in the direction of $\( \mathbf{d} \)$.
- **Interpretation**:
  - **Positive Value**: The function $\( f \)$ is increasing in the direction of $\( \mathbf{d} \)$ from the point $\( x \)$.
  - **Negative Value**: The function $\( f \)$ is decreasing in that direction.
- **Rate of Change**: The rate of change in any given direction \( \mathbf{u} \) is given by the dot product \( \nabla f \cdot \mathbf{u} \).

### Maximum Positive Change
- **Focus**: The gradient vector indicates the direction of maximum positive rate of change.
- **Maximization**: When the unit vector $\( u \)$ aligns with the gradient vector $\( \nabla f \)$, the dot product $\( \nabla f \cdot \mathbf{u} \)$ is maximized and positive.
- **Convention**: The statement "the gradient vector points to the direction of steepest ascent" is based on this convention.

### Displacement Vector
- **Definition**: A displacement vector is the vector directing towards final position whose length is the shortest distance between the initial and the final point.


Directional Derivative Equals $\Delta f$: The expression $D_{\mathbf{d}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{d}$ represents the directional derivative of the function $f$ at point $\mathbf{x}$ in the direction of the vector $\mathbf{d}$. This quantity approximates the change in the function's value, $\Delta f$, when moving a very small distance from $\mathbf{x}$ in the direction of $\mathbf{d}$, assuming $\mathbf{d}$ is sufficiently small. 

Interpreting Positive and Negative Values:

If the result of $\nabla f(\mathbf{x}) \cdot \mathbf{d}$ is positive, it indicates that the function $f$ is increasing in the direction of 
$\mathbf{d}$ from the point $x$. If the result is negative, it indicates that the function $f$ is decreasing in the direction of $d$ from the point $x$.

The rate of change of the function in any given direction (represented by a unit vector 
$\mathbf{u}$) from that point is given by the directional derivative, which is the dot product of the gradient and $u$. Mathematically, it's expressed as $\nabla f \cdot \mathbf{u}$


The gradient vector's direction corresponds to the direction in which the function experiences the greatest increase. The focus here is on the maximum positive change, not the absolute change. The gradient does not point in the direction of the steepest descent (which would be the maximum absolute rate of change if considering both increases and decreases).

 When the unit vector $u$ is aligned with the gradient vector $\nabla f$, the dot product $\nabla f \cdot \mathbf{u}$ will definitely be a positive value, and it will be maximized.

The statement that "the gradient vector points to the direction of steepest ascent" is a convention based on the fact that the gradient gives the direction of maximum positive rate of change. The direction of steepest descent is indeed the opposite of the gradient vector, but when we refer to the direction the gradient points, we're conventionally speaking of ascent.










In the case of the function $y = x^2$, gradient descent involves updating the x value and then recalculating the y value based on this new x. The y value is not directly manipulated; instead, it is determined by the function itself once you have the new x. 
 
1. **Start with an Initial \( x \) Value**: Begin with an initial guess or starting point for \( x \). This can be any value.

2. **Calculate the Gradient at \( x \)**: The gradient (or derivative) of \( y = x^2 \) with respect to \( x \) is \( 2x \). This gradient tells you the slope of the function at your current \( x \) value.

3. **Update \( x \) Based on the Gradient**: Use the gradient to update the value of \( x \). The update rule is:
   \[ x_{\text{new}} = x_{\text{old}} - \alpha \times \text{gradient} \]
   Here, \( \alpha \) is the learning rate, a small positive number that determines the size of the step you take. If the gradient is positive, \( x \) will decrease, and if the gradient is negative, \( x \) will increase.

4. **Calculate the New \( y \) Value**: After updating \( x \), calculate the new \( y \) value using the function \( y = x^2 \). This new \( y \) is the function value corresponding to your updated \( x \).

5. **Repeat the Process**: Repeat this process of calculating the gradient, updating \( x \), and then calculating \( y \), until you reach a point where the changes in \( x \) (and consequently in \( y \)) are sufficiently small. This indicates that you have reached or are very close to the minimum.


in the context of a single-variable function like $y = x^2$ , the terms "slope," "derivative," and "gradient" essentially refer to the same concept, although they are often used in slightly different contexts or disciplines.

For the function $y = x^2$, when considering the direction of movement along the x-axis, you're right that there are essentially two directions: towards positive x values and towards negative x values. 

