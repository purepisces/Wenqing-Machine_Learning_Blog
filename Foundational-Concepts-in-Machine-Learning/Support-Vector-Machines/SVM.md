# Support Vector Machines (SVMs) and the Max Margin Principle

In the context of Support Vector Machines (SVMs), there are actually **three important planes**: the **decision boundary** (or separating hyperplane) and two **marginal hyperplanes** on either side of it. Let's clarify what each of these planes represents and why we impose the constraint $w \cdot x_i + b = \pm 1$ for points on the margin.

## 1. The Decision Boundary (Separating Hyperplane)

The decision boundary is the main hyperplane that separates the two classes. This hyperplane is defined by:

$$w \cdot x + b = 0$$

This equation represents the set of points \( x \) where the classifier's output is zero, meaning it lies directly between the two classes.

## 2. Marginal Hyperplanes: $w \cdot x + b = +1$ and $w \cdot x + b = -1$

The SVM aims to not only find a hyperplane that separates the classes but also maximize the margin—the distance between the closest points from each class to the decision boundary.

To define this margin, we introduce two additional planes, called marginal hyperplanes:

- The hyperplane $w \cdot x + b = +1$ represents the set of points on the edge of one class.
- The hyperplane $w \cdot x + b = -1$ represents the set of points on the edge of the other class.

These marginal hyperplanes are at equal distances from the decision boundary and define the width of the margin.

## 3. Constraints on Points (Support Vectors) on the Margin

The points that lie **exactly on** these marginal hyperplanes are called **support vectors**. For these points $x_i$, we impose the constraint:

$$w \cdot x_i + b = \pm 1$$

This means:
- If $x_i$ belongs to the positive class, it lies on the hyperplane $w \cdot x_i + b = +1$.
- If $x_i$ belongs to the negative class, it lies on the hyperplane $w \cdot x_i + b = -1$.

These constraints ensure that:
1. The support vectors (points closest to the decision boundary) lie precisely on the marginal hyperplanes, defining the boundary of the margin.
2. All other points are either farther from the margin or correctly classified within the margin bounds.

## 4. Why These Constraints Lead to the Max Margin

The **distance between the two marginal hyperplanes** $w \cdot x + b = +1$ and $w \cdot x + b = -1$ is $\frac{2}{\|w\|}$. By minimizing $\|w\|$, we maximize this margin. Hence, the SVM objective is to minimize $\frac{1}{2} \|w\|^2$ under the constraints $w \cdot x_i + b \geq 1$ for the positive class and $w \cdot x_i + b \leq -1$ for the negative class.

## Summary

- The **decision boundary** is given by $w \cdot x + b = 0$.
- The **marginal hyperplanes** are $w \cdot x + b = +1$ and $w \cdot x + b = -1$.
- The **constraints** $w \cdot x_i + b = \pm 1$ ensure that the closest points (support vectors) are on the margins, defining the maximum-margin boundaries between classes.

