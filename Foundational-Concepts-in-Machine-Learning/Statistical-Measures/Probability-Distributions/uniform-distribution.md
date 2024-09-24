# Uniform Distribution

The uniform distribution is a probability distribution in which all outcomes are equally likely within a given range. It can be either continuous or discrete.

## Continuous Uniform Distribution

A continuous uniform distribution refers to a situation where any number between `a` and `b` is equally likely to occur. The probability density function (PDF) is given by:

$$f(x) =
\begin{cases}
\frac{1}{b - a}, & a \leq x \leq b \\
0, & \text{otherwise}
\end{cases}$$

- **Mean**: $\mu = \frac{a + b}{2}$
- **Variance**: $\sigma^2 = \frac{(b - a)^2}{12}$

## Discrete Uniform Distribution

In a discrete uniform distribution, a finite number of equally likely outcomes occur. For example, a fair die roll, where each of the six faces has an equal chance of landing:

$$P(X = x) = \frac{1}{n}, \quad \text{for } x \in \{x_1, x_2, \dots, x_n\}$$


### Examples:

1. **Rolling a fair die**: Each number from 1 to 6 has a probability of \( \frac{1}{6} \).
2. **Random number generator**: A random number generator that picks a value between 0 and 1 from a uniform distribution would select each number with equal probability.
