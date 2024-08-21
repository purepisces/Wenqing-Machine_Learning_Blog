# Bernoulli Distribution

The Bernoulli distribution is a discrete probability distribution that models a random experiment with exactly two possible outcomes: success and failure. It is named after the Swiss mathematician Jacob Bernoulli.

## Key Characteristics

- **Outcomes**: The Bernoulli distribution has only two possible outcomes:
  - **Success**: Typically represented by 1.
  - **Failure**: Typically represented by 0.

- **Probability of Success**: The probability of success is denoted by \( p \), where \( 0 \leq p \leq 1 \).

- **Probability of Failure**: The probability of failure is \( 1 - p \).

## Probability Mass Function (PMF)

The probability mass function of a Bernoulli-distributed random variable \( X \) is given by:

\[
P(X = x) =
\begin{cases} 
p & \text{if } x = 1 \\
1 - p & \text{if } x = 0 
\end{cases}
\]

In other words:
- The probability that \( X = 1 \) (success) is \( p \).
- The probability that \( X = 0 \) (failure) is \( 1 - p \).

## Examples of Bernoulli Distribution

- **Coin Toss**: A fair coin toss is a classic example of a Bernoulli distribution. If you define success as getting heads, then \( p = 0.5 \). The outcome is either heads (1) with probability 0.5 or tails (0) with probability 0.5.

- **Binary Decision**: Any scenario with a binary outcome, such as whether an email is spam (1) or not spam (0), can be modeled by a Bernoulli distribution, with \( p \) representing the probability that the email is spam.

## Relationship to Other Distributions

- **Binomial Distribution**: The Bernoulli distribution is a special case of the binomial distribution where the number of trials \( n = 1 \). In a binomial distribution, the random variable represents the number of successes in \( n \) independent Bernoulli trials.

## Applications

- **Machine Learning**: In machine learning, especially in neural networks, the Bernoulli distribution is often used in dropout, where each neuron is independently dropped out with a probability \( p \), meaning that the neuron’s output is set to zero with that probability.

- **Statistical Modeling**: The Bernoulli distribution is used in various statistical models that involve binary outcomes, such as logistic regression.

## Summary

The Bernoulli distribution is a simple yet fundamental distribution that models situations with two possible outcomes, often referred to as "success" and "failure." It forms the basis for more complex distributions and is widely used in probability theory, statistics, and various applications in machine learning and data science.
