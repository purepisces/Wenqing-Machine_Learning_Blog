The **Cumulative Distribution Function (CDF)** and the **Probability Density Function (PDF)** are both important concepts in probability and statistics, often used to describe the distribution of continuous random variables.

### 1. **CDF (Cumulative Distribution Function)**:

The **CDF**, denoted as F(x)F(x)F(x), gives the probability that a random variable XXX takes a value less than or equal to xxx. Mathematically, it's defined as:

F(x)=P(X≤x)F(x) = P(X \leq x)F(x)=P(X≤x)

For a continuous random variable, the CDF is the integral of the PDF from −∞-\infty−∞ to xxx:

F(x)=∫−∞xf(t) dtF(x) = \int_{-\infty}^{x} f(t) \, dtF(x)=∫−∞x​f(t)dt

Here, f(t)f(t)f(t) is the **PDF** of the random variable XXX.

### 2. **PDF (Probability Density Function)**:

The **PDF**, denoted as f(x)f(x)f(x), describes the relative likelihood of the random variable taking on a particular value. For continuous random variables, the PDF itself is not a probability (since P(X=x)=0P(X = x) = 0P(X=x)=0 for continuous variables), but the area under the curve of the PDF over an interval represents the probability.

The relationship between the CDF and PDF is that the PDF is the derivative of the CDF:

f(x)=ddxF(x)f(x) = \frac{d}{dx}F(x)f(x)=dxd​F(x)

In simpler terms, the **PDF** gives the rate of change of the **CDF**, showing how rapidly the cumulative probability increases with xxx.

### Key Relations:

-   The CDF is the integral of the PDF.
-   The PDF is the derivative of the CDF.

### Example:

For the normal distribution:

-   The **PDF** is the bell-shaped curve that describes the density of probability around the mean.
-   The **CDF** starts at 0 for very low values and increases to 1 as xxx approaches infinity, representing the cumulative probability that the variable takes on a value up to xxx.



the **cumulative distribution function (CDF)** applies to both **continuous** and **discrete** random variables. The CDF concept is universal across both types of variables, though the way it is computed differs slightly depending on whether the variable is discrete or continuous.

### 1. **CDF for a Discrete Random Variable:**

For a **discrete random variable**, the CDF is the sum of the probabilities of the variable taking on all values less than or equal to a certain value. If XXX is a discrete random variable, and pX(xi)p_X(x_i)pX​(xi​) represents the probability mass function (PMF) of XXX, then the CDF of XXX at a value xxx is:

FX(x)=P(X≤x)=∑xi≤xpX(xi)F_X(x) = P(X \leq x) = \sum_{x_i \leq x} p_X(x_i)FX​(x)=P(X≤x)=xi​≤x∑​pX​(xi​)

#### Example:

Consider a discrete random variable XXX representing the outcome of a dice roll, where each possible outcome xi∈{1,2,3,4,5,6}x_i \in \{1, 2, 3, 4, 5, 6\}xi​∈{1,2,3,4,5,6} has probability pX(xi)=16p_X(x_i) = \frac{1}{6}pX​(xi​)=61​. The CDF for XXX at x=3x = 3x=3 is:

FX(3)=P(X≤3)=pX(1)+pX(2)+pX(3)=16+16+16=36=0.5F_X(3) = P(X \leq 3) = p_X(1) + p_X(2) + p_X(3) = \frac{1}{6} + \frac{1}{6} + \frac{1}{6} = \frac{3}{6} = 0.5FX​(3)=P(X≤3)=pX​(1)+pX​(2)+pX​(3)=61​+61​+61​=63​=0.5

This gives the cumulative probability that the outcome is 3 or less.

### 2. **CDF for a Continuous Random Variable:**

For a **continuous random variable**, the CDF is the integral of the probability density function (PDF) from the lower bound of the distribution (typically −∞-\infty−∞) up to the given value xxx. If fX(x)f_X(x)fX​(x) represents the PDF of the continuous random variable XXX, the CDF is:

FX(x)=P(X≤x)=∫−∞xfX(t) dtF_X(x) = P(X \leq x) = \int_{-\infty}^{x} f_X(t) \, dtFX​(x)=P(X≤x)=∫−∞x​fX​(t)dt

#### Example:

Consider a continuous random variable XXX with the PDF fX(x)=14f_X(x) = \frac{1}{4}fX​(x)=41​ for x∈[0,4]x \in [0, 4]x∈[0,4] (a uniform distribution between 0 and 4). The CDF for XXX at x=3x = 3x=3 is:

FX(3)=∫0314 dt=14×(3−0)=0.75F_X(3) = \int_0^3 \frac{1}{4} \, dt = \frac{1}{4} \times (3 - 0) = 0.75FX​(3)=∫03​41​dt=41​×(3−0)=0.75

This gives the cumulative probability that the outcome is 3 or less.

### Key Differences:

-   **Discrete Case**: The CDF is a **sum** of probabilities from the PMF.
-   **Continuous Case**: The CDF is an **integral** of the PDF.

### 3. **CDF Properties (for Both Discrete and Continuous Variables):**

-   **Monotonicity**: The CDF is a non-decreasing function because probabilities accumulate.
-   **Range**: The CDF always ranges from 0 to 1. At −∞-\infty−∞, FX(x)=0F_X(x) = 0FX​(x)=0, and at +∞+\infty+∞, FX(x)=1F_X(x) = 1FX​(x)=1.
-   **Right-continuous**: The CDF is continuous from the right for both discrete and continuous cases, though for discrete variables, it may have jumps corresponding to the probabilities of specific values.

### Summary:

-   The CDF can be applied to **both discrete and continuous** random variables.
-   For **discrete variables**, it's the cumulative sum of the probabilities (PMF).
-   For **continuous variables**, it's the cumulative integral of the probability density function (PDF).
