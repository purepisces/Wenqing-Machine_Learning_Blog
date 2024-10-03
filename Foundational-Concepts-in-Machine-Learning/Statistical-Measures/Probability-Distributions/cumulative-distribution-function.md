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
