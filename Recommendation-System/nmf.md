# Non-Negative Matrix Factorization (NMF) in Recommendation Systems

**Non-Negative Matrix Factorization (NMF)** is a matrix factorization technique widely used in recommendation systems and data analysis. By decomposing a larger matrix (such as a user-item matrix in recommendation systems) into two smaller matrices with non-negative values, NMF reveals underlying patterns and hidden features, known as **latent factors**, that capture essential user and item characteristics.

## How NMF Works

Let’s consider a matrix **R** where:
- Each row represents a user,
- Each column represents an item,
- Each entry represents the user's rating for that item.

NMF approximates **R** by decomposing it into two matrices:

1. **User Matrix (U)**: Represents each user in terms of a set of `n` hidden features, or **latent factors**.
2. **Item Matrix (V)**: Represents each item in terms of the same `n` latent factors.

The product of **U** and the transpose of **V** approximates the original matrix **R**:

$$R \approx U \times V^T$$

This decomposition uncovers **latent factors**—hidden characteristics of users and items, such as movie genres, product qualities, or song characteristics. Each user and item is represented by scores for these factors, making it easier to identify similarities and predict preferences.

### Why Non-Negative?

All elements in the matrices **U** and **V** must be non-negative (greater than or equal to zero), which is beneficial for data where negative values are meaningless, such as ratings. This constraint also enhances interpretability, as all features positively contribute to each user's or item's profile.

## Example: Movie Recommendation System

Suppose we have the following user-item matrix **R** with ratings for three users and three movies:

|         | Movie 1 | Movie 2 | Movie 3 |
|---------|---------|---------|---------|
| User 1  | 5       | 3       | 0       |
| User 2  | 4       | 0       | 0       |
| User 3  | 1       | 1       | 0       |

The `0`s represent missing ratings. Using NMF, we aim to fill these gaps by learning **latent factors** that describe both users and movies.

### Step 1: Decompose the Matrix

1. Choose the number of latent factors, say `n = 2`.
2. Decompose **R** into:
   - **U (User Matrix)** with shape `(3 x 2)`.
   - **V (Item Matrix)** with shape `(3 x 2)`.

### Step 2: Matrix Approximation

After training, the matrices might look like this:

**User Matrix (U):**

| User   | Feature 1 | Feature 2 |
|--------|-----------|-----------|
| User 1 | 0.8       | 1.2       |
| User 2 | 1.1       | 0.0       |
| User 3 | 0.3       | 0.5       |

**Item Matrix (V):**

| Movie   | Feature 1 | Feature 2 |
|---------|-----------|-----------|
| Movie 1 | 2.0       | 0.3       |
| Movie 2 | 1.5       | 1.0       |
| Movie 3 | 0.5       | 0.2       |

By taking the dot product \( U \times V^T \), we approximate the original matrix **R**, filling in the missing ratings based on the patterns learned by the model.

### Step 3: Predict Missing Ratings

Using the learned latent factors, we can estimate the missing values:

|         | Movie 1 | Movie 2 | Movie 3 |
|---------|---------|---------|---------|
| User 1  | 5       | 3       | 0.86    |
| User 2  | 4       | 1.5     | 0.55    |
| User 3  | 1       | 1       | 0.25    |

These estimated values are derived from the hidden preferences of users and characteristics of items, as captured by the latent factors.

## Example Code Using `Surprise` Library

Here’s a practical example using the **Surprise** library to perform NMF:

```python
import pandas as pd
from surprise import Dataset, Reader, NMF

# Sample ratings data
ratings_df = pd.DataFrame({
    "user_id": [1, 2, 1, 3],
    "movie_id": ["A", "B", "C", "A"],
    "rating": [4.5, 3.0, 5.0, 2.5]
})

# Convert the DataFrame to a Surprise Dataset
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings_df[["user_id", "movie_id", "rating"]], reader)

# Create trainset and train the NMF model
trainset = data.build_full_trainset()
model = NMF(n_factors=2)
model.fit(trainset)

# Predict the rating for a specific user and movie
prediction = model.predict(uid=1, iid="B")
print(f"Predicted rating for user 1 on movie B: {prediction.est}")
```
In this code:

-   We used `NMF` with 2 latent factors to model preferences.
-   The `predict` method estimates a rating for user `1` on movie `B`.

## Latent Factors in Recommendation Systems

A **latent factor** is an underlying feature or characteristic inferred from the data, which helps explain patterns in user preferences or item attributes. In recommendation systems, latent factors capture essential but unobserved qualities, such as a user's interest in specific movie genres or music styles.

### How Latent Factors Work

1. **Matrix Factorization**: Matrix factorization algorithms like NMF decompose a user-item rating matrix into two smaller matrices: a **User Matrix (U)** and an **Item Matrix (V)**.
2. **Interpreting Latent Factors**: The values in **U** and **V** are not directly observable, but they represent abstract features or themes. For example, in a movie-rating matrix, one latent factor might correlate with a preference for action movies, while another might correlate with romantic elements.
3. **Predicting Preferences**: A user’s rating for an item can be predicted by the dot product of their vector of latent factors and the item’s vector, giving a score that reflects the user’s interest in that item based on these hidden features.

### Example of Latent Factors

Consider the following trained matrices with two latent factors:

**User Matrix (U)**:

| User   | Action Factor | Romance Factor |
|--------|---------------|----------------|
| User 1 | 0.8           | 0.2            |
| User 2 | 0.4           | 0.6            |
| User 3 | 1.0           | 0.1            |

**Item Matrix (V)**:

| Movie   | Action Factor | Romance Factor |
|---------|---------------|----------------|
| Movie A | 0.9           | 0.2            |
| Movie B | 0.1           | 0.8            |
| Movie C | 0.6           | 0.3            |

To predict **User 1**'s preference for **Movie A**:
1. Take the dot product of **User 1**’s vector `[0.8, 0.2]` and **Movie A**’s vector `[0.9, 0.2]`.
2. The result will be a score indicating how much **User 1** might like **Movie A** based on the learned latent factors.

## Benefits of NMF and Latent Factors

1. **Interpretability**: The non-negative constraint in NMF makes it easier to interpret the results, as positive values indicate positive associations with each factor.
2. **Dimensionality Reduction**: NMF reduces the original matrix to a smaller number of latent factors, simplifying the data and uncovering important features.
3. **Efficient and Personalized Recommendations**: By capturing hidden preferences, NMF can quickly predict ratings for unseen items, enabling personalized and accurate recommendations.

In summary, NMF leverages latent factors to provide a compact, interpretable, and powerful approach to recommendation systems, revealing hidden patterns that drive user-item interactions.

