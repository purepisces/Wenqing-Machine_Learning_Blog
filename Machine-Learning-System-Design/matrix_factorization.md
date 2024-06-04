# Matrix Factorization

Let's clarify why we use matrix factorization to generate the predicted ratings matrix (R') instead of directly using the original interaction matrix (R) for predictions.

## Key Reasons for Using Matrix Factorization

### Sparsity of the Original Matrix:
- The original interaction matrix (R) is usually very sparse. This means that most users have only watched or rated a small fraction of the total available videos. For example, in a large video streaming platform, a user might have watched only a few dozen videos out of millions available.
- Matrix factorization helps to fill in these missing values by identifying patterns and similarities among users and items. This allows us to predict interactions for videos that a user hasn't watched yet.

### Latent Factors:
- Matrix factorization decomposes the interaction matrix into latent factors that capture underlying patterns in user preferences and video characteristics. These latent factors can represent abstract concepts like a user's preference for certain genres or themes, which are not directly visible in the original data.
- By learning these latent factors, the model can generalize and make more accurate predictions for unseen interactions.

### Dimensionality Reduction:
- The original interaction matrix can be extremely large, with dimensions equal to the number of users multiplied by the number of videos. Matrix factorization reduces this to a lower-dimensional representation, making it computationally efficient to work with.

### Example with Sparse Matrix
Let's revisit the simplified example:

**Original Interaction Matrix (R):**

|       | Video 1 | Video 2 | Video 3 |
|-------|---------|---------|---------|
| User 1 |    5    |    3    |    0    |
| User 2 |    4    |    0    |    0    |
| User 3 |    1    |    1    |    0    |

Notice that many entries are zero, meaning those interactions are unknown.

**Predicted Ratings Matrix (R') (After Matrix Factorization):**

|       | Video 1 | Video 2 | Video 3 |
|-------|---------|---------|---------|
| User 1 |   4.8   |   3.1   |   2.0   |
| User 2 |   4.1   |   2.5   |   1.7   |
| User 3 |   1.2   |   1.1   |   0.9   |

- **User 1 and Video 3**: In the original matrix (R), we don't have a rating for Video 3 by User 1. However, in the predicted matrix (R'), we have a predicted rating of 2.0, which indicates the estimated preference based on latent factors.
- **Filling in the Gaps**: The predicted matrix provides ratings for all user-video pairs, including those not explicitly rated in the original matrix.

### Why Not Use R Directly?
- **Incomplete Data**: The original matrix (R) lacks information for many user-video pairs. Using it directly would mean we can't make recommendations for videos a user hasn't interacted with.
- **Discovery of Preferences**: Matrix factorization uncovers hidden patterns and preferences that aren't apparent from the sparse data alone.
- **Generalization**: It allows the model to generalize from known interactions to predict unknown ones, improving recommendation quality and user experience.

In summary, matrix factorization leverages the incomplete interaction data to generate a comprehensive set of predicted ratings, enabling the recommendation system to suggest relevant videos even for those a user hasn't previously watched.

## How Matrix Factorization with SGD Works

### Initialization:
- Initially, the user and item matrices are filled with random values. These matrices represent latent factors for users and items. The dimensions of these matrices depend on the number of latent factors we choose to use.

### Optimization:
- The algorithm iteratively adjusts these values to minimize the difference between the observed interactions (original matrix R) and the predicted interactions (predicted matrix R').

### Learning Latent Factors:
- **User Matrix (U)**: Represents users in terms of latent factors.
- **Item Matrix (V)**: Represents items (videos) in terms of latent factors.

The decomposition can be mathematically represented as:
$$R \approx U \times V^T$$
where $U$ is the user matrix, $V$ is the item matrix, and $R$ is the interaction matrix.

### Optimization Process
The goal is to minimize the reconstruction error, which is the difference between the observed ratings (in R) and the predicted ratings (in R'). This is typically done using a loss function, such as Mean Squared Error (MSE):

$$\text{Loss} = \sum_{(i, j) \in \text{observed}} (R_{ij} - (U_i \cdot V_j)^T)^2 + \lambda (||U||^2 + ||V||^2)$$

Here:
- $R_{ij}$ is the actual rating of user $i$ for video $j$.
- $U_i$ is the latent factor vector for user $i$.
- $V_j$ is the latent factor vector for video $j$.
- $\lambda$ is a regularization term to prevent overfitting.

### Example
Consider a small example with 3 users and 3 videos, with 2 latent factors (k=2):

#### Step 1: Initialize Matrices
Randomly initialize the user (U) and item (V) matrices:

**User Matrix (U):**
|       | Factor 1 | Factor 2 |
|-------|----------|----------|
| User 1 |    0.1   |    0.3   |
| User 2 |    0.4   |    0.2   |
| User 3 |    0.2   |    0.5   |

**Item Matrix (V):**
|         | Factor 1 | Factor 2 |
|---------|----------|----------|
| Video 1 |    0.3   |    0.6   |
| Video 2 |    0.1   |    0.4   |
| Video 3 |    0.5   |    0.2   |

#### Step 2: Compute Predicted Ratings
Multiply the user matrix (U) by the item matrix (V) to get the predicted ratings matrix (R'):

$$R' = U \times V^T$$

**Predicted Ratings Matrix (R'):**
|       | Video 1 | Video 2 | Video 3 |
|-------|---------|---------|---------|
| User 1 |   0.21  |   0.15  |   0.11  |
| User 2 |   0.26  |   0.14  |   0.23  |
| User 3 |   0.39  |   0.23  |   0.23  |

#### Step 3: Optimize Matrices
Adjust the values in U and V to minimize the loss function. This is done using techniques like Stochastic Gradient Descent (SGD).

#### Step 4: Updated Matrices
After several iterations, the matrices U and V are adjusted to better predict the original matrix R.

**User Matrix (U) (Adjusted):**
|       | Factor 1 | Factor 2 |
|-------|----------|----------|
| User 1 |    0.4   |    0.5   |
| User 2 |    0.3   |    0.4   |
| User 3 |    0.2   |    0.7   |

**Item Matrix (V) (Adjusted):**
|         | Factor 1 | Factor 2 |
|---------|----------|----------|
| Video 1 |    0.5   |    0.6   |
| Video 2 |    0.3   |    0.4   |
| Video 3 |    0.4   |    0.5   |

### Step 5: Generate Recommendations
Using the adjusted matrices, compute the predicted ratings for all user-video pairs. These predicted ratings help generate a list of recommended videos for each user based on the latent factors captured during the matrix factorization process.

**Predicted Ratings Matrix (R') (Final):**
|       | Video 1 | Video 2 | Video 3 |
|-------|---------|---------|---------|
| User 1 |   4.8   |   3.1   |   2.0   |
| User 2 |   4.1   |   2.5   |   1.7   |
| User 3 |   1.2   |   1.1   |   0.9   |

### Conclusion
Matrix factorization does not directly extract latent factors from video metadata. Instead, it learns latent factors through an iterative process that minimizes the difference between observed interactions and predicted interactions. These latent factors emerge from the patterns in the user-item interaction data, capturing abstract concepts that explain user preferences and item characteristics. While SVD and matrix factorization with SGD both aim to decompose the interaction matrix into latent factors, SVD is a direct decomposition method, whereas SGD is an iterative optimization technique. In practice, matrix factorization with SGD is often preferred for large-scale recommendation systems due to its scalability and efficiency in handling sparse data.


> #### Differences and Use Cases
> ##### SVD:
> - Provides an exact decomposition.
> - Computationally intensive for very large datasets.
> - Useful when an exact solution is needed and the dataset is manageable in size.
> ##### Matrix Factorization with SGD:
> - Provides an approximate solution through iterative optimization.
> - More scalable and efficient for very large and sparse datasets.
> - Commonly used in practical recommendation systems where the focus is on scalability and handling large datasets.
