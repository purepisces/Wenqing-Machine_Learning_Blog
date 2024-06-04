# Collaborative Filtering

## Collaborative Filtering Example

### User-Item Interaction Matrix
Let's consider a small dataset of user interactions with videos. The interaction matrix \( R \) contains user ratings for different videos.

|       | Video 1 | Video 2 | Video 3 | Video 4 | Video 5 |
|-------|---------|---------|---------|---------|---------|
| User 1 |    5    |    4    |    0    |    0    |    3    |
| User 2 |    3    |    0    |    4    |    2    |    1    |
| User 3 |    4    |    2    |    5    |    3    |    0    |
| User 4 |    0    |    5    |    4    |    4    |    2    |
| User 5 |    1    |    0    |    0    |    5    |    4    |

### Step-by-Step Calculation

#### Step 1: Identify Common Rated Videos
For User 1 and User 3, the common rated videos are Video 1, Video 2, and Video 5.

- **User 1's ratings**: $R_{U1} = [5, 4, 3]$ for Videos 1, 2, and 5.
- **User 3's ratings**: $R_{U3} = [4, 2, 0]$ for Videos 1, 2, and 5.

#### Step 2: Calculate the Dot Product
The dot product of the ratings vectors is calculated as follows:

$R_{U1} \cdot R_{U3} = (5 \times 4) + (4 \times 2) + (3 \times 0)$
$R_{U1} \cdot R_{U3} = 20 + 8 + 0$
$R_{U1} \cdot R_{U3} = 28$

#### Step 3: Calculate the Magnitudes of the Vectors
Calculate the magnitude of User 1's and User 3's ratings vectors.

$$\|R_{U1}\| = \sqrt{(5^2) + (4^2) + (3^2)}$$

$$\|R_{U1}\| = \sqrt{25 + 16 + 9}$$

$$\|R_{U1}\| = \sqrt{50}$$

$$\|R_{U1}\| \approx 7.07$$

$$\|R_{U3}\| = \sqrt{(4^2) + (2^2) + (0^2)}$$

$$\|R_{U3}\| = \sqrt{16 + 4 + 0}$$

$$\|R_{U3}\| = \sqrt{20}$$

$$\|R_{U3}\| \approx 4.47$$

#### Step 4: Calculate Cosine Similarity
Using the dot product and magnitudes, calculate the cosine similarity.

$$\text{similarity}(U1, U3) = \frac{R_{U1} \cdot R_{U3}}{\|R_{U1}\| \times \|R_{U3}\|}$$

$$ \text{similarity}(U1, U3) = \frac{28}{7.07 \times 4.47}$$

$$\text{similarity}(U1, U3) = \frac{28}{31.61}$$

$$ \text{similarity}(U1, U3) \approx 0.89$$

### Conclusion
The cosine similarity between User 1 and User 3 is approximately 0.89, indicating a high degree of similarity between their ratings for the common videos. This similarity score can be used in collaborative filtering algorithms to recommend videos to users based on the preferences of similar users.

### Predict Missing Ratings
Use the similarity scores to predict ratings for the missing items. For User 1, we need to predict ratings for Video 3 and Video 4.

#### Weighted Sum Approach

$$\hat{R}(U1, V3)= \frac{\sum (\text{similarity}(U1, Ux) \cdot R_{Ux, V3})}{\sum \text{similarity}(U1, Ux)}$$

Where $Ux$ are the users similar to $U1$ who have rated $V3$.

### Example Calculation
Assuming the similarity scores and user ratings, we calculate the predicted rating for User 1 for Video 3 using the ratings from similar users (User 2 and User 4).

For simplicity, let's assume:
- Similarity(User 1, User 2) = 0.9
- Similarity(User 1, User 4) = 0.7

Ratings:
- User 2's rating for Video 3 = 4
- User 4's rating for Video 3 = 4

Predicted Rating for User 1 for Video 3:

$$\hat{R}_{U1, V3} = \frac{(0.9 \cdot 4) + (0.7 \cdot 4)}{0.9 + 0.7} = \frac{3.6 + 2.8}{1.6} = \frac{6.4}{1.6} = 4.0 $$

Thus, the predicted rating for User 1 for Video 3 is 4.0. We can similarly predict ratings for other missing values.

### Final Recommendations
Based on the predicted ratings, User 1 can be recommended Video 3 and Video 4, as these now have high predicted ratings.

This collaborative filtering approach leverages the collective preferences of similar users to make recommendations, enhancing the user's experience by suggesting videos they are likely to enjoy based on their similarity to other users' tastes.
