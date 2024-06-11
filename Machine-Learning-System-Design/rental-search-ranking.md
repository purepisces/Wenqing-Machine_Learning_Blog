# Airbnb Rental Search Ranking

## 1. Problem Statement

Airbnb users search for available homes at a particular location. The system should sort stays from multiple homes in the search result so that the most frequently booked homes appear on top.

<img src="Search_ranking_system_for_Airbnb.png" alt="Search_ranking_system_for_Airbnb" width="600" height="300"/>

The naive approach would be to craft a custom score ranking function, for example, a score based on text similarity given a query. This wouldn’t work well because similarity doesn’t guarantee a booking.

The better approach would be to sort results based on the likelihood of booking. We can build a supervised ML model to predict booking likelihood. This is a binary classification model, i.e., classify booking and not-booking.

## 2. Metrics Design and Requirements

### Metrics

**Offline Metrics**

- **Discounted Cumulative Gain (DCG):**

  
  $$DCG_p = \sum_{i=1}^{p} \frac{rel_i}{\log_2(i + 1)}$$
  - $p$ is the position in the ranking list.
  - $rel_i$ stands for the relevance of the result at position $i$.

- **Ideal Discounted Cumulative Gain (IDCG):**
  
  $$IDCG_p = \sum_{i=1}^{|REL_p|} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$
  - $|REL_p|$ represents the number of relevance scores considered for the top $p$ positions in the ideally sorted list. Essentially, it is the count of relevance scores in the ideal ranking list up to position $p$.

- **Normalized Discounted Cumulative Gain (nDCG):**
  
  $$nDCG_p = \frac{DCG_p}{IDCG_p}$$
  

> DCG is calculated based on the predicted ranking of the results. The relevance scores used in the DCG calculation are the ground truth relevance scores corresponding to the predicted positions.

> IDCG is calculated based on the ideal ranking of the results, which means sorting the results by the ground truth relevance scores in descending order. The relevance scores used in the IDCG calculation are also the ground truth relevance scores.

> nDCG is the ratio of DCG to IDCG and provides a normalized score to compare different ranking systems. The value of nDCG ranges between 0 and 1, where 1 indicates a perfect ranking.

**Online Metrics**

- **Conversion Rate and Revenue Lift:** This measures the number of bookings per number of search results in a user session.

  
  $$conversionrate = \frac{numberofbookings}{numberofsearchresults}$$


### Requirements

**Training**

- **Imbalanced Data and Clear-cut Session:** An average user might do extensive research before deciding on a booking. As a result, the number of non-booking labels has a higher magnitude than booking labels.
- **Train/Validation Data Split:** Split data by time to mimic production traffic. For example, select one specific date to split training and validation data. Use a few weeks of data before that date as training data and a few days of data after that date as validation data.

> A clear-cut session is typically defined by a period of continuous user activity without significant breaks. The end of a session can be marked by a user's inactivity for a specific duration, logging out, or other criteria defined by the application.
> 
> Mimicking Production Traffic: When splitting data for training and validation, it is important to simulate real-world conditions as closely as possible. In the context of Airbnb's search ranking system, we want to ensure that the model is trained on past data and validated on future data to mimic how it would perform in production. By splitting data based on time, we ensure that the model is tested on data it has not seen before, similar to how it would encounter new user interactions in a live environment.

**Inference**

- **Serving:** Low latency (50ms - 100ms) for search ranking.
- **Under-predicting for New Listings:** Brand new listings might not have enough data for the model to estimate likelihood, potentially leading to under-prediction for new listings.

### Summary

| Type       | Desired Goals                                                       |
|------------|---------------------------------------------------------------------|
| **Metrics**| Achieve high normalized discounted Cumulative Gain metric           |
| **Training**| Ability to handle imbalanced data                                  |
|            | Split training data and validation data by time                     |
| **Inference**| Latency from 50ms to 100ms                                        |
|            | Ability to avoid under-predicting for new listings                  |

## 3. Model
### Feature Engineering
- **Geolocation of listing (latitude/longitude):** Taking raw latitude and raw longitude features is very tough to model as feature distribution is not smooth. One way around this is to take a log of the distance from the center of the map for latitude and longitude separately.
- **Favorite place:** Store user’s favorite neighborhood place in a 2-dimensional grid. For example, users add Pier 39 as their favorite place, we encode this place into a specific cell, then use embedding before training/serving.

| Features                   | Feature Engineering                             | Description                                                                      |
|----------------------------|-------------------------------------------------|----------------------------------------------------------------------------------|
| **Listing ID**             | Listing ID embedding                            | |
| **Listing feature**        | Number of bedrooms, list of amenities, listing city |                                                                                  |
| **Location**               | Measure lat/long from the center of the user map, then normalize |                                           |
| **Historical search query**| Text embedding                                  |                                                                                  |
| **User associated features: age, gender** | Normalization or Standardization  |                                                                                  |
| **Number of previous bookings** | Normalization or Standardization           |                                                                                  |
| **Previous length of stays** | Normalization or Standardization              |                                                                                  |
| **Time related features**  | Month, week of year, holiday, day of week, hour of day |                                                                                  |

### Training Data
- **Source:** User search history, view history, and bookings.
- **Selection:** We can start by selecting a period of data: last month, last 6 months, etc., to find the balance between training time and model accuracy.
- **Experimentation:** In practice, we decide the length of training data by running multiple experiments. Each experiment will pick a certain time period to train data. We then compare model accuracy and training time across different experimentations.

### Model Architecture
- **Input:** User data, search query, and Listing data.
- **Output:** This is a binary classification model, i.e., user books a rental or not.
- **Baseline:** We can start with deep learning with fully connected layers as a baseline. The model outputs a number within [0, 1] and presents the likelihood of booking.
- **Improvement:** To further improve the model, we can also use other more modern network architectures, i.e., Variational AutoEncoder or Denoising AutoEncoder. Read more about [Variational Autoencoder](https://arxiv.org/abs/1312.6114).

# Appendix

## Example: Offline Metrics

### Step 1: Define Relevance Scores

Assume we have a list of 5 search results with the following relevance scores (higher is better):
| Position (i) | Listing ID | Relevance Score (rel_i) |
|--------------|------------|-------------------------|
| 1            | A          | 3                       |
| 2            | B          | 2                       |
| 3            | C          | 3                       |
| 4            | D          | 0                       |
| 5            | E          | 1                       |

### Step 2: Calculate DCG

Using the formula for DCG:
$$DCG_p = \sum_{i=1}^{p} \frac{rel_i}{\log_2(i + 1)}$$

For our 5 results:
$$DCG_5 = \frac{3}{\log_2(1 + 1)} + \frac{2}{\log_2(2 + 1)} + \frac{3}{\log_2(3 + 1)} + \frac{0}{\log_2(4 + 1)} + \frac{1}{\log_2(5 + 1)}$$

Calculating each term:
$$\frac{3}{\log_2(2)} = 3$$
$$\frac{2}{\log_2(3)} \approx 1.26186$$
$$\frac{3}{\log_2(4)} = 1.5$$
$$\frac{0}{\log_2(5)} = 0$$
$$\frac{1}{\log_2(6)} \approx 0.38685$$

Summing these values:
$$DCG_5 = 3 + 1.26186 + 1.5 + 0 + 0.38685 \approx 6.14871$$

### Step 3: Calculate IDCG

Using the formula for IDCG:
$$IDCG_p = \sum_{i=1}^{p} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

First, sort the relevance scores in descending order: [3, 3, 2, 1, 0]

Using the sorted relevance scores, we calculate IDCG:
$$IDCG_5 = \frac{2^3 - 1}{\log_2(1 + 1)} + \frac{2^3 - 1}{\log_2(2 + 1)} + \frac{2^2 - 1}{\log_2(3 + 1)} + \frac{2^1 - 1}{\log_2(4 + 1)} + \frac{2^0 - 1}{\log_2(5 + 1)}$$

Calculating each term:
$$\frac{2^3 - 1}{\log_2(2)} = \frac{7}{1} = 7$$
$$\frac{2^3 - 1}{\log_2(3)} \approx \frac{7}{1.58496} \approx 4.41651$$
$$\frac{2^2 - 1}{\log_2(4)} = \frac{3}{2} = 1.5$$
$$\frac{2^1 - 1}{\log_2(5)} \approx \frac{1}{2.32193} \approx 0.43068$$
$$\frac{2^0 - 1}{\log_2(6)} = \frac{0}{2.58496} = 0$$

Summing these values:
$$IDCG_5 = 7 + 4.41651 + 1.5 + 0.43068 + 0 \approx 13.34719$$

### Step 4: Calculate nDCG

Finally, nDCG is the ratio of DCG to IDCG:
$$nDCG_p = \frac{DCG_p}{IDCG_p}$$

For our example:
$$nDCG_5 = \frac{6.14871}{13.34719} \approx 0.4605$$

### Summary

- **DCG**: 6.14871
- **IDCG**: 13.34719
- **nDCG**: 0.4605


## Example: Using Both Online Metrics: Conversion Rate and Revenue Lift

Let’s take a practical example of implementing a new search ranking algorithm and measure its impact using both conversion rate and revenue lift.

#### Data Collection Before and After Change

| Metric                    | Before Change | After Change |
|---------------------------|---------------|--------------|
| Number of Search Results  | 10,000        | 10,000       |
| Number of Bookings        | 1,000         | 1,200        |
| Average Revenue per Booking | \$200       | \$220        |

#### Conversion Rate Calculation

**Before Change:**

$$\text{conversion rate}_{\text{before}} = \frac{1,000}{10,000} = 10\%$$

**After Change:**

$$\text{conversion rate}_{\text{after}} = \frac{1,200}{10,000} = 12\%$$

#### Revenue Lift Calculation

**Total Revenue Before Change:**

$$\text{total revenue}_{\text{before}} = 1,000 \times \$200 = \$200,000$$

**Total Revenue After Change:**

$$\text{total revenue}_{\text{after}} = 1,200 \times \$220 = \$264,000$$

**Revenue Lift:**

$$\text{revenue lift} = \frac{\$264,000 - \$200,000}{\$200,000} \times 100\% = 32\%$$

### Summary

- **Conversion Rate** increased from 10% to 12%, indicating that more users are booking properties.
- **Revenue Lift** increased by 32%, showing that the new algorithm not only resulted in more bookings but also increased the total revenue significantly.
