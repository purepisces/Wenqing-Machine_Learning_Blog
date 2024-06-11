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
