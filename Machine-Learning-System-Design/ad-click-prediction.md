# Ad Click Prediction

## 1. Problem Statement

Build a machine learning model to predict if an ad will be clicked.

> For the sake of simplicity, we will not focus on the cascade of classifiers that is commonly used in AdTech.

<img src="Ads_recommendation_system.png" alt="Ads_recommendation_system" width="600" height="300"/>


- Let’s understand the ad serving background before moving forward. The ad request goes through a waterfall model where publishers try to sell its inventory through direct sales with high CPM (Cost Per Million). If it is unable to do so, the publishers pass the impression to other networks until it is sold.
  
<img src="Waterfall_Revenue_Model.png" alt="Waterfall_Revenue_Model" width="600" height="300"/>

> A cascade of classifiers involves multiple stages, each with its own classifier, where the output of one classifier becomes the input for the next. This can help in:
> - **Filtering:** Early stages filter out the majority of negative cases (e.g., unlikely clicks), allowing subsequent stages to focus on harder-to-classify cases.
> - **Specialization:** Different stages can specialize in different aspects of the prediction task, improving overall performance.
> - **Efficiency:** By progressively reducing the number of cases to be considered in each stage, the cascade can improve computational efficiency.

> In the waterfall model, the publisher (e.g., The New York Times) would first try to sell the ad space directly to advertisers for a high CPM. If it fails, it would then pass the opportunity to ad networks like Google AdSense, Facebook Audience Network, etc., in sequence until the ad space is filled.

>**Publisher's Direct Sale**:
> *Scenario*: The New York Times negotiates directly with Samsung for a banner ad on its homepage for a month at a high CPM.
>
> **Premium Ad Network**:
> *Scenario*: An advertiser uses Google Marketing Platform to place ads on top-tier sites like ESPN, ensuring their ads appear on reputable, high-traffic sites.
>
> **Vertical Ad Network**:
> *Scenario*: A travel agency uses Travel Ad Network to place ads specifically on travel-related websites, reaching an audience interested in travel.
>
> **Remnant Ad Network**:
> *Scenario*: A smaller advertiser uses Google Ads to bid on leftover ad inventory across Google's network, allowing their ads to appear on various sites at a lower cost.

> **Publisher**: A publisher is an entity (individual, organization, or company) that creates and owns content on which advertisements can be displayed. They provide the platform where ads are shown.
> 
> **Ad Network**: An ad network is a company that connects advertisers with publishers. It aggregates ad inventory from multiple publishers and sells it to advertisers, facilitating the buying and selling of ads.
> 
> **Impression**: An impression is a single instance of an ad being displayed on a web page, app, or any other digital medium. It is a metric used to measure how many times an ad is seen.


## 2. Metrics Design and Requirements

### Metrics

During the training phase, we can focus on machine learning metrics instead of revenue metrics or CTR metrics. Below are the two metrics:

#### Offline Metrics

- **Normalized Cross-Entropy (NCE):** NCE is the predictive log loss divided by the cross-entropy of the background CTR. This way NCE is insensitive to background CTR. This is the NCE formula:

$$ NCE = \frac{-\frac{1}{N} \sum\limits_{i=1}^{n} \left( \frac{1 + y_i}{2} \log(p_i) + \frac{1 - y_i}{2} \log(1 - p_i) \right)}{- \left( p \log(p) + (1 - p) \log(1 - p) \right)} $$

  >   **N**: Total number of samples.
  > 
  >   **$y_i$**: Actual label for the $i$-th sample. $y_i$ is 1 if the item was clicked and 0 if it was not clicked.
  > 
  > **$p_i$**: Predicted probability of the $i$-th item being clicked.
  > 
  >**$p$**: Background CTR (average probability of an item being clicked).
  >
  > Background CTR refers to the average Click Through Rate (CTR) observed in the dataset or a specific subset of data. It reflects the inherent likelihood of clicks across all items without considering individual item characteristics. For example, if the overall CTR of the platform is 0.02, it means that, on average, 2% of all shown items are clicked.
> 
#### Online Metrics

- **Revenue Lift:** Percentage of revenue changes over a period of time. Upon deployment, a new model is deployed on a small percentage of traffic. The key decision is to balance between percentage traffic and the duration of the A/B testing phase.

> A/B Testing involves splitting the traffic into two groups:
> - Control Group: Uses the existing model.
> - Test Group: Uses the new model.

> **Revenue Lift**: It is the percentage change in revenue over a period of time when a new model is deployed compared to the existing model or baseline performance.
> #### Formula:
> **Revenue Lift**:
>
> $\text{Revenue Lift} = \frac{\text{Revenue test} - \text{Revenue control}}{\text{Revenue}_{\text{control}}} \times 100$ \%
> 
> Where:
> - $\text{Revenue}_{\text{test}}$ is the revenue generated by the test group.
> - $\text{Revenue}_{\text{control}}$ is the revenue generated by the control group.

> ### Balancing Percentage Traffic:
>- **Small Percentage**: Minimizes risk but may take longer to gather statistically significant data.
>- **Large Percentage**: Provides quicker results but increases the risk if the new model performs poorly.

> ### Duration of A/B Testing:
>- **Short Duration**: Faster decision-making but may not capture long-term user behavior changes.
>- **Long Duration**: More comprehensive data but delays implementation of the new model if it performs well.

> ### Balance between Percentage Traffic and the Duration of the A/B Testing Phase
> #### Risk Tolerance:
> - **More Risk-Averse**: You might prefer a smaller percentage of traffic and a longer duration.
  > - Example: Allocating 10% of traffic to the test group and running the test for 4 weeks.
> - **More Comfortable with Risk**: You might choose a larger percentage of traffic for a shorter duration.
  > - Example: Allocating 50% of traffic to the test group and running the test for 1 week.
> #### Need for Speed:
> - **Quick Decision Needed**: You might accept the higher risk of using a larger percentage of traffic over a shorter period.
  > - Example: Allocating 40% of traffic to the test group and running the test for 2 weeks.
> - **Time Is Less of an Issue**: You can afford to run the test longer with a smaller percentage of traffic.
  > - Example: Allocating 20% of traffic to the test group and running the test for 6 weeks.

### Requirements

#### Training

- **Imbalance data:** The Click Through Rate (CTR) is very small in practice (1%-2%), which makes supervised training difficult. We need a way to train the model that can handle highly imbalanced data.
- **Retraining frequency:** The ability to retrain models many times within one day to capture the data distribution shift in the production environment.
- **Train/validation data split:** To simulate a production system, the training data and validation data is partitioned by time.

> ### Partitioning by Time
> The statement "Train/validation data split: To simulate a production system, the training data and validation data is partitioned by time" means that when dividing the data into training and validation sets, the split is done based on time rather than randomly.
> 
> Training Data: This set includes data from an earlier time period.
> 
> Validation Data: This set includes data from a later time period, following the training data period.
> 
> This approach ensures that the model is trained on past data and tested on future data, which is more representative of the production environment where the model will encounter new, unseen data.
#### Inference

- **Serving:** Low latency (50ms - 100ms) for ad prediction.
- **Latency:** Ad requests go through a waterfall model, therefore, recommendation latency for ML model needs to be fast.
- **Overspent:** If the ad serving model repeatedly serves the same ads, it might end up over-spending the campaign budget and publishers lose money.

### Summary

| Type     | Desired Goals                                                                 |
|----------|-------------------------------------------------------------------------------|
| Metrics  | Reasonable normalized cross-entropy and click through rate                    |
| Training | Ability to handle imbalance data                                              |
|          | High throughput with the ability to retrain many times per day                |
| Inference| Latency from 50 to 100ms                                                      |
|          | Ability to control or avoid overspent campaign budget while serving ads       |

## 3. Model

### Feature Engineering

| Features                              | Feature Engineering              | Description                                            |
|---------------------------------------|----------------------------------|--------------------------------------------------------|
| **AdvertiserID**                      | Use Embedding or feature hashing | It’s easy to have millions of advertisers              |
| **User’s historical behavior,i.e. Number of clicks on ads over a period of time**        | Feature scaling, i.e., normalization |          |
| **Temporal: time_of_day, day_of_week**| One hot encoding                 |                                                        |
| **Cross features**                    | Combine multiple features        | See example in the Machine Learning System Design Primer |

### Training Data

Before building any ML models we need to collect training data. The goal here is to collect data across different types of posts while simultaneously improving the user experience. As you recall from the previous lesson about the waterfall model, we can collect a lot of data about ad clicks. We can use this data for training the Ad Click model.

We can start to use data for training by selecting a period of data: last month, last six months, etc. In practice, we want to find a balance between training time and model accuracy. We also downsample the negative data to handle the imbalanced data.

### Model Selection

We can use deep learning in distributed settings. We can start with fully connected layers with the Sigmoid activation function applied to the final layer. Because the CTR is usually very small (less than 1%), we would need to resample the training data set to make the data less imbalanced. It’s important to leave the validation and test sets intact to have accurate estimations about model performance.

### Evaluation

One approach is to split the data into training data and validation data. Another approach is to replay evaluation to avoid biased offline evaluation. Assume the training data we have up until time \( t \). We use test data from time \( t+1 \) and reorder their ranking based on our model during inference. If there is an accurate click prediction, we record a match. The total match will be considered as total clicks.

During evaluation, we will also evaluate how big our training data set should be and how frequently we need to retrain the model among many other hyperparameters.

## 4. Calculation and Estimation

### Assumptions

- 40K ad requests per second or 100 billion ad requests per month.
- Each observation (record) has hundreds of features, and it takes 500 bytes to store.

### Data Size

- Data: Historical ad click data includes [user, ads, click_or_not]. With an estimated 1% CTR, it has 1 billion clicked ads.
- We can start with 1 month of data for training and validation. Within a month, we have:
  \[
  100 \times 10^{12} \times 500 = 5 \times 10^{16} \text{ bytes or 50 PB}
  \]
- One way to make it more manageable is to downsample the data, i.e., keep only 1%-10% or use 1 week of data for training data and use the next day for validation data.

### Scale

- Supports 100 million users.

## 5. High-Level Design

### AdClick Prediction High-Level Design

- **Data Lake:** Store data that is collected from multiple sources, i.e., logs data or event-driven data (Kafka).
- **Batch Data Prep:** Collections of ETL (Extract, Transform, and Load) jobs that store data in Training Data Store.
- **Batch Training Jobs:** Organize scheduled jobs as well as on-demand jobs to retrain new models based on training data storage.
- **Model Store:** Distributed storage like S3 to store models.
- **Ad Candidates:** Set of Ad candidates provided by upstream services (refer back to waterfall model).
- **Stream Data Prep Pipeline:** Processes online features and stores features in key-value storage for low latency downstream processing.
- **Model Serving:** Standalone service that loads different models and provides Ad Click probability.

### Flow of the System

1. User visits the homepage and sends an Ad request to the Candidate Generation Service. Candidate Generation Service generates a list of Ads Candidates and sends them to the Aggregator Service.
2. The Aggregator Service splits the list of candidates and sends it to the Ad Ranking workers to score.
3. Ad Ranking Service gets the latest model from Model Repos, gets the correct features from the Feature Store, produces ad scores, then returns the list of ads with scores to the Aggregator Service.
4. The Aggregator Service selects top K ads (For example, k = 10, 100, etc.) and returns to upstream services.

## 6. Scale the Design

Given a latency requirement of 50ms-100ms for a large volume of Ad Candidates (50k-100k), if we partition one serving instance per request, we might not achieve Service Level Agreement (SLA). For this, we scale out Model Serving and put Aggregator Service to spread the load for Model Serving components.

One common pattern is to have the Aggregator Service. It distributes the candidate list to multiple serving instances and collects results. [Read more about it here](#).

## 7. Follow-Up Questions

| Question | Answer |
|----------|--------|
| How do we adapt to user behavior changing over time? | Retrain the model as frequently as possible. One example is to retrain the model every few hours with new data (collected from user clicked data). |
| How do we handle the Ad Ranking Model being under-explored? | We can introduce randomization in Ranking Service. For example, 2% of requests will get random candidates, and 98% will get sorted candidates from Ad Ranking Service. |

## 8. Summary

- We first learned to choose Normalize Entropy as the metric for the Ad Click Prediction Model.
- We learned how to apply the Aggregator Service to achieve low latency and overcome imbalance workloads.
- To scale the system and reduce latency, we can use Kubeflow so that Ad Generation services can directly communicate with Ad Ranking services.
- We can also learn more about how companies scale their design [here](#).


