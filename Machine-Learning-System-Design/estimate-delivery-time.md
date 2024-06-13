# Estimate Delivery Time

## 1. Problem Statement
Build a model to estimate the total delivery time given order details, market conditions, and traffic status.

<img src="Food_Delivery_flow.png" alt="Food_Delivery_flow" width="650" height="450"/>


To keep it simple, we do not consider batching (group multiple orders at restaurants) in this exercise.

> In the context of building a model to estimate total delivery time, "market conditions" refer to the various external factors that can influence the delivery process.  Some examples of market conditions include:
> 
> Day of the Week: Delivery times may vary between weekdays and weekends.
> 
>Time of the Day: Delivery times can be influenced by whether it is peak dining hours (e.g., lunch or dinner time) or off-peak hours.
> 
>Weather Conditions: Adverse weather such as rain, snow, or extreme heat can slow down deliveries.
> 
>Special Events: Local events such as sports games, concerts, or festivals can cause increased traffic and longer delivery times.
> 
>Holidays: Delivery times might be longer during holidays due to higher order volumes and different traffic patterns.
> 
>Promotions and Discounts: Times when there are special promotions or discounts might lead to a higher volume of orders, affecting delivery times.

### Delivery Time Calculation
$$\text{DeliveryTime} = \text{PickupTime} + \text{PointtoPointTime} + \text{DropoffTime}$$

## 2. Metrics Design and Requirements

### Metrics

**Offline Metrics:** Use Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{\sum\limits_{k=1}^{n} (\text{predict} - y)^2}{n}}$$
where,
- $n$ is the total number of samples,
- $\text{predict}$ is the estimated wait time,
- $y$ is the actual wait time.

**Online Metrics:** Use A/B testing and monitor RMSE, customer engagement, customer retention, etc.


> **RMSE** is a commonly used metric for regression problems, including estimating delivery times. RMSE measures the average magnitude of the error between predicted values and actual values.By taking the square root of MSE, RMSE converts the error back into the original units of the target variable. This means RMSE is in the same units as the delivery time (minutes), making it more interpretable. For example, an RMSE of 2.9 minutes directly tells us that, on average, the predictions are off by about 2.9 minutes.

> **A/B Testing**: Split users into groups where one group uses the new delivery time estimation model and the other uses the existing model. Compare the outcomes between these groups.
> 
> **Customer Engagement**: Measure how customers interact with the app, such as checking delivery times, order frequency, and app usage duration.
> 
> **Customer Retention**: Track whether customers continue using the service over time, indicating satisfaction with the delivery experience.

### Requirements

#### Training
- During training, we need to handle a large amount of data. For this, the training pipeline should have a high throughput. To achieve this purpose, data can be organized in Parquet files.
- The model should undergo retraining every few hours. Delivery operations are under a dynamic environment with many external factors: traffic, weather conditions, etc. So, it is important for the model to learn and adapt to the new environment. For example, on game day, traffic conditions can get worse in certain areas. Without a retrained model, the current model will consistently underestimate delivery time. Schedulers are responsible for retraining models many times throughout the day.
- Balance between overestimation and underestimation. To help with this, retrain multiple times per day to adapt to market dynamics and traffic conditions.

> **High Throughput**: In the context of machine learning, "high throughput" refers to the ability to process a large amount of data quickly and efficiently.
> 
>  To achieve high throughput, the training pipeline should be optimized for speed and efficiency. This involves using data storage and processing techniques that allow for rapid reading and writing of data. So we can consider using **Parquet Files** : Apache Parquet is a columnar storage file format designed for efficient data processing.
> 
>  Here's why using Parquet files can help achieve high throughput:
> 
> Parquet stores data in a columnar format, meaning that all the values for a particular column are stored together. This contrasts with row-based formats (like CSV), where each row's data is stored together. For many machine learning tasks, operations are performed on entire columns at once (e.g., calculating averages, normalizing data). Columnar storage allows these operations to be performed more efficiently.
> 
> Parquet files support various compression algorithms (e.g., Snappy, Gzip), which reduce the size of the data on disk.
Speed: Smaller file sizes mean less data needs to be read from disk into memory, speeding up data loading times.
#### Inference
- For every delivery, the system needs to make real-time estimations as frequently as possible. For simplicity, we can assume we need to make 30 predictions per delivery.
- Near real-time updates: any changes in status need to go through model scoring as fast as possible, i.e., the restaurant starts preparing meals, the driver starts driving to customers.
- Whenever there are changes in delivery, the model runs a new estimate and sends an update to the customer.
- Capture near real-time aggregated statistics, i.e., feature pipeline aggregates data from multiple sources (Kafka, database) to reduce latency.
- Latency from 100ms to 200ms

### Summary
| Type      | Desired goals                                                                                                                                       |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| Metrics   | Optimized for low RMSE. Estimation should be less than 10-15 minutes. If we overestimate, customers are less likely to make orders. Underestimation can cause customers upset. |
| Training  | High throughput with the ability to retrain many times per day                                                                                      |
| Inference | Latency from 100ms to 200ms                                                                                                                         |

# Appendix 
## Offline Metric Example

Let's say we have 5 deliveries with the following actual and predicted delivery times (in minutes):

| Delivery | Actual Time (y) | Predicted Time (predict) |
|----------|------------------|--------------------------|
| 1        | 30               | 32                       |
| 2        | 25               | 28                       |
| 3        | 40               | 36                       |
| 4        | 35               | 38                       |
| 5        | 20               | 22                       |

First, calculate the squared differences:
$$(32 - 30)^2 = 4$$
$$(28 - 25)^2 = 9$$
$$(36 - 40)^2 = 16$$
$$(38 - 35)^2 = 9$$
$$(22 - 20)^2 = 4$$

Sum these squared differences:
$$4 + 9 + 16 + 9 + 4 = 42$$

Divide by the number of samples (n = 5) and take the square root:
$$\text{RMSE} = \sqrt{\frac{42}{5}} = \sqrt{8.4} \approx 2.9$$

So, the RMSE in this case is approximately 2.9 minutes.
