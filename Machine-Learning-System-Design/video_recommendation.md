
# Video Recommendations

## 1. Problem Statement
The goal is to build a video recommendation system for YouTube users to maximize engagement and suggest new types of content. The system aims to:

- Increase user engagement.
- Introduce users to new and diverse content.
  
insert video recommendation png

## 2. Metrics Design and Requirements

### Metrics

**Offline Metrics:**

- **Precision:** The ratio of relevant videos recommended to the total recommended videos.
- **Recall:** The ratio of relevant videos recommended to the total relevant videos available.
- **Ranking Loss:** A metric to measure the error in the predicted order of recommended videos.
- **Logloss:** A measure of the accuracy of the probabilistic predictions.

**Online Metrics:**

- **Click Through Rates (CTR):** The ratio of users who click on a recommended video to the total users who see the recommendation.
- **Watch Time:** The total time users spend watching the recommended videos.
- **Conversion Rates:** The ratio of users who perform a desired action (e.g., subscribing to a channel) after watching the recommended videos.

### Requirements

**Training:**

- **Frequency:** Train the model multiple times a day to capture temporal changes, as user behavior can be unpredictable, and videos can become viral quickly.
- **Throughput:** Ensure the training process can handle high volumes of data efficiently.

**Inference:**

- **Latency:** Recommendations must be generated within 200ms, ideally under 100ms, for each user visit to the homepage.
- **Balance:** Find the right balance between exploration (showing new content) and exploitation (showing historically relevant content). Over-exploitation of historical data can prevent new videos from being exposed to users.

### Summary

| Type       | Desired Goals                                      |
|------------|----------------------------------------------------|
| **Metrics**| Reasonable precision, high recall                  |
| **Training**| High throughput, ability to retrain frequently    |
| **Inference**| Latency from 100ms to 200ms, flexible exploration vs. exploitation control |

The recommendation system should ensure a balance between providing relevant content and introducing fresh content to keep the users engaged and discovering new videos. The system needs to be responsive and adaptable to the rapidly changing user preferences and viral trends.


## 3. Multi-stage Models

insert architecture_diagram_for_the_video_recommendation_system.png

The architecture of the video recommendation system is divided into two main stages: **Candidate Generation** and **Ranking**. This two-stage approach helps in scaling the system efficiently and effectively.

It’s a common pattern that you will see in many ML systems.

We will explore the two stages in the section below.

### Candidate Generation Model

The candidate model will find the relevant videos based on user watch history and the type of videos the user has watched.

#### Feature Engineering

Each user has a list of video watches (videos, minutes_watched).

#### Training Data

User-video watch space: Use data from a selected period (e.g., last month, last 6 months) to balance training time and model accuracy.

#### Model

- The candidate generation can be done by Matrix factorization. The purpose of candidate generation is to generate “somewhat” relevant content to users based on their watched history. The candidate list needs to be big enough to capture potential matches for the model to perform well with desired latency.

- One solution is to use collaborative algorithms because the inference time is fast, and it can capture the similarity between user taste in the user-video space.

> In practice, for large scale system (Facebook, Google), we don’t use Collaborative Filtering and prefer low latency method to get candidate. One example is to leverage Inverted Index (commonly used in Lucene, Elastic Search). Another powerful technique can be found FAISS or Google ScaNN.

### Ranking Model

During inference, the ranking model receives a list of video candidates given by the Candidate Generation model. For each candidate, the ranking model estimates the probability of that video being watched. It then sorts the video candidates based on that probability and returns the list to the upstream process.

#### Feature Engineering

| Features                     | Feature Engineering                  |
|------------------------------|--------------------------------------|
| Watched video IDs            | Video embedding                      |
| Historical search query      | Text embedding                       |
| Location                     | Geolocation embedding                |
| User associated features: age, gender | Normalization or Standardization |
| Previous impression          | Normalization or Standardization     |
| Time related features        | Month, week_of_year, holiday, day_of_week, hour_of_day |

#### Training Data

We can use User Watched History data. Normally, the ratio between watched vs. not-watched is 2/98. So, for the majority of the time, the user does not watch a video.

#### Model

At the beginning, it’s important that we started with a simple model, as we can add complexity later.

- A fully connected neural network is simple yet powerful for representing non-linear relationships, and it can handle big data.

- We start with a fully connected neural network with sigmoid activation at the last layer. The reason for this is that the Sigmoid function returns value in the range [0, 1]; therefore it’s a natural fit for estimating probability.

> For deep learning architecture, we can use relu, (Rectified Linear Unit), as an activation function for hidden layers. It’s very effective in practice.

- The loss function can be cross-entropy loss.

![Model prediction](path/to/model_prediction.png)

