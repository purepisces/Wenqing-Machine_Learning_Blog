**NER (Named Entity Recognition)** is performed at the **token level**, not at the sentence level.

### What is NER (Named Entity Recognition)?

**Named Entity Recognition (NER)** is a sub-task of Natural Language Processing (NLP) that focuses on identifying and classifying named entities in text into predefined categories. Named entities often include:

-   **Person Names**: e.g., "Albert Einstein"
-   **Organizations**: e.g., "Google"
-   **Locations**: e.g., "New York City"
-   **Dates/Time**: e.g., "January 1, 2024"
-   **Monetary Values**: e.g., "$1000"
-   **Other Specific Entities**: e.g., "COVID-19" (medical condition), "Tesla Model 3" (product name)

### NER at the Token Level

NER involves identifying **specific entities** within a text, such as names, locations, dates, and so on. Since these entities are embedded in the sentence as individual words or phrases, the process requires labeling each token in the text.

For example, consider the sentence:

> "Albert Einstein was born in Germany."

The task of NER is to assign a label to **each token** as follows:

| Token      | NER Label |
|------------|-----------|
| Albert     | B-PER     |
| Einstein   | I-PER     |
| was        | O         |
| born       | O         |
| in         | O         |
| Germany    | B-LOC     |
| .          | O         |

Here:
- **B-PER**: Beginning of a "Person" entity.
- **I-PER**: Inside a "Person" entity (continuation of the same entity).
- **B-LOC**: Beginning of a "Location" entity.
- **O**: Outside any named entity (no label).


#### Use Cases:

1.  **Information Extraction**: Extracting key entities from documents or articles.
2.  **Search Engine Optimization**: Improving search results by understanding the entities in user queries.
3.  **Customer Support**: Automatically identifying important entities in customer complaints or requests.
4.  **Financial Analysis**: Extracting company names, stock prices, or dates from financial news.

#### How NER Works:

1.  **Tokenization**: Break the text into smaller units (words or subwords).
2.  **Feature Extraction**: Identify patterns (e.g., capitalization, context, and word position).
3.  **Classification**: Assign a label to each token (e.g., "B-PER" for the beginning of a person's name).

#### Common Algorithms for NER:

-   **Rule-Based Systems**: Using handcrafted rules to identify entities.
-   **Statistical Models**: e.g., Hidden Markov Models (HMMs), Conditional Random Fields (CRFs).
-   **Deep Learning Models**: Using architectures like BiLSTM-CRF or Transformer-based models (e.g., BERT) for high accuracy.

NER is widely used in tasks like text mining, chatbots, sentiment analysis, and other NLP applications.
