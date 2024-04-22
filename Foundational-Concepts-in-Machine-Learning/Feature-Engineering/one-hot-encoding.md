
One Hot Encoding
Overview
One hot encoding converts categorical variables into a binary matrix representation. It's particularly useful for categorical features with medium cardinality.
> Note: Cardinality itself measures the number of elements in a set, so when applied to data features, it indicates the number of unique values that a feature can take.
>

Common Problems
Memory Consumption: Can be extensive if there are many unique categories, leading to high-dimensional feature vectors.

> Note: Each number in a vector represents a dimension:
> [2] is a one-dimensional vector because it contains only one element.
> [3, 4] is a two-dimensional vector because it contains two elements.
> [3, 4, 7] is a three-dimensional vector because it contains three elements.
>
> 
Computational Efficiency: Processing large one-hot encoded vectors can be computationally expensive.

Unsuitability for NLP: Due to the large vocabulary size, it's impractical for natural language processing tasks.

Best Practices

Handling Rare Categories: Group infrequent categories into a single "Other" category.

Managing Unseen Categories: Ensure that your one hot encoding implementation can handle new, unseen categories during testing or in production.

Coding:
pandas.get_dummies
Functionality: This function converts categorical variable(s) into dummy/indicator variables. It is straightforward and quick for exploratory data analysis and smaller projects.
Limitation: pandas.get_dummies doesn't inherently "remember" the mapping from categorical values to dummy variables. This means if you encode your training data, and later receive new data for prediction (like a test set), there could be inconsistencies in the encoded columns if the new data contains categories not present in the training data.
sklearn.preprocessing.OneHotEncoder
Functionality: Part of the Scikit-learn library, OneHotEncoder is designed to be used in machine learning pipelines. It converts categorical features into a 2D array of one hot encoded vectors and is capable of handling unseen categories.
Persistence: OneHotEncoder can remember the encoding scheme by fitting the encoder to the training data. This allows it to handle new data during testing or in production environments consistently, by either ignoring unseen categories or throwing an error, depending on how you configure it.
Integration in Pipelines: It can be incorporated into a preprocessing pipeline, ensuring that all steps from encoding to model training are aligned and consistent.
Choosing Between pandas.get_dummies and OneHotEncoder
Use pandas.get_dummies when you are doing quick data transformations for analysis or visualizations where the exact alignment of encoded features between different datasets (like training and testing) is not critical.
Use OneHotEncoder when building machine learning models that need to be robust and handle new, unseen data after being deployed. It's particularly useful in production environments or when the data is split into training and testing sets, and consistency across these splits is crucial.

Dummy:
In statistics and machine learning, "dummy variable" and "indicator variable" are terms often used interchangeably to refer to variables created to represent categorical data in numeric form.

Dummy Variable
A dummy variable is a binary variable that has been created to represent a category. For each category in a categorical feature, a new binary variable is generated. These variables take the value:

1 if the observation belongs to that category
0 if it does not.

