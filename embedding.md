## Dense Vector
A dense vector refers to a vector where most of the elements are non-zero, contrasted with a “sparse vector”, where most elements are zero (or some other default value). Real-world vectors can often be somewhere in between purely dense and purely sparse.

## Embedding
An embedding is a dense vector representation of some object, such as a word or a phrase. Embeddings represent data like words, images, graphs, etc., in a continuous vector space where semantically or structurally similar items are mapped close together. Embedding values are continuous and are learned from data during model training.

### Example:
Consider three words: ‘king’, ‘queen’, and ‘man’. In a hypothetical embedding space of 2 dimensions, their embeddings could be:
- king: [1.2, 2.1]
- queen: [1.1, 1.9]
- man: [1.0, 2.0]

### Semantic Relationship:
The semantic relationship captured by embeddings can be illustrated as:
\[ \text{vector(“king”)} - \text{vector(“man”)} + \text{vector(“woman”)} \]
This might be close to the vector for ‘queen’.

## One-hot Encoding vs Word Embedding
- **One-hot Encoding**: Each word in the vocabulary is represented by a binary vector of length equal to the vocabulary size. This vector is sparse, containing a “1” at the position corresponding to the word’s index in the vocabulary and ‘0’s everywhere else.
- **Word Embedding**: Word embeddings are dense vectors of fixed size (e.g., 50, 100, 200, 300 dimensions) that represent words. They are preferable as they reduce dimensionality and capture semantic meaning, which is not possible with one-hot vectors.

## Word2Vec
Word2Vec is a method or algorithm for creating word embeddings. It uses neural networks and can be trained in two main modes: Skip-gram and Continuous Bag of Words (CBOW).

## Relationship Summary:
Embedding is a broad term encompassing various methods of representing categorical data as vectors. Specific subtypes related to natural language processing (NLP) include:
- **Word Embedding**: Representations specifically for words. Methods to generate word embeddings include:
	- Word2Vec
	- GloVe
	- FastText
- **One-Hot Embedding**: Binary, sparse representations lacking semantic context.

## Tokenization
Tokenization is the process of breaking down text into smaller units, called “tokens”. Tokenization can produce different results depending on the rules defined, such as splitting on punctuation, special characters, etc., and can be more complex in languages with more intricate sentence structures or no spaces between words.

### Example:
Given the sentence "I love ChatGPT!", tokenization would typically produce the following tokens:
\[ \text{[“I”,”love”,”ChatGPT”,”!”]} \]
