## Dense Vector and Sparse Vector

- **Dense Vector**: A vector where most of the elements are non-zero. 
- **Sparse Vector**: In contrast to a dense vector, a sparse vector is where most of the elements are zero (or some other default value). In practice, most real-world vectors can be somewhere between purely dense and purely sparse.

## Embedding

Embedding is a broad term that encompasses various ways of representing categorical data as continuous vectors. 

- **Word Embedding**: A dense vector representation of words. Semantically or structurally similar words are represented close together in the vector space. Popular methods for generating word embeddings include:
    * Word2Vec
    * GloVe
    * FastText
- **One-Hot Encoding**: Represents each word in the vocabulary as a vector where the position corresponding to the word's index is marked as "1", and all other positions are "0". This representation is binary, sparse, and lacks semantic context.

## Word2Vec

Word2Vec is a method for creating word embeddings using neural networks. It can be trained in two main modes:
    * Skip-gram
    * Continuous Bag of Words (CBOW)

### Semantic Relationships in Embeddings

A classic illustration of the semantic relationships that embeddings can capture is:
`vector(“king”) - vector(“man”) + vector(“woman”)` might be close to the vector for 'queen'. However, capturing such relationships may require specific training data or fine-tuning.

## Tokenization

Tokenization is the process of breaking down a piece of text into smaller units, known as "tokens". For example, given the sentence "I love ChatGPT!", tokenization would typically produce: 
\[“I”,”love”,”ChatGPT”,”!”\].
Note: Tokenization complexity can vary based on the language and whether punctuation or special characters are considered.

