# Dense vector

A dense vector refers to a vector where most of the elements are non-zero. This is in contrast to a “sparse vector”, where most of the elements are zero (or some other default value).

# Embedding

An embedding is a dense vector representation of some object, such as a word or a phrase. The general idea of embedding is to represent some form of data (words, images, graphs, etc.), in a continuous vector space where semantically or structurally similar items are close together.

Consider three words: ‘king’, ‘queen’, and ‘man’. In a hypothetical embedding space of 2 dimensions, their embeddings could be:
- king: [1.2, 2.1]
- queen: [1.1, 1.9]
- man: [1.0, 2.0]

These are just illustrative numbers, but in practice, embeddings capture relationships such that operations like `vector(“king”) - vector(“man”) + vector(“woman”)` might be close to the vector for ‘queen’, capturing a semantic relationship.

# One-hot encoding vs Word Embedding

- **One-hot encoding**: In one-hot encoding, each word in the vocabulary is represented by a vector of length equal to the vocabulary size. This vector contains a “1” at the position corresponding to the word’s index in the vocabulary and ‘0’s everywhere else. (Sparse: most elements in the vector are zeros.)
  
- **Word Embedding**: Word embeddings are dense vectors of fixed size (e.g., 50, 100, 200, 300 dimensions) that represent words. These vectors are learned such that they capture semantic relationships between words. (Dense: Most elements in the vector carry some value).

# Word2vec

Word2vec is a method or algorithm for creating word embeddings.

**Relationship Summary**:
Embedding is a broad term that encompasses various ways of representing categorical data as vectors. Word embeddings and one-hot embeddings are common types related to natural language processing (NLP).
- **Word Embedding** is a subtype of embeddings specifically for words. Methods to generate word embeddings include:
  - Word2Vec
  - GloVe
  - FastText
  
- **One-Hot Embedding** is another subtype of embeddings for words, but it's binary, sparse, and lacks semantic context.

# Tokenize

Tokenize is the process of breaking down a piece of text into smaller units, which are referred to as “tokens”. For example, given the sentence ”I love ChatGPT!”, tokenization would typically produce the following tokens: [“I”, ”love”, ”ChatGPT”, ”!”].
