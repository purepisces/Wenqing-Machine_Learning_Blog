## `Dictionary` and `Corpus` for Penn Treebank Dataset

### Overview

In word-level language modeling tasks, the goal is to predict the probability of the next word in a sequence, given the words that have already been observed. The Penn Treebank dataset, consisting of stories from the Wall Street Journal, is commonly used for training and evaluating language models on word-level prediction.

We will implement the following classes:

1.  **`Dictionary`**: A class that creates a word-to-index mapping.
2.  **`Corpus`**: A class that reads text data (e.g., train and test files) and converts it into sequences of word indices using the `Dictionary`.

### Code Implementation

Below are the two main classes: `Dictionary` and `Corpus`.


```python
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
```
```python
class Corpus(object):
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        with open(path, 'r') as f:
            ids = []
            line_idx = 0
            for line in f:
                if max_lines is not None and line_idx >= max_lines:
                    break
                words = line.split() + ['<eos>']
                for word in words:
                    ids.append(self.dictionary.add_word(word))
                line_idx += 1
        return ids
```
----------

### Class: `Dictionary`

The `Dictionary` class is responsible for creating a word-to-index mapping. This is crucial because language models require words to be converted into numeric indices before they can be processed.

#### `__init__` Method
```python
def __init__(self):
    self.word2idx = {}
    self.idx2word = []
```
-   **`self.word2idx`**: A dictionary where each word is mapped to a unique index.
-   **`self.idx2word`**: A list that stores words, where each position corresponds to the index of the word.

Example:

-   Before any words are added:
```python
word2idx = {}
idx2word = []
```
`add_word` Method
```python
def add_word(self, word):
    if self.word2idx.get(word) is None:
        self.word2idx[word] = len(self.idx2word)
        self.idx2word.append(word)
    return self.word2idx[word]
```

-   **If the word doesn't exist in `word2idx`**: It assigns the next available index to the word (the current length of `idx2word`), then appends the word to `idx2word`.
-   **If the word exists**: It simply returns the existing index.

Example:

-   Adding the word `"cat"`:
```python
word2idx = {"cat": 0}
idx2word = ["cat"]
```
   Calling `add_word("cat")` returns `0`.
    
-   Adding the word `"dog"`:
```python
word2idx = {"cat": 0, "dog": 1}
idx2word = ["cat", "dog"]
```

Calling `add_word("dog")` returns `1`.
    

#### `__len__` Method
```python
def __len__(self):
    return len(self.idx2word)
```
-   Returns the number of unique words in the dictionary (the length of `idx2word`).

Example:

-   After adding `"cat"` and `"dog"`
```python
len(dictionary)  # Returns 2
```

### Class: `Corpus`

The `Corpus` class processes text data and converts it into sequences of word indices using the `Dictionary` class.

#### `__init__` Method
```python
def __init__(self, base_dir, max_lines=None):
    self.dictionary = Dictionary()
    self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
    self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)
```

-   **`self.dictionary = Dictionary()`**: Initializes a `Dictionary` object.
-   **`self.train` and `self.test`**: Calls the `tokenize` function to process the `train.txt` and `test.txt` files and convert them into word indices.

#### `tokenize` Method
```python
def tokenize(self, path, max_lines=None):
    with open(path, 'r') as f:
        ids = []
        line_idx = 0
        for line in f:
            if max_lines is not None and line_idx >= max_lines:
                break
            words = line.split() + ['<eos>']
            for word in words:
                ids.append(self.dictionary.add_word(word))
            line_idx += 1
    return ids
```
-   **`path`**: The file path for the text data (e.g., `train.txt` or `test.txt`).
-   **`ids = []`**: Initializes an empty list to store the word indices.
-   **`words = line.split() + ['<eos>']`**: Splits each line of text into words and appends an end-of-sentence (`<eos>`) token.
-   **`self.dictionary.add_word(word)`**: Converts each word into its corresponding index using the `Dictionary`.
-   **Returns**: A list of word indices for the entire file.

----------

### Example: Tokenizing `train.txt`

Consider the following example for `train.txt`:
```python
the cat sat on the mat
the dog barked loudly
```
1. **After tokenization**:
```python
ids = [0, 1, 2, 3, 0, 4, '<eos>_idx', 0, 5, 6, 7, '<eos>_idx']
```

2.  **Explanation**:
    -   `'the' = 0`
    -   `'cat' = 1`
    -   `'sat' = 2`
    -   `'on' = 3`
    -   `'mat' = 4`
    -   `'dog' = 5`
    -   `'barked' = 6`
    -   `'loudly' = 7`
    -   `'<eos>'` is the end-of-sentence token.

----------

### Summary

The `Dictionary` class creates a mapping from words to unique indices, and the `Corpus` class tokenizes text files, converting words into their corresponding indices using the dictionary. These classes are foundational for building a word-level language model on the Penn Treebank dataset.
