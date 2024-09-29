## Question 1

**What is NLP?**

NLP stands for Natural Language Processing. The subfield of Artificial intelligence and computational linguistics deals with the interaction between computers and human languages. It involves developing algorithms, models, and techniques to enable machines to understand, interpret, and generate natural languages in the same way as a human does.

NLP encompasses a wide range of tasks, including language translation, sentiment analysis, text categorization, information extraction, speech recognition, and natural language understanding. NLP allows computers to extract meaning, develop insights, and communicate with humans in a more natural and intelligent manner by processing and analyzing textual input.


## Question 2

**What are the main challenges in NLP?**

The complexity and variety of human language create numerous difficult problems for the study of Natural Language Processing (NLP). The primary challenges in NLP are as follows:

- Semantics and Meaning: It is a difficult undertaking to accurately capture the meaning of words, phrases, and sentences. The semantics of the language, including word sense disambiguation, metaphorical language, idioms, and other linguistic phenomena, must be accurately represented and understood by NLP models.
- Ambiguity: Language is ambiguous by nature, with words and phrases sometimes having several meanings depending on context. Accurately resolving this ambiguity is a major difficulty for NLP systems.
- Contextual Understanding: Context is frequently used to interpret language. For NLP models to accurately interpret and produce meaningful replies, the context must be understood and used. Contextual difficulties include, for instance, comprehending referential statements and resolving pronouns to their antecedents.
- Language Diversity: NLP must deal with the world’s wide variety of languages and dialects, each with its own distinctive linguistic traits, lexicon, and grammar. The lack of resources and knowledge of low-resource languages complicates matters.
- Data Limitations and Bias: The availability of high-quality labelled data for training NLP models can be limited, especially for specific areas or languages. Furthermore, biases in training data might impair model performance and fairness, necessitating careful consideration and mitigation.
- Real-world Understanding: NLP models often fail to understand real-world knowledge and common sense, which humans are born with. Capturing and implementing this knowledge into NLP systems is a continuous problem.


## 🌟 Question 3

**What are the different tasks in NLP?**

Natural Language Processing (NLP) includes a wide range of tasks involving understanding, processing, and creation of human language. Some of the most important tasks in NLP are as follows:

-   [Text Classification](https://www.geeksforgeeks.org/classification-of-text-documents-using-the-approach-of-naive-bayes)
-   [Named Entity Recognition (NER)](https://www.geeksforgeeks.org/python-named-entity-recognition-ner-using-spacy)
-   [Part-of-Speech Tagging (POS)](https://www.geeksforgeeks.org/nlp-part-of-speech-default-tagging)
-   [Sentiment Analysis](https://www.geeksforgeeks.org/what-is-sentiment-analysis)
-   [Language Modeling](https://www.geeksforgeeks.org/videos/what-is-language-modelling-in-nlp)
-   [Machine Translation](https://www.geeksforgeeks.org/machine-translation-of-languages-in-artificial-intelligence)
-   [Chatbots](https://www.geeksforgeeks.org/battle-of-ai-chatbots-which-chatbot-will-rule-the-present-and-future)
-   [Text Summarization](https://www.geeksforgeeks.org/python-extractive-text-summarization-using-gensim)
-   [Information Extraction](https://www.geeksforgeeks.org/difference-between-information-retrieval-and-information-extraction)
-   [Text Generation](https://www.geeksforgeeks.org/text-generation-using-recurrent-long-short-term-memory-network)
-   [Speech Recognition](https://www.geeksforgeeks.org/text-generation-using-recurrent-long-short-term-memory-network)

## Question 4

**What do you mean by Corpus in NLP?**

In NLP, a [corpus](https://www.geeksforgeeks.org/nlp-wordlist-corpus) is a huge collection of texts or documents. It is a structured dataset that acts as a sample of a specific language, domain, or issue. A corpus can include a variety of texts, including books, essays, web pages, and social media posts. Corpora are frequently developed and curated for specific research or NLP objectives. They serve as a foundation for developing language models, undertaking linguistic analysis, and gaining insights into language usage and patterns.


## Question 5

**What do you mean by text augmentation in NLP and what are the different text augmentation techniques in NLP?**

[Text augmentation](https://www.geeksforgeeks.org/text-augmentation-techniques-in-nlp) in NLP refers to the process that generates new or modified textual data from existing data in order to increase the diversity and quantity of training samples. Text augmentation techniques apply numerous alterations to the original text while keeping the underlying meaning.

Different text augmentation techniques in NLP include:

1.  ****Synonym Replacement:**** Replacing words in the text with their synonyms to introduce variation while maintaining semantic similarity.
2.  ****Random Insertion/Deletion:**** Randomly inserting or deleting words in the text to simulate noisy or incomplete data and enhance model robustness.
3.  ****Word Swapping:**** Exchanging the positions of words within a sentence to generate alternative sentence structures.
4.  ****Back translation:**** Translating the text into another language and then translating it back to the original language to introduce diverse phrasing and sentence constructions.
5.  ****Random Masking:**** Masking or replacing random words in the text with a special token, akin to the approach used in masked language models like BERT.
6.  ****Character-level Augmentation:**** Modifying individual characters in the text, such as adding noise, misspellings, or character substitutions, to simulate real-world variations.
7.  ****Text Paraphrasing:**** Rewriting sentences or phrases using different words and sentence structures while preserving the original meaning.
8.  ****Rule-based Generation:**** Applying linguistic rules to generate new data instances, such as using grammatical templates or syntactic transformations.

### Example 
**1. Synonym Replacement**

In this technique, words in a sentence are replaced with their synonyms to introduce variation while keeping the meaning intact.

**Original Sentence**:  
"The quick brown fox jumps over the lazy dog."

**Synonym Replacement**:  
"The fast brown fox leaps over the lazy dog."

**2. Random Insertion**

This method involves randomly inserting words that are semantically relevant into the sentence.

**Original Sentence**:  
"The quick brown fox jumps over the lazy dog."

**Random Insertion**:  
"The quick brown fox jumps swiftly over the lazy dog."

**3. Random Deletion**

Random words are deleted from the sentence to simulate incomplete or noisy data.

**Original Sentence**:  
"The quick brown fox jumps over the lazy dog."

**Random Deletion**:  
"Quick brown fox jumps over dog."

**4. Word Swapping**

In word swapping, words are swapped within a sentence to create different sentence structures without changing the overall meaning.

**Original Sentence**:  
"The quick brown fox jumps over the lazy dog."

**Word Swapping**:  
"Over the lazy dog, the quick brown fox jumps."

**5. Back Translation**

A sentence is translated into another language and then translated back into the original language, resulting in slightly different phrasing.

**Original Sentence (English)**:  
"The quick brown fox jumps over the lazy dog."

**Translated to French**:  
"Le renard brun rapide saute par-dessus le chien paresseux."

**Back to English**:  
"The fast brown fox leaps over the lazy dog."

 **6. Random Masking**

In this technique, random words in a sentence are replaced by a special token, similar to the technique used in models like BERT.

**Original Sentence**:  
"The quick brown fox jumps over the lazy dog."

**Random Masking**:  
"The quick [MASK] fox jumps over the [MASK] dog."

**7. Character-level Augmentation**

At the character level, random modifications are made to individual characters to simulate typos or real-world text variations.

**Original Sentence**:  
"The quick brown fox jumps over the lazy dog."

**Character-level Augmentation**:  
"The quick bronw fox jmups over the lazy dog."

**8. Text Paraphrasing**

Paraphrasing changes the sentence structure and wording while keeping the same meaning.

**Original Sentence**:  
"The quick brown fox jumps over the lazy dog."

**Paraphrased Text**:  
"A fast, brown fox leaps across the lazy dog."

**9. Rule-based Generation**

This involves applying specific linguistic rules to create new text. For example, you can change verb tenses or rearrange sentence structures.

**Original Sentence**:  
"The quick brown fox jumps over the lazy dog."

**Rule-based Generation (change tense)**:  
"The quick brown fox jumped over the lazy dog."

## Question 6

**What are some common pre-processing techniques used in NLP?**

[Natural Language Processing (NLP)](https://www.geeksforgeeks.org/natural-language-processing-nlp-pipeline) preprocessing refers to the set of processes and techniques used to prepare raw text input for analysis, modelling, or any other NLP tasks. The purpose of preprocessing is to clean and change text data so that it may be processed or analyzed later.

Preprocessing in NLP typically involves a series of steps, which may include:

-   [Tokenization](https://www.geeksforgeeks.org/tokenize-text-using-nltk-python)
-   [Stop Word Removal](https://www.geeksforgeeks.org/removing-stop-words-nltk-python)
-   [Text Normalization](https://www.geeksforgeeks.org/normalizing-textual-data-with-python)
    -   Lowercasing
    -   Lemmatization
    -   Stemming
    -   Date and Time Normalization
-   [Removal of Special Characters and Punctuation](https://www.geeksforgeeks.org/removing-punctuations-given-string)
-   [Removing HTML Tags or Markup](https://www.geeksforgeeks.org/program-to-remove-html-tags-from-a-given-string)
-   [Spell Correction](https://www.geeksforgeeks.org/correcting-words-using-nltk-in-python)
-   [Sentence Segmentation](https://www.geeksforgeeks.org/python-perform-sentence-segmentation-using-spacy)


1. **Tokenization**

Tokenization is the process of breaking down a sentence or document into individual words or phrases, known as tokens. This helps in analyzing the text at a more granular level.

**Example**:  
Original Text: "The quick brown fox jumps over the lazy dog."  
After Tokenization: `["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]`

2. **Stop Word Removal**

Stop words are common words (such as "the", "is", "in", "at") that do not carry significant meaning and are removed to focus on the important words.

**Example**:  
Original Text: "The quick brown fox jumps over the lazy dog."  
After Stop Word Removal: `["quick", "brown", "fox", "jumps", "lazy", "dog"]`

3. **Text Normalization**

Text normalization involves converting text into a standard format. This process includes several steps:

-   **Lowercasing**: Converting all text to lowercase to ensure uniformity.
    
    **Example**:  
    Original Text: "The QUICK Brown Fox."  
    After Lowercasing: `"the quick brown fox"`
    
-   **Lemmatization**: Reducing words to their base form (lemma) while maintaining the context.
    
    **Example**:  
    Original Word: "running"  
    After Lemmatization: `"run"`
    Original Word: "better"
    After Lemmatization: `"good"`
    
-   **Stemming**: Reducing words to their root form, often leading to truncation.
    
    **Example**:  
    Original Word: "running"  
    After Stemming: `"run"`
    Original Word: "studies"
    After Stemming: `"studi"`
 In the second example, stemming produced `"studi"`, which is not a valid word.
    
-   **Date and Time Normalization**: Converting dates and times into a consistent format.
    
    **Example**:  
    Original Text: "I was born on 20th June, 1995."  
    After Date Normalization: `"I was born on [DATE]."`
    

4. **Removal of Special Characters and Punctuation**

Special characters and punctuation marks are usually removed as they do not carry significant meaning in most NLP tasks.

**Example**:  
Original Text: "Hello! How are you doing today?"  
After Removal of Special Characters: `"Hello How are you doing today"`

5. **Removing HTML Tags or Markup**

When working with web-scraped data, removing HTML tags is essential to extract clean text for analysis.

**Example**:  
Original Text: "<p>This is a paragraph.</p>"  
After Removing HTML Tags: `"This is a paragraph."`

6. **Spell Correction**

This involves correcting spelling mistakes in the text.

**Example**:  
Original Text: "The quik brown fox jmps over the lazi dog."  
After Spell Correction: `"The quick brown fox jumps over the lazy dog."`

7. **Sentence Segmentation**

Sentence segmentation is the process of dividing text into individual sentences.

**Example**:  
Original Text: "The quick brown fox jumps. It is very agile."  
After Sentence Segmentation:  
`["The quick brown fox jumps.", "It is very agile."]`

## Question 7
**What is text normalization in NLP?**

Text normalization, also known as text standardization, is the process of transforming text data into a standardized or normalized form It involves applying a variety of techniques to ensure consistency, reduce variations, and simplify the representation of textual information.

The goal of text normalization is to make text more uniform and easier to process in Natural Language Processing (NLP) tasks. Some common techniques used in text normalization include:

-   ****Lowercasing****: Converting all text to lowercase to treat words with the same characters as identical and avoid duplication.
-   ****Lemmatization****: Converting words to their base or dictionary form, known as lemmas. For example, converting “running” to “run” or “better” to “good.”
-   ****Stemming****: Reducing words to their root form by removing suffixes or prefixes. For example, converting “playing” to “play” or “cats” to “cat.”
-   ****Abbreviation Expansion****: Expanding abbreviations or acronyms to their full forms. For example, converting “NLP” to “Natural Language Processing.”
-   ****Numerical Normalization****: Converting numerical digits to their written form or normalizing numerical representations. For example, converting “100” to “one hundred” or normalizing dates.
-   ****Date and Time Normalization****: Standardizing date and time formats to a consistent representation.



## Question 8

**What is tokenization in NLP?**

[Tokenization](https://www.geeksforgeeks.org/tokenization-using-spacy-library) is the process of breaking down text or string into smaller units called tokens. These tokens can be words, characters, or subwords depending on the specific applications. It is the fundamental step in many natural language processing tasks such as sentiment analysis, machine translation, and text generation. etc.

Some of the most common ways of tokenization are as follows:

-   ****Sentence tokenization:**** In Sentence tokenizations, the text is broken down into individual sentences. This is one of the fundamental steps of tokenization.
-   ****Word tokenization:**** In word tokenization, the text is simply broken down into words. This is one of the most common types of tokenization. It is typically done by splitting the text into spaces or punctuation marks.
-   ****Subword tokenization:**** In subword tokenization, the text is broken down into subwords, which are the smaller part of words. Sometimes words are formed with more than one word, for example, Subword i.e Sub+ word, Here sub, and words have different meanings. When these two words are joined together, they form the new word “subword”, which means “a smaller unit of a word”. This is often done for tasks that require an understanding of the morphology of the text, such as stemming or lemmatization.
-   ****Char-label tokenization:**** In Char-label tokenization, the text is broken down into individual characters. This is often used for tasks that require a more granular understanding of the text such as text generation, machine translations, etc.


## Question 9

**What is NLTK and How it’s helpful in NLP?**

[NLTK](https://www.geeksforgeeks.org/python-nltk-tokenize-regexp) stands for Natural Language Processing Toolkit. It is a suite of libraries and programs written in Python Language for symbolic and statistical natural language processing. It offers tokenization, stemming, lemmatization, POS tagging, Named Entity Recognization, parsing, semantic reasoning, and classification.

NLTK is a popular NLP library for Python. It is easy to use and has a wide range of features. It is also open-source, which means that it is free to use and modify.

### Key Features of NLTK:

1.  **Tokenization**: NLTK allows you to break down text into smaller units, such as sentences or words.
2.  **Stemming**: It provides stemmers like the Porter Stemmer to reduce words to their root form.
3.  **Lemmatization**: NLTK offers tools to reduce words to their dictionary form, accounting for context.
4.  **POS Tagging**: NLTK can assign part-of-speech (POS) tags to words in a sentence, such as noun, verb, adjective, etc.
5.  **Named Entity Recognition (NER)**: NLTK can recognize proper nouns and categorize them as organizations, people, or locations.
6.  **Parsing**: It allows for syntactic parsing of sentences to understand sentence structure.
7.  **Classification**: NLTK provides tools to classify text into categories (e.g., spam or non-spam).
8.  **Text Corpora**: NLTK includes many well-known corpora (collections of text data) such as the Brown Corpus, Gutenberg Corpus, and WordNet for word relations.

### Example of NLTK in Action:

#### 1. **Tokenization Example**:

```python
import nltk
nltk.download('punkt')  # Download necessary data
from nltk.tokenize import word_tokenize

text = "Natural Language Processing with NLTK is easy."
tokens = word_tokenize(text)
print(tokens)
```

**Output**:
```css
['Natural', 'Language', 'Processing', 'with', 'NLTK', 'is', 'easy', '.']
```
#### 2.**POS Tagging Example**:
```python
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag

tokens = word_tokenize("NLTK helps in processing text.")
tagged = pos_tag(tokens)
print(tagged)
```
**Output**:
```css
[('NLTK', 'NNP'), ('helps', 'VBZ'), ('in', 'IN'), ('processing', 'VBG'), ('text', 'NN'), ('.', '.')]
```
Here, NLTK has tagged each word with its corresponding part of speech (e.g., NNP for proper noun, VBZ for verb).

#### 3. **Named Entity Recognition Example**:
```python
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk import ne_chunk

sentence = "Apple is looking at buying a startup in New York."
tokens = word_tokenize(sentence)
tagged = pos_tag(tokens)
entities = ne_chunk(tagged)
print(entities)
```
**Output**:
```css
(S
  (ORGANIZATION Apple/NNP)
  is/VBZ
  looking/VBG
  at/IN
  buying/VBG
  a/DT
  startup/NN
  in/IN
  (GPE New/NNP York/NNP)
  ./.)
```
Here, NLTK identifies "Apple" as an **organization** and "New York" as a **geopolitical entity (GPE)**.

## Questoin 10

**What is stemming in NLP, and how is it different from lemmatization?**

Stemming and lemmatization are two commonly used word normalization techniques in NLP, which aim to reduce the words to their base or root word. Both have similar goals but have different approaches.

In [stemming](https://www.geeksforgeeks.org/python-stemming-words-with-nltk), the word suffixes are removed using the heuristic or pattern-based rules regardless of the context of the parts of speech. The resulting stems may not always be actual dictionary words. Stemming algorithms are generally simpler and faster compared to lemmatization, making them suitable for certain applications with time or resource constraints.

In [lemmatization](https://www.geeksforgeeks.org/python-lemmatization-with-nltk), The root form of the word known as lemma, is determined by considering the word’s context and parts of speech. It uses linguistic knowledge and databases (e.g., wordnet) to transform words into their root form. In this case, the output lemma is a valid word as per the dictionary. For example, lemmatizing “running” and “runner” would result in “run.” Lemmatization provides better interpretability and can be more accurate for tasks that require meaningful word representations.


## Question 11

**How does part-of-speech tagging work in NLP?**

[Part-of-speech tagging](https://www.geeksforgeeks.org/part-speech-tagging-stop-words-using-nltk-python) is the process of assigning a part-of-speech tag to each word in a sentence. The POS tags represent the syntactic information about the words and their roles within the sentence.

There are three main approaches for POS tagging:

-   ****Rule-based POS tagging:**** It uses a set of handcrafted rules to determine the part of speech based on morphological, syntactic, and contextual patterns for each word in a sentence. For example, words ending with ‘-ing’ are likely to be a verb.
-   ****Statistical POS tagging:**** The statistical model like Hidden Markov Model (HMMs) or Conditional Random Fields (CRFs) are trained on a large corpus of already tagged text. The model learns the probability of word sequences with their corresponding POS tags, and it can be further used for assigning each word to a most likely POS tag based on the context in which the word appears.
-   ****Neural network POS tagging:**** The neural network-based model like RNN, LSTM, Bi-directional RNN, and transformer have given promising results in POS tagging by learning the patterns and representations of words and their context.


Part-of-speech (POS) tagging is the process of labeling each word in a sentence with its corresponding part of speech, such as noun, verb, adjective, etc. This helps machines understand the syntactic role of each word in context. There are three main approaches to POS tagging: rule-based, statistical, and neural network-based methods.

### Example Sentence:

_"The quick brown fox jumps over the lazy dog."_

#### 1. **Rule-based POS Tagging:**

Rule-based POS tagging uses hand-crafted linguistic rules to assign tags based on word patterns and context. For example:

-   "The" is identified as a determiner because it commonly precedes nouns.
-   Words ending with "-ing" or "-ed" are often tagged as verbs.
-   "Quick" is tagged as an adjective because it modifies "fox."

In the sentence:

-   **The** (determiner)
-   **quick** (adjective)
-   **brown** (adjective)
-   **fox** (noun)
-   **jumps** (verb)
-   **over** (preposition)
-   **the** (determiner)
-   **lazy** (adjective)
-   **dog** (noun)

The rules here identify parts of speech based on word structure and position in the sentence.

#### 2. **Statistical POS Tagging:**

Statistical methods, such as Hidden Markov Models (HMMs) or Conditional Random Fields (CRFs), use a probabilistic model that has been trained on large datasets. The model looks at the likelihood of a word being a specific POS tag based on its context. For example, the word "fox" is more likely to be a noun if it follows an adjective like "quick."

For the sentence:

-   The statistical model might predict "jumps" is a verb because the likelihood of seeing a verb after the noun "fox" is high, based on training data.

#### 3. **Neural Network POS Tagging:**

Neural networks, especially models like Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), or Transformers, have achieved state-of-the-art results in POS tagging. These models can capture long-term dependencies in a sentence, understanding both the word itself and its context within the entire sentence.

For example, a Bi-directional LSTM might:

-   Look at the word "jumps" and analyze not just the preceding words ("fox") but also the following words ("over") to make a more accurate prediction that "jumps" is a verb.

Neural models can also handle ambiguous words better. For example, "fox" could be either a noun or a verb, but based on context, neural models predict it as a noun in this case.

### Summary of Example:

-   **Rule-based:** Relies on fixed linguistic patterns (e.g., suffixes or word order).
-   **Statistical:** Predicts based on the likelihood of word sequences from previously tagged data.
-   **Neural networks:** Understands the deeper context of a word's role in the sentence using advanced machine learning techniques.


## Question 12

**What is named entity recognition in NLP?**

[Named Entity Recognization (NER)](https://www.geeksforgeeks.org/named-entity-recognition) is a task in natural language processing that is used to identify and classify the named entity in text. Named entity refers to real-world objects or concepts, such as persons, organizations, locations, dates, etc. NER is one of the challenging tasks in NLP because there are many different types of named entities, and they can be referred to in many different ways. The goal of NER is to extract and classify these named entities in order to offer structured data about the entities referenced in a given text.

The approach followed for Named Entity Recognization (NER) is the same as the POS tagging. The data used while training in NER is tagged with persons, organizations, locations, and dates.

### Example:

For the sentence:  
_"Barack Obama was born in Hawaii on August 4, 1961, and became the President of the United States."_

NER would identify:

-   **"Barack Obama"** as a **Person**,
-   **"Hawaii"** as a **Location**,
-   **"August 4, 1961"** as a **Date**, and
-   **"President of the United States"** as an **Organization** or title.

### How NER Works:

NER uses similar techniques as Part-of-Speech (POS) tagging, but instead of tagging grammatical roles, it focuses on identifying named entities. There are three main approaches for NER:

1.  **Rule-based NER**: Uses manually crafted rules such as regular expressions or linguistic patterns to identify entities. For example, names may start with a capital letter, or dates might follow certain formats (e.g., "DD-MM-YYYY"). This approach lacks flexibility for unseen cases but works well for highly structured data.
    
2.  **Statistical NER**: Similar to POS tagging, statistical methods like Hidden Markov Models (HMMs) or Conditional Random Fields (CRFs) are trained on large datasets where entities are already labeled. The model learns the probability of a word belonging to a certain entity type based on its context and can predict entity tags in unseen text. For instance, it might learn that "Barack" followed by "Obama" is likely a person’s name.
    
3.  **Neural Network-based NER**: More recent methods use deep learning techniques like Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), or Transformer models like BERT. These models can capture complex patterns and long-range dependencies in the text. For example, a neural network model could use the broader context of "President" to correctly identify "United States" as a location or entity even if it's used in a less obvious form.


## Question 13
**What is parsing in NLP?**

In NLP, [parsing](https://www.geeksforgeeks.org/difference-between-top-down-parsing-and-bottom-up-parsing) is defined as the process of determining the underlying structure of a sentence by breaking it down into constituent parts and determining the syntactic relationships between them according to formal grammar rules. The purpose of parsing is to understand the syntactic structure of a sentence, which allows for deeper learning of its meaning and encourages different downstream NLP tasks such as semantic analysis, information extraction, question answering, and machine translation. it is also known as syntax analysis or syntactic parsing.

The formal grammar rules used in parsing are typically based on Chomsky’s hierarchy. The simplest grammar in the Chomsky hierarchy is regular grammar, which can be used to describe the syntax of simple sentences. More complex grammar, such as context-free grammar and context-sensitive grammar, can be used to describe the syntax of more complex sentences.


### 14. What are the different types of parsing in NLP?

In natural language processing (NLP), there are several types of parsing algorithms used to analyze the grammatical structure of sentences. Here are some of the main types of parsing algorithms:

-   [****Constituency Parsing****](https://www.geeksforgeeks.org/constituency-parsing-and-dependency-parsing): Constituency parsing in NLP tries to figure out a sentence’s hierarchical structure by breaking it into constituents based on a particular grammar. It generates valid constituent structures using context-free grammar. The parse tree that results represents the structure of the sentence, with the root node representing the complete sentence and internal nodes representing phrases. Constituency parsing techniques like as CKY, Earley, and chart parsing are often used for parsing. This approach is appropriate for tasks that need a thorough comprehension of sentence structure, such as semantic analysis and machine translation. When a complete understanding of sentence structure is required, constituency parsing, a classic parsing approach, is applied.
-   [****Dependency Parsing****](https://www.geeksforgeeks.org/constituency-parsing-and-dependency-parsing)****:**** In NLP, dependency parsing identifies grammatical relationships between words in a sentence. It represents the sentence as a directed graph, with dependencies shown as labelled arcs. The graph emphasises subject-verb, noun-modifier, and object-preposition relationships. The head of a dependence governs the syntactic properties of another word. Dependency parsing, as opposed to constituency parsing, is helpful for languages with flexible word order. It allows for the explicit illustration of word-to-word relationships, resulting in a clear representation of grammatical structure.
-   [****Top-down parsing:****](https://www.geeksforgeeks.org/difference-between-top-down-parsing-and-bottom-up-parsing) Top-down parsing starts at the root of the parse tree and iteratively breaks down the sentence into smaller and smaller parts until it reaches the leaves. This is a more natural technique for parsing sentences. However, because it requires a more complicated language, it may be more difficult to implement.
-   [****Bottom-up parsing:****](https://www.geeksforgeeks.org/difference-between-top-down-parsing-and-bottom-up-parsing) Bottom-up parsing starts with the leaves of the parse tree and recursively builds up the tree from smaller and smaller constituents until it reaches the root. Although this method of parsing requires simpler grammar, it is frequently simpler to implement, even when it is less understandable.


### Example:

Consider the sentence:  
_"The quick brown fox jumps over the lazy dog."_

-   **Parsing** this sentence involves identifying the syntactic components (constituents), such as the **subject** ("The quick brown fox"), **verb** ("jumps"), and **prepositional phrase** ("over the lazy dog").
-   The parse tree would show the hierarchical structure where "The quick brown fox" is the **noun phrase**, "jumps" is the **verb**, and "over the lazy dog" is a **prepositional phrase** that modifies the verb.

### Types of Parsing in NLP

#### 1. **Constituency Parsing**:

Constituency parsing focuses on dividing a sentence into **constituents** (phrases or sub-phrases) based on a formal grammar, often using context-free grammar (CFG). The goal is to build a **parse tree** that represents the sentence structure with nodes for each phrase and subphrase.

-   **Example:** For the sentence _"The cat sits on the mat,"_ constituency parsing breaks it down into:
    -   "The cat" (Noun Phrase)
    -   "sits" (Verb Phrase)
    -   "on the mat" (Prepositional Phrase)

The parse tree would start from the sentence at the root, with branches representing the **Noun Phrase (NP)** and **Verb Phrase (VP)**.

#### 2. **Dependency Parsing**:

Dependency parsing analyzes the **dependencies** between words in a sentence. It represents the sentence as a **directed graph**, where the nodes are words and the edges represent grammatical relationships, such as **subject-verb** or **verb-object**.

-   **Example:** For the sentence _"The cat chased the mouse,"_ dependency parsing would show:
    -   "cat" as the **subject** of the verb "chased"
    -   "mouse" as the **object** of the verb "chased"

The dependency graph shows how each word depends on others (e.g., the verb "chased" governs both "cat" and "mouse").

#### 3. **Top-down Parsing**:

In top-down parsing, the parsing process starts at the **root of the parse tree** (representing the whole sentence) and progressively breaks the sentence into smaller parts (phrases or words) according to grammar rules, moving down toward the leaves (individual words).

-   **Example:** For the sentence _"She ate an apple,"_ the top-down parser would start by dividing it into a **Noun Phrase (She)** and **Verb Phrase (ate an apple)**. It would then further divide the verb phrase into a verb ("ate") and an object ("an apple").

#### 4. **Bottom-up Parsing**:

In bottom-up parsing, the parser starts at the **leaves** (the individual words of the sentence) and attempts to combine them into larger constituents until the entire sentence is covered and the root of the tree is formed.

-   **Example:** For the sentence _"They watch movies,"_ the bottom-up parser would start with the individual words:
    -   "They" (Pronoun)
    -   "watch" (Verb)
    -   "movies" (Noun)

It then gradually combines these into phrases (e.g., "watch movies" as a verb phrase), eventually reaching the full parse tree that represents the sentence.

### Comparing Top-down and Bottom-up Parsing:

-   **Top-down parsing** breaks the sentence from the root into smaller components, which makes it more intuitive but can struggle with ambiguous or complex grammars.
-   **Bottom-up parsing** builds the tree from individual words, making it easier to implement but potentially less intuitive when trying to understand the entire sentence structure.

## Reference
- https://www.geeksforgeeks.org/nlp-interview-questions/

