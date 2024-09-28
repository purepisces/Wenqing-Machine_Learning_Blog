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

## Reference
- https://www.geeksforgeeks.org/nlp-interview-questions/

