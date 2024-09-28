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




## The Difference Between Stemming and Lemmatization

The primary difference between **stemming** and **lemmatization** is how they reduce words to their base form, as well as the level of linguistic knowledge used in each process.

### **Stemming**:
- **Definition**: Stemming is a rule-based process of reducing a word to its root form, often by chopping off prefixes or suffixes. It doesn’t necessarily produce a valid word, and it doesn’t consider the context in which the word is used.
- **Process**: Stemming uses heuristic methods to cut off word endings (e.g., "-ing," "-ed," "-s"). This often results in truncated forms of words.
- **Output**: Stems are not always valid words in the language.
- **Purpose**: Stemming is a quick and crude method, useful when performance is more important than accuracy.

**Example**:
- **Original Word**: "running"  
  **After Stemming**: `"run"`
  
- **Original Word**: "studies"  
  **After Stemming**: `"studi"`

In the second example, stemming produced `"studi"`, which is not a valid word.

### **Lemmatization**:
- **Definition**: Lemmatization reduces a word to its lemma (base or dictionary form) while maintaining its meaning within the context of the sentence. It involves more linguistic knowledge, such as part of speech, to ensure that the lemma is valid.
- **Process**: Lemmatization considers the word's part of speech (e.g., verb, noun, adjective) and requires looking up the lemma in a dictionary or morphological database.
- **Output**: Lemmas are always valid words in the language.
- **Purpose**: Lemmatization is more precise but computationally more expensive than stemming.

**Example**:
- **Original Word**: "running"  
  **After Lemmatization**: `"run"`
  
- **Original Word**: "better"  
  **After Lemmatization**: `"good"`

In this example, "better" is lemmatized to "good," as it considers context and linguistic rules.

### Key Differences:

| Aspect                | Stemming                          | Lemmatization                      |
|-----------------------|------------------------------------|------------------------------------|
| **Method**            | Rule-based, simple heuristics      | Dictionary-based, linguistic rules |
| **Output**            | Often not a valid word (e.g., "studies" -> "studi") | Always a valid word (e.g., "studies" -> "study") |
| **Context Awareness**  | Ignores context                   | Considers the context and part of speech |
| **Speed**             | Faster, less computationally expensive | Slower, more computationally intensive |
| **Use Case**          | Quick text processing, lower accuracy | Accurate text processing with proper word forms |

### Summary:
- **Stemming** is a fast, approximate method that might not return valid words (e.g., "playing" → "play" or "play" → "play"), whereas **lemmatization** returns the actual base form of a word (e.g., "running" → "run" and "better" → "good").
- **Lemmatization** is more sophisticated and requires more resources, but it ensures that the word retains its meaning and is a valid word in the language.

---

## Lemmatization: Context and Part of Speech

When we say **lemmatization considers the context and part of speech**, it means that the process analyzes the role of a word in a sentence (whether it’s a noun, verb, adjective, etc.) before reducing it to its base form, called the **lemma**. This is important because the same word can have different base forms depending on its part of speech.

### **Example 1: Word "bats"**
- **Noun**: "The bats flew across the sky."  
  **Lemmatized form**: `"bat"` (because it's a noun referring to the flying mammal).
  
- **Verb**: "He bats the ball with force."  
  **Lemmatized form**: `"bat"` (because it's a verb, meaning to hit).

Even though both instances of "bats" look identical, they have different parts of speech, and lemmatization returns the same lemma "bat," but only after understanding whether the word is a noun or a verb.

### **Example 2: Word "saw"**
- **Noun**: "The carpenter picked up the saw."  
  **Lemmatized form**: `"saw"` (because it's a noun, referring to a tool).
  
- **Verb**: "She saw the movie yesterday."  
  **Lemmatized form**: `"see"` (because it's a verb, meaning the past tense of "see").

Here, "saw" is lemmatized to "saw" when it's a noun (tool) but lemmatized to "see" when it's a verb in the past tense.

### **Example 3: Word "better"**
- **Adjective**: "He is feeling better today."  
  **Lemmatized form**: `"good"` (because "better" is the comparative form of the adjective "good").
  
- **Verb**: "She wants to better her performance."  
  **Lemmatized form**: `"better"` (because "better" here is a verb meaning to improve).

### **Why It Matters**
In each of these examples, the lemmatizer needs to know the **context** and **part of speech** of the word to reduce it to the correct lemma. Without this context, it might return the wrong base form. This makes **lemmatization** more accurate than **stemming**, which does not consider context or part of speech.

---

### **Summary**
- **Context**: Lemmatization analyzes how the word is used in the sentence.
- **Part of Speech**: Lemmatization uses the role of the word (noun, verb, adjective, etc.) to choose the correct base form.

---

## Example of Part of Speech and Context in Lemmatization:

In this example, lemmatizing the word **"saw"** as either the verb **"see"** or the noun **"saw"** involves both **part of speech** and **context**:

1. **Part of Speech**: The lemmatizer identifies whether "saw" is functioning as a noun or a verb in the sentence. This distinction is crucial because the lemma depends on whether the word is a noun (where the lemma remains "saw") or a verb (where the lemma becomes "see").

   - **Noun**: In "The carpenter picked up the saw," the word "saw" is a noun, referring to a tool.
   - **Verb**: In "She saw the movie yesterday," the word "saw" is a verb (past tense of "see").

2. **Context**: The **context** helps the lemmatizer understand how the word is being used in a specific sentence. For example, the word "saw" in the sentence "She saw the movie yesterday" is understood as the past tense of the verb "see" due to the context provided by the sentence.

### In summary:
- **Part of Speech** is the primary factor that allows the lemmatizer to determine whether "saw" should be reduced to "saw" (noun) or "see" (verb).
- **Context** (the words around "saw") helps the lemmatizer correctly assign the part of speech. For example, "picked up" suggests that "saw" is a noun, while "the movie yesterday" suggests that "saw" is a verb.

Thus, **both part of speech and context** work together to ensure accurate lemmatization.
