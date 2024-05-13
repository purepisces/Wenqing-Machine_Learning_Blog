
# The Transformer Architecture

The transformer architecture, introduced in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762), has revolutionized generative AI. This novel approach can be scaled efficiently using multi-core GPUs and can parallel process input data, allowing it to handle much larger training datasets. Crucially, it can learn to pay attention to the meaning of the input.

Building large language models with the transformer architecture has dramatically improved the performance of natural language tasks compared to earlier generations of RNNs, leading to a surge in generative capabilities.

## Understanding Self-Attention

The power of the transformer architecture lies in its ability to learn the relevance and context of all the words in a sentence. It applies attention weights to the relationships between words, enabling the model to understand the importance of each word in relation to every other word in the input. For example, in the sentence: 

"The teacher taught the student with the book,"

the model can learn who has the book, who could have the book, and if it's relevant to the wider context.

<img src="every_other_word.png" alt="every_other_word" width="400" height="300"/>

These attention weights are learned during LLM training, illustrated by an attention map showing the attention weights between each word and every other word.

<img src="diagram_teacher.png" alt="diagram_teacher" width="400" height="300"/> <img src="diagram_book.png" alt="diagram_book" width="400" height="300"/>

In this stylized example, you can see that the word "book" is strongly connected with or paying attention to the words "teacher" and "student." This is called self-attention, and the ability to learn attention in this way across the whole input significantly improves the model's ability to encode language.

<img src="stylized_example.png" alt="stylized_example" width="400" height="300"/>


## Transformer Architecture Overview

Here's a simplified diagram of the transformer architecture to focus on where these processes take place.

<img src="transformer.png" alt="transformer" width="400" height="300"/> <img src="simple_transformer.png" alt="simple_transformer" width="400" height="300"/>

The transformer architecture is split into two distinct parts: the encoder and the decoder. These components work in conjunction with each other and share a number of similarities. The diagram below is derived from the original "Attention is All You Need" paper.


## Tokenization and Embeddings

Machine learning models are statistical calculators that work with numbers, not words. Before passing texts into the model, you must first tokenize the words, converting them into numbers with each number representing a position in a dictionary of all possible words the model can work with. There are multiple tokenization methods, such as matching token IDs to complete words or parts of words.

<img src="complete_word.png" alt="complete_word" width="400" height="300"/> <img src="parts_of_word.png" alt="parts_of_words" width="400" height="300"/>

Once you've selected a tokenizer to train the model, you must use the same tokenizer when generating text.

Now that your input is represented as numbers, you can pass it to the embedding layer, this layer is a trainable vector embedding space, a high dimensional space where each token is represented as a vector and occpies a unique location within that space. Each token id in the vocabulary is matched to a multi-dimensional vector and the intuition is that these vectors learn to encode the meaning and context of individual tokens in the input sequence. Embedding vector spaces have been used in natural language processing for some time, previous generation language algorithms like word2vec use this concept.

Looking back at the sample sequence, you can see that in this simple case each word has been matched to a token ID and each token is mapped into a vector. In the original transformer paper, the vector size was actually 512, so much bigger than we can fit onto this image.

<img src="sample_sequence.png" alt="sample_sequence" width="400" height="300"/>

For simplicity, if you imagine a vector size of just three, you could plot the words into a three-dimensional space and see the relationships between thoese words, you can see now how you can relate words that are located close to each other in the embedding space and how you can calculate the distance between the words as an angle, which gives the model the ability to mathematically understand language.

<img src="angle_measure.png" alt="angle_measure" width="400" height="300"/>


## Positional Encoding

As you add token vectors into the base of the encoder or the decoder, you also add positional encoding. The model processes each of the input tokens in parallel, so adding this preserves information about word order, ensuring the relevance of the word position in the sentence is not lost.

<img src="positional_encoding.png" alt="positional_encoding" width="400" height="300"/> <img src="add_positional_encoding.png" alt="add_positional_encoding" width="400" height="300"/>


## Self-Attention and Multi-Head Attention

Once you've summed the input tokens and the positional encodings, you pass the resulting vectors to the self-attention layer. Here, the model analyzes the relationships between the tokens in your input sequence. As you saw earlier, this allows the model to attend to different parts of the input sequence to better capture the contextual dependencies between the words. The self-attention weights that are learned during training and stored in these layers reflect the importance of each word in that input sequence to all other words in the sequence. But this does not happen just once, the transformer architecture actually has multi-headed self-attention. This means that multiple sets of self-attention weights or heads are learned in parallel independently of each other. The number of attention heads included in the attention layer varies from model to model, but numbers in the range of 12-100 are common. The intuition here is that each self-attention head will learn a different aspect of language. For example, one head may see the relationship between the people entities in our sentence, while another head may focus on the activity of the sentence. while yet another head may focus on some other properties such as the words rhyme. It's important to note that you don't dictate ahead of time what aspects of language the attentino heads will learn. The weights of each head are randomly initialized and given sufficient traning data and time, each will learn differnet aspects of language. While some attention maps are easy to interpret others may not be.


<img src="self-attention.png" alt="self-attention" width="400" height="300"/> <img src="multi_headed_self_attention.png" alt="multi_headed_self_attention" width="400" height="300"/> <img src="learn_different.png" alt="learn_different" width="400" height="300"/>

## Feed-Forward Network and Output

Now that all of the attention weights have been applied to your input data, the output is processed through a fully connected feed-forward network. The output of this layer is a vector of logits proportional to the probability score for each and every token in the tokenizer dictionary. You can then pass these logits to a final softmax layer where they are normalized into a probability score for each word. This output includes a probability for every single word in the vocabulary, so there's likely to be thousands of scores here. One single token will have a score higher than the rest. This is the most likely predicted token. There are a number of methods that you can use to vary the final selection from this vector of probabilities.

<img src="feed_forward.png" alt="feed_forward" width="400" height="300"/> <img src="vector_probabilities.png" alt="vector_probabilities" width="400" height="300"/> <img src="final_transformers.png" alt="final_transformers" width="400" height="300"/>

## Reference:
- Generative AI with large language models coursera



