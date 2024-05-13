The transformer architecture had arrived in the paper "attention is all you need" , this novel approach unlocked the progress in generative AI that we see today, it can be scaled efficiently to use multi-core gpus,it can parallel process input data making use of much larger training datasets and crucially its able to learn to pay attention to input meaning. 


Building large language models using the transformer architecture dramatically improved the performance of natural languagle tasks over the earlier generation of RNNs and led to an explosiion in regenerative capability.

The power of the transformer architecture lies in its ability to learn the relevance and context of all of the words in a sentence. Not just as you see here to each word next to its neighbor but to every other word in a sentance and to apply attention wrights to those relationships so that the model learns the relevance of each word to each other words no matter where they are in the input. For example, for sentence: "The teacher taught the student with the book", This gives the algorithm the ability to learn who has the book, who could have the book. and if it's even relevant to the wider context of the document.

insert every_other_word.png

These attention weights are learned during llm training, the diagram is called an attention map and can be useful to illustrate the attetion weights between each word and every other word.

insert diagram_teacher.png
insert diagram_book.png



Here in this stylized example, you can see that the word book is strongly connected with or paying attention to the word teacher and the word student. This is called self attention, and the ability to learn attention in this way across the whole input significantly improves the model's ability to encode language.

insert stylized_example.png

Here's a simplified diagram of the transformer atchitecture so that you can focus at a high level on where these processes are taking place.

insert transformer.png

insert simple_transformer.png

The transformer architecture is split into two distinct parts, the encoder and the decoder. These components work in conjuction with each other and they share a number of similarities. Also, note here, the diagram you see is derived from the original attention is all you need paper.


Machine learning models are just big statistical calculators, and they work with numbers not words. so before passing texts into the model to process, you must first tokenize the words. Simply put, this converts the words into numbers with each number representing a position in a dictionary of all the possible words that the model can work with. You can choose from multiple tokenization methods. For example, token IDs matching two complete words, or using token IDs to represent parts of words.

insert complete_word.png

insert parts_of_words.png

what's important is that once you've selected a tokenizer to train the model, you must use the same tokenizer when you generate text.

Now that your input is represented as numbers, you can pass it to the embedding layer, this layer is a trainable vector embedding space, a high dimensional space where each token is represented as a vector and occpies a unique location within that space. Each token id in the vocabulary is matched to a multi-dimensional vector and the intuition is that these vectors learn to encode the meaning and context of individual tokens in the input sequence. Embedding vector spaces have been used in natural language processing for some time, previous generation language algorithms like word2vec use this concept.



Looking back at the sample sequence, you can see that in this simple case each word has been matched to a token ID and each token is mapped into a vector. In the original transformer paper, the vector size was actually 512, so much bigger than we can fit onto this image.

insert sample_sequence.png

For simplicity, if you imagine a vector size of just three, you could plot the words into a three-dimensional space and see the relationships between thoese words, you can see now how you can relate words that are located close to each other in the embedding space and how you can calculate the distance between the words as an angle, which gives the model the ability to mathematically understand language.

insert angle_measure.png

As you add the token vectors into the base of the encoder or the decoder, you also add positional encoding. The model processes each of the input tokens in parallel, so by adding the positional encoding, you preserve the information about the word order and don't lose the relevance of the position of the word in the sentence. 

insert positional_encoding.png

insert add_positional_encoding.png


Once you've summed the input tokens and the positional encodings, you pass the resulting vectors to the self-attention layer. Here, the model analyzes the relationships between the tokens in your input sequence. As you saw earlier, this allows the model to attend to different parts of the input sequence to better capture the contextual dependencies between the words. The self-attention weights that are learned during training and stored in these layers reflect the importance of each word in that input sequence to all other words in the sequence. But this does not happen just once, the transformer architecture actually has multi-headed self-attention. Thsi means that multiple sets of self-attention weights or heads are learned in parallel independently of each other. The number of attention heads included in the attention layer varies from model to model, but numbers in the range of 12-100 are common. The intuition here is that each self-attention head will learn a different aspect of language. For example, one head may see the relationship between the people entities in our sentence, while another head may focus on the activity of the sentence. while yet another head may focus on some other properties such as the words rhyme. It's important to note that you don't dictate ahead of time what aspects of language the attentino heads will learn. The weights of each head are randomly initialized and given sufficient traning data and time, each will learn differnet aspects of language. While some attention maps are easy to interpret others may not be.


now that all of the attention weights have been applied to your input data, the output is processed through a fully connected feed-forward network. The output of this layer is a vector of logits proportional to the probability score for each and every token in the tokenizer dictionary. You can then pass these logits to a final softmax layer where they are normalized into a probability score for each word. This output includes a probability for every single word in the vocabulary, so there's likely to be thousands of scores here. One single token will have a score higher than the rest. This is the most likely predicted token. There are a number of methods that you can use to vary the final selection from this vector of probabilities.


