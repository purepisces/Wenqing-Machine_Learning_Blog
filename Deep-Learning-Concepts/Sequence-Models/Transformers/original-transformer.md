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


Machine learning models are just big statistical calculators, and they work with numbers not words. so before passing texts into the model to process, you must first tokenize the words. Simply put, this converts the words into numbers with each number representing a position in a dictionary of all the possible words that the model can work with. You can choose from multiple tokenization methods. For example, token IDs matching two complete words 


