The transformer architecture had arrived in the paper "attention is all you need" , this novel approach unlocked the progress in generative AI that we see today, it can be scaled efficiently to use multi-core gpus,it can parallel process input data making use of much larger training datasets and crucially its able to learn to pay attention to input meaning. 


Building large language models using the transformer architecture dramatically improved the performance of natural languagle tasks over the earlier generation of RNNs and led to an explosiion in regenerative capability.

The power of the transformer architecture lies in its ability to learn the relevance and context of all of the words in a sentence. Not just as you see here to each word next to its neighbor but to every other word in a sentance and to apply attention wrights to those relationships so that the model learns the relevance of each word to each other words no matter where they are in the input. For example, for sentence: "The teacher taught the student with the book", This gives the algorithm the ability to learn who has the book, who could have the book. and if it's even relevant to the wider context of the document.

insert every_other_word.png

