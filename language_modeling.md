self.batch_size * self.sequence_length is the total number of elements (words, tokens, etc.) in each batch of the dataset. 
self.batch_size = number of sequences per batch
self.sequence_length = length of each sequence
For example 
Batch 1:
Sequence 1: [word1, word2, word3, word4, word5, word6]
Sequence 2: [word7, word8, word9, word10, word11, word12]
Sequence 3: [word13, word14, word15, word16, word17, word18]
Sequence 4: [word19, word20, word21, word22, word23, word24]
