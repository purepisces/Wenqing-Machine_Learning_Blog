import csv
import sys
import numpy as np

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################

# dataset: [(1, 'This is a good review') (0, 'This is a bad review') (1, 'Another positive review') (0, 'Another negative review')]
def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    # 'comments=None': Ensures that no comment character is treated specially.
    # By default, np.loadtxt treats the # character as a comment character. This means that any line in the file that begins with #, or any text following a # character on a line, will be ignored by np.loadtxt.
    # dtype='l,O': Specifies that the data type is a long integer for labels and an object for reviews.
    # encoding='utf-8': Ensures the file is read using UTF-8 encoding.  UTF-8 is a standard encoding for text files that supports all Unicode characters, making it widely used for internationalization and handling text in multiple languages. Unicode is a universal character encoding standard that provides a unique number for every character, no matter the platform, program, or language. 
    # dtype='l,O': Specifies that the data type is a long integer for labels and an object for reviews. Since np.loadtxt is trying to interpret all the data as numerical values by default, so it cannot convert text reviews into float.

    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset

def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

class Model:
    def __init__(self, file_in, glovevec):
        self.dataset = load_tsv_dataset(file_in)
        self.labels = []
        self.reviews = []
        for row in self.dataset:
            self.labels.append(row[0])
            self.reviews.append(row[1])
        self.glove_embeddings = glovevec

    def word_appear_count(self, review):
        word_count_dict = {}
        total_count = 0
        for word in review:
            if word not in self.glove_embeddings:
                continue
            else:
                total_count += 1
                if word not in word_count_dict:
                    word_count_dict[word] = 1
                else:
                    word_count_dict[word] += 1
        return word_count_dict, total_count

    def generate_review_embeddings(self):
        total_reviews = []
        for i in range(len(self.reviews)):
            # Initialize single_review with the label of the current review
            single_review = '{0:.6f}'.format(self.labels[i]) # 1 or 0
            review = self.reviews[i].split() #['dayjobbers', 'rejoice', 'here', 'is']
            bow_dict = self.word_appear_count(review)[0]
            length = self.word_appear_count(review)[1]
            aggregated_vector  = np.zeros((VECTOR_LEN,))
            for word, count in bow_dict.items():
                single_vec = (count / length) * self.glove_embeddings[word]
                aggregated_vector += single_vec
            # It rounds the elements of the array to 6 decimal places.
            final_vec = np.round(aggregated_vector, 6)
            for item in final_vec:
                single_review += '\t{0:.6f}'.format(item)
            single_review += '\n'
            total_reviews.append(single_review)
        return  total_reviews

if __name__ == '__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    feature_dictionary_input = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_validation_out = sys.argv[6]
    formatted_test_out = sys.argv[7]

    glovevec = load_feature_dictionary(feature_dictionary_input)
    trainModel = Model(train_input, glovevec)
    valModel = Model(validation_input, glovevec)
    testModel = Model(test_input, glovevec)
    with open(formatted_train_out, 'w') as f:
        f.writelines(trainModel.generate_review_embeddings())
    with open(formatted_validation_out, 'w') as f:
        f.writelines(valModel.generate_review_embeddings())
    with open(formatted_test_out, 'w') as f:
        f.writelines(testModel.generate_review_embeddings())
