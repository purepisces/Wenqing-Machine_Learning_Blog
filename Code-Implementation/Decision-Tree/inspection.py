import math
import numpy as np
import sys


# load data from tsv
def load_data(filename):
    return np.loadtxt(filename, delimiter='\t', skiprows=1)

# get the label column
def get_last_column(data):
    return data[:, -1]


# major vote algorithm
class MajorityVoteClassifier:

    def __init__(self, input_file, output_file):
        self.infile = input_file
        self.outfile = output_file
        self.train_data = load_data(infile)

    # calculate majority label
    def majority_vote(self):
        labels = get_last_column(self.train_data)
        count0 = np.sum(labels == 0)
        count1 = np.sum(labels == 1)
        return 0 if count0 > count1 else 1

    # predict train and test set separately
    def predict(self):
        majority_vote = self.majority_vote()
        predictions = [majority_vote] * len(self.train_data)
        with open(self.outfile, 'w') as file:
            for prediction in predictions:
                file.write(str(prediction) + '\n')
        return predictions

    # calculate error rate
    def error_rate(self):
        majority_vote = self.majority_vote()
        true_values = get_last_column(self.train_data)
        error_count = np.sum(true_values != majority_vote)
        return error_count / len(true_values)

    def entropy(self):
        majority_vote = self.majority_vote()
        true_values = get_last_column(self.train_data)
        probs = np.sum(true_values == majority_vote)/len(true_values)
        if probs == 0 or probs == 1:
            return 0
        result = -probs * math.log2(probs) - (1 - probs) * math.log2(1 - probs)
        return result


if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    classifier = MajorityVoteClassifier(infile, outfile)
    train_error = classifier.error_rate()
    train_entropy = classifier.entropy()
    train_predict = classifier.predict()

    with open(outfile, 'w') as f:
        f.writelines("entropy: {}\n".format(train_entropy))
        f.writelines("error: {}".format(train_error))
