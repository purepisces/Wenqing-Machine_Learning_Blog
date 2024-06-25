import numpy as np
import csv
import math
import sys

# Firstly, deal with data
# load data from tsv
# [[1,1,0,0,1],[1,1,1,0,1],...]
# Loads data from a TSV file and returns it as a numpy array.
def load_data(filename):
    return np.loadtxt(filename, delimiter='\t', skiprows=1)

# get the label column
# [1,1,0,0,1,0]
# label_cnt: {0: 23, 1: 119}
def label_data(dataset):
    label_column = []
    for row in dataset:
        label_column.append(row[-1])
    count0 = label_column.count(0)
    count1 = label_column.count(1)
    label_cnt = {0: count0, 1: count1}
    return label_column, label_cnt

# all_attributes:[chest_pain, thelassemia,...]
# attr_info: each dict is each row in the dataset [{sex:1,chest_pain:1,...},{sex:0,chest_pain:1,...},...]
def attr_data(input_file):
    with open(input_file, "r") as newfile:
        # Convert entire TSV file as a list of lists, where each inner list represents a row in the file.
        tsv_file = list(csv.reader(newfile, delimiter="\t"))
        # [:-1] slices the list to exclude the last label column
        all_attributes = tsv_file[0][:-1]
    attr_info = []
    #load the dataset as a numpy array.
    dataset = load_data(input_file)
    for data in dataset:
        # print(data) #[1. 1. 0. 0. 0. 0. 0. 1. 0.]
        attr_name_attr_value = {}
        for i in range(len(all_attributes)):
            attr_name_attr_value[all_attributes[i]] = data[i]
        #print(attr_name_attr_value) # {'sex': 1.0, 'chest_pain': 1.0, 'high_blood_sugar': 0.0, 'abnormal_ecg': 0.0, 'angina': 0.0, 'flat_ST': 0.0, 'fluoroscopy': 0.0, 'thalassemia': 1.0}
        attr_info.append(attr_name_attr_value)
    return all_attributes, attr_info

# total_num: length of dataset, also lenght of attr_value_label
# attr_label_dict: attr's is the key, and value is label count {1.0: {0.0: 5, 1.0: 4}, 0.0: {0.0: 37, 1.0: 12}}
def attr_label_data(dataset, col_index):
    #attr_value_label: Extract Attribute Values and Labels: [[1. 1.], [1. 1.],[0. 1.],...]
    attr_value_label = dataset[:, [col_index, -1]]
    total_num = len(dataset)
    attr_label_dict = {}
    for item in attr_value_label:
        attr_value = item[0]
        label = item[1]
        if attr_value not in attr_label_dict.keys():
            attr_label_dict[attr_value] = {}
        if label not in attr_label_dict[attr_value].keys():
            attr_label_dict[attr_value][label] = 1
        else:
            attr_label_dict[attr_value][label] += 1
    return total_num, attr_label_dict


class Node:
    """
    Here is an arbitrary Node class that will form the basis of your decision
    tree.
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired
    """

    def __init__(self, training_data, used_attrs, depth, msg=""):
        self.left = None
        self.right = None
        self.attr = None #The attribute used to split the data at this node.
        self.vote = None #The majority class vote at this node.

        self.training_data = training_data
        self.used_attrs = used_attrs #List of attributes that have been used for splitting up to this point.
        self.depth = depth
        self.msg = msg #A message for printing the tree.
        self.majority_value = self.majority_vote()

    def majority_vote(self):
        label_cnt = label_data(self.training_data)[1] # {0: 23, 1: 119}
        count0 = label_cnt[0]
        count1 = label_cnt[1]
        if count0 > count1:
            return 0
        return 1

    def entropy(self):
        majority_value = self.majority_vote()
        true_values = label_data(self.training_data)[0] #label_column
        probs = true_values.count(majority_value) / len(true_values)
        if probs == 0 or probs == 1:
            result = 0
        else:
            result = -probs * math.log2(probs) - (1 - probs) * math.log2(1 - probs)
        return result

    # attr_label_dict is, {0.0: {1.0: 46, 0.0: 2}, 1.0: {1.0: 10}}
    # attr_value is, 0.0
    # attr_label_dict[attr_value] is, {1.0: 46, 0.0: 2}
    # value is,  46
    # value is,  2
    # separate_sum is 48
    # total_num is 58

    # attr_value is, 1.0
    # attr_label_dict[attr_value] is, {1.0: 10}
    # value is,  10
    # separate_sum is 10
    # total_num is 58
    def conditional_entropy(self, total_num, attr_label_dict):  # {sunny: {1.0: 15, 0.0: 2}, rain: {1.0: 41}}
        result = 0
        # print("attr_label_dict is,", attr_label_dict) {0.0: {1.0: 46, 0.0: 2}, 1.0: {1.0: 10}}
        for attr_value in attr_label_dict.keys():
            # print("attr_value is,",attr_value) #attr_value is, 0.0 second loop is 1.0
            separate_entropy = 0
            # print("attr_label_dict[attr_value] is,",attr_label_dict[attr_value]) #{1.0: 46, 0.0: 2}
            separate_sum = sum(attr_label_dict[attr_value].values())
            for value in attr_label_dict[attr_value].values():
                # print("value is, ", value) # 46 second 2 ... second loop is 10
                separate_entropy += -(value / separate_sum) * math.log2(value / separate_sum)
            # print("separate_sum is", separate_sum) #48 second 10
            # print("total_num is", total_num) #58
            prob = separate_sum / total_num
            result += prob * separate_entropy
        return result

class DecisionTree:
    def __init__(self, training_data_file, max_depth, all_attrs):
        self.training_data = load_data(training_data_file)
        self.max_depth = max_depth
        self.all_attrs = all_attrs
        self.root = Node(self.training_data, [], 0)

    def train(self, node):
        if len(node.training_data) == 0:
            return
        if node.depth == self.max_depth:
            return
        if len(self.all_attrs) == len(node.used_attrs):
            return
        if node.entropy() == 0:
            return
        label_entropy = node.entropy()
        max_mutual = -1
        attr_index = -1
        for i in range(len(self.all_attrs)):
            if self.all_attrs[i] not in node.used_attrs:
                num = attr_label_data(node.training_data, i)[0] #length of dataset, also lenght of attr_value_label
                dict = attr_label_data(node.training_data, i)[1] # attr_label_dict: attr's is the key, and value is label count {1.0: {0.0: 5, 1.0: 4}, 0.0: {0.0: 37, 1.0: 12}}
                condition_entropy = node.conditional_entropy(num, dict)
                #Determine the information gain (label entropy - conditional entropy).
                #Update attr_index and max_mutual if the current attribute provides a higher information gain.
                if label_entropy - condition_entropy > max_mutual:
                    attr_index = i
                    max_mutual = label_entropy - condition_entropy

        node.attr = self.all_attrs[attr_index]
        #Create two subsets of the data: set0 for instances where the best attribute is 0 and set1 for instances where the best attribute is 1.
        set0 = []
        set1 = []
        for data in node.training_data:
            if data[attr_index] == 0:
                set0.append(data)
            else:
                set1.append(data)

        new_depth = node.depth + 1
        node.left = Node(np.array(set0), node.used_attrs + [node.attr], new_depth, "{} = {}: ".format(node.attr, 0))
        node.right = Node(np.array(set1), node.used_attrs + [node.attr], new_depth, "{} = {}: ".format(node.attr, 1))

        self.train(node.left)
        self.train(node.right)
        return

    def print(self, node):
        label_cnt = label_data(node.training_data)[1]
        print("{}{}".format("| " * node.depth + node.msg, label_cnt))
        if node.left is not None:
            self.print(node.left)
        if node.right is not None:
            self.print(node.right)
        return

    def predict(self, node, single_attr_info):
        node.vote = node.majority_value
        #If node.attr is None, it means that the node is a leaf node.
        # print("node.attr is", node.attr)
        # print("node.vote is", node.vote)
        if node.attr is None:
            return node.vote
        #The method retrieves the value of the attribute at node.attr from the input instance single_attr_info.
        attr_value = single_attr_info[node.attr]
        if attr_value == 0:
            return self.predict(node.left, single_attr_info)
        if attr_value == 1:
            return self.predict(node.right, single_attr_info)

    def output_predict(self, input_file, output_file):
        dataset = load_data(input_file)
        outfile = open(output_file, "w")
        count = 0
        attr_info = attr_data(input_file)[1] #all_attributes:[chest_pain, thelassemia,...]
        # attr_info: each dict is each row in the dataset [{sex:1,chest_pain:1,...},{sex:0,chest_pain:1,...},...]
        label_col = label_data(dataset)[0]
        pred_col = []
        for single_attr_info in attr_info:
            pred_col.append(self.predict(root, single_attr_info))
        for i in range(len(pred_col)):
            if pred_col[i] != label_col[i]:
                count += 1
            outfile.write("{}\n".format(pred_col[i]))
        outfile.close()
        error_rate = count / len(label_col)
        return error_rate


if __name__ == "__main__":
    train_infile = sys.argv[1]
    test_infile = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_outfile = sys.argv[4]
    test_outfile = sys.argv[5]
    metrics_out = sys.argv[6]
    all_attrs = attr_data(train_infile)[0]
    attr_info = attr_data(train_infile)[1]
    tree = DecisionTree(train_infile, max_depth, all_attrs)
    root = tree.root
    tree.train(root)
    tree.print(root)
    train_error = tree.output_predict(train_infile, train_outfile)
    test_error = tree.output_predict(test_infile, test_outfile)
    file = open(metrics_out, "w")
    file.write("error(train): {}\n".format(train_error))
    file.write("error(test): {}\n".format(test_error))
    file.close()
