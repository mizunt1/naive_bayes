from sklearn.datasets import load_iris
import numpy as np
from collections import defaultdict
#nbc = NBC(feature_types=['b', 'r', 'b'], num_classes=4)

class NBC:
    def __init__(self, feature_types,num_classes):
        self.feature_types = feature_types
        self.num_classes = num_classes
        self.likelihood_table = np.array(num_classes, len(feature_types))

    def apply_berboulli(classes):
        print("apply b")
    
    def applu_gauss(classes):
        print("apply g")
    def make_likelihood_table(dict_of_classes):
        """
        create a (class, features) shaped matrix of likelihoods
        """
        likelihoods=np.array(classes, features)
        classes = 0
        features = 1
        # features starts at one as dict of classes contains the first element
        # being y_train
        for feature_type in self.feature_types:
            for classes in dict_of_classes:
                # calculate for one feature the likelihoods for each class
                # then move on to the next feature
                # probabilities must be calculated separately for each class
                if feature_type == "b":
                    likelihood = apply_berboulli(classes)
                elif feature_type == "r":
                    mean, sd = apply_gauss(classes)
                else:
                    print("only B or r allowed")
                likelihoods[classes][features] = likelihood
                classes += 1
            features +=1
        return likelihoods
        
    def fit(x_train, y_train):
        """will make a matrix of size (classes, features) for the liklihoods:
        p(x=x'|features, class)
        and a vector of priors: p(c)
        each row of the matrix can be used to calculate a posterior distribution
        for a single class
        x_train: data of form (N_points,num_features)
        y_train: "answers", classes of shape (N_points)
        """ 
        self.num_points = len(y_train)
        #tudo: make function which changed input categories in to ints?
        both_data = np.concatenate(y_train, x_train, axis=1)
        both_data.view('i8,i8,i8').sort(order=['f1'], axis=0)

        num_items_in_class = defaultdict(None)
        # dictionary of class: number of data points in that class
        for classes in y_train:
            num_items_in_class[classes] += 1
        self.prior_vector = np.array(num_items_in_class.values())
        data_as_dict = defaultdict(None)
        # creating data as fictionary. Key is class,
        # value is a matrix taken from both_data. each array in dictionary 
        # will have dimensions (num_data_points_in_that_class, features +1)
        # first column is the label, i.e. the class, which should be the same
        # for each row within each object in the dictionary
        total_index = 0
        for classes, num_items in num_items_in_class:
            data_as_dict[classes] = both_data[total_index:num_items]
            total_index += num_items
        
        # count number of data points in each class 
        # ??must iterate through all data points, no other way right?
        self.make_likelihood_table(data_as_dict)
        
        return None
        
    def predict():
        return None


def main():
    iris = load_iris()
    x, y = iris['data'], iris['target']
    print(x.shape)
    print(y.shape)
    num_classes = 3
    num_features = x.shape[1]
    feature_types = ['b', 'r', 'b']
    nbc = NBC(feature_types, num_classes)
    fit(x, y)

main()

# Next to do
# check dictionary is correct
# check number of classes
# fix bugs
# apply bernoulli and gaussian
# write prediction function

