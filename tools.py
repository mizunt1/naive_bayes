from sklearn.datasets import load_iris
import numpy as np
from collections import defaultdict
#nbc = NBC(feature_types=['b', 'r', 'b'], num_classes=4)

def apply_bernoulli(data, num_classes):
    """
    bernoulli with laplace smoothing.
    add one to each numerator, add num_classes to each denom
    """
    num_ones = (data == 1).sum()
    total = len(data)
    return ((num_ones + 1 / total+num_classes), None)
    
def apply_gauss(data):
    """
    TODO: Must make mean non zero
    """
    mean = np.mean(data)
    sd = np.std(data)
    return (mean, sd)

class NBC:
    def __init__(self, feature_types, num_classes):
        self.feature_types = feature_types
        self.num_features = len(feature_types)
        self.num_classes = num_classes
        self.num_data = None

    def make_likelihood_table(self, dict_of_classes):
        """
        create a (class, features) shaped matrix of likelihoods
        """
        likelihood_table = np.zeros((self.num_classes, self.num_features), dtype=(float,2))
        feature = 0
        # features starts at one as dict of classes contains the first element
        # being y_train
        print("dict", dict_of_classes.keys())
        for feature_type in self.feature_types:
            classes_counter = 0
            for classes in dict_of_classes.keys():
                # calculate for one feature the likelihoods for each class
                # then move on to the next feature
                # probabilities must be calculated separately for each class
                class_is = dict_of_classes[classes]
                # this is data for a single class.
                # each column is a feature
                if feature_type == "b":
                    likelihood = apply_bernoulli(class_is[:, feature+1], self.num_classes)
                elif feature_type == "r":
                    likelihood = apply_gauss(class_is[:, feature+1])
                    
                else:
                    print("only B or r allowed")
                likelihood_table[classes_counter][feature] = likelihood
                classes_counter += 1
            feature +=1
        return likelihood_table
        
    def fit(self, x_train, y_train):
        """will make a matrix of size (classes, features) for the liklihoods:
        p(x=x'|features, class)
        and a vector of priors: p(c)
        each row of the matrix can be used to calculate a posterior distribution
        for a single class
        x_train: data of form (N_points,num_features)
        y_train: "answers", classes of shape (N_points)
        outputs: likelihood table of form (num_classes, features)
                 priors which show the number of items in each class (num_classes)
        """ 
        self.num_points = len(y_train)
        #tudo: make function which changed input categories in to ints?
        both_data = np.concatenate((y_train.reshape(len(y_train), 1), x_train), axis=1)
        both_data[both_data[:,1].argsort()]
        num_items_in_class = defaultdict(int)
        # dictionary of class: number of data points in that class
        for classes in y_train:
            num_items_in_class[classes] += 1
        prior_vector = np.asarray(list(num_items_in_class.values()))
        data_as_dict = defaultdict(None)
        # creating data as fictionary. Key is class,
        # value is a matrix taken from both_data. each array in dictionary 
        # will have dimensions (num_data_points_in_that_class, features +1)
        # first column is the label, i.e. the class, which should be the same
        # for each row within each object in the dictionary
        total_index = 0
        for classes, num_items in num_items_in_class.items():
            data_as_dict[classes] = both_data[total_index:total_index+num_items, :]
            total_index += num_items
        # count number of data points in each class 
        # ??must iterate through all data points, no other way right?
        likelihood_table = self.make_likelihood_table(data_as_dict)
        return likelihood_table, prior_vector
        
        
    def predict(self, in_data, likelihood_table, prior_vector):
        
        def f(x, num_data):
            # return math.sqrt(x)
            x* np.log(x/num_data)

        vf = np.vectorize(f)

        log_likelihood = np.log(likelihood_table)
        prior_vector = vf(prior_vector)
        
        print(type(likelihood_table))
        print(likelihood_table.shape)
        return None


def main():
    iris = load_iris()
    x, y = iris['data'], iris['target']
    y.reshape(150,1)

    num_classes = 3
    num_features = x.shape[1]
    feature_types = ['b', 'r', 'b', 'r']
    nbc = NBC(feature_types, num_classes)
    likelihood_table, prior_vector = nbc.fit(x, y)
    
    print(likelihood_table)
    print(prior_vector)
    nbc.predict(x[0], likelihood_table, prior_vector)
    
main()

# Next to do
# smoothing
# figure out bernoulli


