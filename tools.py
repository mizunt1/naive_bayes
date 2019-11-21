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

def columns_of_br(features):
    columns_of_bin = []
    columns_of_r = []
    i = 0
    for f in features:
        if f == 'b':
            columns_of_bin.append(i)
        else:
            columns_of_r.append(i)
        i+=1
    return np.array(columns_of_bin), np.array(columns_of_r)

def sep_data(data, columns_of_bin, columns_of_r):
    binary = np.delete(data, [columns_of_r], 1)
    real = np.delete(data, [columns_of_bin], 1)
    return binary, real

class NBC:
    def __init__(self, feature_types, num_classes):
        self.feature_types = feature_types
        self.num_features = len(feature_types)
        self.num_classes = num_classes
        self.num_data = None

    def make_likelihood_table(self, data_3d_b, data_3d_r, b, r):
        """
        input data is data_3d_b: this is data[classes][num_datapoints][num_binary features]
        b is list of indexes of where binary features are in the data
        create a (class, features) shaped matrix of likelihoods
        two of them, depending on binary or real
        """
        num_bin_features = len(b)
        num_r_features = len(r)
        likelihood_table_bin = np.zeros(
            (self.num_classes, self.num_bin_features), dtype=(float))
        likelihood_table_r = np.zeros(
            (self.num_classes, self.num_r_features), dtype=(float,2))       
        for classes in data_3d_b:
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
        data = np.delete(both_data, 0, 1)
        num_items_in_class = defaultdict(int)
        # dictionary of class: number of data points in that class
        for classes in y_train:
            num_items_in_class[classes] += 1
        prior_vector = np.asarray(list(num_items_in_class.values()))
        b, r = columns_of_br(self.feature_types)
        binary_data, real_data = sep_data(data, b, r)
        data_3d_b = np.array((self.num_classes, self.num_data, len(b)))
        data_3d_r = np.array((self.num_classes, self.num_data, len(r)))
        # creating data as fictionary. Key is class,
        # value is a matrix taken from both_data. each array in dictionary 
        # will have dimensions (num_data_points_in_that_class, features +1)
        # first column is the label, i.e. the class, which should be the same
        # for each row within each object in the dictionary
        total_index = 0
        for classes, num_items in num_items_in_class.items():
            data_3d_b[classes] = binary_data[total_index:total_index+num_items, :]
            data_3d_r[classes] = real_data[total_index:total_index+num_items, :]
            total_index += num_items
        ####### TODO CHECK ABOVE IS WORKING ############ 
        # count number of data points in each class 
        #print(data)
        #print(data_3d_b[0])
        #print(data_3d_r[0])
        
        # ??must iterate through all data points, no other way right?
        print(data_3d_b.shape)
        print(data_3d_b[0].shape)
        print(data_3d_b[0][0].shape)
        likelihood_table_b = self.make_likelihood_table(data_3d_b)
        likelihood_table_r = self.make_likelihood_table(data_3d_r)
        return likelihood_table_b, likelihood_table_r, prior_vector
        
        
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
# check indices after separating and classing
# do make likelighood table, vectorise functions
# do pred fun 


