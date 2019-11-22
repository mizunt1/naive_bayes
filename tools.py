from sklearn.datasets import load_iris
import numpy as np
import scipy.stats
from collections import defaultdict
#nbc = NBC(feature_types=['b', 'r', 'b'], num_classes=4)


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

    def apply_bernoulli(self, data):
        """
        bernoulli with laplace smoothing.
        add one to each numerator, add num_classes to each denom
        """
        num_ones = (data == 1).sum()
        total = len(data)
        return np.log(num_ones + 1 / total+self.num_classes)

    def apply_gauss(self,data):
        """
        TODO: Must make mean non zero
        """
        mean = np.mean(data)
        sd = np.std(data)
        return mean, sd

    def make_likelihood_table(
            self, data_classes_r, data_classes_b, num_b_features, num_r_features):
        """
        input data is data_3d_b: this is data[classes][num_datapoints][num_binary features]
        b is list of indexes of where binary features are in the data
        create a (class, features) shaped matrix of likelihoods
        two of them, depending on binary or real
        """        
        likelihood_table_b = np.zeros(
            (self.num_classes, num_b_features), dtype=(float))
        likelihood_table_r_mean = np.zeros(
            (self.num_classes, num_r_features))       
        likelihood_table_r_sd = np.zeros(
            (self.num_classes, num_r_features))       
            # calculate for one feature the likelihoods for each class
            # then move on to the next feature
            # probabilities must be calculated separately for each class
        i = 0
        for a_class in data_classes_b.values():
            likelihood_table_b[i] = np.apply_along_axis(self.apply_bernoulli, 0, a_class)
            i += 1
        i = 0
        for a_class in data_classes_r.values():
            likelihood_table_r_mean[i]  = np.apply_along_axis(
                np.mean, 0, a_class)
            likelihood_table_r_sd[i] = np.apply_along_axis(np.std, 0, a_class)
            i += 1
        return likelihood_table_b, likelihood_table_r_mean, likelihood_table_r_sd
        
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
        data_classes_b = dict()
        data_classes_r = dict()
        # creating data as fictionary. Key is class,
        # value is a matrix taken from both_data. each array in dictionary 
        # will have dimensions (num_data_points_in_that_class, features +1)
        # first column is the label, i.e. the class, which should be the same
        # for each row within each object in the dictionary
        total_index = 0
        for classes, num_items in num_items_in_class.items():
            data_classes_b[classes] = binary_data[total_index:total_index+num_items, :]
            data_classes_r[classes] = real_data[total_index:total_index+num_items, :]
            total_index += num_items
        ####### TODO CHECK ABOVE IS WORKING ############ 

        num_b_features = len(b)
        num_r_features = len(r)
        self.likelihood_table_b, self.likelihood_table_r_mean, self.likelihood_table_r_sd  = self.make_likelihood_table(data_classes_b, data_classes_r, num_b_features, num_r_features)
        

    def predict(self, in_data):
        log_likelihood_b = np.log(self.likelihood_table_b)
        def pdf(in_data,means,stds):
            prob = scipy.stats.norm(stds, means).pdf(in_data)
            return log(prob)
        def format_priors(in_prior):
            return self.num_points*np.log(in_prior/self.num_points)
            
        vec_p = np.vectorize(format_priors)
        vec = np.vectorize(pdf)
        log_real = vec(in_data,self.likelihood_table_r_mean, self.likelihood_table_r_sd)
        priors = vec_p(self.prior_vector)
        log_bin  = self.likelihood_table_b
        log_table = np.concatenate((priors, log_bin, log_table), axis=1)
        pred_class = np.argmax(np.sum(log_table, axis=1))
        return pred_class

def main():
    iris = load_iris()
    x, y = iris['data'], iris['target']
    N_points, D_features = x.shape
    Ntrain = int(0.8 * N_points)
    def train_loop():
        k = 1
        for k in range(11):
            Ntrain_small = int(k*0.1*Ntrain)
            print("num train points", Ntrain_small)
            shuffler = np.random.permutation(Ntrain_small)
            xtrain = x[shuffler[:Ntrain_small]]
            # select N points out of shuffled array
            ytrain = y[shuffler[:Ntrain_small]]
            xtest = x[shuffler[Ntrain_small:]]
            ytest = y[shuffler[Ntrain_small:]]
            feature_types = ['b', 'r', 'b', 'r']
            nbc = NBC(feature_types, D_features)
            nbc.fit(xtrain, ytrain)
            nbc.predict(x_test)
            #train
            #predict
            #append error
            k += 1
    train_loop()

    y.reshape(150,1)

    num_classes = 3
    num_features = x.shape[1]

    nbc.fit(x, y)
    nbc.predict(x[0,0:2])
    
    
main()

# Next to do

# maybe do some numerical checks
# make predict so that it can run
# sort out predict for binary features
# find out what to do with priors and laplace smoothing
# add logs


