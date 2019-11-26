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
            self, data_classes_b, data_classes_r, num_b_features, num_r_features):
        print("num_r_features", num_r_features)
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
        if num_b_features == 0:
            likelihood_table_b = None
        else:
            for a_class in data_classes_b.values():
                likelihood_table_b[i] = np.apply_along_axis(self.apply_bernoulli, 0, a_class)
                i += 1
        if num_r_features == 0:
            likelihood_table_r_mean = None
        else:
            i = 0
            for a_class in data_classes_r.values():

                likelihood_table_r_mean[i]  = np.apply_along_axis(np.mean, 0, a_class)
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
        self.num_items_in_class = defaultdict(int)
        # dictionary of class: number of data points in that class
        for classes in y_train:
            self.num_items_in_class[classes] += 1
        b, r = columns_of_br(self.feature_types)
        binary_data, real_data = sep_data(data, b, r)
        data_classes_b = dict()
        data_classes_r = dict()
        self.num_b_features = len(b)
        self.num_r_features = len(r)
        # creating data as fictionary. Key is class,
        # value is a matrix taken from both_data. each array in dictionary 
        # will have dimensions (num_data_points_in_that_class, features +1)
        # first column is the label, i.e. the class, which should be the same
        # for each row within each object in the dictionary
        total_index = 0
        for classes, num_items in self.num_items_in_class.items():
            if self.num_b_features == 0:
                data_classes_b == None
            else:
                data_classes_b[classes] = binary_data[total_index:total_index+num_items, :]
            if self.num_r_features == 0:
                data_classes_r = None
            else:
                data_classes_r[classes] = real_data[total_index:total_index+num_items, :]
            total_index += num_items
        self.likelihood_table_b, self.likelihood_table_r_mean, self.likelihood_table_r_sd  = self.make_likelihood_table(data_classes_b, data_classes_r, self.num_b_features, self.num_r_features)
        

    def predict(self,in_data):
        # TUDO FIND OUT THE ACTUAL FUNCTION for periors
        self.num_data = in_data.shape[0]
        def pdf(in_data,means,stds):
            prob = scipy.stats.norm(means, stds).pdf(in_data)
            return np.log(prob)

        def format_priors(in_prior):
           #not being used
            return self.num_points*np.log(in_prior/self.num_points)
            
        def binary(prob, in_data):
            i = [1,0]
            return np.log(abs(prob - i[int(in_data)]))

        vec = np.vectorize(pdf)
        vecb = np.vectorize(binary)
        answ_r = np.zeros((self.num_classes, self.num_r_features), dtype=object)
        answ_b = np.zeros((self.num_classes, self.num_b_features), dtype=object)
        # np array of objects
        # size (num classes, features)
        # each item contains an array of len (num_data)
        for j in range(self.num_r_features):
            i = 0
            dataj = in_data[:,j]
            for i in range(self.num_classes):
                meani = self.likelihood_table_r_mean[:,j][i]
                sdi = self.likelihood_table_r_sd[:,j][i]
                ans = vec(dataj, meani, sdi)
                answ_r[i][j] = ans
                i+=1
            j +=1
            
        j=0
        for j in range(self.num_b_features):
            i = 0
            dataj = in_data[:,j]
            for i in range(self.num_classes):
                b = self.likelihood_table_b[:,j][i]
                ans = vecb(b, dataj)
                answ_b[i][j] = ans
                i+=1
            j +=1

        collapsed_r = np.array((self.num_classes, self.num_data))
        c=0
        if self.num_r_features != 0:
            for c in range(self.num_classes):
                for f in range(1, self.num_r_features):
                    answ_r[c][1] += answ_r[c][f]
            answ_r = answ_r[:,0]
            
        c = 0
        if self.num_b_features != 0:
            for c in range(self.num_classes):
                for f in range(1, self.num_b_features):
                    answ_b[c][1] += answ_b[c][f]
            answ_b = answ_b[:,0]
        c = 0
        if self.num_b_features != 0:
            for c in range(self.num_classes):
                answ_b[c] += answ_r[c]
            final_ans = answ_b

        else:
            final_ans = answ_r
        # answ_r and answ_b are now numpy arrays of shape (num_classes)
        # each class contains 
        i = 0
        o = 0
        array_is = np.zeros((self.num_classes, self.num_data))
        for i in range(self.num_classes):
            array_is[i, 0:self.num_data] = (final_ans[i] + np.log(self.num_items_in_class[i])/self.num_data)
            i += 1
        
        return np.argmax(array_is, axis = 0)
def main():
    iris = load_iris()
    x, y = iris['data'], iris['target']
    print(y)
    print(x.shape)
    num_classes = 3
    print(y.shape)
    N_points, D_features = x.shape
    Ntrain = int(0.8 * N_points)
    feature_types = ['r', 'r', 'r', 'r']
    nbc = NBC(feature_types,num_classes)
    nbc.fit(x, y)
    print("x shape", x.shape)
    prediction = nbc.predict(x)
    print("pred", prediction)
    print("y is", y)
    correct = np.count_nonzero(y==prediction)
    print("correct", correct)
    print("accuracy", correct/len(y))
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
            feature_types = ['r', 'r', 'r', 'r']
            nbc = NBC(feature_types, D_features)
            nbc.fit(xtrain, ytrain)
            prediction = nbc.predict(xtest)
            correct = np.count_nonzero(ytest==prediction)
            print("fraction correct ", correct/Ntrain_small)
            #train
            #predict
            #append error
            k += 1
    train_loop()
    y.reshape(150,1)

    num_classes = 3
    num_features = x.shape[1]

    nbc.fit(x, y)
    nbc.predict(x)
    
    
main()

# Next to do

# maybe do some numerical checks
# make predict so that it can run
# sort out predict for binary features
# find out what to do with priors and laplace smoothing
# add logs


