# Intro: The solution uses the "One Cluster, One Model" method to predict 
#the Y values for a given set of X value data set. The code has been 
#written as a function to easily enter the data required.
# Set-up
import csv
import numpy as np
import pandas as pd
from collections import namedtuple
# Creating function load_data that reads the data sets in the csv file 
# and sorts them into an array. (To be used in training/testing)
def load_data(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        data = []
        output = []
        for row in reader:
            data.append([float(row['x{}'.format(idx+1)]) for idx in range(5)])
            output.append(float(row['y']))
    data = np.array(data)
    output = np.array(output)
    return data, output, make_zipped_iterator(data, output)
ErrorComputation = namedtuple('ErrorType', ['component', 'finish'])
error_types = {
    'RMSE': ErrorComputation(lambda gap: gap * gap, lambda agg, N: np.sqrt(agg / N)),
    'MAE': ErrorComputation(lambda gap: gap, lambda agg, N: agg / N),
}
def make_zipped_iterator(data, output):
    def zipped_iterator():
        for x, y in zip(data, output):
            yield x, y
    return zipped_iterator
def evaluate_predictor(predict_one, test_items, err_type='RMSE',
                       DEBUG = False, DEBUG_SHOW_ALL = False, DEBUG_SHOW_INTERVAL = 100
                      ):
    total_error = 0
    total_items = 0
    err_component = error_types[err_type].component
    err_finish = error_types[err_type].finish
    for idx, items in enumerate(test_items()):
        data, result = items
        prediction = predict_one(data)
        
        gap = np.abs(result - prediction)
        current_error = err_component(gap)
        total_error += current_error
        total_items += 1
        if DEBUG and (DEBUG_SHOW_ALL or idx > 0 and ((idx + 1) % DEBUG_SHOW_INTERVAL == 0)):
            print('[{:6}] actual = {}, predicted = {:.2f}, gap = {:.2f}'.format(idx + 1, result, prediction, gap))
    overall_error = err_finish(total_error, total_items)
    if DEBUG:
        print('[Evaluation Done] Overall {} Error = {} (# Items: {})'.format(err_type, overall_error, total_items))
    return overall_error
# Importing required libraries for training
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sklearn.base
import sklearn.cluster
import sklearn.linear_model
import sklearn.preprocessing
# Defining the data for training 
training_data, training_output, _ = load_data('assessment_c01_train.csv')
# Variables used for training and/or testing using the given data
# A test size of 0.25 is being used as to manage a possibility of 
#over fitting of the data sets.
X_train, X_test, y_train, y_test = train_test_split(training_data, training_output, test_size=0.25)
# Creating Dataframe used for machine learning
features = ['x1','x2','x3','x4','x5']
df = pd.DataFrame(X_train, columns=features)
df['y'] = y_train
# Scaling of X data
ss = sklearn.preprocessing.StandardScaler()
ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
# Clustering is being used as a form of unsupervised learning as we 
#assume that data is a recorded behavior. As such, the data points may be 
#associated with entities or populations that behave differently and 
#clustering would thus aid with prediction.
# K-means is being used as a method of clustering. The goal of this
#algorithm is to find groups in the data, with the number of groups 
#represented by the variable K, group according to the nearest mean. This
#is useful for us due to the fact that the data given are unlabeled, 
#meaning that the best way in clustering them is using the mean. 
## Upon Plotting the training data, we noticed that there are 3 distinct 
#clusters formed. As such, the optimal number of clusters used in KMeans 
#was 3.
y = y_train
km = sklearn.cluster.KMeans(n_clusters=3)
km.fit(X_train_scaled, y)
# Base model
base_model = sklearn.linear_model.LinearRegression()
train_clusters = km.labels_
## There are several methods for us to use as predictors for the machine 
#learning of the data sets. In this case, we try out the One-cluster, One- 
#model method 
models = { 
    c: sklearn.base.clone(base_model).fit(X_train_scaled[train_clusters == c, :], y_train[train_clusters == c])
    for c in train_clusters
}
# Using the One Cluster, One Model to predict the Y value
# Predict function: Function predicts which cluster data is going to be in 
#based on input x values.
def predict(X):                    # new data goes through the pipeline
    X_scaled = ss.transform(X)     # scale
    labels = km.predict(X_scaled)  # predict cluster label
    return np.array([              # go through each point
        models[c].predict([x])[0]  # predict using the relevant model
        for x, c in zip(X_scaled, labels)
    ])
# test_size values of 0.05 and 0.1 give greater ranges of RMSE values
# which usually exceeds 25. 
#As such,increasing the test_size to 0.25 would improve performance, 
#returning a RMSE of < 25.
# the entire set of training data is then used:
X_train = training_data
y_train = training_output