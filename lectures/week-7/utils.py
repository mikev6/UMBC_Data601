import requests  # for loading data from an online resource
from io import StringIO  # for reading inputs


def get_data_from_drive(url):
    """
    For a given shared link of a csv file, this function edits the link and reads the data.
    :param url: string. Sharable link to a csv file from google drive
    :return: raw file. Expected to read by pandas dataframe.
    """
    orig_url = url
    file_id = orig_url.split('/')[-2]
    dwn_url = 'https://drive.google.com/uc?export=download&id=' + file_id
    url = requests.get(dwn_url).text
    csv_raw = StringIO(url)
    return csv_raw


import matplotlib.pyplot as plt
from sklearn import tree
def draw_tree(estimator, figsize =(15, 5), feature_names = ["Hits", "Years"]):
    """
    Takes a decision  tree estimator and plots it's tree structure
    :param estimator: A sklearn decision tree estimator. Should be fitted.
    :param figsize: tuple. (int, int).
    :param feature_names:
    :return: It returns a plot. The image is not saved.
    """
    fig = plt.figure(figsize= figsize)
    _ = tree.plot_tree(estimator,
                       feature_names= feature_names,
                       filled=True)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
def heart_data_ohe(categorical_variables, remaining_variables, data):
    '''
    Takes the heart dataset (ISLR) and uses One Hot Encoding to encode categorical variables.
    Uses OneHotEncoder and ColumnTransformer from sklearn
    :param categorical_variables: list. List of categorical variables to encode with OHE
    :param remaining_variables: list. Remaining list of columns.
    :param data: DataFrame. Make sure that 'target' variable is not in this dataset.
    :return: DataFrame. A new dataframe with the columns are transformed.
    '''
    ## instantiate the encoder
    encoder = OneHotEncoder(handle_unknown='error',
                            drop='first',
                            categories='auto')

    #instantiate the ColumTransformer
    ct = ColumnTransformer(transformers=[('ohe', encoder, categorical_variables)],
                           remainder='passthrough', )
    ct.fit(data)
    X = ct.transform(data)
    ## create a dataframe with new column names
    ohe = ct.named_transformers_['ohe'] ##Trained OHE object
    feature_names_created= ohe.get_feature_names(input_features = categorical_variables)
    column_names = list(feature_names_created) + remaining_variables
    return pd.DataFrame(X, columns=column_names)

def gini(zero_total, one_total, left_zero, left_one ):
    total = zero_total + one_total
    root_p0 = zero_total/total
    root_p1 = one_total/total
    root_gini = 1 - (root_p0)**2 - root_p1**2
    left_total = left_zero + left_one
    right_total = total - left_total
    right_zero = zero_total - left_zero
    right_one = one_total - left_one
    left_p0= left_zero/left_total
    left_p1 = left_one/left_total
    left_gini = 1 - left_p0**2 - left_p1**2
    right_p0 = right_zero/right_total
    right_p1 = right_one/right_total
    right_gini = 1 - right_p0**2 - right_p1**2
    weighted_gini = (left_total/total)*left_gini + (right_total/total)*right_gini
    net_gain = root_gini - weighted_gini
    return net_gain,weighted_gini, root_gini, left_gini, right_gini

import numpy as np
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def draw_sigmoid():
    z = np.arange(-7, 7, 0.1)
    phi_z = sigmoid(z)
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$h(z)$')  # y axis ticks and gridline
    plt.yticks([0.0, 0.5, 1.0])
    ax = plt.gca()
    ax.yaxis.grid(True)
    plt.title('Sigmoid Function')
    plt.tight_layout()
    plt.show()

def cost_1(z):
    return - np.log(sigmoid(z))
def cost_0(z):
    return - np.log(1 - sigmoid(z))
def draw_likelihood():
    z = np.arange(-10, 10, 0.1)
    phi_z = sigmoid(z)
    c1 = [cost_1(x) for x in z]
    plt.plot(phi_z, c1, label='J(w) if y=1')
    c0 = [cost_0(x) for x in z]
    plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')
    plt.ylim(0.0, 5.1)
    plt.xlim([0, 1])
    plt.xlabel('$h$(z)')
    plt.ylabel('J(w)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o', s=100, label='test set')