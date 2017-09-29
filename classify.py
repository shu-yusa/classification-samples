import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

def boundary(x, y):
    #return - 0.5 * x**3 + x**2
    return y + 0.5*x*x

def generate(min_x, max_x, min_y, max_y, num_samples, overlap_width=0.5):
    samples = np.empty((0, 2), float)
    labels = np.array([])
    for i in range(num_samples):
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        samples = np.append(samples, np.array([[x, y]]), axis=0)
        if abs(boundary(x, y)) < overlap_width:
            labels = np.append(labels, 1 if random.random() >= 0.5 else 0)
        else:
            labels = np.append(labels, 1 if boundary(x, y) > 0 else 0)
    return samples, labels


def show_figure(samples, labels, num_samples, clf, title=''):
    labeled_samples = pd.DataFrame(np.hstack((samples, labels.reshape(num_samples, 1))), columns=('x', 'y', 'label'))

    sns.lmplot(x='x', y='y', data=labeled_samples, fit_reg=False, hue='label')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    num_samples = 1000
    # generate samples
    # samples, labels = generate(-10, 10, -10, 10, num_samples, 3)
    samples, labels = make_classification(n_features=2, n_redundant=0, n_informative=2,
        random_state=6, n_clusters_per_class=1, n_samples=num_samples, n_classes=2)

    #### Create a model for SVM
    """
    rbf ... Gaussian kernel
    linear ... Linear kernel
    poly ... Polynomial kernel
    sigmoid ... Sigmoid kernel
    """
    clf = svm.SVC(C=2., kernel='rbf', gamma=0.05)
    # learning
    clf.fit(samples, labels)
    # show results
    show_figure(samples, labels, num_samples, clf, 'SVM')

    #### Linear Discriminant Analysis
    clf = LinearDiscriminantAnalysis()
    clf.fit(samples, labels)
    show_figure(samples, labels, num_samples, clf, 'Linear Discriminant Analysis')

    #### Quadratic Discriminant Analysis
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(samples, labels)
    show_figure(samples, labels, num_samples, clf, 'Quadratic Discriminant Analysis')

    ## # prediction for a new point
    ## prediction = clf.predict([[2., 2.]])
    ## print('prediction: ', prediction)

    ## # print support vectors
    ## print('support vectors: ', clf.support_vectors_)
    ## # indices of support vectors
    ## print('indices of support vectors: ', clf.support_) 
    ## # get number of support vectors for each class
    ## print('# of support vectors for each class: ', clf.n_support_)

