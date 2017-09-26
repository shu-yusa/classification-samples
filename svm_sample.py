import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm

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
            pass
        else:
            labels = np.append(labels, 1 if boundary(x, y) > 0 else 0)
    return samples, labels

def generate_normal_samle(mu1, var1, mu2, var2, num_samples):
    std1 = np.sqrt(var1)
    std2 = np.sqrt(var2)
    samples = np.empty((0, 2), float)
    a = np.random.normal(mu1, var1, size=[num_samples/2, 2])
    b = np.random.normal(mu2, var2, size=[num_samples/2, 2])
    return np.r_[a, b], np.r_[np.ones(num_samples/2), np.zeros(num_samples/2)]


def show_figure(samples, labels, num_samples, clf):
    labeled_samples = pd.DataFrame(np.hstack((samples, labels.reshape(num_samples, 1))), columns=('x', 'y', 'label'))

    sns.lmplot(x='x', y='y', data=labeled_samples, fit_reg=False, hue='label')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
    plt.show()


if __name__ == '__main__':
    num_samples = 1000
    # generate samples
    samples, labels = generate(-10, 10, -10, 10, num_samples, 3)

    # Create a model for SVM
    """
    rbf ... Gaussian kernel
    linear ... Linear kernel
    poly ... Polynomial kernel
    sigmoid ... Sigmoid kernel
    """
    clf = svm.SVC(C=2., kernel='rbf', gamma=0.01)

    # learning
    clf.fit(samples, labels)

    # show results
    show_figure(samples, labels, num_samples, clf)

    ## # prediction for a new point
    ## prediction = clf.predict([[2., 2.]])
    ## print('prediction: ', prediction)

    ## # print support vectors
    ## print('support vectors: ', clf.support_vectors_)
    ## # indices of support vectors
    ## print('indices of support vectors: ', clf.support_) 
    ## # get number of support vectors for each class
    ## print('# of support vectors for each class: ', clf.n_support_)
