from sklearn import svm
# data points
X = [[0, 0], [1, 1]]
# labels
y = [0, 1]
clf = svm.SVC()
# learning
clf.fit(X, y)

# prediction for a new point
prediction = clf.predict([[2., 2.]])
print('prediction: ', prediction)

# print support vectors
print('support vectors: ', clf.support_vectors_)
# indices of support vectors
print('indices of support vectors: ', clf.support_) 
# get number of support vectors for each class
print('# of support vectors for each class: ', clf.n_support_)
