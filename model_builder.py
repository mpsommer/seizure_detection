import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import sys

################################################################################|
#####                         Splitting data into test and train
################################################################################
# loadfile1 = 'labels_train_1.npz'
# data = np.load(loadfile1)
 # X = numpy.ones((1, 1301))
labels = []
features = []
with np.load('labels_train_1.npz') as data:
    labels = data['labels']
#
print(len(labels))
#
with np.load('features_train_1_mean.npz') as data:
    features = data['features']
#
print(len(features))
#
# if len(labels[0]) != len(features[0]):
#     print("Length of feature set and label set do not match")
#     sys.exit()


# X_temp = np.array(labels)
# temp_array = []
# for x in X_temp[0]:
#     temp_array.append(x)
#     print(x)

y = np.array(labels)
X = np.array(features)
# y = np.asarray(y)[0,:,:]
# X = np.asarray(X)[0,:]
n_samples = y.shape[0]
# X.reshape(-1, 1)
print("n_samples = ", n_samples)
print("X Shape = ", X.shape)
print("y.shape = ", y.shape)
#
# print


#n_samples =
#X[] = feature set
#y[] = labels


X_train, y_train = X[:n_samples / 2], y[:n_samples / 2]
X_test, y_test = X[n_samples / 2 +1:], y[n_samples / 2+1:]

print("y_train.shape = ", y_train.shape) #shape = (650,)
# print("y_train = ", y_train) #[] of floating points
print("y_train.type() = ", type(y_train)) #<class 'numpy.ndarray'>

print("X_train.shape = ", X_train.shape) #shape = (650, 104)
# print("X_train = ", X_train) #[[]] of floating points
print("X_train.type() = ", type(X_train)) #<class 'numpy.ndarray'>

print("y_test.shape = ", y_test.shape) #shape = (650,)
# print("y_test = ", y_test) #[] of floating points
print("y_test.type() = ", type(y_test)) #<class 'numpy.ndarray'>

print("X_test.shape = ", X_test.shape) #shape = (650, 104)
# print("X_test = ", X_test) #[[]] of floating points
print("X_test.type() = ", type(X_test)) #<class 'numpy.ndarray'>

# X_train, y_train = X[:650], y[0][:650]
# X_test, y_test = X[651:], y[651:]
# for i in range(0, int(n_samples[1] / 2)):
#     X_train.append(X[0][i])
#     y_train.append(y[i])

# print(len(X_train))
# print("X_train shape = ", X_train.shape)
# print(len(y_train))
# print("y_train shape = ", y_train.shape)
# print(len(X_test))
# print("X_test shape = ", X_test.shape)
# print(len(y_test))
# print("y_test shape = ", y_test.shape)
# print(type(y_test))
#
# print(X_train.shape)
# print(X_train.transpose().shape)



################################################################################|
#####                         Pipeline for Linear Model Lasso
################################################################################
#
# clf = Lasso(alpha=0.1)
# clf.fit(X_train, y_train)
# Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
#    normalize=False, positive=False, precompute=False, random_state=None,
#    selection='cyclic', tol=0.0001, warm_start=False)
# print(clf.coef_)
# print(clf.intercept_)



####Version 2
# alpha = 0.1
# lasso = Lasso(alpha=alpha, max_iter=1000000000, tol=0.1)
# y_pred_lasso = lasso.fit(X_train, y_train)
# results = y_pred_lasso.predict(X_test)
# classificationReport_predictiveAccuracy = metrics.classification_report(y_test, results)
# print(classificationReport_predictiveAccuracy)


# predictiveAccuracy_Lasso = np.mean(results == y_test)
# print(predictiveAccuracy_Lasso)
# for i in range(0, len(results)):
#     print(type(results[i]))

# r2_score_lasso = r2_score(y_test, results)
# print(lasso)
# print("r^2 on test data : %f" % r2_score_lasso)

####   Version 1
alpha = 0.1
lasso = Lasso(alpha=alpha, max_iter=100000)
y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)

# print("X_test.shape = ", X_test.shape) #shape = (650,104)
# print("X_test = ", X_test) #[] of floating points
print("y_test.type() = ", type(y_test)) #<class 'numpy.ndarray'>

# print("y_pred_lasso.shape = ", y_pred_lasso.shape) #shape = (650,)
# print("y_pred_lasso = ", y_pred_lasso) #[[]] of floating points
print("y_pred_lasso.type() = ", type(y_pred_lasso)) #<class 'numpy.ndarray'>

# r2_score_lasso = r2_score(y_test, y_pred_lasso)
# print(lasso)
# print("r^2 on test data : %f" % r2_score_lasso)

################################################################################|
#####                         Pipeline for Linear Model ElasticNet
################################################################################
# alpha = 0.1
# enet = ElasticNet(alpha=alpha, l1_ratio=0.7, max_iter=100000)
#
# y_pred_enet = enet.fit(X_train, y_train)
# .predict(X_test)
# r2_score_enet = r2_score(y_test, y_pred_enet)
# # print(enet)
# # print("r^2 on test data : %f" % r2_score_enet)
# plt.plot(enet.coef_, color='lightgreen', linewidth=2,
#          label='Elastic net coefficients')
# plt.plot(lasso.coef_, color='gold', linewidth=2,
#          label='Lasso coefficients')
# plt.plot(coef, '--', color='navy', label='original coefficients')
# plt.legend(loc='best')
# plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
#           % (r2_score_lasso, r2_score_enet))
# plt.show()
