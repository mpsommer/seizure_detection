import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import sys
from sklearn.grid_search import GridSearchCV

################################################################################|
#####                         Splitting data into test and train
################################################################################
# loadfile1 = 'labels_train_1.npz'
# data = np.load(loadfile1)
 # X = numpy.ones((1, 1301))

with np.load('labels_train_1.npz') as data:
    #print data.keys()
    labels = data['labels']
#
#print(len(labels))
#
with np.load('features_train_1_mean.npz') as data:
    features = data['features']
#
# print(len(features))
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
# print("n_samples = ", n_samples)
# print("X Shape = ", X.shape)
# print("y.shape = ", y.shape)
#
# print


#n_samples =
#X[] = feature set
#y[] = labels


X_train, y_train = X[:n_samples / 2], y[:n_samples / 2]
X_test, y_test = X[n_samples / 2 :], y[n_samples / 2:]
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



####Version 2
alpha = 0.1
# lasso = Lasso(alpha=alpha, max_iter=10000000, tol=0.01) #84.7926267281
# lasso = Lasso(alpha=0.001, copy_X=False, fit_intercept=True, normalize=True, positive=False, selection='random',
#               tol=0.01, warm_start=True, max_iter=10000000 ) #90.7834101382
lasso = Lasso(alpha=0.001, copy_X=True, fit_intercept=True, normalize=True, positive=False, selection='random',
              tol=0.01, warm_start=True, max_iter=10000000 ) #90.7834101382
y_pred_lasso = lasso.fit(X_train, y_train)
results = y_pred_lasso.predict(X_test)
y_test = y_test.astype(float)
results = np.round(results)

percent_correct = 100.0 * np.sum(results == y_test)/np.float(len(results))

# plt.subplot(2,1,1)
# plt.title('results')
# plt.hist(results)
# plt.xlim([-.25,1.25])
# plt.ylim([0,700])
#
# plt.subplot(2,1,2)
# plt.hist(y_test)
# plt.title('test')
# plt.xlim([-.25,1.25])
# plt.ylim([0,700])
# plt.show()

print("% correctly classified :" + str(percent_correct))
r2_score_lasso = r2_score(y_test, results)
print("r^2 on test data : %f" % r2_score_lasso)


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
################################################################################|
#####                         Parameter Tuning
###############################################################################
y_train = y_train.astype(float)
parameters = {'alpha': (1e-2, 1e-3), 'fit_intercept': (True, False), 'normalize': (True, False), 'copy_X': (True, False), 'tol': (1e-2, 1e-3),
              'warm_start': (True, False), 'positive': (True, False),'selection': ('cyclic', 'random'), }



gs_clf = GridSearchCV(lasso, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X_train[:600], y_train[:600])
best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
print(score)
