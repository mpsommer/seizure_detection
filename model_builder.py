import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import sys
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.preprocessing import label_binarize

################################################################################|
#####                         Splitting data into test and train
################################################################################
with np.load('labels_train_1.npz') as data:
    labels = data['labels']

with np.load('features_train_1_mean.npz') as data:
    features = data['features']

y = np.array(labels)
print(y)
y = y.astype(int)
# y = label_binarize(y, classes=[0, 1])
n_classes = 100
# print("n_classes", n_classes)
X = np.array(features)

n_samples = y.shape[0]


X_train, y_train = X[:int(n_samples / 2)], y[:int(n_samples / 2)]
X_test, y_test = X[int(n_samples / 2) :], y[int(n_samples / 2):]


################################################################################|
#####                         Pipeline for Linear Model Lasso
################################################################################

##### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   #########

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
auc = roc_auc_score(y_test, results)
print(auc)


fpr, tpr, _ = roc_curve(y_test, results)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.plot(fpr, tpr)
plt.show()
# roc_auc = auc(fpr, tpr)

# for i in range(651):
#     print(i)
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], results[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), results.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

################################################################################|
#####                         Pipeline for LinearSVC
################################################################################

# from sklearn.svm import LinearSVC
# #
# lsvc = LinearSVC(C=0.01, dual=False, fit_intercept=True, intercept_scaling=0.01, multi_class='ovr', tol=0.01)
# y_pred_lsvc = lsvc.fit(X_train, y_train)
# results = y_pred_lsvc.predict(X_test)
# # print(results)
# results = results.astype(int)
# # results = np.round(results)
#
# percent_correct = 100.0 * np.sum(results == y_test)/np.float(len(results))
# print("% correctly classified LinearSVC:" + str(percent_correct))
# r2_score_lsvc = r2_score(y_test, results)
# print("r^2 on test data : %f" % r2_score_lsvc)


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
#LASS0
# y_train = y_train.astype(float)
# parameters = {'alpha': (1e-2, 1e-3), 'fit_intercept': (True, False), 'normalize': (True, False), 'copy_X': (True, False), 'tol': (1e-2, 1e-3),
#               'warm_start': (True, False), 'positive': (True, False),'selection': ('cyclic', 'random'), }
#
#
#
# gs_clf = GridSearchCV(lasso, parameters, n_jobs=-1)
# gs_clf = gs_clf.fit(X_train[:600], y_train[:600])
# best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, best_parameters[param_name]))
# print(score)

#LSVC
# parameters = {'C': (1e-2, 1e-3), 'tol': (1e-2, 1e-3),  'dual': (True, False),
#               'multi_class': ('ovr', 'crammer_singer'), 'fit_intercept': (True, False), 'intercept_scaling': (1e-2, 1e-3),
#                }
#
# gs_clf = GridSearchCV(lsvc, parameters, n_jobs=-1)
# gs_clf = gs_clf.fit(X_train[:600], y_train[:600])
# best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, best_parameters[param_name]))
# print(score)


################################################################################|
#####                         Parameter Tuning
###############################################################################
