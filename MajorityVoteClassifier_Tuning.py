from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from itertools import product

import matplotlib.pyplot as plt
import numpy as np 
import MajorityVoteClassifier_predict as MVCP

colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']

for clf, label, clr, ls in zip(MVCP.all_clf, MVCP.clf_labels, colors, linestyles):
    y_pred = clf.fit(MVCP.X_train, MVCP.y_train).predict_proba(MVCP.X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_true =MVCP.y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.2f)' % (label, roc_auc))
    
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.5)
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.show()

sc = StandardScaler()
X_train_std = sc.fit_transform(MVCP.X_train)

x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(7, 5))

for idx, clf, tt in zip(product([0, 1], [0, 1]),MVCP.all_clf, MVCP.clf_labels):
    clf.fit(X_train_std, MVCP.y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[MVCP.y_train==0, 0], X_train_std[MVCP.y_train==0, 1], c='blue', marker='^',s=50)
    axarr[idx[0], idx[1]].scatter(X_train_std[MVCP.y_train==1, 0], X_train_std[MVCP.y_train==1, 1], c='green', marker='o',s=50)
    axarr[idx[0], idx[1]].set_title(tt)

plt.text(-3.5, -5.,  s='Sepal width [standardized]', ha='center', va='center', fontsize=12)
plt.text(-12.5, 4.5, s='Petal length [standardized]', ha='center', va='center', fontsize=12, rotation=90)
plt.show()

print(MVCP.mv_clf.get_params())


params = {'decisiontreeclassifier__max_depth': [1, 2], 'pipeline-1__clf__C': [0.001, 0.1, 100.0]}

grid = GridSearchCV(estimator=MVCP.mv_clf, param_grid=params, cv=10, scoring='roc_auc')

grid.fit(MVCP.X_train, MVCP.y_train)

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r" % (grid.cv_results_['mean_test_score'][r], grid.cv_results_['std_test_score'][r] / 2.0, grid.cv_results_['params'][r]))

print('최적의 매개변수: %s' % grid.best_params_)
print('정확도: %.2f' % grid.best_score_)

print(grid.best_estimator_.classifiers)

mv_clf = grid.best_estimator_

print(mv_clf.set_params(**grid.best_estimator_.get_params()))

print(mv_clf)