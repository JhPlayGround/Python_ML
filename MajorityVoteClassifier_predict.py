import MajorityVoteClassifier as MVC 
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

iris = datasets.load_iris()
X,y = iris.data[50 :, [1,2]], iris.target[50:]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=1,stratify=y)

#서로 다른 세개의 분류기를 훈련
#1. 로지스틱 
#2. 결정 트리
#3. KNN

clf1 = LogisticRegression(solver='liblinear', penalty='l2', C=0.001, random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy',random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()], ['clf',clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf',clf3]])

clf_labels = ['Logistic regression', 'Decision tree', 'KNN']
print('10겹 교차 검증 : \n')

for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print("ROC AUC : %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

print('\n')
mv_clf = MVC.MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels += ['Majority voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print("ROC AUC : %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

