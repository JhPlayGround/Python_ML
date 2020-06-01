from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt 
import plot_decision_regions as prd

iris = datasets.load_iris()

X = iris.data[:, [2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1, stratify = y) # 계층화 기능 사용
X_combined = np.vstack((X_train,X_test))
y_combined = np.hstack((y_train, y_test))

forset = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)
forset.fit(X_train, y_train)

prd.plot_decision_regions(X_combined, y_combined, classifier=forset, test_idx=range(105,150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()