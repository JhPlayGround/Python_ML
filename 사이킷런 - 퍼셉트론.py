from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #특성 스케일 조정 - 사이킷런의 preprocessing 모듈의 StandardScaler 클래스 사용
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score # 성능 지표
from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data[:, [2,3]]
y = iris.target
print('클래스 레이블', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1, stratify = y) # 계층화 기능 사용
print('y의 레이블 카운트 :', np.bincount(y))
print('y_train의 레이블 카운트 :', np.bincount(y_train))
print('y_test의 레이블 카운트 :', np.bincount(y_test))

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(max_iter = 40, eta0 = 0.01, tol = 1e-3, random_state = 1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('잘못 분류된 샘플 개수 : %d' % (y_test != y_pred).sum())

print('정확도 : %.2f' % accuracy_score(y_test, y_pred))

print('정확도 : %.2f' %ppn.score(X_test_std, y_test))

def plot_decision_regions(X, y, classifier, test_idx = None, resolution=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl,1], alpha= 0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolors='black')

    if test_idx :
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:,1], c='', edgecolors='black', alpha=1.0, linewidths=1, marker='o', s=100, label = 'test set')

X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined_std = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined_std, classifier=ppn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.tight_layout()
plt.show()