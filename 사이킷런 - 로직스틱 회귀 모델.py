from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler #특성 스케일 조정 - 사이킷런의 preprocessing 모듈의 StandardScaler 클래스 사용
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt

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



iris = datasets.load_iris()

X = iris.data[:, [2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1, stratify = y) # 계층화 기능 사용

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined_std = np.hstack((y_train, y_test))

lr = LogisticRegression(solver='liblinear', multi_class= 'auto', C= 100.0 , random_state= 1)
# solver 기본값이 liblinear -> lbfgs로 변경
# solver = liblinear일 경우 -> ovr 선택, 그 외에는 multinomial 선택됨
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined_std, classifier= lr, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

#클래스 소속 확률 
predict_proba = lr.predict_proba(X_test_std[:3, :])
print(predict_proba)

predict_proba = lr.predict_proba(X_test_std[:3, :]).argmax(axis= 1)
print(predict_proba)

#직접 predict 메소드 사용 
predict_proba = lr.predict(X_test_std[:3, :])
print(predict_proba)


#사이킷런은 입력 데이터로 2차원 배열을 기대
#하나의 행을 2차원 포맷으로 먼저 변경해야 함 
predict_reshape = lr.predict(X_test_std[0,:].reshape(1,-1))
print(predict_reshape)


#규제를 사용하여 과대적합 피하기, 과대적합일때 분산이 크다, 과소적합일때 편향이 크다 
#규제는 공선성(특성간의 높은 상관관계)을 다루거나 데이터에서 잡음을 제거
weights, params = [], []

for c in np.arange(-5,5):
    lr = LogisticRegression(solver = 'liblinear', multi_class = 'auto', C = 10.**c, random_state= 1)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)

weights = np.array(weights)
plt.plot(params, weights[:,0], label='petal length')
plt.plot(params, weights[:,1], linestyle='--', label='petal width')
plt.ylabel('weight cofficient')
plt.xlabel('C')
plt.legend(loc = 'upper left')
plt.xscale('log')
plt.show()