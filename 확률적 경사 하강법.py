import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

df = pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data',
                  header=None)
df.tail()
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values

def plot_decision_regions(X, y, classifier, resolution=0.02):
    
 # 마커와 컬러맵을 설정합니다
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계를 그립니다
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 샘플의 산점도를 그립니다
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

class AdalineSGD(object):
    """
        ADAptive Linear Neuron 분류기 

        매개변수
        eta  : float
            학습률 (0.0과 0.1 사이)
        n_iter : int 
            훈련 데이터셋 반복 횟수
        shuffle : bool (default : True)
            True로 설정하면 같은 반복이 되지 않도록 에포크마다 훈련 데이터를 섞음 
        random_stat : int
            가중치 무작위 초기화를 위한 난수 생성기 시드

        ----
        속성
        w_ : 1d-array
            학습된 가중치
        cost_ : list
            모든 훈련 샘플에 대해 에포크마다 누적된 평균 비용 함수의 제곱합
    """
    def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialize = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self,X,y):
        """
            훈련 데이터 학습

            매개변수
            X : {array-like}, shape = [n_samples, n_features]
                n_samples개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터
            y : array-like, shape = [n_samples]
                타깃 벡터 

            -----
            반환값 
            self : object
        """

        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X,y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        return self

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return  X[r], y[r]

    def _initialize_weights(self, m):
        #랜덤한 작은 수로 가중치 초기화
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = 0.0, scale = 0.01, size = 1+m)
        self.w_initialize = True
    
    def _update_weights(self, xi, target):
        #아달린 학습 규칙을 적용하여 가중치 업데이트 
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost 

    def net_input(self, X):
        #최종 입력 계산
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        #선형 활성화 계산 
        return X

    def predict (self, X):
        #단위 계단 함수를 사용하여 클래스 레이블을 반환
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std,y)



plot_decision_regions(X_std, y, classifier= ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('Sepal length [standardized')
plt.ylabel('Petal length [standardized')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()