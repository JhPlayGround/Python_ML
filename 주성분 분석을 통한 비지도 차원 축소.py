#PCA
#특성의 스케일이 다르고 모든 특성의 중요도를 동일하게 취급하려면 PCA 적용전 데이터를 표준화 전처리를 해야함

#주성분 추출 단계
"""
    1. 데이터를 표준화 전처리
    2. 공분산 행렬을 구성
    3. 공분산 행렬의 고유값과 고유 백터 계산
    4. 고유 값을 내림 차순으로 정렬하여 고유 벡터의 순위 측정
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plot_decision_regions as prd 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

df_wine =  pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header= None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0, stratify=y)

#1. 데이터를 표준화 전처리
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#2~3. 공분산 행렬을 구성,  공분산 행렬의 고유값과 고유 백터 계산
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat) 
print('\n고윳값 \n%s' % eigen_vals)
#np.linalg.eig는 대칭, 비대칭 정방 행렬을 모두 다를 수 있지만, 복소수 고유값을 반환하기도 함
#np.linalg.eigh는 대칭 행렬을 다룰 때 수치적으로 더 안정된 결과를 생성, 실수 고유값만을 반환
#eigen_vals2, eigen_vecs2 = np.linalg.eigh(cov_mat) 



#총분산과 설명된 분산
"""
    가장 많은 정보(분산)을 가진 고유 벡터(주성분) 일부만을 선택
    고유값은 고유 벡터의 크기를 결정하므로, 고유값을 내림차순으로 정렬
    고유값 순서에 따라 최상위 k개의 고유 벡터를 선택
"""

tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#특성 변환
#고유 값, 고유 벡터 튜플 리스트 생성 후 내림차순 정렬
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

eigen_pairs.sort(key= lambda k: k[0], reverse=True)

#계산 효율성과 모델 성능 사이의 절충점을 찾아 주성분 개수를 결정해야 함
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print('투영 행렬 : \n', w)

X_train_pca = X_train_std.dot(w)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


#사이킷런의 주성분 분석 
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


lr = LogisticRegression(solver='liblinear', multi_class='auto')
lr = lr.fit(X_train_pca, y_train)

prd.plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

#테스트 세트로 변환
prd.plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

#전체 주성분의 설명된 분산 비율 계산 방법
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
print(pca.explained_variance_ratio_)