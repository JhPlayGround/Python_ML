import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_wine =  pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header= None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']


X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0, stratify=y)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)



#일반화 오차를 감소시키 위한 방법
#1. 더 많은 훈련 데이터
#2. 규제를 통해 복잡도를 제한
#3. 파라미터 개수가 적은 간단한 모델을 선택
#4. 데이터 차원을 줄임


#L1 규제 : 가중치 제곱 -> 가중치 절대값으로 변경

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='liblinear', multi_class='auto', penalty='l1', C=1.0, random_state=1)

lr.fit(X_train_std,y_train)
print('훈련 정확도 ', lr.score(X_train_std,y_train))

print('테스트 정확도', lr.score(X_test_std, y_test))

#절편 확인
print(lr.intercept_)

#가중치 배열 확인
print(lr.coef_)


#규제 강도를 달리하여 특성의 가중치 변화 확인
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)

colors = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(solver='liblinear', multi_class='auto',
                            penalty='l1', C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', 
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
plt.show()
#L2 규제 : 비용함수에 패털티 항을 추가