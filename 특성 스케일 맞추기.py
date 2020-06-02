import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split

df_wine =  pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header= None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
print("클래스 레이블", np.unique(df_wine['Class label']))

print(df_wine.head())


X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0, stratify=y)

#정규화 -> 특성의 스케일을 [0,1] 범위에 맞추는 것 
#최소-최대 스케일 변환
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)

#표준화 -> 특성의 평균을 0에 맞추고 표준 편차를 1로 만들어 정규 분포와 같은 특징을 가지도록 함
#가중치를 더 쉽게 학습, 이상치 정보가 유지되기 때문에
#제한된 범위로 데이터를 조정하는 최소-최대 스케일 변환에 비해 알고리즘이 이상치에 덜 민감
from sklearn.preprocessing import StandardScaler

sc = StandardScaler
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)