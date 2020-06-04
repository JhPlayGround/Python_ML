import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

print(df.head())


X = df.loc[:, 2:].values
y = df.loc[:, 1].values

#레이블 M,B를 정수로 변환
le = LabelEncoder()
y = le.fit_transform(y)

#학습, 테스트 세트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.30, stratify=y,random_state=1)

#StandardScaler, PCA, LogisticRegression 객체를 하나의 파이프 라인으로 연결
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(StandardScaler(), KernelPCA(n_components=2), LogisticRegression(solver='liblinear', random_state=1))
pipe_lr.fit(X_train,y_train)

y_pred = pipe_lr.predict(X_test)

print('테스트 정확도 : %.3f' % pipe_lr.score(X_test,y_test))
