#경험적으로 보았을때 k-겹 교차 검증에서 좋은 기본값은 k=10이다.
import numpy as np 
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

print(df.head())


X = df.loc[:, 2:].values
y = df.loc[:, 1].values

#레이블 M,B를 정수로 변환
le = LabelEncoder()
y = le.fit_transform(y)

#학습, 테스트 세트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.30, stratify=y,random_state=1)

pipe_lr = make_pipeline(StandardScaler(), KernelPCA(n_components=2), LogisticRegression(solver='liblinear', random_state=1))

kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)

scores = []

for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        print('폴드 : %2d, 클래스 분포 %s, 정확도 %.3f' % (k+1, np.bincount(y_train[train]), score))

print('\nCV 정확도 : %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#사이킷 런 K-겹 교차 검증 함수
from sklearn.model_selection import cross_val_score, cross_validate

scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10,  n_jobs=1)
print('CV 정확도 점수: %s' % scores)
print('CV 정확도: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


scores = cross_validate(estimator=pipe_lr, X=X_train, y=y_train,  scoring=['accuracy'],  cv=10,  n_jobs=-1, return_train_score=False)
print('CV 정확도 점수: %s' % scores['test_accuracy'])
print('CV 정확도: %.3f +/- %.3f' % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])))