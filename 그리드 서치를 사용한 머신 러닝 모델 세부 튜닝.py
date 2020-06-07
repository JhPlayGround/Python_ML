import numpy as np 
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values

#레이블 M,B를 정수로 변환
le = LabelEncoder()
y = le.fit_transform(y)

#학습, 테스트 세트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.30, stratify=y,random_state=1)

pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C':param_range,
                'svc__kernel' : ['linear']},
                {'svc__C':param_range,
                'svc__gamma':param_range,
                'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy',cv=10, n_jobs=1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('테스트 정확도 %.3f' %clf.score(X_test, y_test))
#모든 매개변수 조합을 평가 하기 위해 계산 비용이 많이 듬
#랜덤 서치가 있음 RadnomizedSearchCV 클래스를 사용하여 제한된 횟수 안에서 샘플링 분포로부터 랜덤한 매개벼수 조합을 뽑음


#중첩 교차 검증
from sklearn.model_selection import cross_val_score, cross_validate

scores = cross_val_score(gs, X=X_train, y=y_train, cv=5, scoring='accuracy')
print('CV 정확도 점수: %s' % scores)
print('CV 정확도: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
