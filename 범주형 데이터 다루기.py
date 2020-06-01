import pandas as pd
import numpy as np

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
])
df.columns = ['color', 'size', 'price', 'classlabel']

print(df)

#순서 특성 매핑
size_mapping = {
    'XL' : 3,
    'L' : 2,
    'M' : 1
}

df['size'] = df['size'].map(size_mapping)
print(df)

#정수값을 다시 원래 문자열 표현으로 바꾸는 방법
#inv_size_mapping = {v:k for k,v in size_mapping.items()}
#df['size'] = df['size'].map(inv_size_mapping)
print(df)

#클래스 레이블 인코딩
class_mapping = {label : idx for idx, label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)

df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

#정수값을 다시 원래 문자열 표현으로 바꾸는 방법
#inv_class_mapping = {v:k for k,v in class_mapping.items()}
#df['classlabel'] = df['classlabel'].map(inv_class_mapping)
#print(df)

#사이킷 런 클래스 사용
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()

y = class_le.fit_transform(df['classlabel'].values)
print(y)

y = class_le.inverse_transform(y)
print(y)



#순서가 없는 특성에 원핫 인코딩 사용
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)


#여러 열을 한번에 정수로 변환 하는 방법 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder(dtype=np.int)
col_trans = ColumnTransformer([('ord_enc', ord_enc, ['color'])])
X_trans = col_trans.fit_transform(df)
print(X_trans)

#여러개의 열을 정수에서 문자로 변환하는 방법
inv_X_trans = col_trans.named_transformers_['ord_enc'].inverse_transform(X_trans)
print(inv_X_trans)

#원 핫 인코딩 
#순서 없는 특성에 들어 있는 고유한 값마다 새로운 더미 특성을 만드는 것
from sklearn.preprocessing import OneHotEncoder

oh_enc = OneHotEncoder(categories='auto')
col_trans = ColumnTransformer([('oh_enc', oh_enc, [0])], remainder='passthrough')
ohe = col_trans.fit_transform(X)
print(ohe)

#pandas의 get_dummies 사용
pd_dummies =pd.get_dummies(df[['price','color','size']])
print(pd_dummies)

#다중 공선성 문제를 해결하기 위해, 원핫 인코딩 된 배열에서 특성 열 하나를 제거
pd_dummies =pd.get_dummies(df[['price','color','size']], drop_first=True)
print(pd_dummies)