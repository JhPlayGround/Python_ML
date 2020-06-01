import pandas as pd
import numpy as np

from io import StringIO

csv_data = ''' A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
df
print(df)

df_null_sum = df.isnull().sum()
print(df_null_sum)

df_values = df.values
print(df_values)


#누락된 값이 있는 샘플이나 특성 제외
df_dropna_row = df.dropna(axis = 0) #행 제거 
print(df_dropna_row)

df_dropna_col = df.dropna(axis = 1) #열 제거
print(df_dropna_col)

df_dropna_all = df.dropna(how = 'all') #모든 열이 NaN일때만 행 삭제
print(df_dropna_all)

df_dropna_thresh = df.dropna(thresh = 4) #실수 값이 4개 보다 작은 행을 삭제
print(df_dropna_thresh)

df_dropna_subset = df.dropna(subset = ['C']) #특정 열에 NaN이 있는 행만 삭제 
print(df_dropna_subset)


#보간 기법 -> 누락된 값을 추정
from sklearn.impute import SimpleImputer

#열 방향
simr = SimpleImputer(missing_values=np.nan, strategy='mean') #mean, median, most_frequent, constant
simr = simr.fit(df.values)

imputed_data = simr.transform(df.values)
print(imputed_data)

#행 방향 
from sklearn.preprocessing import FunctionTransformer
ftr_simr = FunctionTransformer(lambda X : simr.fit_transform(X.T).T, validate=False)

imputed_data = ftr_simr.fit_transform(df.values)
print(imputed_data)


