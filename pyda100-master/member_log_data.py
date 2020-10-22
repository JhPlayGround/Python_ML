import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from dateutil.relativedelta import relativedelta
from datetime import datetime

customer = pd.read_csv('pyda100-master/data/customer_join.csv')
uselog_months = pd.read_csv('pyda100-master/data/use_log_months.csv')

year_months = list(uselog_months["연월"].unique())
uselog = pd.DataFrame()
for i in range(1, len(year_months)):
    tmp = uselog_months.loc[uselog_months["연월"]==year_months[i]]
    tmp.rename(columns={"count":"count_0"}, inplace=True)
    tmp_before = uselog_months.loc[uselog_months["연월"]==year_months[i-1]]
    del tmp_before["연월"]
    tmp_before.rename(columns={"count":"count_1"}, inplace=True)
    tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")
    uselog = pd.concat([uselog, tmp], ignore_index=True)

print(uselog.head())

#탈퇴전월의 탈퇴고객데이터 작성

exit_customer = customer.loc[customer["is_deleted"]==1]
exit_customer["exit_date"] = None
exit_date = []
exit_customer["end_date"] = pd.to_datetime(exit_customer["end_date"])
for i in range(len(exit_customer)):
    exit_customer["exit_date"].iloc[i] = str(exit_customer["end_date"].iloc[i] - relativedelta(months=1))
    exit_customer["exit_date"].iloc[i] = datetime.strptime(exit_customer["exit_date"].iloc[i], '%Y-%m-%d %H:%M:%S')
    exit_customer["exit_date"].iloc[i] = exit_customer["exit_date"].iloc[i].strftime("%Y%m")
    exit_date.append(exit_customer["exit_date"].iloc[i])


exit_customer["연월"] = exit_date
uselog["연월"] = uselog["연월"].astype(str)
exit_uselog = pd.merge(uselog, exit_customer, on=["customer_id", "연월"], how="left")
print(len(uselog))
print(exit_uselog.head())

exit_uselog = exit_uselog.dropna(subset=["name"])
print(len(exit_uselog))
print(len(exit_uselog["customer_id"].unique()))
print(exit_uselog.head())
