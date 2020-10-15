import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from dateutil.relativedelta import relativedelta

uselog = pd.read_csv('pyda100-master/data/use_log.csv')
print(len(uselog))
print(uselog.head())

customer = pd.read_csv('pyda100-master/data/customer_master.csv')
print(len(customer))
print(customer.head())

class_master = pd.read_csv('pyda100-master/data/class_master.csv')
print(len(class_master))
print(class_master.head())

campaign_master = pd.read_csv('pyda100-master/data/campaign_master.csv')
print(len(campaign_master))
print(campaign_master.head())


customer_join = pd.merge(customer, class_master, on="class", how="left")
customer_join = pd.merge(customer_join, campaign_master, on="campaign_id", how="left")
print(len(customer_join))
print(customer_join.head())
print(customer_join.isnull().sum())

#집계
print(customer_join.groupby("class_name").count()["customer_id"])
print(customer_join.groupby("campaign_name").count()["customer_id"])
print(customer_join.groupby("gender").count()["customer_id"])
print(customer_join.groupby("is_deleted").count()["customer_id"])


customer_join["start_date"] = pd.to_datetime(customer_join["start_date"])
customer_start = customer_join.loc[customer_join["start_date"]>pd.to_datetime("20180401")]
print(len(customer_start))

customer_join["end_date"] = pd.to_datetime(customer_join["end_date"])
customer_newer = customer_join.loc[(customer_join["end_date"]>=pd.to_datetime("20190331"))|(customer_join["end_date"].isna())]
print(len(customer_newer))
print(customer_newer["end_date"].unique())

print(customer_newer.groupby("class_name").count()["customer_id"])
print(customer_newer.groupby("campaign_name").count()["customer_id"])
print(customer_newer.groupby("gender").count()["customer_id"])

uselog["usedate"] = pd.to_datetime(uselog["usedate"])
uselog["연월"] = uselog["usedate"].dt.strftime("%Y%m")
uselog_months = uselog.groupby(["연월","customer_id"],as_index=False).count()
uselog_months.rename(columns={"log_id":"count"}, inplace=True)
del uselog_months["usedate"]
print(uselog_months.head())

uselog_customer = uselog_months.groupby("customer_id").agg(["mean", "median", "max", "min" ])["count"]
uselog_customer = uselog_customer.reset_index(drop=False)
print(uselog_customer.head())

uselog["weekday"] = uselog["usedate"].dt.weekday
uselog_weekday = uselog.groupby(["customer_id","연월","weekday"], 
                                as_index=False).count()[["customer_id","연월", "weekday","log_id"]]
uselog_weekday.rename(columns={"log_id":"count"}, inplace=True)
print(uselog_weekday.head())

uselog_weekday = uselog_weekday.groupby("customer_id",as_index=False).max()[["customer_id", "count"]]
uselog_weekday["routine_flg"] = 0
uselog_weekday["routine_flg"] = uselog_weekday["routine_flg"].where(uselog_weekday["count"]<4, 1)
print(uselog_weekday.head())

customer_join = pd.merge(customer_join, uselog_customer, on="customer_id", how="left")
customer_join = pd.merge(customer_join, uselog_weekday[["customer_id", "routine_flg"]], on="customer_id", how="left")
print(customer_join.head())

print(customer_join.isnull().sum())


customer_join["calc_date"] = customer_join["end_date"]
customer_join["calc_date"] = customer_join["calc_date"].fillna(pd.to_datetime("20190430"))
customer_join["membership_period"] = 0

for i in range(len(customer_join)):
    delta = relativedelta(customer_join["calc_date"].iloc[i], customer_join["start_date"].iloc[i])
    customer_join["membership_period"].iloc[i] = delta.years*12 + delta.months

print(customer_join.head())

print(customer_join[["mean", "median", "max", "min"]].describe())

print(customer_join.groupby("routine_flg").count()["customer_id"])

plt.hist(customer_join["membership_period"])
plt.show()

customer_end = customer_join.loc[customer_join["is_deleted"]==1]
print(customer_end.describe())

customer_stay = customer_join.loc[customer_join["is_deleted"]==0]
print(customer_stay.describe())

customer_join.to_csv("pyda100-master/data/customer_join.csv", index=False)