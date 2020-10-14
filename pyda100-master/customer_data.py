import warnings
import pandas as pd
import os
warnings.filterwarnings('ignore')


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


