import pandas as pd 

#데이터 import
uriage_data = pd.read_csv('pyda100-master/data/uriage.csv')
print(uriage_data.head())

kokyaku_data = pd.read_excel('pyda100-master/data/kokyaku_daicho.xlsx')
print(kokyaku_data.head())

#데이터 오류 수정
#print(len(pd.unique(uriage_data['item_name'])))
uriage_data['item_name'] = uriage_data['item_name'].str.upper()
uriage_data['item_name'] = uriage_data['item_name'].str.replace(' ','')
uriage_data['item_name'] = uriage_data['item_name'].str.replace('  ','')
print(uriage_data.sort_values(by=['item_name'], ascending=True))
#print(len(pd.unique(uriage_data['item_name'])))

#결측치 수정
print(uriage_data.isnull().any(axis=0))
flg_is_null = uriage_data["item_price"].isnull()
for trg in list(uriage_data.loc[flg_is_null, "item_name"].unique()):
    price = uriage_data.loc[(~flg_is_null) & (uriage_data["item_name"] == trg), "item_price"].max()
    uriage_data["item_price"].loc[(flg_is_null) & (uriage_data["item_name"]==trg)] = price
print(uriage_data.isnull().any(axis=0))

for trg in list(uriage_data["item_name"].sort_values().unique()):
    print(trg + "의최고가：" + str(uriage_data.loc[uriage_data["item_name"]==trg]["item_price"].max()) 
          + "의최저가：" + str(uriage_data.loc[uriage_data["item_name"]==trg]["item_price"].min(skipna=False)))

#오류 수정2
print(kokyaku_data["고객이름"].head())
kokyaku_data["고객이름"] = kokyaku_data["고객이름"].str.replace("　", "")
kokyaku_data["고객이름"] = kokyaku_data["고객이름"].str.replace(" ", "")
print(kokyaku_data["고객이름"].head())

#오류 수정3
flg_is_serial = kokyaku_data["등록일"].astype("str").str.isdigit()
print(kokyaku_data["등록일"])

fromSerial = pd.to_timedelta(kokyaku_data.loc[flg_is_serial, "등록일"].astype("float"), unit="D") + pd.to_datetime("1900/01/01")
print(fromSerial)

fromString = pd.to_datetime(kokyaku_data.loc[~flg_is_serial, "등록일"])
print(fromString)

kokyaku_data["등록일"] = pd.concat([fromSerial, fromString])
print(kokyaku_data)

kokyaku_data["등록연월"] = kokyaku_data["등록일"].dt.strftime("%Y%m")
rslt = kokyaku_data.groupby("등록연월").count()["고객이름"]
print(rslt)
print(len(kokyaku_data))

flg_is_serial = kokyaku_data["등록일"].astype("str").str.isdigit()
print(flg_is_serial.sum())

join_data = pd.merge(uriage_data, kokyaku_data, left_on="customer_name", right_on="고객이름", how="left")
join_data = join_data.drop("customer_name", axis=1)
print(join_data)

dump_data = join_data[["purchase_date", "purchase_month", "item_name", "item_price", "고객이름", "지역", "등록일"]]
dump_data
dump_data.to_csv("pyda100-master/data/dump_data.csv", index=False)

import_data = pd.read_csv("pyda100-master/data/dump_data.csv")
print(import_data)

byItem = import_data.pivot_table(index="purchase_month", columns="item_name", aggfunc="size", fill_value=0)
print(byItem)

byPrice = import_data.pivot_table(index="purchase_month", columns="item_name", values="item_price", aggfunc="sum", fill_value=0)
print(byPrice)

byCustomer = import_data.pivot_table(index="purchase_month", columns="고객이름", aggfunc="size", fill_value=0)
print(byCustomer)

byRegion = import_data.pivot_table(index="purchase_month", columns="지역", aggfunc="size", fill_value=0)
print(byRegion)

away_data = pd.merge(uriage_data, kokyaku_data, left_on="customer_name", right_on="고객이름", how="right")
print(away_data)
print(away_data[away_data["purchase_date"].isnull()][["고객이름", "등록일"]])