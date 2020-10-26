import pandas as pd
import sklearn.model_selection
import warnings
warnings.filterwarnings('ignore')

from dateutil.relativedelta import relativedelta
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier

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

#탈퇴전월의 탈퇴고객데이터를 작성
exit_customer = customer.loc[customer["is_deleted"]==1]
exit_customer["exit_date"] = None
exit_customer["end_date"] = pd.to_datetime(exit_customer["end_date"])
exit_date = []

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

#지속 회원
conti_customer = customer.loc[customer["is_deleted"]==0]
conti_uselog = pd.merge(uselog, conti_customer, on=["customer_id"], how="left")
print(len(conti_uselog))
conti_uselog = conti_uselog.dropna(subset=["name"])
print(len(conti_uselog))

conti_uselog = conti_uselog.sample(frac=1).reset_index(drop=True)
conti_uselog = conti_uselog.drop_duplicates(subset="customer_id")
print(len(conti_uselog))
print(conti_uselog.head())

predict_data = pd.concat([conti_uselog, exit_uselog],ignore_index=True)
print(len(predict_data))
print(predict_data.head())

predict_data["period"] = 0
predict_data["now_date"] = pd.to_datetime(predict_data["연월"], format="%Y%m")
predict_data["start_date"] = pd.to_datetime(predict_data["start_date"])
for i in range(len(predict_data)):
    delta = relativedelta(predict_data["now_date"][i], predict_data["start_date"][i])
    predict_data["period"][i] = int(delta.years*12 + delta.months)
print(predict_data.head())

print(predict_data.isna().sum())

predict_data = predict_data.dropna(subset=["count_1"])
print(predict_data.isna().sum())

target_col = ["campaign_name", "class_name", "gender", "count_1", "routine_flg", "period", "is_deleted"]
predict_data = predict_data[target_col]
print(predict_data.head())

predict_data = pd.get_dummies(predict_data)
print(predict_data.head())

del predict_data["campaign_name_2_일반"]
del predict_data["class_name_2_야간"]
del predict_data["gender_M"]
print(predict_data.head())

#의사결정나무를 사용해서 탈퇴예측모델
exit = predict_data.loc[predict_data["is_deleted"]==1]
conti = predict_data.loc[predict_data["is_deleted"]==0].sample(len(exit))

X = pd.concat([exit, conti], ignore_index=True)
y = X["is_deleted"]
del X["is_deleted"]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
print(y_test_pred)

results_test = pd.DataFrame({"y_test":y_test ,"y_pred":y_test_pred })
print(results_test.head())

correct = len(results_test.loc[results_test["y_test"]==results_test["y_pred"]])
data_count = len(results_test)
score_test = correct / data_count
print(score_test)

print(model.score(X_test, y_test))
print(model.score(X_train, y_train))

X = pd.concat([exit, conti], ignore_index=True)
y = X["is_deleted"]
del X["is_deleted"]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)

model = DecisionTreeClassifier(random_state=0, max_depth=5)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.score(X_train, y_train))

importance = pd.DataFrame({"feature_names":X.columns, "coefficient":model.feature_importances_})
print(importance)


count_1 = 3
routing_flg = 1
period = 10
campaign_name = "입회비무료"
class_name = "종일"
gender = "M"

campaign_name_list = []
class_name_list = []
gender_list = []

if campaign_name == "입회비반값할인":
    campaign_name_list = [1, 0]
elif campaign_name == "입회비무료":
    campaign_name_list = [0, 1]
elif campaign_name == "일반":
    campaign_name_list = [0, 0]
if class_name == "종일":
    class_name_list = [1, 0]
elif class_name == "주간":
    class_name_list = [0, 1]
elif class_name == "야간":
    class_name_list = [0, 0]
if gender == "F":
    gender_list = [1]
elif gender == "M":
    gender_list = [0]
input_data = [count_1, routing_flg, period]
input_data.extend(campaign_name_list)
input_data.extend(class_name_list)
input_data.extend(gender_list)

print(model.predict([input_data]))
print(model.predict_proba([input_data]))


