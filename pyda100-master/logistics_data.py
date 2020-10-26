import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 경고(worning)비표시
import warnings
warnings.filterwarnings('ignore')


# 공장데이터 불러오기
factories = pd.read_csv("pyda100-master/data/tbl_factory.csv", index_col=0)
print(factories)

# 창고데이터 불러오기
warehouses = pd.read_csv("pyda100-master/data/tbl_warehouse.csv", index_col=0)
print(warehouses)

# 비용 테이블
cost = pd.read_csv("pyda100-master/data/rel_cost.csv", index_col=0)
print(cost.head())

# 운송 실적 테이블
trans = pd.read_csv("pyda100-master/data/tbl_transaction.csv", index_col=0)
print(trans.head())

# 운송실적 테이블에 각 테이블을 조인
# 비용 데이터추가
join_data = pd.merge(trans, cost, left_on=["ToFC","FromWH"], right_on=["FCID","WHID"], how="left")
print(join_data.head())

# 공장정보 추가
join_data = pd.merge(join_data, factories, left_on="ToFC", right_on="FCID", how="left")
print(join_data.head())

# 창고정보 추가
join_data = pd.merge(join_data, warehouses, left_on="FromWH", right_on="WHID", how="left")
print(join_data.head())

# 컬럼 정리
join_data = join_data[["TransactionDate","Quantity","Cost","ToFC","FCName","FCDemand","FromWH","WHName","WHSupply","WHRegion"]]
print(join_data.head())

# 북부 데이터 추출
north = join_data.loc[join_data["WHRegion"]=="북부"]
print(north.head())

# 남부데이터 추출
south = join_data.loc[join_data["WHRegion"]=="남부"]
print(south.head())

# 지사의 비용합계 계산
print("북부지사 총비용: " , north["Cost"].sum() , "만원")
print("남부지사 총비용: " , south["Cost"].sum() , "만원")

# 지사의 총운송개수
print("북부지사의 총부품 운송개수: " , north["Quantity"].sum(),  "개")
print("남부지사의 총부품 운송개수: " , south["Quantity"].sum(), "개")

# 부품 1개당 운송비용
tmp = (north["Cost"].sum() / north["Quantity"].sum()) * 10000
print("북부지사의 부품 1개당 운송 비용: " , tmp , "원")
tmp = (south["Cost"].sum() / south["Quantity"].sum()) * 10000
print("남부지사의 부품 1개당 운송 비용: " , tmp , "원")

# 비용을 지사별로 집계
cost_chk = pd.merge(cost, factories, on="FCID", how="left")

# 평균
print("북부지사의 평균 운송 비용：" , cost_chk["Cost"].loc[cost_chk["FCRegion"]=="북부"].mean(), "원")
print("남부지사의 평균 운송 비용：" , cost_chk["Cost"].loc[cost_chk["FCRegion"]=="남부"].mean(), "원")

# 그래프 객체생성
G=nx.Graph()

# 노드 설정
G.add_node("nodeA")
G.add_node("nodeB")
G.add_node("nodeC")

# 엣지 설정
G.add_edge("nodeA","nodeB")
G.add_edge("nodeA","nodeC")
G.add_edge("nodeB","nodeC")

# 좌표 설정
pos={}
pos["nodeA"]=(0,0)
pos["nodeB"]=(1,1)
pos["nodeC"]=(0,1)

# 그리기
nx.draw(G,pos)

# 표시
plt.show()

# 그래프 객체 생성．
G=nx.Graph()

# 노드 설정
G.add_node("nodeA")
G.add_node("nodeB")
G.add_node("nodeC")
G.add_node("nodeD")

# 엣지 설정
G.add_edge("nodeA","nodeB")
G.add_edge("nodeA","nodeC")
G.add_edge("nodeB","nodeC")
G.add_edge("nodeA","nodeD")

# 좌표 설정
pos={}
pos["nodeA"]=(0,0)
pos["nodeB"]=(1,1)
pos["nodeC"]=(0,1)
pos["nodeD"]=(1,0)

# 그리기
nx.draw(G,pos, with_labels=True)

# 표시
plt.show()

# 데이터 불러오기
df_w = pd.read_csv('pyda100-master/data/network_weight.csv')
df_p = pd.read_csv('pyda100-master/data/network_pos.csv')

# 엣지 가중치 리스트화
size = 10
edge_weights = []
for i in range(len(df_w)):
    for j in range(len(df_w.columns)):
        edge_weights.append(df_w.iloc[i][j]*size)

# 그래프 객체 생성
G = nx.Graph()

# 노드 설정
for i in range(len(df_w.columns)):
    G.add_node(df_w.columns[i])

# 엣지 설정
for i in range(len(df_w.columns)):
    for j in range(len(df_w.columns)):
        G.add_edge(df_w.columns[i],df_w.columns[j])

# 좌표 설정
pos = {}
for i in range(len(df_w.columns)):
    node = df_w.columns[i]
    pos[node] = (df_p[node][0],df_p[node][1])

# 그리기
nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)

# 표시
plt.show()

df_tr = pd.read_csv('pyda100-master/data/trans_route.csv', index_col="공장")
print(df_tr.head())

df_pos = pd.read_csv('pyda100-master/data/trans_route_pos.csv')
print(df_pos.head())

# 그래프 객체 생성
G = nx.Graph()

# 노드 설정
for i in range(len(df_pos.columns)):
    G.add_node(df_pos.columns[i])

# 엣지 설정 및 가중치 리스트화
num_pre = 0
edge_weights = []
size = 0.1
for i in range(len(df_pos.columns)):
    for j in range(len(df_pos.columns)):
        if not (i==j):
            # 엣지 추가
            G.add_edge(df_pos.columns[i],df_pos.columns[j])
            # 엣지 가중치 추가
            if num_pre<len(G.edges):
                num_pre = len(G.edges)
                weight = 0
                if (df_pos.columns[i] in df_tr.columns)and(df_pos.columns[j] in df_tr.index):
                    if df_tr[df_pos.columns[i]][df_pos.columns[j]]:
                        weight = df_tr[df_pos.columns[i]][df_pos.columns[j]]*size
                elif(df_pos.columns[j] in df_tr.columns)and(df_pos.columns[i] in df_tr.index):
                    if df_tr[df_pos.columns[j]][df_pos.columns[i]]:
                        weight = df_tr[df_pos.columns[j]][df_pos.columns[i]]*size
                edge_weights.append(weight)
                

# 좌표 설정
pos = {}
for i in range(len(df_pos.columns)):
    node = df_pos.columns[i]
    pos[node] = (df_pos[node][0],df_pos[node][1])
    
# 그리기
nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)

# 표시
plt.show()


# 데이터 불러오기
df_tc = pd.read_csv('pyda100-master/data/trans_cost.csv', index_col="공장")

# 운송 비용 함수
def trans_cost(df_tr,df_tc):
    cost = 0
    for i in range(len(df_tc.index)):
        for j in range(len(df_tr.columns)):
            cost += df_tr.iloc[i][j]*df_tc.iloc[i][j]
    return cost

print("총 운송 비용:" ,trans_cost(df_tr,df_tc))

# 데이터 불러오기
df_demand = pd.read_csv('pyda100-master/data/demand.csv')
df_supply = pd.read_csv('pyda100-master/data/supply.csv')

# 수요측 제약조건
for i in range(len(df_demand.columns)):
    temp_sum = sum(df_tr[df_demand.columns[i]])
    print(df_demand.columns[i], "으로 운송량:", temp_sum, " (수요량:" , df_demand.iloc[0][i] , ")")
    if temp_sum>=df_demand.iloc[0][i]:
        print("수요량을 만족시키고있음")
    else:
        print("수요량을 만족시키지 못하고 있음. 운송경로 재계산 필요")

# 공급측 제약조건
for i in range(len(df_supply.columns)):
    temp_sum = sum(df_tr.loc[df_supply.columns[i]])
    print(df_supply.columns[i], "부터의 운송량:" , temp_sum ," (공급한계:" , df_supply.iloc[0][i], ")")
    if temp_sum<=df_supply.iloc[0][i]:
        print("공급한계 범위내")
    else:
        print("공급한계 초과. 운송경로 재계산 필요")

# 데이터 불러오기
df_tr_new = pd.read_csv('pyda100-master/data/trans_route_new.csv', index_col="공장")
print(df_tr_new)

# 총 운송비용 재계산 
print("총 운송 비용(변경 후):", trans_cost(df_tr_new,df_tc))

# 제약조건 계산함수
# 수요측
def condition_demand(df_tr,df_demand):
    flag = np.zeros(len(df_demand.columns))
    for i in range(len(df_demand.columns)):
        temp_sum = sum(df_tr[df_demand.columns[i]])
        if (temp_sum>=df_demand.iloc[0][i]):
            flag[i] = 1
    return flag
            
# 공급측
def condition_supply(df_tr,df_supply):
    flag = np.zeros(len(df_supply.columns))
    for i in range(len(df_supply.columns)):
        temp_sum = sum(df_tr.loc[df_supply.columns[i]])
        if temp_sum<=df_supply.iloc[0][i]:
            flag[i] = 1
    return flag

print("수요조건 계산결과:", condition_demand(df_tr_new,df_demand))
print("공급조건 계산결과:", condition_supply(df_tr_new,df_supply))