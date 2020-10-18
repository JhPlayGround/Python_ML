import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

uselog = pd.read_csv('pyda100-master/data/use_log.csv')
print(uselog.isnull().sum())

customer = pd.read_csv('pyda100-master/data/customer_join.csv')
print(customer.isnull().sum())

customer_clustering = customer[["mean", "median","max", "min", "membership_period"]]
print(customer_clustering.head())


sc = StandardScaler()
customer_clustering_sc = sc.fit_transform(customer_clustering)

kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit(customer_clustering_sc)
customer_clustering["cluster"] = clusters.labels_
print(customer_clustering["cluster"].unique())
print(customer_clustering.head())

customer_clustering.columns = ["월평균값","월중앙값", "월최댓값", "월최솟값","회원기간", "cluster"]
print(customer_clustering.groupby("cluster").count())
print(customer_clustering.groupby("cluster").mean())


X = customer_clustering_sc
pca = PCA(n_components=2)
pca.fit(X)
x_pca = pca.transform(X)
pca_df = pd.DataFrame(x_pca)
pca_df["cluster"] = customer_clustering["cluster"]



for i in customer_clustering["cluster"].unique():
    tmp = pca_df.loc[pca_df["cluster"]==i]
    plt.scatter(tmp[0], tmp[1])
    plt.show()