import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv('movie_data.csv', encoding='utf-8')
print(df.head())
count = CountVectorizer(stop_words='english',max_df=.1,max_features=5000)

X= count.fit_transform(df['review'].values)
lda = LatentDirichletAllocation(n_components=10, random_state=1, learning_method='batch') #learning_method='online'가능
X_topics = lda.fit_transform(X)
print(lda.components_.shape)

n_top_words = 5
feature_names = count.get_feature_names()

for topic_idx, topic in enumerate(lda.components_):
    print("토픽 %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i] for i in topic.argsort() [:-n_top_words - 1:-1]]))


horror = X_topics[:, 5].argsort()[::-1]

for iter_idx, movie_idx in enumerate(horror[:3]):
    print('\n공포 영화 #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:300], '...')