import tarfile
import pyprind
import pandas as pd
import os
import numpy as np
import re


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

"""
with tarfile.open('aclImdb_v1.tar.gz','r:gz') as tar:
    tar.extractall()

basepath = 'aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], 
                           ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')
"""

df = pd.read_csv('movie_data.csv', encoding='utf-8')
print(df.head(3))
print(df.shape)
#단어를 특성 벡터로 변환
count = CountVectorizer(ngram_range=(1,1))

docs = np.array(['The sun is shining', 'The weather is sweet', 'The sun is shining, the weather is sweet, and one and one is two'])

bag = count.fit_transform(docs)
#어휘 사전 내용 출력
print(count.vocabulary_)

#특성 벡터 출력 
print(bag.toarray())

#tf-idf를 사용하여 단어 적합성 평가
#tf-idf는 단어 빈도와 역문서 빈도의 곱으로 정의

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

#텍스트 데이터 정제
print(df.loc[0, 'review'][-50:])

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text
print(preprocessor(df.loc[0, 'review'][-50:]))

df['review'] = df['review'].apply(preprocessor)
df['review'].map(preprocessor)

print(df)

#텍스트를 토큰으로 나누기
porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

print(tokenizer('runners like running and thus they run'))
print(tokenizer_porter('runners like running and thus they run'))

#불용어 제거
#nltk.download('stopwords')

stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop])


#문서 분류를 위한 로지스틱 회귀 모델 훈련
X_train = df.loc[:2500, 'review'].values
y_train = df.loc[:2500, 'sentiment'].values
X_test = df.loc[2500:, 'review'].values
y_test = df.loc[2500:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(solver='liblinear', random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=1)
gs_lr_tfidf.fit(X_train, y_train)

print('최적의 매개변수 조합: %s ' % gs_lr_tfidf.best_params_)
print('CV 정확도: %.3f' % gs_lr_tfidf.best_score_)


clf = gs_lr_tfidf.best_estimator_
print('테스트 정확도: %.3f' % clf.score(X_test, y_test))