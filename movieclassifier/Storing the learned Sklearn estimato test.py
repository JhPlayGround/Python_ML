import pickle
import re
import os 
import numpy as np

from vectorizer import vect 

cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'rb'))

label = {0:'음성', 1:'양성'}

example = ['I like this movie']
X = vect.transform(example)
print('예측: %s\n확률: %.2f%%' % (label[clf.predict(X)[0]],  np.max(clf.predict_proba(X))*100))