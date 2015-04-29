from sklearn.feature_extraction import DictVectorizer
import numpy as np

def sparsify(x):
	xnew=[]
	for row in x:
		d={}
		for tup in row:
			d[tup[0]]=tup[1]
		xnew.append(d)
	dv = DictVectorizer(sparse=True,dtype=np.float)
	return dv.fit_transform(xnew)
