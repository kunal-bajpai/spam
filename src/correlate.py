from sklearn.feature_extraction import DictVectorizer
import numpy as np

def correlate(x):
	xnew=[]
	for ind,row in enumerate(x):
		dnew={}
		rn=0
		for i in range(x.shape[1]):
			for j in range(i+1,x.shape[1]):
				rn = rn + 1
				if (x[ind,i]!=0 and x[ind,j]!=0):
					dnew[rn]=x[ind,i]*x[ind,j]
		xnew.append(dnew)
		print ind
	dv = DictVectorizer(sparse=True,dtype=np.float)
	return dv.fit_transform(xnew)
