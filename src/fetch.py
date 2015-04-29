import numpy as np

from sklearn.feature_extraction import DictVectorizer

def getData(vectors,labels):
    x=[]
    y=[]

    print "\nImporting data..."
    file = open(vectors)
    j=0
    for row in file:
        j=0
        temp = row.split(',')
        d = dict()
        for i in temp[2:]:
            d[i.split(':')[0]] = float(i.split(':')[1])
        file1 = open(labels)
        for row in file1:
            temp1 = row.split(' ')
            if (temp1[0] == temp[1]) or temp[1] in temp1[0]:
                if temp1[1] == 'spam':
                    y.append(1)
                elif temp1[1] == 'normal':
                    y.append(0)
                if temp1[1]=='spam' or temp1[1]=='normal':
	                j=1
	                x.append(d)
                break
        file1.close()
    file.close()
    print "\nConverting to sparse matrix..."
    dv = DictVectorizer(sparse=True,dtype=np.float)
    x = dv.fit_transform(x)
    print "Obtained %d samples with %d features" % x.shape
    return x, y
