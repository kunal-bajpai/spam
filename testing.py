import sys,gensim,pickle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
sys.path.append("src")
from fetch import *
from sparsify import *
from correlate import *

#read testing data and parse into sparse vectors
x,y = getData("","")

#convert to gensim compatible corpus
corpus = gensim.matutils.Sparse2Corpus(x,False)

#load lda model saved while training
lda = gensim.utils.SaveLoad.load('lda_model')

#derive topic proportions for test data
corpus_lda = lda[corpus]

#convert to sparse representation
topDistSparse = sparsify(corpus_lda)

#find correlation between topics
corrTopics = correlate(topDistSparse)

#load previously trained classifier
clf = pickle.load(open("classifier"))

#predict labels and find (precision, recall, f1-score)
res = clf.predict(corrTopics.toarray())
print (metrics.precision_score(y,res),metrics.recall_score(y,res),metrics.f1_score(y,res))
