import sys,gensim,pickle
from sklearn.ensemble import RandomForestClassifier
sys.path.append("src")
from fetch import *
from sparsify import *
from correlate import *

#Read training data from files and parse into sparse vectors
x,y = getData("small_tfidfvector_byhost_onlybody.csv.txt","webspam-uk2006-set1-labels.txt")

#Convert to gensim compatible corpus
corpus = gensim.matutils.Sparse2Corpus(x,False)

#derive topic model
lda = gensim.models.ldamodel.LdaModel(corpus,num_topics=100)

#save model to load while testing
lda.save('lda_model')

#derive topic proportions for training data
corpus_lda = lda[corpus]

#convert topic proportions to sparse representation
topDistSparse = sparsify(corpus_lda)

#find correlation between topics
corrTopics = correlate(topDistSparse)

#train and save classifier
clf = RandomForestClassifier(n_estimators=50)
clf.fit(corrTopics.toarray(),y)
pickle.save(clf,open("classifier","wb"))
