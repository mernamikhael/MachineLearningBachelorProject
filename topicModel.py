from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction import text
from sklearn.pipeline import Pipeline
import numpy
import scipy
import csv
from langdetect import detect
import nltk
import sys
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import NMF, LatentDirichletAllocation

# message=['lolo','tes','lolo','tes','tes','ko']
# def topic(message):
# lang=detect(messa)
def topic(user,userA,scoreA,scoreB,weightTopic):

    message=[]
    t=[]
    path='dataset\user_likes_'+user+'.csv'
    pathA='dataset\user_likes_'+userA+'.csv'
    with open(path,'rb') as f:
        reader=csv.reader(f)
        for row in reader:
            message.append(row[2])
    # print message
        
    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            l= "Topic %d:" % (topic_idx)
            l= " ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]])
            t.append(feature_names[i])

    vectorizer = TfidfVectorizer(min_df=0.02,max_df=0.7,stop_words='english')
    # print(vectorizer.get_feature_names())
    X = vectorizer.fit_transform(message)

    no_topics = 2
    # Run NMF
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(X)
    display_topics(nmf, vectorizer.get_feature_names(), 1)

    # anomalyUser
    tA=[]
    messageA=[]
    with open(pathA,'rb') as fa:
        readerA=csv.reader(fa)
        for rowA in readerA:
            messageA.append(rowA[2])
        
    def display_topicsA(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            lA= "Topic %d:" % (topic_idx)
            lA= " ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]])
            tA.append(feature_names[i])

    vectorizerA = TfidfVectorizer(min_df=0.02,max_df=0.7,stop_words='english')
    # print(vectorizer.get_feature_names())
    XA = vectorizerA.fit_transform(messageA)

    no_topicsA =2
    # Run NMF
    # print vectorizerA.get_feature_names()
    nmfA = NMF(n_components=no_topicsA, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(XA)
    display_topicsA(nmfA, vectorizerA.get_feature_names(), 1)

    for i in tA:
        if(not(t.__contains__(i))):
            scoreA+=(1*weightTopic)
            # break
        else:
            scoreB+=(1*weightTopic)
            
    return scoreA,scoreB