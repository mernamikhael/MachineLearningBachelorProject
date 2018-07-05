from pymongo import MongoClient
import json
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
client = MongoClient("mongodb://merna:Merna123@ds151558.mlab.com:51558/sos")
db = client['sos']
coll = db['face']
kinds=[]
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i].encode("utf-8")
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
with open('users.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        faceID=row[2]
        export= db.posts.find({"users_id":faceID})
        out= csv.writer(open('user_posts_'+faceID+'.csv', 'wb'),delimiter=(','))
        b=[]
        a=[]
        for items in export:
            a.append(items['message'].encode("utf-8"))
            a.append(items['story'].encode("utf-8"))
            a.append(items['_id'].encode("utf-8"))
            a.append(items['created_time'])
            if(not(items['message']=="below")):
                if(items['story'].__contains__("updated")):
                    a.append("updated")
                    kinds.append("updated")
                else:
                    if(items['story'].__contains__("shared")):
                        a.append("shared")
                        kinds.append("shared")
                    else:
                        if(items['story'].__contains__("added")):
                            a.append("added")
                            kinds.append("added")
                        else:
                            if(items['story'].__contains__("celebrating")):
                                a.append("shared")
                                kinds.append("shared")
                            else:
                                a.append("posted")
                                kinds.append("posted")
            else:
                if(items['story'].__contains__("shared")):
                    a.append("shared")
                    kinds.append("shared")
                else:
                    if(items['story'].__contains__("updated")):
                        a.append("updated")
                        kinds.append("updated")
                    else:
                        if(items['story'].__contains__("added")):
                            a.append("added")
                            kinds.append("added")
            b.append(a)
            out.writerows(b)
            a=[]
            b=[]
# print(kinds)
vectorizer = TfidfVectorizer()
# print(vectorizer.get_feature_names())
X = vectorizer.fit_transform(kinds)
# a = np.array(X)
# a.all(X)
print(len(X.toarray()[1]))
# # print(vectorizer.vocabulary_)
# idf = vectorizer.idf_
# print(vectorizer._tfidf)
# no_topics = 3
# # Run NMF
# nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(X)
# display_topics(nmf, vectorizer.get_feature_names(), 1)

    
     
