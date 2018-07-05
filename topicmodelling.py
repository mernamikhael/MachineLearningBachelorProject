
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.decomposition import NMF, LatentDirichletAllocation

# def display_topics(model, feature_names, no_top_words):
#     for topic_idx, topic in enumerate(model.components_):
#         print "Topic %d:" % (topic_idx)
#         print " ".join([feature_names[i]
#                         for i in topic.argsort()[:-no_top_words - 1:-1]])

# dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
# documents = dataset.data

# no_features = 1000

# # NMF is able to use tf-idf
# tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
# tfidf = tfidf_vectorizer.fit_transform(documents)
# tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
# tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
# tf = tf_vectorizer.fit_transform(documents)
# tf_feature_names = tf_vectorizer.get_feature_names()

# no_topics = 20

# # Run NMF
# nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# # Run LDA
# lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

# no_top_words = 10
# display_topics(nmf, tfidf_feature_names, no_top_words)
# display_topics(lda, tf_feature_names, no_top_words)

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

# usedhash=[]
# def display_topics(model, feature_names, no_top_words):
#     for topic_idx, topic in enumerate(model.components_):
#         l= "Topic %d:" % (topic_idx)
#         l= " ".join([feature_names[i].encode("utf-8")
#                         for i in topic.argsort()[:-no_top_words - 1:-1]])
#         usedhash.append(feature_names[i].encode("utf-8"))
# # from nltk.corpus import wordnet
# # from nltk import pos_tag,word_tokenize
# # from nltk.corpus import stopwords
# # from nltk.stem.porter import PorterStemmer
# # from nltk.stem import  WordNetLemmatizer
# # import csv
# # def clean(words):
# #         lemmatiser = WordNetLemmatizer()
# #         words = re.sub('[^a-zA-Z]', '__', words.lower()).split()

# #         words_tag = dict(pos_tag(words))
# #         words = [lemmatiser.lemmatize(word, get_wordnet_pos(words_tag.get(word)))for word in words if
# #                  not word in set(stopwords.words('english'))]
# #         words = ' '.join(words)
# #         return words


# # corpus = []
# # with open('status.csv',newline='') as File:
# #      spamreader = csv.reader(File)
# #      for row in spamreader:
# #          print("l")
# #          corpus.append(clean(row[1]))
# #      print(corpus)

# corpus=['tes','tes','tes','lolo','ko','lolo','ko','lolo','lolo']
# vectorizer = TfidfVectorizer(min_df=0.02,max_df=0.7,stop_words='english')
# # print(vectorizer.get_feature_names())
# X = vectorizer.fit_transform(corpus)
# idf = vectorizer.idf_

# no_topics = 3
# # Run NMF
# nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(X)
# display_topics(nmf, vectorizer.get_feature_names(), 1)
# print usedhash
