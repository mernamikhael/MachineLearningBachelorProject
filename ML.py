from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction import text
from sklearn.pipeline import Pipeline
import numpy
import scipy
import csv
import pandas as pd
from langdetect import detect
import nltk
import sys
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import NMF, LatentDirichletAllocation


# def _calculate_languages_ratios(text):
#     languages_ratios = {}
#     tokens = wordpunct_tokenize(text)
#     words = [word.lower() for word in tokens]
#     # Compute per language included in nltk number of unique stopwords appearing in analyzed text
#     for language in stopwords.fileids():
#         stopwords_set = set(stopwords.words(language))
#         words_set = set(words)
#         common_elements = words_set.intersection(stopwords_set)

#         languages_ratios[language] = len(common_elements) # language "score"

#     return languages_ratios


# #----------------------------------------------------------------------
# def detect_language(text):
#     ratios = _calculate_languages_ratios(text)
#     most_rated_language = max(ratios, key=ratios.get)
#     return most_rated_language

# text="Je t'aime, i love u"
# language = detect_language(text)
# print language

# text_clf = Pipeline([('vect', TfidfVectorizer()),('tfidf',TfidfTransformer()),('clf', MultinomialNB())])

# text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
############################
# ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
# NON_ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words()) - ENGLISH_STOPWORDS
 
# STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}
 
# def get_language(text):
#     words = set(nltk.wordpunct_tokenize(text.lower()))
#     return max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key = lambda x: x[1])[0]
 
 
# def is_english(text):
#     text = text.lower()
#     words = set(nltk.wordpunct_tokenize(text))
#     return len(words & ENGLISH_STOPWORDS) > len(words & NON_ENGLISH_STOPWORDS)

# lang = detect("Je t'aime, i love u")

# print lang
# # to read from csv

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i].encode("utf-8")
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
corpus = []
with open('user_likes_10214181385024102.csv') as File:
    tfidfReader = csv.reader(File)
    for row in tfidfReader:
        corpus.append((row[1]))
# print(corpus)
vectorizer = TfidfVectorizer(min_df=0.02,max_df=0.7,stop_words='english')
# print(vectorizer.get_feature_names())
X = vectorizer.fit_transform(corpus)
print(vectorizer.vocabulary_)
idf = vectorizer.idf_

no_topics = 10
# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(X)
display_topics(nmf, vectorizer.get_feature_names(), 4)

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(X)
# display_topics(lda,vectorizer.get_feature_names(), 1)



# text_clf = text_clf.fit(corpus,corpus)

# print(vectorizer.vocabulary_)
# print(X.toarray())
# print dict(zip(vectorizer.get_feature_names(), idf))

# list of text documents
# text1 = ["<html> The quick brown fox jumped over the lazy dog. </html>",
# 		"The DOG.",
# 		"The fox"]

# # create the transform  
# vectorizer = TfidfVectorizer()

# # tokenize and build vocab
# # to edit stopword
# # stop_words = text.ENGLISH_STOP_WORDS.union(["dog"])
# # vectorizer.stop_words=stop_words
# vectorizer.stop_words='english'
# print(vectorizer.fit_transform(text1))
# # summarize
# print(vectorizer.vocabulary_)
# print(vectorizer.idf_)
# # encode document
# vector = vectorizer.transform([text1[0]])
# # summarize encoded vector
# print(vector.shape)
# print(vector.toarray())