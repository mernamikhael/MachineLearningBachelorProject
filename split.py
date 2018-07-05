from sklearn.svm import SVC
import numpy as np
import nltk
import glob
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten
import csv
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
from nltk.corpus import wordnet
from datetime import datetime
import idna
import uritools
import urlextract
from googletrans import Translator
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.preprocessing import normalize
# from sklearn.model_selection import GridSearchCV
import scipy
import scipy.stats
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from svc import grid_search

stopwords = stopwords.words('english')

def get_wordnet_pos(treebank_tag):
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return  wordnet.NOUN

def lima(word, words):
    
    # print(word)
    lemmatiser = WordNetLemmatizer()
    words_tag = dict(pos_tag(words))
    return  lemmatiser.lemmatize(word, get_wordnet_pos(words_tag.get(word)))

def clean(words):
    tknzr = TweetTokenizer()

    words = tknzr.tokenize(words)
    exclude = set(string.punctuation)
    words2 = [word for word in words if
            not word in exclude]
    words_tag = dict(pos_tag(words))
    words = [word.lower() for word in words2 if
            not word in nltk.corpus.stopwords.words('english') and not word.isdigit()]
    words = [lima(word, words) for word in words]
    words = ' '.join(words)
    return words

extractor = urlextract.URLExtract()
translator = Translator()
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
chapters=[]
houra=[]
minutea=[]
urlLength=[]
shared=[]
updated=[]
posted=[]
added=[]
lang=[]
hashtag=[]
label=[]
daya=[]
montha=[]

with open('csv/merna.csv','r') as File:
    tfidfReader = csv.reader(File)
    for row in tfidfReader:
        chapters.append(clean(row[0]).encode('utf-8'))
        datetime_object = datetime.strptime(row[3][:-4],"%Y-%m-%dT%H:%M:%S+")
        urls = extractor.find_urls(row[0])
        hour=datetime_object.hour
        minute=datetime_object.minute
        hashcount=row[0].count('#')
        hashtag.append(hashcount)
        day=datetime_object.day
        month=datetime_object.month
        # t=translator.detect(json.dumps(row[0].decode('utf-8')))
        # lang.append(t.lang)
        houra.append(hour)
        minutea.append(minute)
        daya.append(day)
        montha.append(month)
        urlLength.append(len(urls))
        
        if(row[4]=='posted'):
            posted.append(1)
            added.append(0)
            updated.append(0)
            shared.append(0)
        elif(row[4]=='added'):
            posted.append(0)
            added.append(1)
            updated.append(0)
            shared.append(0)
        elif(row[4]=='shared'):
            posted.append(0)
            added.append(0)
            updated.append(0)
            shared.append(1)
        elif(row[4]=='updated'):
            posted.append(0)
            added.append(0)
            updated.append(1)
            shared.append(0)

        label.append(row[6])

# print(label)
num_chapters = len(chapters)
fvs_lexical = np.zeros((len(chapters), 13), np.float64)
i=0
# print('t',hashtag)   
for e, ch_text in enumerate(chapters):
    fvs_lexical[e, 3] = houra[i]
    fvs_lexical[e, 4] = minutea[i]
    fvs_lexical[e, 5] = urlLength[i]
    fvs_lexical[e, 6] = posted[i]
    fvs_lexical[e, 7] = added[i]
    fvs_lexical[e, 8] = shared[i]
    fvs_lexical[e, 9] = updated[i]
    fvs_lexical[e, 10]= hashtag[i]
    fvs_lexical[e, 11]= montha[i]
    fvs_lexical[e, 12]= daya[i]
    ch_text = unicode(ch_text, errors='ignore')

    # note: the nltk.word_tokenize includes punctuation
    # print(ch_text)
    if(ch_text):
        # ch_text.encode('utf-8')
        tokens = nltk.word_tokenize(ch_text.lower())
        words = word_tokenizer.tokenize(ch_text.lower())
        words=[word for word in words if
         not word in stopwords]
        sentences = sentence_tokenizer.tokenize(ch_text)
        if(words):
            vocab = set(words)
            # print(i)
            words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
                                        for s in sentences])
            # print('w',words_per_sentence)
            # average number of words per sentence
            fvs_lexical[e, 0] = words_per_sentence.mean()
            # sentence length variation
            fvs_lexical[e, 1] = words_per_sentence.std()
            # Lexical diversity
            fvs_lexical[e, 2] = len(vocab) / float(len(words))

    i+=1

# chapters=[]
# houra=[]
# minutea=[]
# urlLength=[]
# shared=[]
# updated=[]
# posted=[]
# added=[]
# lang=[]
# hashtag=[]
# with open('postout.csv') as File:
#     tfidfReader = csv.reader(File)
#     for row in tfidfReader:
#         chapters.append(clean(row[0]).encode('utf-8'))
#         datetime_object = datetime.strptime(row[3][:-4],"%Y-%m-%dT%H:%M:%S+")
#         urls = extractor.find_urls(row[0])
#         hour=datetime_object.hour
#         minute=datetime_object.minute
#         hashcount=row[0].count('#')
#         hashtag.append(hashcount)
#         # t=translator.detect(json.dumps(row[0].decode('utf-8')))
#         # lang.append(t.lang)
#         houra.append(hour)
#         minutea.append(minute)
#         urlLength.append(len(urls))
#         if(row[4]=='posted'):
#             posted.append(1)
#             added.append(0)
#             updated.append(0)
#             shared.append(0)
#         elif(row[4]=='added'):
#             posted.append(0)
#             added.append(1)
#             updated.append(0)
#             shared.append(0)
#         elif(row[4]=='shared'):
#             posted.append(0)
#             added.append(0)
#             updated.append(0)
#             shared.append(1)
#         elif(row[4]=='updated'):
#             posted.append(0)
#             added.append(0)
#             updated.append(1)
#             shared.append(0)


# num_chapters = len(chapters)
# fvs_lexicalT = np.zeros((len(chapters), 11), np.float64)
# fvs_punctT = np.zeros((len(chapters), 11), np.float64)
# i=0
    
# for e, ch_text in enumerate(chapters):
#     fvs_lexicalT[e, 3] = houra[i]
#     fvs_lexicalT[e, 4] = minutea[i]
#     fvs_lexicalT[e, 5] = urlLength[i]
#     fvs_lexicalT[e, 6] = posted[i]
#     fvs_lexicalT[e, 7] = added[i]
#     fvs_lexicalT[e, 8] = shared[i]
#     fvs_lexicalT[e, 9] = updated[i]
#     fvs_lexicalT[e, 10]= hashtag[i]
#     ch_text = unicode(ch_text, errors='ignore')

#     # note: the nltk.word_tokenize includes punctuation
#     # print(ch_text)
#     if(ch_text):
#         # ch_text.encode('utf-8')
#         tokens = nltk.word_tokenize(ch_text.lower())
#         words = word_tokenizer.tokenize(ch_text.lower())
#         words=[word for word in words if
#          not word in stopwords]
#         sentences = sentence_tokenizer.tokenize(ch_text)
#         if(words):
#             vocab = set(words)
#             # print(i)
#             words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
#                                         for s in sentences])
#             # print('w',words_per_sentence)
#             # average number of words per sentence
#             fvs_lexicalT[e, 0] = words_per_sentence.mean()
#             # sentence length variation
#             fvs_lexicalT[e, 1] = words_per_sentence.std()
#             # Lexical diversity
#             fvs_lexicalT[e, 2] = len(vocab) / float(len(words))

#     i+=1


# Cs = [0.001, 0.01, 0.1, 1, 10,100,1000,1005,1200,2000,1500,1700,558.7589367831655]
# gammas = [0.001, 0.01, 0.1, 1,0.5,0.0007922028396163162]
# kernel=['linear','rbf']
# param_grid = {'C': Cs, 'gamma' : gammas}
# tuned_parameters = {'C': [0.0001, 2000],
#                     "gamma": [3, 6, 9, None],
#                     "max_features": ["auto","log2",None],
#                     "class_weight": [None, 'balanced']}
# l={'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
#   'kernel': ['rbf','linear'], 'class_weight':['balanced', None]}

X_train, X_test, y_train, y_test = train_test_split(fvs_lexical,label,test_size=0.2)
# grid_search = RandomizedSearchCV(svm.SVC(),l,cv=7)
grid_search= svm.SVC(C=200,kernel='linear')
grid_search.fit(X_train, y_train)
# print grid_search.best_params_
# print grid_search.best_score_
# print classification_report(,label)
y_pred=grid_search.predict(X_test)
print accuracy_score(y_test, y_pred)
print(f1_score(y_test, y_pred, average="macro"))
print(precision_score(y_test, y_pred, average="macro"))
print(recall_score(y_test, y_pred, average="macro"))   
print(y_pred)

