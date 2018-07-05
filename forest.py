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
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# import sys  
# reload(sys)  # Reload does the trick!
# sys.setdefaultencoding('utf-8')
# print(sys.getdefaultencoding())
# stop_words = text.ENGLISH_STOP_WORDS.union(["xD","xd","XD"])
# words = [word.lower() for word in words if
#             not word in set(stopwords.words('english')) and not word.isdigit()]
# stop_words.append('.')
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
    # print(wordnet.synsets(word))
    # print(get_wordnet_pos(words_tag.get(word)))
    # if word.isalpha() and wordnet.synsets(word):
    return  lemmatiser.lemmatize(word, get_wordnet_pos(words_tag.get(word)))
    # else:
    #     return word

def clean(words):
    # words = re.sub('[^a-zA-Z]', '', words.lower()).split()
    tknzr = TweetTokenizer()
    # tokenizer = RegexpTokenizer('\w+|\S+')
    # words=nltk.word_tokenize(words.lower())
    words = tknzr.tokenize(words)
    exclude = set(string.punctuation)
    words2 = [word for word in words if
            not word in exclude]
    words_tag = dict(pos_tag(words))
    words = [word.lower() for word in words2 if
            not word in nltk.corpus.stopwords.words('english') and not word.isdigit()]
    # print(words)
    words = [lima(word, words) for word in words]
    # print(words)
    words = ' '.join(words)
    # print(words)
    return words

stopwords = stopwords.words('english')
# stopwords.union('sally')
# operators = set(('sally'))
# stop = set(nltk.corpus.stopwords.words('english')) + operators
# print(stopwords)
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
with open('poststrain.csv') as File:
    tfidfReader = csv.reader(File)
    for row in tfidfReader:
        chapters.append(clean(row[0]).encode('utf-8'))
        datetime_object = datetime.strptime(row[3][:-4],"%Y-%m-%dT%H:%M:%S+")
        urls = extractor.find_urls(row[0])
        hour=datetime_object.hour
        minute=datetime_object.minute
        # t=translator.detect(json.dumps(row[0].decode('utf-8')))
        # lang.append(t.lang)
        houra.append(hour)
        minutea.append(minute)
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


num_chapters = len(chapters)
fvs_lexical = np.zeros((len(chapters), 11), np.float64)
fvs_punct = np.zeros((len(chapters), 11), np.float64)
i=0
    
for e, ch_text in enumerate(chapters):
    fvs_lexical[e, 3] = houra[i]
    fvs_lexical[e, 4] = minutea[i]
    fvs_lexical[e, 5] = urlLength[i]
    fvs_lexical[e, 6] = posted[i]
    fvs_lexical[e, 7] = added[i]
    fvs_lexical[e, 8] = shared[i]
    fvs_lexical[e, 9] = updated[i]
    # fvs_lexical[e, 10]= lang[i]
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


        
            # # Commas per sentence
            # fvs_punct[e, 0] = tokens.count(',') / float(len(sentences))
            # # Semicolons per sentence
            # fvs_punct[e, 1] = tokens.count(';') / float(len(sentences))
            # # Colons per sentence
            # fvs_punct[e, 2] = tokens.count(':') / float(len(sentences))
    i+=1

chapters=[]
houra=[]
minutea=[]
urlLength=[]
shared=[]
updated=[]
posted=[]
added=[]
lang=[]
with open('postTest.csv') as File:
    tfidfReader = csv.reader(File)
    for row in tfidfReader:
        chapters.append(clean(row[0]).encode('utf-8'))
        datetime_object = datetime.strptime(row[3][:-4],"%Y-%m-%dT%H:%M:%S+")
        urls = extractor.find_urls(row[0])
        hour=datetime_object.hour
        minute=datetime_object.minute
        # t=translator.detect(json.dumps(row[0].decode('utf-8')))
        # lang.append(t.lang)
        houra.append(hour)
        minutea.append(minute)
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


num_chapters = len(chapters)
fvs_lexicalT = np.zeros((len(chapters), 11), np.float64)
fvs_punctT = np.zeros((len(chapters), 11), np.float64)
i=0
    
for e, ch_text in enumerate(chapters):
    fvs_lexicalT[e, 3] = houra[i]
    fvs_lexicalT[e, 4] = minutea[i]
    fvs_lexicalT[e, 5] = urlLength[i]
    fvs_lexicalT[e, 6] = posted[i]
    fvs_lexicalT[e, 7] = added[i]
    fvs_lexicalT[e, 8] = shared[i]
    fvs_lexicalT[e, 9] = updated[i]
    # fvs_lexical[e, 10]= lang[i]
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
            fvs_lexicalT[e, 0] = words_per_sentence.mean()
            # sentence length variation
            fvs_lexicalT[e, 1] = words_per_sentence.std()
            # Lexical diversity
            fvs_lexicalT[e, 2] = len(vocab) / float(len(words))


        
            # # Commas per sentence
            # fvs_punct[e, 0] = tokens.count(',') / float(len(sentences))
            # # Semicolons per sentence
            # fvs_punct[e, 1] = tokens.count(';') / float(len(sentences))
            # # Colons per sentence
            # fvs_punct[e, 2] = tokens.count(':') / float(len(sentences))
    i+=1

chapters=[]
houra=[]
minutea=[]
urlLength=[]
shared=[]
updated=[]
posted=[]
added=[]
lang=[]
with open('postout.csv') as File:
    tfidfReader = csv.reader(File)
    for row in tfidfReader:
        chapters.append(clean(row[0]).encode('utf-8'))
        datetime_object = datetime.strptime(row[3][:-4],"%Y-%m-%dT%H:%M:%S+")
        urls = extractor.find_urls(row[0])
        hour=datetime_object.hour
        minute=datetime_object.minute
        # t=translator.detect(json.dumps(row[0].decode('utf-8')))
        # lang.append(t.lang)
        houra.append(hour)
        minutea.append(minute)
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


num_chapters = len(chapters)
fvs_lexicalO = np.zeros((len(chapters), 11), np.float64)
fvs_punctO = np.zeros((len(chapters), 11), np.float64)
i=0
# print(chapters)
for e, ch_text in enumerate(chapters):
    fvs_lexicalO[e, 3] = houra[i]
    fvs_lexicalO[e, 4] = minutea[i]
    fvs_lexicalO[e, 5] = urlLength[i]
    fvs_lexicalO[e, 6] = posted[i]
    fvs_lexicalO[e, 7] = added[i]
    fvs_lexicalO[e, 8] = shared[i]
    fvs_lexicalO[e, 9] = updated[i]
    # fvs_lexical[e, 10]= lang[i]
    ch_text = unicode(ch_text, errors='ignore')

    # note: the nltk.word_tokenize includes punctuation
    # print(ch_text)
    if(ch_text):
        # ch_text.encode('utf-8')
        tokens = nltk.word_tokenize(ch_text.lower())
        words = word_tokenizer.tokenize(ch_text.lower())
        words=[word for word in words if
         not word in stopwords]
        # print(words)
        sentences = sentence_tokenizer.tokenize(ch_text)
        if(words):
            vocab = set(words)
            # print(i)
            words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
                                        for s in sentences])
            # print('w',words_per_sentence)
            # average number of words per sentence
            fvs_lexicalO[e, 0] = words_per_sentence.mean()
            # sentence length variation
            fvs_lexicalO[e, 1] = words_per_sentence.std()
            # Lexical diversity
            fvs_lexicalO[e, 2] = len(vocab) / float(len(words))


        
            # # Commas per sentence
            # fvs_punct[e, 0] = tokens.count(',') / float(len(sentences))
            # # Semicolons per sentence
            # fvs_punct[e, 1] = tokens.count(';') / float(len(sentences))
            # # Colons per sentence
            # fvs_punct[e, 2] = tokens.count(':') / float(len(sentences))
    i+=1
# print(fvs_lexical)
# print(fvs_punct)
# apply whitening to decorrelate the features
# Before running k-means, it is beneficial to rescale each feature dimension of the observation set with whitening.
#  Each feature is divided by its standard deviation across all observations to give it unit variance.
# fvs_lexical = whiten(fvs_lexical)
# fvs_lexicalT=whiten(fvs_lexicalT)
# fvs_punct = whiten(fvs_punct)
fvs_lexical= normalize(fvs_lexical,norm='l2', axis=1, copy=True, return_norm=False)
fvs_lexicalT=normalize(fvs_lexicalT,norm='l2', axis=1, copy=True, return_norm=False)
fvs_lexicalO=normalize(fvs_lexicalO,norm='l2', axis=1, copy=True, return_norm=False)
# km = KMeans(n_clusters=1, init='k-means++', n_init=10, verbose=0)
# k=km.fit(fvs_lexical)
# print(k)

xx, yy= np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate train data

X_train = fvs_lexical
print(X_train)
# print('train',X_train)
# Generate some regular novel observations
X = np.random.rand(10, 11)
X_test = fvs_lexicalT
# print('test',X_test)
# Generate some abnormal novel observations
X= 0.3 * np.random.randn(10, 11)
X_outliers = fvs_lexicalO
# X_outliers = np.random.uniform(low=-4, high=4, size=(20, 11))
# print('outliers',X_outliers)
# fit the model
print(X_outliers)
clf = IsolationForest(max_samples=160, random_state=rng,contamination=0.2)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
print(y_pred_outliers)
# print(y_pred_test)
# print(y_pred_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

print('n_error_train: ',n_error_train)
print('n_error_outliers: ',n_error_outliers)
print('n_error_test: ',n_error_test)

# plot the line, the points, and the nearest vectors to the plane
# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

# Z = Z.reshape(xx.shape)

# plt.title("Novelty Detection")
# plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
# a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
# plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

# s = 40
# b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
# b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
#                  edgecolors='k')
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
#                 edgecolors='k')
# plt.axis('tight')
# plt.xlim((-5, 5))
# plt.ylim((-5, 5))
# plt.legend([a.collections[0], b1, b2, c],
#            ["learned frontier", "training observations",
#             "new regular observations", "new abnormal observations"],
#            loc="upper left",
#            prop=matplotlib.font_manager.FontProperties(size=11))
# plt.xlabel(
#     "error train: %d/200 ; errors novel regular: %d/40 ; "
#     "errors novel abnormal: %d/40"
#     % (n_error_train, n_error_test, n_error_outliers))
# plt.show()