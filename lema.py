from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from langdetect import detect
import numpy
import scipy
import re
import csv
import string
from sklearn.decomposition import NMF, LatentDirichletAllocation


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i].encode("utf-8")
                        for i in topic.argsort()[:-no_top_words - 1:-1]])

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
    words = [word for word in words if
            not word in exclude]
    words_tag = dict(pos_tag(words))
    words = [word.lower() for word in words if
            not word in set(stopwords.words('english')) and not word.isdigit()]
    # print(words)
    words = [lima(word, words) for word in words]
    # print(words)
    words = ' '.join(words)
    print(words)
    return words

    # 'groups.csv', 'w', newline=''
    #  lemmatiser.lemmatize(word, get_wordnet_pos(words_tag.get(word)))
    # words = re.sub('[^a-zA-Z]', 'pp', words.lower()).split()
corpus = []
with open('user_posts.csv') as File:
    spamreader = csv.reader(File)
    for row in spamreader:
        corpus.append(clean(row[0]))
    # print(corpus)
vectorizer = TfidfVectorizer(max_df=0.7,min_df=5)
# tokenize and build vocab
# to edit stopword
stop_words = text.ENGLISH_STOP_WORDS.union(["xD","xd","XD"])
vectorizer.stop_words=stop_words

# vectorizer.ngram_range=(1,2)
X = vectorizer.fit_transform(corpus)

idf = vectorizer.idf_

no_topics = 10
# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(X)
# display_topics(nmf, vectorizer.get_feature_names(), 2)
# summarize
# print(vectorizer.get_feature_names())


# print(vectorizer.vocabulary_)
# vectorizer.use_idf
# print(vectorizer.idf_)
# # encode document
# vector = vectorizer.transform(corpus)
# # # summarize encoded vector
# print(vector.shape)
# print(vector.toarray())
