from googletrans import Translator
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def language(lang,langA,total_post,total_postA,scoreA,scoreB,weightlang):
    if(len(lang)>0):
        vectorizer = TfidfVectorizer(min_df=0.05)
        X = vectorizer.fit_transform(lang)
        z=zip(vectorizer.get_feature_names(),np.ravel(X.sum(axis=0)))
        # sort descending according to count of each language
        z.sort(key = lambda t: t[1],reverse=True)
        langSum=np.ravel(X.sum(axis=0))
        langUsed=vectorizer.get_feature_names()
        maxlang=max(z,key=lambda x:x[1])
        
        if(len(langA)>0):
            vectorizerA = TfidfVectorizer(min_df=0.05)
            XA = vectorizerA.fit_transform(langA)
            zA=zip(vectorizerA.get_feature_names(),np.ravel(XA.sum(axis=0)))
            # sort descending according to count of each language
            zA.sort(key = lambda t: t[1],reverse=True)
            langSumA=np.ravel(XA.sum(axis=0))
            langUsedA=vectorizerA.get_feature_names()
            maxlangA=max(zA,key=lambda x:x[1])

            # print langUsed
            # print langUsedA

            # print langSum
            # print langSumA
            # calculate the score 
            
            for row in langUsedA:
                if(not(langUsed.__contains__(row))):
                    scoreA+=2  # first time to write with it is def. a hack
                else:
                    scoreB+=1
            normalLang=0
            anomlayLang=0
            while(anomlayLang<len(langUsedA) and normalLang<len(langUsed)):
                if(langUsedA[anomlayLang]==langUsed[normalLang]):
                    if(langSum[normalLang]<langSumA[anomlayLang]):
                        scoreA+=(1*weightlang)
                    else:
                        scoreB+=(1*weightlang)
                    normalLang+=1
                    anomlayLang+=1
                elif(langUsedA[anomlayLang]>langUsed[normalLang]):
                    normalLang+=1
                elif(langUsedA[anomlayLang]<langUsed[normalLang]):
                    anomlayLang+=1
    else:
        scoreA+=(1*weightlang)
    return scoreA,scoreB

