# from googletrans import Translator
# import json
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np
# import csv

# translator = Translator()
# user_id=[]
# with open('users.csv', 'rb') as fil:
#     user = csv.reader(fil)
#     for row in user:
#         user_id.append(row[2]) 
# pathA='user_posts_'+user_id[6]+'.csv'
# languA=[]
# with open(pathA, 'rb') as files:
#     postsA = csv.reader(files)
#     for anomaly in postsA:
#         tA=translator.detect(json.dumps(anomaly[0].decode('utf-8')))
#         # print tA.lang
#         languA.append(tA.lang)

#     # print languA
#     vectorizerA = TfidfVectorizer()
#     XA = vectorizerA.fit_transform(languA)
#     zA=zip(vectorizerA.get_feature_names(),np.ravel(XA.sum(axis=0)))
#     # sort descending according to count of each language
#     zA.sort(key = lambda t: t[1],reverse=True)
#     langSumA=np.ravel(XA.sum(axis=0))
#     langUsedA=vectorizerA.get_feature_names()
#     maxlangA=max(zA,key=lambda x:x[1])

#     print langSumA
#     print langUsedA

# print (0.1794871794871795 and 0.13157894736842105 and 0.3)

# from scipy.stats import beta
# import numpy as np
# import matplotlib.pyplot as plt
# x = np.linspace(0, 1.0, 100)
# y=beta.pdf(x, 39,23)
# k=beta.pdf(x,9,57)
# l= y - k

# plt.plot(x,l)
# plt.show()

stri='momo'+str(0)
print stri