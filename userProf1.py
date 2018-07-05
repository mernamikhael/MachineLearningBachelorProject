import csv
from googletrans import Translator
from google.cloud import translate
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from googletrans import Translator
import numpy as np
import nltk
from nltk.corpus import stopwords
import idna
import uritools
import urlextract
from datetime import date, datetime
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
import calendar

def user(userID,per,weightTopic,weightLang,weightpost,weighttag,weightfre):
    # stopwords = stopwords.words('english')
    extractor = urlextract.URLExtract()
    translator = Translator()
    # stopwords = stopwords.words('english')


    # user_id=[]
    # with open('users.csv', 'rb') as fil:
    #     user = csv.reader(fil)
    #     for row in user:
    #         user_id.append(row[2]) 

    #intilization
    # total_posts=0
    # total_postsA=0



    total_posts_per_year=0
    total_posts_per_yearA=0

    shared=0
    added=0
    posted=0
    updated=0 
    sharedA=0
    addedA=0
    postedA=0
    updatedA=0 

    langu=[]
    languA=[]

    months=[0]*12
    monthsA=[0]*12


    daysX_monthsY=np.zeros([12,31])
    daysX_monthsYA=np.zeros([12,31])


    # sharedRatio=0
    # updateRatio=0
    # addRatio=0
    # postRatio=0
    # sharedRatioA=0
    # updateRatioA=0
    # addRatioA=0
    # postRatioA=0

    season=[0]*4 # winter spring summer autumn
    seasonA=[0]*4 # winter spring summer autumn

    hashTags=[] # store hashtags used
    number_hash=0

    weekends=0
    postsWeekends52=0
    postsWeekends51=0
    postsWeekends50=0
    postsWeekends49=0
    postsWeekends48=0
    postsWeekends47=0
    weekendsA=0
    postsWeekends52A=0
    postsWeekends51A=0
    postsWeekends50A=0
    postsWeekends49A=0
    postsWeekends48A=0
    postsWeekends47A=0

    tophash=[]
    tags=[]
    tagnbr=[]
    tagsA=[]
    tagnbrA=[]
    taggedposts=0
    taggedpostsA=0


    weekposts52=0
    weekposts51=0
    weekposts50=0
    weekposts52A=0
    weekposts51A=0
    weekposts50A=0
    weekposts48=0
    weekposts49=0
    weekposts47=0
    weekposts48A=0
    weekposts49A=0
    weekposts47A=0

    daysweek=np.zeros([3,7])*4  # week 52
    daysweekA=np.zeros([3,7])
    daysweek2=np.zeros([3,7])*4 # week 50
    daysweekA2=np.zeros([3,7])
    daysweek1=np.zeros([3,7])*4 #week 51
    daysweekA1=np.zeros([3,7])
    daysweek3=np.zeros([3,7])*4  # week 49
    daysweekA3=np.zeros([3,7])


    activityM52=0    
    activityN52=0
    activityM51=0
    activityN51=0 
    activityM50=0
    activityN50=0
    activityM49=0    
    activityN49=0
    activityM48=0
    activityN48=0 
    activityM47=0
    activityN47=0


    activityM52A=0    
    activityN52A=0
    activityM51A=0
    activityN51A=0 
    activityM50A=0
    activityN50A=0 
    activityM49A=0    
    activityN49A=0
    activityM48A=0
    activityN48A=0 
    activityM47A=0
    activityN47A=0 
    url=[]
    urlA=[]
    urlSize=0
    urlSizeA=0

    # user=0
    # userA=0
    personalityActivityTime=[1,1,0,1,0,1,2,2,1,1,1,1,0,0,0,2,0,1,2,1,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,2,1,0,0,2,2,2,1,2,1,1]
    # shows whether they  none 0 or add 1 share 2 update 3 post 4
    personalityTypePost=[2,0,2,2,0,2,0,0,4,0,2,4,2,2,4,2,2,2,2,0,2,0,0,2,2,2,2,4,2,2,2,4,4,2,2,2,2,2,2,2,1,2,2,2,2,2,4,2]
    # shows how many times they post none 0 or hour 1 couples of day 2 once per day 3 rarely 4
    personalityDay=[4,4,3,4,2,4,1,4,2,4,4,4,4,4,2,4,3,4,4,4,4,4,2,2,2,2,2,2,2,2,4,2,2,2,4,4,4,4,4,4,2,2,2,4,2,4,4]

    path='dataset\user_posts_'+userID+'.csv'
    pathA='dataset\user_posts_'+userID+'.csv'

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

        lemmatiser = WordNetLemmatizer()
        words_tag = dict(pos_tag(words))
        return  lemmatiser.lemmatize(word, get_wordnet_pos(words_tag.get(word)))


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


    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            l= "Topic %d:" % (topic_idx)
            l= " ".join([feature_names[i].encode("utf-8")
                            for i in topic.argsort()[:-no_top_words - 1:-1]])
            tophash.append(feature_names[i].encode("utf-8"))


    def topic_hash(hashtags):
        vectorizer = TfidfVectorizer(min_df=0.2,stop_words='english')
        X = vectorizer.fit_transform(hashtags)
        no_topics = min(10,len(hashtags))
        nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(X)
        display_topics(nmf, vectorizer.get_feature_names(), 1)



    def extract_hash(word):
        i=0
        h=''
        while(i<len(word)):
            s=word[i]
            if(s=='#'):
                i+=1
                while(i<len(word) and not(word[i]=='#' or word[i]==' ' or word[i]=="\n")):
                    s=word[i]
                    h+=s
                    i+=1
                hashTags.append(h)
                h=''
            else:
                i+=1


    def get_season(now):
        if isinstance(now, datetime):
            now = now.date()
        return next(s for s, (start, end) in seasons
                    if start <= now <= end)

    def get_tags(tag):
        tag=tag.replace('u','')
        tag=tag.replace('[','')
        tag=tag.replace(']','')
        tag=tag.replace('\'','')
        tag= (tag.split(','))
        tagnbr.append(len(tag))
        i=0
        while(i<len(tag)):
            if(int(tag[i]) not in tags):
                tags.append(int(tag[i]))
            i+=1
        return tags

    def get_tags_anoamly(tagA):
        tagL=tagA.replace('u','')
        tagL=tagL.replace('[','')
        tagL=tagL.replace(']','')
        tagL=tagL.replace('\'','')
        tagL= (tagL.split(','))
        tagnbrA.append(len(tagL))
        l=0
        while(l<len(tagL)):
            if(int(tagL[l]) not in tagsA):
                tagsA.append(int(tagL[l]))
                
            l+=1
        return tagsA
        
                
    with open(path, 'rb') as f:
        posts = csv.reader(f)
        for items in posts:
            # check date time
            datetime_object = datetime.strptime(items[3],"%Y-%m-%d %H:%M:%S")    
            hour=datetime_object.hour
            month=datetime_object.month
            year=datetime_object.year
            day=datetime_object.day
            dates=datetime.date(datetime_object)
            
            seasons = [('winter', (date(year,  1,  1),  date(year,  3, 20))),
                        ('spring', (date(year,  3, 21),  date(year,  6, 20))),
                        ('summer', (date(year,  6, 21),  date(year,  9, 22))),
                        ('autumn', (date(year,  9, 23),  date(year, 12, 20))),
                        ('winter', (date(year, 12, 21),  date(year, 12, 31)))]
        
                
            # activity in 2017 
            if(year==2017):
                
                total_posts_per_year+=1
                # posts/month
                months[month-1]+=1
                # posts /day
                daysX_monthsY[month-1][day-1]+=1
                # week number
                weekNumber = dates.isocalendar()[1]
                # activity of last 3 weeks in 2017
                # total nbr of posts/each week
                # posts/day in each week
                # activity time in each day in each week
                if(weekNumber==51):
                    weekend=datetime_object.weekday()
                    if(weekend==4 or weekend==5):
                        # weekends+=1
                        postsWeekends51+=1
                    weekposts51+=1
                    if(hour>=6 and hour<18):
                        activityM51+=1
                    elif(hour>=18 and hour <24):
                        activityN51+=1
                    elif(hour >=0 and hour <6):
                        activityN51+=1
                    daysweek1[0][(dates.weekday())-1]+=1


                if(weekNumber==50):
                    weekend=datetime_object.weekday()
                    if(weekend==4 or weekend==5):
                        # weekends+=1
                        postsWeekends50+=1
                    weekposts50+=1
                    if(hour>=6 and hour<18):
                        activityM50+=1
                    elif(hour>=18 and hour <24):
                        activityN50+=1
                    elif(hour >=0 and hour <6):
                        activityN50+=1
                    daysweek2[0][(dates.weekday())-1]+=1
                    daysweek1[1][(dates.weekday())-1]+=1
                
                if(weekNumber==49):
                    weekend=datetime_object.weekday()
                    if(weekend==4 or weekend==5):
                        # weekendsA+=1
                        postsWeekends49+=1
                    weekposts49+=1
                    if(hour>=6 and hour<18):
                        activityM49+=1
                    elif(hour>=18 and hour <24):
                        activityN49+=1
                    elif(hour >=0 and hour <6):
                        activityN49+=1
                    daysweek3[0][(dates.weekday())-1]+=1
                    daysweek2[1][(dates.weekday())-1]+=1
                    daysweek1[2][(dates.weekday())-1]+=1

                if(weekNumber==48):
                    weekend=datetime_object.weekday()
                    if(weekend==4 or weekend==5):
                        # weekendsA+=1
                        postsWeekends48+=1
                    weekposts48+=1
                    if(hour>=6 and hour<18):
                        activityM48+=1
                    elif(hour>=18 and hour <24):
                        activityN48+=1
                    elif(hour >=0 and hour <6):
                        activityN48+=1
                    daysweek3[1][(dates.weekday())-1]+=1
                    daysweek2[2][(dates.weekday())-1]+=1
                
                if(weekNumber==47):
                    weekend=datetime_object.weekday()
                    if(weekend==4 or weekend==5):
                        # weekendsA+=1
                        postsWeekends47+=1
                    weekposts47+=1
                    if(hour>=6 and hour<18):
                        activityM47+=1
                    elif(hour>=18 and hour <24):
                        activityN47+=1
                    elif(hour >=0 and hour <6):
                        activityN47+=1
                    daysweek3[2][(dates.weekday())-1]+=1

                # which type they use most share/add/update/post
                if(items[4]=='added'):
                    added+=1
                elif(items[4]=='updated'):
                    updated+=1
                elif(items[4]=='posted'):
                    posted+=1
                else:
                    shared+=1
                
            # season
                if(get_season(datetime_object)=='winter'):
                    season[0]+=1
                elif(get_season(datetime_object)=='spring'):
                    season[1]+=1
                elif(get_season(datetime_object)=='summer'):
                    season[2]+=1
                elif(get_season(datetime_object)=='autumn'):
                    season[3]+=1
                weekends= len([1 for i in calendar.monthcalendar(2017,
                                    12) if i[5] != 0])

                weekends+=len([1 for i in calendar.monthcalendar(2017,
                                    12) if i[4] != 0])


                        
                
                # lang detector
                if(items[0]):
                    t=translator.detect(json.dumps(items[0].decode('utf-8')))
                    langu.append(t.lang)
                    # msg=items[0]
                    # if(t.lang=="en"):
                    #     message.append(clean(msg).encode('utf-8'))
                # tags
                k=items[5]
                if(len(k)>2):
                    taggedposts+=1
                    get_tags(k)

                urls = extractor.find_urls(items[0])
                url.append(urls)
                urlSize+=len(urls)
            
            # nbr of words
            # word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
            # mess=items[0]
            # mess = unicode(mess, errors='ignore')
            # words1 = word_tokenizer.tokenize(mess.lower())
            # words=[word for word in words1 if
            #     not word in stopwords]
            # postLength.append(len(words))

            # nbr of urls
            
            
            
            
            # extract hashTags
            extract_hash(items[0])

            # count hash tags
            number_hash+=items[0].count('#')



    #  anomaly user
    with open(pathA, 'rb') as files:
        postsA = csv.reader(files)
        for anomaly in postsA:
            # check date time
            datetime_object = datetime.strptime(anomaly[3][:-4],"%Y-%m-%d %H:%M:%S")    
            hour=datetime_object.hour
            month=datetime_object.month
            year=datetime_object.year
            day=datetime_object.day
            dates=datetime.date(datetime_object)
            
            seasons = [('winter', (date(year,  1,  1),  date(year,  3, 20))),
                        ('spring', (date(year,  3, 21),  date(year,  6, 20))),
                        ('summer', (date(year,  6, 21),  date(year,  9, 22))),
                        ('autumn', (date(year,  9, 23),  date(year, 12, 20))),
                        ('winter', (date(year, 12, 21),  date(year, 12, 31)))]
        
                
            # activity in 2017 
            if(year==2017):
                
                total_posts_per_yearA+=1
                # posts/month
                monthsA[month-1]+=1
                # posts /day
                daysX_monthsYA[month-1][day-1]+=1
                # week number
                weekNumber = dates.isocalendar()[1]
                # activity of last 3 weeks in 2017
                # total nbr of posts/each week
                # posts/day in each week
                # activity time in each day in each week
                if(weekNumber==51):
                    weekend=datetime_object.weekday()
                    if(weekend==4 or weekend==5):
                        # weekendsA+=1
                        postsWeekends51A+=1
                    weekposts51A+=1
                    if(hour>=6 and hour<18):
                        activityM51A+=1
                    elif(hour>=18 and hour <24):
                        activityN51A+=1
                    elif(hour >=0 and hour <6):
                        activityN51A+=1
                    daysweekA1[0][(dates.weekday())-1]+=1


                if(weekNumber==50):
                    weekend=datetime_object.weekday()
                    if(weekend==4 or weekend==5):
                        # weekendsA+=1
                        postsWeekends50A+=1
                    weekposts50A+=1
                    if(hour>=6 and hour<18):
                        activityM50A+=1
                    elif(hour>=18 and hour <24):
                        activityN50A+=1
                    elif(hour >=0 and hour <6):
                        activityN50A+=1
                    daysweekA2[0][(dates.weekday())-1]+=1
                    daysweekA1[1][(dates.weekday())-1]+=1
                
                if(weekNumber==49):
                    weekend=datetime_object.weekday()
                    if(weekend==4 or weekend==5):
                        # weekendsA+=1
                        postsWeekends49A+=1
                    weekposts49A+=1
                    if(hour>=6 and hour<18):
                        activityM49A+=1
                    elif(hour>=18 and hour <24):
                        activityN49A+=1
                    elif(hour >=0 and hour <6):
                        activityN49A+=1
                    daysweekA2[1][(dates.weekday())-1]+=1
                    daysweekA3[0][(dates.weekday())-1]+=1
                    daysweekA1[2][(dates.weekday())-1]+=1


                if(weekNumber==48):
                    weekend=datetime_object.weekday()
                    if(weekend==4 or weekend==5):
                        # weekendsA+=1
                        postsWeekends48A+=1
                    weekposts48A+=1
                    if(hour>=6 and hour<18):
                        activityM48A+=1
                    elif(hour>=18 and hour <24):
                        activityN48A+=1
                    elif(hour >=0 and hour <6):
                        activityN48A+=1
                    daysweekA3[1][(dates.weekday())-1]+=1
                    daysweekA2[2][(dates.weekday())-1]+=1

                if(weekNumber==47):
                    weekend=datetime_object.weekday()
                    if(weekend==4 or weekend==5):
                        # weekendsA+=1
                        postsWeekends47A+=1
                    weekposts47A+=1
                    if(hour>=6 and hour<18):
                        activityM47A+=1
                    elif(hour>=18 and hour <24):
                        activityN47A+=1
                    elif(hour >=0 and hour <6):
                        activityN47A+=1
                    daysweekA3[2][(dates.weekday())-1]+=1



                # which type they use most share/add/update/post
                if(anomaly[4]=='added'):
                    addedA+=1
                elif(anomaly[4]=='updated'):
                    updatedA+=1
                elif(anomaly[4]=='posted'):
                    postedA+=1
                else:
                    sharedA+=1
                
            # season
                if(get_season(datetime_object)=='winter'):
                    seasonA[0]+=1
                elif(get_season(datetime_object)=='spring'):
                    seasonA[1]+=1
                elif(get_season(datetime_object)=='summer'):
                    seasonA[2]+=1
                elif(get_season(datetime_object)=='autumn'):
                    seasonA[3]+=1
                
                weekendsA= len([1 for i in calendar.monthcalendar(2017,
                                    12) if i[5] != 0])

                weekendsA+=len([1 for i in calendar.monthcalendar(2017,
                                    12) if i[4] != 0])
                


                # tags
                kA=anomaly[5]
                if(len(kA)>2):
                    taggedpostsA+=1
                    get_tags_anoamly(kA)
            
                        
                
                
                
                # lang detector
                tA=translator.detect(json.dumps(anomaly[0].decode('utf-8')))
                languA.append(tA.lang)

                # msgA=anomaly[0]
                # if(tA.lang=='en'):
                #     messageA.append(clean(msgA).encode('utf-8'))

                urlsA = extractor.find_urls(items[0])
                urlA.append(urlsA)
                urlSizeA+=len(urlsA)
                



            # # nbr of words
            # word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
            # mess=anomaly[0]
            # mess = unicode(mess, errors='ignore')
            # words1 = word_tokenizer.tokenize(mess.lower())
            # words=[word for word in words1 if
            #     not word in stopwords]
            # postLength.append(len(words))

            # nbr of urls
            urls = extractor.find_urls(anomaly[0])
            # urls_Size+=len(urls)
            
            
            
            # extract hashTags
            extract_hash(anomaly[0])

            # count hash tags
            number_hash+=anomaly[0].count('#')


    scoreA1=2
    scoreB1=2
    import topicModel
    scoreTA,scoreTB= topicModel.topic(userID,userID,scoreA1,scoreB1,weightTopic)


    import lang
    scoreLA,scoreLB= lang.language(langu, languA,total_posts_per_year,total_posts_per_yearA,scoreTA,scoreTB,weightLang)
    # print added,updated,posted,shared
    # print addedA,updatedA,postedA,sharedA


    import typeOfPost
    scorePA,scorePB= typeOfPost.type_post(added,shared,updated,posted,total_posts_per_year,addedA,sharedA,updatedA,postedA,total_posts_per_yearA,personalityTypePost[per],scoreLA,scoreLB,weightpost)

    
    import tagged

    scoreTagA,scoreTagB= tagged.tagged(tags,tagnbr,taggedposts,total_posts_per_year,tagsA,tagnbrA,taggedpostsA,total_posts_per_yearA,scorePA,scorePB,weighttag)

    import freq
    # print freq.activityTime(activityM52, activityM51, activityM50, activityN52, activityN51, activityN50, activityM52A, activityM51A, activityM50A, activityN52A, activityN51A, activityN50A, personalityActivityTime[0])

    # print freq.day(daysweek, daysweekA)

    # print freq.weekend(postsWeekends52,postsWeekends51,postsWeekends50,weekends,postsWeekends52A,postsWeekends51A,postsWeekends50A,weekendsA)


    scoreFA51,scoreFB51= freq.frequency(postsWeekends51,postsWeekends50,postsWeekends49,weekends,postsWeekends51A,postsWeekends50A,
    postsWeekends49A,weekendsA,activityM51, activityM50, activityM49, activityN51, activityN50, activityN49,
    activityM51A, activityM50A, activityM49A, activityN51A, activityN50A, activityN49A, personalityActivityTime[per]
    ,daysweek, daysweekA,season,total_posts_per_year,seasonA,total_posts_per_yearA,weekposts51,weekposts50,weekposts49,weekposts51A,
    weekposts50A,weekposts49A,personalityDay[per],scoreTagA,scoreTagB,weightfre)

    scoreFA49,scoreFB49= freq.frequency(postsWeekends49,postsWeekends48,postsWeekends47,weekends,postsWeekends49A,postsWeekends48A,
    postsWeekends47A,weekendsA,activityM49, activityM48, activityM47, activityN49, activityN48, activityN47,
    activityM49A, activityM48A, activityM47A, activityN49A, activityN48A, activityN47A, personalityActivityTime[per]
    ,daysweek, daysweekA,season,total_posts_per_year,seasonA,total_posts_per_yearA,weekposts49,weekposts48,weekposts47,weekposts49A,
    weekposts48A,weekposts47A,personalityDay[per],scoreTagA,scoreTagB,weightfre)

    scoreFA50,scoreFB50= freq.frequency(postsWeekends50,postsWeekends49,postsWeekends48,weekends,postsWeekends50A,postsWeekends49A,
    postsWeekends48A,weekendsA,activityM51, activityM49, activityM48, activityN51, activityN49, activityN48,
    activityM51A, activityM49A, activityM48A, activityN51A, activityN49A, activityN48A, personalityActivityTime[per]
    ,daysweek, daysweekA,season,total_posts_per_year,seasonA,total_posts_per_yearA,weekposts51,weekposts49,weekposts48,weekposts51A,
    weekposts49A,weekposts48A,personalityDay[per],scoreTagA,scoreTagB,weightfre)

    # import links

    # links.link(url, urlSize, total_posts_per_year, urlA, urlSizeA, total_posts_per_yearA)

    return scoreFA51,scoreFB51,scoreFA50,scoreFB50,scoreFA49,scoreFB49