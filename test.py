from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import time
import tweepy
import csv
# from test.test_decimal import init
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from datetime import datetime, date, time, timedelta
from collections import Counter
import sys

access_token = "1012234479832895488-B8LBfisf1FvgnnPuHigo98434GxoXs"
access_token_secret = "KB1TFhcC4H9NcjwkbHnE7C05vsc90qu9VZZLKTHDXtlkK"
consumer_key = "xrqojmbzrzRqR1BXV9ORp75Zt"
consumer_secret = "lhhNNw77O534Ck1yXcbUKJ76km6h5y6lgF96SE3S4EPmLjfjj2"

class StdOutListener(StreamListener):
    
    def on_data(self, data):
        #x = json.loads(data)
        #if(not isReply(x)):
        print data
        return True

    def on_error(self, status):
        print status

if __name__ == '__main__':

    #This handles Twitter authentication and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)
    # 200 tweets to be extracted
    number_of_tweets=200
    idlevel=[]
    mainID=[]

    with open("id university of south carolina.csv","rb") as f:
        t=csv.reader(f)
        for row in t:
            for k in row:
                idlevel.append(k)
                mainID.append(idlevel)
                try:
                    u = api.get_user(k)
                    screen_name=u.screen_name
                    print k,screen_name
                    levelout= csv.writer(open("Dataset/ids.csv","wb"),delimiter=(','))
                    levelout.writerow(mainID)
                    idlevel=[]
                    mainID=[]
                
                    tweets= Cursor(api.user_timeline, id=k).items()
                    if(tweets):
                        out= csv.writer(open("Dataset/tweets_"+k+"_"+screen_name+".csv","wb"),delimiter=(','))
                        a=[]
                        b=[]
                        hashtags = []
                        mentions = []
                        urls = []
                        tweet_count = 0
                        # end_date = datetime.utcnow() - timedelta(days=30)
                        for status in tweets:
                            a.append(status.text.encode('utf-8'))
                            a.append(" ")
                            a.append(status.id)
                            a.append(status.created_at)
                            a.append(" ")
                            tweet_count += 1
                            if hasattr(status, "entities"):
                                entities = status.entities
                                if "hashtags" in entities:
                                    for ent in entities["hashtags"]:
                                        if ent is not None:
                                            if "text" in ent:
                                                hashtag = ent["text"]
                                                if hashtag is not None:
                                                    hashtags.append(hashtag)
                                if "user_mentions" in entities:
                                    for ent in entities["user_mentions"]:
                                        if ent is not None:
                                            if "screen_name" in ent:
                                                name = ent["screen_name"]
                                                if name is not None:
                                                    mentions.append(name)
                                if "urls" in entities:
                                    for url in entities["urls"]:
                                        if url is not None:
                                            # urls.append(url)
                                            if "expanded_url" in url:
                                                expUrl = url["expanded_url"]
                                                if expUrl is not None:
                                                    urls.append(expUrl)
                            
                            a.append(mentions)
                            a.append(hashtags)
                            a.append(status.lang)
                            a.append(urls)
                            b.append(a)
                            out.writerows(b)
                            a=[]
                            b=[]
                            mentions= []
                            hashtags= []
                except:
                    
                #     print "not authori"
                    continue
