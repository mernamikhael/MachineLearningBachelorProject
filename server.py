import json
from flask import Flask
from flask import request
import requests
import facebook
from pymongo import MongoClient
import urllib
from datetime import datetime
from flask.json import dump
import csv
from googletrans import Translator
import sys  
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('utf-8')
print(sys.getdefaultencoding())
#https://teamtreehouse.com/community/can-someone-help-me-understand-flaskname-a-little-better
app = Flask(__name__) # flask is webframwork for python __name__ from root 

appID="1583861328394595"
appSecret="b218fd24b40469f2af993d24435eff0d"

client = MongoClient("mongodb://merna:Merna123@ds151558.mlab.com:51558/sos")
db = client['sos']
coll = db['face']
print(coll)
translator = Translator()

@app.route("/")
def hello():
    #here request is request of flask not the requests library and it return the attribute specified f
    #flask.Request.argsA MultiDict with the parsed contents of the query string. (The part in the URL after the question mark).
    code = request.args.get('code')
   
    #Exchanging Code for an Access Token
    r=requests.get('https://graph.facebook.com/v2.12/oauth/access_token?client_id={}&redirect_uri={}&client_secret={}&code={}'.format(appID,'http://localhost:8080/',appSecret,code))
    
    data = r.json()
    if(data.__contains__('access_token')):
        
        access_token=data['access_token']
        #print(access_token)
        graph = facebook.GraphAPI(access_token)
    
        declined=[]
        permission= requests.get("https://graph.facebook.com/me/permissions?access_token="+access_token)
        permissionJson=permission.json()
        i = 0
        while i < len(permissionJson['data']):
            if permissionJson['data'][i]['status']=='declined':
                declined.append(permissionJson['data'][i]['permission'])
            i += 1

        #print(declined)
        profile = graph.get_object("/me?fields=id,name,feed.limit(200){story,with_tags,created_time,message,message_tags,place},likes.limit(200){name,about,category},birthday,email")
        profile2 =graph.get_object("/me?fields=languages,devices,albums{id,name,privacy,place,created_time}")
        deviceFound=False
        if declined.__contains__('devices'):
            deviceFound=False
        else:
            if(profile2.__contains__('devices')):
                deviceFound=True
        if declined.__contains__('name'):
            name = ""
        else:
            name = profile['name']
                
        if declined.__contains__('user_birthday'):
            bd=""
        else:
            bd=profile['birthday']   
        
        if declined.__contains__('user_posts'):
            posts=""  
        else:
            posts=profile['feed']
            postFound=True 

        if declined.__contains__('user_likes'):
            likes=""  
        else:
            likes=profile['likes'] 
            likesFound=True

        if declined.__contains__('email'):
            email=""  
        else:
            email=profile['email']  
        os=0
        andriod=0
        if(deviceFound):
            os=0
            andriod=0
            count=0
            while(count<len(profile2['devices'])):
                if(profile2['devices'][count].get('os')=='Android'):
                    andriod=1
                else:
                    os=1
                count=count+1
                
        id=profile['id']
        #p=posts['data']
        db.face.update_one({"_id":id},{
            "$set":{
            "_id":id,
            "name":name,
            "email":email,
            "birthday":bd,
            "os":os,
            "andriod":andriod
        }}, upsert=True)
        
        if(postFound):
            postCount = 0
            lang=""
            while postCount < len(posts['data']):
                message= posts['data'][postCount].get("message", "empty")
                story =  posts['data'][postCount].get("story", "empty")
                if((message=="empty")):
                    lang=""
                # else:
                    # t=translator.detect(json.dumps(message.decode()))
                    # lang=t.lang
                place=""
                if(posts['data'][postCount].__contains__('place')):
                    place=posts['data'][postCount]['place'].get("id")
                tags=[]
                if(posts['data'][postCount].__contains__('with_tags')):
                    withTagCount=0
                    while(withTagCount<len(posts['data'][postCount]['with_tags']['data'])):
                        name=posts['data'][postCount]['with_tags']['data'][withTagCount].get("name")
                        idTag=posts['data'][postCount]['with_tags']['data'][withTagCount].get("id")
                        if(~(tags.__contains__(idTag))):
                            tags.append(idTag)
                        withTagCount+=1
                #########################
                if(posts['data'][postCount].__contains__('message_tags')):
                    messageTagCount=0
                    while(messageTagCount<len(posts['data'][postCount]['message_tags'])):
                        name=posts['data'][postCount]['message_tags'][messageTagCount].get("name")
                        idTag=posts['data'][postCount]['message_tags'][messageTagCount].get("id")
                        if(~(tags.__contains__(idTag))):
                            tags.append(idTag)
                        messageTagCount+=1

                if(story=="empty"):
                    db.posts.update_one({"_id":posts['data'][postCount].get("id")},{
                        "$set": {
                        "_id":posts['data'][postCount].get("id"),
                        "message": message ,
                        "story": "",
                        "created_time": posts['data'][postCount].get("created_time"),
                        "users_id": id,
                        "Tags":tags,
                        "place":place,
                        "language":lang
                    }},  upsert=True)

                elif(message=="empty"):
                    db.posts.update_one({"_id": posts['data'][postCount].get("id")},{
                        "$set": {
                        "_id": posts['data'][postCount].get("id"),
                        "message": "",
                        "story": story,
                        "created_time": posts['data'][postCount].get("created_time"),
                        "users_id": id,
                        "Tags":tags,
                        "place":place,
                        "language":lang
                    }},  upsert=True)
                
                else:
                    db.posts.update_one({"_id": posts['data'][postCount].get("id")},{
                        "$set": {
                        "_id": posts['data'][postCount].get("id"),
                        "message": message,
                        "story": story,
                        "created_time":posts['data'][postCount].get("created_time"),
                        "users_id": id,
                        "Tags":tags,
                        "place":place,
                        "language":lang
                    }},  upsert=True)

                postCount+=1                    
        if(likesFound):
            likeCount=0
            while(likeCount<len(likes['data'])):
                name=likes['data'][likeCount].get("name","empty")
                about=likes['data'][likeCount].get("name","empty")
                category=likes['data'][likeCount].get("category")

                if(name=="empty"):
                    db.likes.update_one({"_id": likes['data'][likeCount].get("id")},{
                        "$set": {
                        "_id": likes['data'][likeCount].get("id"),
                        "name": "",
                        "about":about,
                        "users_id": id,
                        "category":category
                    }},  upsert=True)

                elif(about=="empty"):
                    db.likes.update_one({"_id": likes['data'][likeCount].get("id")},{
                        "$set": {
                        "_id": likes['data'][likeCount].get("id"),
                        "name": name,
                        "about":"below",
                        "users_id": id,
                        "category":category
                    }},  upsert=True)
                else:
                    db.likes.update_one({"_id": likes['data'][likeCount].get("id")},{
                        "$set": {
                        "_id": likes['data'][likeCount].get("id"),
                        "name": name,
                        "about":about,
                        "users_id": id,
                        "category":category
                    }},  upsert=True)


                likeCount+=1 
        
        return "done"
    else:
        return"U did not grant your permission"


app.run(host="0.0.0.0", port=int("8080"), debug=True)




# {u'data': [{u'status': u'granted', u'permission': u'user_birthday'}, {u'status': u'granted', u'permission': u'user_religion_politics'}, {u'status': u'granted', u'permission': u'user_relationships'}, {u'status': u'granted', u'permission': u'user_relationship_details'}, {u'status': u'granted', u'permission': u'user_hometown'}, {u'status': u'granted', u'permission': u'user_location'}, {u'status': u'granted', u'permission': u'user_likes'}, {u'status': u'granted', u'permission': u'user_education_history'}, {u'status': u'granted', u'permission': u'user_work_history'}, {u'status': u'granted', u'permission': u'user_website'}, {u'status': u'granted', u'permission': u'user_photos'}, {u'status': u'granted', u'permission': u'user_videos'}, {u'status': u'granted', u'permission': u'user_friends'}, {u'status': u'granted', u'permission': u'user_about_me'}, {u'status': u'granted', u'permission': u'user_status'}, {u'status': u'granted', u'permission': u'user_games_activity'}, {u'status': u'granted', u'permission': u'user_tagged_places'}, {u'status':
# u'granted', u'permission': u'user_posts'}, {u'status': u'granted', u'permission': u'email'}, {u'status': u'granted', u'permission': u'publish_actions'}, {u'status': u'granted', u'permission': u'manage_pages'}, {u'status': u'granted', u'permission': u'pages_show_list'}, {u'status': u'granted', u'permission': u'public_profile'}]}
# 10/29/1995
# Merna Michel
# 127.0.0.1 - - [01/Mar/2018 18:29:00] "GET /?code=AQCqwlo7_bVZ9cBdokeoiiTiZz7ettSfreMySB8rCUdSNC3jxGy5hWPTgs8iHQTioK0_hDzhAc69EvbkqxhXU7JoIg6y0KYuE2uhni2DL_Ljt_F_UupcHUt65qzRNamVDpcnE3N-1mw7nmbs3OpZaLUjfyfdwdbZ3h_9whgnqcHHseWy6yxXtSSsUQ-v6gEI0xC8ydeoOeBKfccB35dwVQgTwQPfET3z9O-VkGLNEXQ9wzOl2hknlJZfZMoAUqe89pbevosdBqwv_Lc1SYNqBLUjE1Ivp5HMSexioQqd2lF0sZ_tTKvVzzJwgdsIctqjW8g HTTP/1.1" 200 -
# 127.0.0.1 - - [01/Mar/2018 18:29:00] "GET /favicon.ico HTTP/1.1" 404 -
# {u'data': [{u'status': u'granted', u'permission': u'user_birthday'}, {u'status': u'granted', u'permission': u'email'}, {u'status': u'granted', u'permission': u'public_profile'}]}
