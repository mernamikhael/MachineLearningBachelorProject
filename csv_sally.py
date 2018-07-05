from pymongo import MongoClient
import json
import csv
client = MongoClient("mongodb://SallyHabib1:Jesus2016@ds119810.mlab.com:19810/mylife")
db = client['mylife']
coll = db['facebook']
kinds=[]

# export= db.facebook.find()
# out= csv.writer(open('users.csv', 'wb'),delimiter=(','))
# b=[]
# a=[]
# for items in export:
#     a.append(items['name'])
#     a.append(items['email'])
#     a.append(items['_id'])
#     a.append(items['birthday'])
#     # a.append(items['os'])
#     # a.append(items['andriod'])
#     b.append(a)
#     out.writerows(b)
#     a=[]
#     b=[]

with open('userssally.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        faceID=row[2]
        export= db.posts.find({"users_id":faceID})
        out= csv.writer(open('user_posts_'+faceID+'.csv', 'wb'),delimiter=(','))
        b=[]
        a=[]
        for items in export:
            a.append(items['message'].encode("utf-8"))
            a.append(items['story'].encode("utf-8"))
            a.append(items['_id'].encode("utf-8"))
            a.append(items['created_time'])
            if(not(items['message']=="below")):
                if(items['story'].__contains__("updated")):
                    a.append("updated")
                    kinds.append("updated")
                else:
                    if(items['story'].__contains__("shared")):
                        a.append("shared")
                        kinds.append("shared")
                    else:
                        if(items['story'].__contains__("added")):
                            a.append("added")
                            kinds.append("added")
                        else:
                            if(items['story'].__contains__("celebrating")):
                                a.append("shared")
                                kinds.append("shared")
                            else:
                                if(items['story'].__contains__("celebrating")):
                                    a.append("shared")
                                    kinds.append("shared")
                                else:
                                    a.append("posted")
                                    kinds.append("posted")
            else:
                if(items['story'].__contains__("shared")):
                    a.append("shared")
                    kinds.append("shared")
                else:
                    if(items['story'].__contains__("updated")):
                        a.append("updated")
                        kinds.append("updated")
                    else:
                        if(items['story'].__contains__("added")):
                            a.append("added")
                            kinds.append("added")
                        else:
                            a.append("posted")
            a.append(items["tags"])
            b.append(a)
            out.writerows(b)
            a=[]
            b=[]

    #print(kinds)
     
     
