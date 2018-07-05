from pymongo import MongoClient
import json
import csv
client = MongoClient("mongodb://joyce.tawfik:Godislove*1@ds163918.mlab.com:63918/fdata")
db = client['fdata']
coll = db['users']
# print coll
# export= db.users.find()
# out= csv.writer(open('dataset/users.csv', 'wb'),delimiter=(','))
# b=[]
# a=[]
# for items in export:
#     a.append(items['name'])
#     a.append(" ")
#     a.append(items['_id'])
#     b.append(a)
#     out.writerows(b)
#     a=[]
#     b=[]

# To read for each user
# posts
# with open('usersjoy.csv', 'rb') as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		faceID=row[2]
# 		user=row[0]
# 		export= db.status.find({"users_id":faceID})
# 		out= csv.writer(open('user_posts_'+faceID+'.csv', 'wb'),delimiter=(','))
# 		b=[]
# 		a=[]
		
# 		for items in export:
# 			a.append(items['story/message'].encode("utf-8"))
# 			a.append(" ")  #items['story'].encode("utf-8")
# 			a.append(items['_id'].encode("utf-8"))
# 			a.append(items['created_time'])
# 			if(items['story/message'].__contains__("updated")):
# 					a.append("updated")
# 			else:
# 				if(items['story/message'].__contains__("shared")):
# 					a.append("shared")
# 				else:
# 					if(items['story/message'].__contains__("added")):
# 						a.append("added")
# 					else:
# 						if(items['story/message'].__contains__("celebrating")):
# 							a.append("shared")
# 						else:
# 							if(items['story/message'].__contains__("is with")):
# 								a.append("shared")
# 							else:
# 								if(items['type']=="status"):
# 									a.append("posted")
# 								elif(items['type'].__contains__("photo") and items['name']==user):
#     									a.append("added")
# 								else:
# 										a.append("shared")

							
			
# 			a.append(items['tags'])
# 			a.append(items['type'].encode("utf-8"))
# 			a.append(items['name'].encode("utf-8"))

# 			b.append(a)
# 			out.writerows(b)
# 			a=[]
# 			b=[]

# likes

with open('usersjoy.csv', 'rb') as f:
	reader = csv.reader(f)
	for row in reader:
		faceID=row[2]
		# user=row[0]
		export= db.pages.find({"user_id":faceID})
		
		out= csv.writer(open('user_likes_'+faceID+'.csv', 'wb'),delimiter=(','))
		b=[]
		a=[]
		
		for items in export:
				a.append(items['name'].encode("utf-8"))
				a.append(items['about'].encode("utf-8"))
				a.append(items['category'].encode("utf-8"))
				b.append(a)
				out.writerows(b)
				a=[]
				b=[]
