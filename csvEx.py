from pymongo import MongoClient
import csv
from langdetect import detect

# import sys  
# reload(sys)  # Reload does the trick!
# sys.setdefaultencoding('utf-8')
# print(sys.getdefaultencoding())
client = MongoClient("mongodb://merna:Merna123@ds151558.mlab.com:51558/sos")
db = client['sos']
coll = db['face']

# # export= db.face.find()
# # out= csv.writer(open('users.csv', 'wb'),delimiter=(','))
# # b=[]
# # a=[]
# # for items in export:
# #     a.append(items['name'])
# #     a.append(items['email'])
# #     a.append(items['_id'])
# #     a.append(items['birthday'])
# #     a.append(items['os'])
# #     a.append(items['andriod'])
# #     b.append(a)
# #     out.writerows(b)
# #     a=[]
# #     b=[]

# # To read for each user
# kinds=[]
# with open('users.csv', 'rb') as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		faceID=row[2]
# 		export= db.posts.find({"users_id":faceID})
# 		out= csv.writer(open('user_posts_'+faceID+'.csv', 'wb'),delimiter=(','))
# 		b=[]
# 		a=[]
		
# 		for items in export:
# 			lang=''
# 			if(items['message']):	
# 				try:
# 					lang= detect(items['message'].decode('utf-8'))
# 				except:
# 					pass
# 			# print lang
# 			if(lang!='ar'):
# 				a.append(items['message'].encode("utf-8"))
# 			a.append(items['story'].encode("utf-8"))
# 			a.append(items['_id'].encode("utf-8"))
# 			a.append(items['created_time'])
# 			if(not(items['message']=="")):
# 					if(items['story'].__contains__("updated")):
# 						a.append("updated")
# 						kinds.append("updated")
# 					else:
# 						if(items['story'].__contains__("shared")):
# 							a.append("shared")
# 							kinds.append("shared")
# 						else:
# 							if(items['story'].__contains__("added")):
# 								a.append("added")
# 								kinds.append("added")
# 							else:
# 								if(items['story'].__contains__("celebrating")):
# 									a.append("shared")
# 									kinds.append("shared")
# 								else:
# 										if(items['story'].__contains__("was at")):
# 												a.append("shared")
# 												kinds.append("shared")
# 										else:
# 											a.append("posted")
# 											kinds.append("posted")
# 			else:
# 				if(items['story'].__contains__("shared")):
# 					a.append("shared")
# 					kinds.append("shared")
# 				else:
# 					if(items['story'].__contains__("updated")):
# 						a.append("updated")
# 						kinds.append("updated")
# 					else:
# 						if(items['story'].__contains__("added")):
# 							a.append("added")
# 							kinds.append("added")
# 						else:
# 								a.append("posted")
# 								kinds.append("posted")	
# 			a.append(items['Tags'])
# 			# a.append(items['place'])
# 			# a.append(items['language'])

# 			b.append(a)
# 			out.writerows(b)
# 			a=[]
# 			b=[]
# 	# likes= db.likes.find({"users_id":faceID})
# 	# out= csv.writer(open('user_likes_'+faceID+'.csv', 'wb'),delimiter=(','))
# 	# b=[]
# 	# a=[]
# 	# for items in likes:
# 	# 	a.append(items['name'].encode("utf-8"))
# 	# 	a.append(items['about'].encode("utf-8"))
# 	# 	a.append(items['category'].encode("utf-8"))
# 	# 	b.append(a)
# 	# 	out.writerows(b)
# 	# 	a=[]
# 	# 	b=[]

with open('users.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        faceID=row[2]
        likes= db.likes.find({"users_id":faceID})
        out= csv.writer(open('user_likes_'+faceID+'.csv', 'wb'))
        b=[]
        a=[]
        for items in likes:
            a.append(items['name'].encode("utf-8"))
            a.append(items['about'].encode("utf-8"))
            a.append(items['category'].encode("utf-8"))
            b.append(a)
            out.writerows(b)
            a=[]
            b=[]
# posts= db.posts.find()
# out= csv.writer(open('user_posts_10214181385024102.csv', 'wb'),delimiter=(','))
# b=[]
# a=[]
# for items in posts:
#     a.append(items['message'].encode('utf-8'))
#     a.append(items['story'].encode("utf-8"))
#     a.append(items['_id'].encode("utf-8"))
#     a.append(items['Tags'])
#     a.append(items['created_time'])
#     b.append(a)
#     out.writerows(b)
#     a=[]
#     b=[]
# likes= db.likes.find()
# out= csv.writer(open('user_likes.csv', 'wb'),delimiter=(','))
# b=[]
# a=[]
# for items in likes:
#     a.append(items['name'].encode("utf-8"))
#     a.append(items['about'].encode("utf-8"))
#     a.append(items['category'].encode("utf-8"))
#     b.append(a)
#     out.writerows(b)
#     a=[]
#     b=[]

