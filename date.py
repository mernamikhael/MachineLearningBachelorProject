from datetime import datetime
import csv
with open('tweets_1013609776641708033_abdullahzezo3.csv') as File:
    tfidfReader = csv.reader(File)
    i=1
    for row in tfidfReader:
        datetime_object = datetime.strptime(row[3],"%Y-%m-%d %H:%M:%S")    
        month=datetime_object.month
        year=datetime_object.year
        day=datetime_object.day
        hour=datetime_object.hour
        minute=datetime_object.minute
        sec=datetime_object.second
        print(month,year,day,hour,minute,sec,i)
        i+=1


date= datetime.strptime("7/4/2018 12:10:43",'%d/%m/%Y %H:%M:%S')

print date.year