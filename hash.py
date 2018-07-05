import csv
import re
# with open('user_posts_10160353867085370.csv') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         m=row[0]

h=''
hash=[]
with open('user_posts_10160353867085370.csv') as f:
    read= csv.reader(f)
    for row in read:
        word=row[0]

        i=0

        while(i<len(word)):
            s=word[i]
            if(s=='#'):
                i+=1
                while(i<len(word) and not(word[i]=='#' or word[i]==' ')):
                    s=word[i]
                    h+=s
                    i+=1
                    # print(i,hash,h)
                    # print("da5el if")
                hash.append(h)
                h=''
            else:
                i+=1
                # print('da5el else',i)


print hash

   
    # i+=1

# a=[m.start() for m in re.finditer('#',word )]


# i=0
# while(i<len(a)):
#     word=word[a[i]:len(word)]
#     a=[m.start() for m in re.finditer('#',word )]
#     print(word)
#     b=[m.start() for m in re.finditer(' ',word)]
#     print('b',b)
#     print('a',a)
#     if(len(b)>0):
        
#         out=word[:b[i]]
#         print(out)
#     i+=1

# # word="bhwfeb #hejbhjb  hj ehb #"
# # print word.find("#")
