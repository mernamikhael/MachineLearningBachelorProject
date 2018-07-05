from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import user
import userProf1
import userprof2
import userProf3
import weight
import csv
from scipy.optimize import fsolve
import uncertainties as u

user_id=[]
userOSAND=[]
UserOSIOS=[]
userAOSAND=[]
UserAOSIOS=[]
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
with open('dataset\users.csv', 'rb') as fil:
    t = csv.reader(fil)
    for row in t:
        user_id.append(row[2]) 
        userOSAND.append(row[4])
        UserOSIOS.append(row[5])
index=4
# while (index <47):
print index
# weightTopic,weightLang,weightpost,weighttag,weightfre=weight.weight(index,user_id,userOSAND,UserOSIOS)
weightTopic=3
weightLang=1
weightpost=2
weighttag=4
weightfre=5
print weightTopic,weightLang,weightpost,weighttag,weightfre
alpha3,beta3,alpha2,beta2,alpha1,beta1=userProf1.user(user_id[index], index,weightTopic,weightLang,weightpost,weighttag,weightfre)

i=0
x = np.linspace(0, 1.0, 100)

week1 = beta.pdf(x, alpha1,beta1) 
mean2, var, skew, kurt = beta.stats( alpha1,beta1, moments='mvsk')
print mean2

week2 = beta.pdf(x,alpha2,beta2) 
mean3, var, skew, kurt = beta.stats( alpha2,beta2, moments='mvsk')
print mean3

week3 = beta.pdf(x,alpha3,beta3) 
mean4, var, skew, kurt = beta.stats( alpha3,beta3, moments='mvsk')
print mean4

while(i<len(user_id)):
    indexA=i
    alphaHack,betaHack= user.user(user_id[index], user_id[indexA],index,indexA,userOSAND[index],UserOSIOS[index],userOSAND[indexA],UserOSIOS[indexA],weightTopic,weightLang,weightpost,weighttag,weightfre)

    print alphaHack,betaHack

    # hack
    hack = beta.pdf(x, alphaHack,betaHack) 
    mean1, var, skew, kurt = beta.stats(alphaHack,betaHack, moments='mvsk')
    print mean1
    plt.plot(x,hack)
    plt.plot(x,week3)
    plt.plot(x,week2)
    plt.plot(x,week1)
    # week1 normal

    plt.xlabel('User Behavior Variation \n a= '+str(alphaHack)+'b= '+str(betaHack), font)
    plt.ylabel('Beta PDF', font)
    plt.savefig("wgraph/"+str(index)+"vs"+str(indexA)+".png")
    # # plotting code
    # plt.show()
    plt.clf()
    # print fsolve(lambda x : hack - week3,0.0)

    # # plotting code
    # plt.show()

    # detect most weird behavior of user
    hack=0
    # d1=mean3-mean2
    # d2=mean4-mean3
    # d3=mean1-mean4
    V1=0.5-mean2
    V2=0.5-mean3
    v3=0.5-mean4
    
    averageV= (V1+v3+V2)/3.0
    vH=0
    if(mean1>0.5):
        hack=1
    else:
        vH=0.5-mean1
    if(mean3==mean2 and mean2==mean4 and mean2<0.05):
        level=averageV*0.8
    else:
        level=averageV*0.75

    # if(d2>=0):
    #     level-=(d2)
    # else:
    #     level+=(d2)
    # if(d1>=0):
    #     level-=(d1)
    # else:
    #     level+=(d1)
    # # if very static user
    # if(d2==0 and d1==0):
    #     level+=(mean2*5)
    # com=0.5
    # if(mean4<0.5):
    #     com-=(d3)
    # else:
    #     com+=(d3)
            
    if(vH<=level):
        hack=1
    # if( vH<=0):
    #     hack=0
    print i , hack , level ,vH

    i+=1
    # index+=1
   

    

