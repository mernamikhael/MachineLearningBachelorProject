from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import featureUsers
# import userFeature1
# import userfeature2
import userFeature3
import csv




def weight(ind,user_id1,userOSAND,UserOSIOS):
    # with open('users.csv', 'rb') as fil:
    #     t = csv.reader(fil)
    #     for row in t:
    #         user_id1.append(row[2]) 
    #         userOSAND.append(row[4])
    #         UserOSIOS.append(row[5])
    # print user_id1
    index=ind
    # scoreTA1,scoreTB1,scoreLA1,scoreLB1,scorePA1,scorePB1,scoreTagA1,scoreTagB1,scoreFA1,scoreFB1= userFeature1.user(user_id[index], index)
    # scoreTA2,scoreTB2,scoreLA2,scoreLB2,scorePA2,scorePB2,scoreTagA2,scoreTagB2,scoreFA2,scoreFB2= userfeature2.user(user_id[index], index)
    scoreTA3,scoreTB3,scoreLA3,scoreLB3,scorePA3,scorePB3,scoreTagA3,scoreTagB3,scoreFA3,scoreFB3= userFeature3.user(user_id1[index], index)
    meantopic3,v,s,k = beta.stats(scoreTA3,scoreTB3, moments='mvsk')
    meanlang3 ,v,s,k= beta.stats(scoreLA3,scoreLB3, moments='mvsk')
    meanpost3 ,v,s,k= beta.stats(scorePA3,scorePB3, moments='mvsk')
    meantag3 ,v,s,k= beta.stats(scoreTagA3,scoreTagB3, moments='mvsk')
    meanfre3 ,v,s,k = beta.stats(scoreFA3,scoreFB3, moments='mvsk')
    # print meantopic3, meanlang3,meanpost3,meantag3 ,meanfre3 
    i=0
    maxLang=0
    maxTopic=0
    maxFre=0
    maxTag=0
    maxpost=0
    while(i<len(user_id1)):
        indexA=i
        scoreTAH,scoreTBH,scoreLAH,scoreLBH,scorePAH,scorePBH,scoreTagAH,scoreTagBH,scoreFAH,scoreFBH= featureUsers.user(user_id1[index], user_id1[indexA],index,indexA,userOSAND[index],UserOSIOS[index],userOSAND[indexA],UserOSIOS[indexA])

        # print alphaHack,betaHack

        x = np.linspace(0, 1.0, 100)
        # hack 
        hack = beta.pdf(x, scoreTAH,scoreTBH) 
        meantopicH,v,s,k = beta.stats(scoreTAH,scoreTBH, moments='mvsk')
        meanlangH,v,s,k = beta.stats(scoreLAH,scoreLBH, moments='mvsk')
        meanpostH,v,s,k = beta.stats(scorePAH,scorePBH, moments='mvsk')
        meantagH,v,s,k = beta.stats(scoreTagAH,scoreTagBH, moments='mvsk')
        meanfreH,v,s,k = beta.stats(scoreFAH,scoreFBH, moments='mvsk')
        # print i,meantopicH,meanlangH,meanpostH,meantagH,meanfreH
        # plt.plot(x,hack)

        # # week1 normal
        # # week1 = beta.pdf(x, alpha1,beta1) 
        # meantopic1 = beta.stats(scoreTA1,scoreTB1, moments='mvsk')
        # meanlang1 = beta.stats(scoreLA1,scoreLB1, moments='mvsk')
        # meanpost1 = beta.stats(scorePA1,scorePB1, moments='mvsk')
        # meantag1 = beta.stats(scoreTagA1,scoreTagB1, moments='mvsk')
        # meanfre1 = beta.stats(scoreFA1,scoreFB1, moments='mvsk')
        # print meantopic1, meanlang1,meanpost1,meantag1 ,meanfre1 
        # # plt.plot(x,week1)

        # # week2 = beta.pdf(x,alpha2,beta2) 
        # meantopic2 = beta.stats(scoreTA2,scoreTB2, moments='mvsk')
        # meanlang2 = beta.stats(scoreLA2,scoreLB2, moments='mvsk')
        # meanpost2 = beta.stats(scorePA2,scorePB2, moments='mvsk')
        # meantag2 = beta.stats(scoreTagA2,scoreTagB2, moments='mvsk')
        # meanfre2 = beta.stats(scoreFA2,scoreFB2, moments='mvsk')
        # print meantopic2, meanlang2,meanpost2,meantag2 ,meanfre2 
        # # plt.plot(x,week2)

        

        
        

        # plt.savefig(str(index)+"vs"+str(indexA)+".png")
        # # plotting code
        # plt.show()
        # plt.clf()
        # print fsolve(lambda x : hack - week3,0.0)

        # # plotting code
        # plt.show()

        # detect most weird behavior of user
        # getting variation
        dlang=meanlangH-meanlang3
        dtopic=meantopicH-meantopic3
        dfre=meanfreH-meanfre3
        dtag=meantagH-meantag3
        dpost=meanpostH-meanpost3
        # checking larger variation from all users
        if(maxLang<dlang):
            maxLang=dlang
        if(maxpost<dpost):
            maxpost=dpost
        if(maxTag<dtag):
            maxTag=dtag
        if(maxTopic<dtopic):
            maxTopic=dtopic
        if(maxFre<dfre):
            maxFre=dfre
        # hack=0
        # level=0.5
        # print maxLang
        # if(mean1>level):
        #     hack=1
        # print hack,level
        # method1
        # if(d2>=0):
        #     level-=(d2*5)
        # else:
        #     level+=(d2*5)
        # if(d1>=0):
        #     level-=(d1*5)
        # else:
        #     level+=(d1*5)
        # # if very static user
        # if(d2==0 and d1==0):
        #     level-=(mean2*5)
        # if(d3>=level):
        #     hack=1
        # if( d3<=0):
        #     hack=0
        # print i , hack , level ,d3


        i+=1
    # print maxLang, maxTopic,maxTag,maxFre,maxpost
    weightLang=0
    weightTopic=0
    weightpost=0
    weighttag=0
    weightfre=0 
    weight=5 
    count=0
    while(count<5):
        # print weightTopic,weightLang,weightpost,weighttag,weightfre
        if(maxLang == max(maxLang,maxFre,maxpost,maxTag,maxTopic)):
            weightLang=weight
            maxLang=0
        elif(maxTopic == max(maxLang,maxFre,maxpost,maxTag,maxTopic)):
            weightTopic=weight
            maxTopic =0
        elif(maxpost == max(maxLang,maxFre,maxpost,maxTag,maxTopic)):
            weightpost=weight
            maxpost =0
        elif(maxTag == max(maxLang,maxFre,maxpost,maxTag,maxTopic)):
            weighttag=weight
            maxTag =0
        elif(maxFre == max(maxLang,maxFre,maxpost,maxTag,maxTopic)):
            weightfre=weight
            maxFre =0
        count+=1
        weight-=1
    return weightTopic,weightLang,weightpost,weighttag,weightfre
        
