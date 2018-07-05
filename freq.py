from datetime import date, datetime

def frequency(postsWeekends52,postsWeekends51,postsWeekends50,weekends,postsWeekends52A,postsWeekends51A,
postsWeekends50A,weekendsA,activityM52, activityM51, activityM50, activityN52, activityN51, activityN50,
 activityM52A, activityM51A, activityM50A, activityN52A, activityN51A, activityN50A,Time,
 daysweek, daysweekA,season,total_posts,seasonA,total_postsA,weekposts52,weekposts51,weekposts50,weekposts52A,
 weekposts51A,weekposts50A,pDay,scoreTagA,scoreTagB,weightfreq):

    scoreweekA,scoreweekB=week(weekposts52,weekposts51,weekposts50,weekposts52A,weekposts51A,weekposts50A,scoreTagA,scoreTagB,weightfreq)
    scoredayA,scoredayB=day(daysweek,daysweekA,pDay,scoreweekA,scoreweekB,weightfreq)
    scoreTimeA,scoreTimeB=activityTime(activityM52,activityM51,activityM50,activityN52,activityN51,activityN50
    ,activityM52A,activityM51A,activityM50A,activityN52A,activityN51A,activityN50A,Time,scoredayA,scoredayB,weightfreq)
    scoreSeaA,scoreSeaB=seasons(season,total_posts,seasonA,total_postsA,scoreTimeA,scoreTimeB,weightfreq)
    FinalscoreA,FinalscoreB=weekend(postsWeekends52,postsWeekends51,postsWeekends50,weekends
    ,postsWeekends52A,postsWeekends51A,postsWeekends50A,weekendsA,scoreSeaA,scoreSeaB,weightfreq)

    return FinalscoreA,FinalscoreB


    
def week(week52,week51,week50,week52A,week51A,week50A,scoreA,scoreB,weightfreq):
    
    # if(week52<week52A):
    #     score+=1
    # if(week51<week51A):
    #     score+=1
    # if(week50<week50A):
    #     score+=1

    if(week52A>max(week51,week50)):
        scoreA+=(1*weightfreq)
    else:
        scoreB+=(1*weightfreq)
    return scoreA,scoreB

def day(daysweek,daysweekA,pDay,scoreA,scoreB,weightfreq):
    
    week50=daysweek[2]
    week50A=daysweekA[2]
    week51=daysweek[1]
    week51A=daysweekA[1]
    week52=daysweek[0]
    week52A=daysweekA[0]
    # check if #posts per day > weeks for rarly any copples of days
    if(pDay==2 or pDay==4):
        if(week52A[0]>(max(sum(week50),sum(week51),sum(week52)))):
            scoreA+=(2*weightfreq)
        else:
            scoreB+=(1*weightfreq)
    
        if(week52A[1]>(max(sum(week50),sum(week51),sum(week52)))):
            scoreA+=(2*weightfreq)
        else:
            scoreB+=(1*weightfreq)

        if(week52A[2]>(max(sum(week50),sum(week51),sum(week52)))):
            scoreA+=(2*weightfreq)
        else:
            scoreB+=(1*weightfreq)
        
        if(week52A[3]>(max(sum(week50),sum(week51),sum(week52)))):
            scoreA+=(1*weightfreq)
        else:
            scoreB+=(1*weightfreq)
        
        if((week52A[6])>(max(sum(week50),sum(week51),sum(week52)))):
            scoreA+=(1*weightfreq)
        else:
            scoreB+=(1*weightfreq)
        
    if(pDay==3):
        if(week52A[0]>2):
            scoreA+=(2*weightfreq)
        else:
            scoreB+=(1*weightfreq)
    
        if(week52A[1]>2):
            scoreA+=(2*weightfreq)
        else:
            scoreB+=(1*weightfreq)

        if(week52A[2]>2):
            scoreA+=(1*weightfreq)
        
        if(week52A[3]>2):
            scoreA+=(1*weightfreq)
        else:
            scoreB+=(1*weightfreq)
        
        if((week52A[6])>2):
            scoreA+=(1*weightfreq)
        else:
            scoreB+=(1*weightfreq)
        

    if(week52A[0]>(max(sum(week51)/7.0,sum(week50/7.0)))):
        # if(max(week50[0],week51[0],week52[0])):
        scoreA+=(1*weightfreq)
    else:
        scoreB+=(1*weightfreq)
    
    if(week52A[1]>(max(sum(week51)/7.0,sum(week50/7.0)))):
        # if(max(week50[1],week51[1],week52[1])):
        scoreA+=(1*weightfreq)
    else:
        scoreB+=(1*weightfreq)

    if(week52A[2]>(max(sum(week51)/7.0,sum(week50/7.0)))):
        # if((max(week50[2],week51[2],week52[2]))):
        scoreA+=(1*weightfreq)
    else:
        scoreB+=(1*weightfreq)
    if(week52A[3]>(max(sum(week51)/7.0,sum(week50/7.0)))):
        # if((max(week50[3],week51[3],week52[3])>0)):
        scoreA+=(1*weightfreq)
    else:
        scoreB+=(1*weightfreq)
    
    if((week52A[6])>(max(sum(week51)/7.0,sum(week50/7.0)))):
        # if((max(week50[6],week51[6],week52[6])>0)):
        scoreA+=(1*weightfreq)
    else:
        scoreB+=(1*weightfreq)

    return scoreA,scoreB

def activityTime(activityM52,activityM51,activityM50,activityN52,activityN51,activityN50
    ,activityM52A,activityM51A,activityM50A,activityN52A,activityN51A,activityN50A,Time,scoreA,scoreB,weightfreq):
    if(Time==1):
        # if(max(activityM50,activityM51,activityM52)<activityM50A):
        #     score+=1
        # if(max(activityM50,activityM51,activityM52)<activityM51A):
        #     score+=1
        if(max(activityM50,activityM51,activityM52)<activityM52A):
            scoreA+=(1*weightfreq)
        else:
            scoreB+=(1*weightfreq)
        
        
    elif(Time==0):
        # if(max(activityN50,activityN51,activityN52)<activityN50A):
        #     score+=1
        # if(max(activityN50,activityN51,activityN52)<activityN51A):
        #     score+=1
        if(max(activityN50,activityN51,activityN52)<activityN52A):
            scoreA+=(2*weightfreq)
        else:
            scoreB+=(1*weightfreq)
    
    total_posts50=activityM50+activityN50
    total_posts51=activityM51+activityN51
    total_posts52=activityM52+activityN52
    total_posts50A=activityM50A+activityN50A
    total_posts51A=activityM51A+activityN51A
    total_posts52A=activityM52A+activityN52A

    if(total_posts50):
        posts_morn50=activityM50/float(total_posts50)
    else:
        posts_morn50=0
    if(total_posts51):
        posts_morn51=activityM51/float(total_posts51)
    else:
        posts_morn51=0
    # if(total_posts52):
    #     posts_morn52=activityM52/float(total_posts52)
    # else:
    #     post_morn52=0
    # posts_morn50A=activityM50A/float(total_posts50A)
    # posts_morn51A=activityM51A/float(total_posts51A)
    if(total_posts52A):
        posts_morn52A=activityM52A/float(total_posts52A)
    else:
        posts_morn52A=0
    if(total_posts50):
        posts_night50=activityN50/float(total_posts50)
    else:
        posts_night50=0
    if(total_posts51):
        posts_night51=activityN51/float(total_posts51)
    else:
        posts_night51=0
    # posts_night52=activityN52/float(total_posts52)
    # posts_night50A=activityN50A/float(total_posts50A)
    # posts_night51A=activityN51A/float(total_posts51A)
    if(total_posts52A):
        posts_night52A=activityN52A/float(total_posts52A)
    else:
        posts_night52A=0
    # if(posts_morn50<posts_morn50A):
    #     score+=(1+posts_morn50)
    # if(posts_morn51<posts_morn51A):
    #     score+=(1+posts_morn51)
    if(posts_morn52A>max(posts_morn51,posts_morn50)):
        scoreA+=(1*weightfreq)
    else:
        scoreB+=(1*weightfreq)
    # if(posts_night50<posts_night50A):
    #     score+=(1+posts_night50)
    # if(posts_night51<posts_night51A):
    #     score+=(1+posts_night51)
    if(posts_night52A>max(posts_night50,posts_night51)):
        scoreA+=(1*weightfreq)
    else:
        scoreB+=(1*weightfreq)

    return  scoreA,scoreB

def weekend(postsWeekends52,postsWeekends51,postsWeekends50,weekends
    ,postsWeekends52A,postsWeekends51A,postsWeekends50A,weekendsA,scoreA,scoreB,weightfreq):
    if(max(postsWeekends51,postsWeekends50)<postsWeekends52A):
        scoreA+=(1*weightfreq)
    else:
        scoreB+=(1*weightfreq)
    # if(postsWeekends52<postsWeekends52A):
    #     score+=1+(float(postsWeekends52/2.0))
    # if(postsWeekends51<postsWeekends51A):
    #     score+=1+(float(postsWeekends51/2.0))
    # if(postsWeekends50<postsWeekends50A):
    #     score+=1+(float(postsWeekends50/2.0))
    return scoreA,scoreB

def seasons(season,total_posts,seasonA,total_postsA,scoreA,scoreB,weightfreq):
    winter=season[0]/float(total_posts)
    winterA=seasonA[0]/float(total_postsA)
    spring=season[1]/float(total_posts)
    springA=seasonA[1]/float(total_postsA)
    summer=season[2]/float(total_posts)
    summerA=seasonA[2]/float(total_postsA)
    autumn=season[3]/float(total_posts)
    autumnA=seasonA[3]/float(total_postsA)

    if(winter<winterA):
        scoreA+=(1*weightfreq)
    else:
        scoreB+=(1*weightfreq)
    if(spring<springA):
        scoreA+=(1*weightfreq)
    else:
        scoreB+=(1*weightfreq)
    if(summer<summerA):
        scoreA+=(1*weightfreq)
    else:
        scoreB+=(1*weightfreq)
    if(autumn<autumnA):
        scoreA+=(1*weightfreq)
    else:
        scoreB+=(1*weightfreq)

    return scoreA,scoreB