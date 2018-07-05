
def tagged(tags,tagnbr,taggedposts,total_posts,tagsA,tagnbrA,taggedpostsA,total_postsA,scorePA,scorePB,weighttag):
    count=0
    countA=0
    for row in tagsA:
        if(tags.__contains__(row)):
            scorePB+=(1*weighttag)
            count+=1
            if(count>2):
                break;
        else:
            scorePA+=(1*weighttag)
            countA+=1
            if(countA>2):
                break;
        
        
            # break # should i break or continue?
    
    # averageNormal=sum(tagnbr)/float(len(tagnbr))
    # averageAnomaly=sum(tagnbrA)/float(len(tagnbrA))

    # print averageAnomaly,averageNormal

    # frequetly tagged or not probabilty of empty 
    # noTag=(total_posts-taggedposts)/float(total_posts)
    # noTagA=(total_postsA-taggedpostsA)/float(total_postsA)

    # if(noTag<noTagA):
    #     scorePA+=1
    # else:
    #     scorePB+=1

    # max number of tag
    if(len(tagnbrA)>0 and len(tagnbr)>0):
        if(max(tagnbr)<(max(tagnbrA)) or sum(tagnbr)/float(taggedposts) < sum(tagnbrA)/float(taggedpostsA)):
            scorePA+=(1*weighttag)
        else:
            scorePB+=(1*weighttag)


    return scorePA,scorePB