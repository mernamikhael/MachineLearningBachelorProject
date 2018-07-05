

def type_post(added,shared,updated,posted,total_posts,addA,shareA,updateA,postA,total_postsA,personality,scoreLA,scoreLB,weightpost):
    # sharedRatio=shared/float(total_posts)
    # updateRatio=updated/float(total_posts)
    # addRatio=added/float(total_posts)
    # postRatio=posted/float(total_posts)

    # sharedRatioA=shareA/float(total_postsA)
    # updateRatioA=updateA/float(total_postsA)
    # addRatioA=addA/float(total_postsA)
    # postRatioA=postA/float(total_postsA)
    # print added,shared,updated,posted
    # print addA,shareA,updateA,postA

    if(personality==1):
        if(addA<max(shared,updated,posted)):
            scoreLA+=(2*weightpost)
        else:
            scoreLB+=(1*weightpost)
    if(personality==2):
        if(shareA<max(added,updated,posted)):
            scoreLA+=(2*weightpost)
        else:
            scoreLB+=(1*weightpost)
    if(personality==3):
        if(updateA<max(added,shared,posted)):
            scoreLA+=(2*weightpost)
        else:
            scoreLB+=(1*weightpost)
    if(personality==4):
        if(postA<max(added,updated,shared)):
            scoreLA+=(2*weightpost)
        else:
            scoreLB+=(1*weightpost)

    
    if(shared<shareA):
        scoreLA+=(1*weightpost)
    else:
        scoreLB+=(1*weightpost)
    if(updated<updateA):
        scoreLA+=(2*weightpost)
    else:
        scoreLB+=(1*weightpost)
    if(added<addA):
        scoreLA+=(1*weightpost)
    else:
        scoreLB+=(1*weightpost)
    if(posted<postA):
        scoreLA+=(1*weightpost)
    else:
        scoreLB+=(1*weightpost)
    return scoreLA,scoreLB





