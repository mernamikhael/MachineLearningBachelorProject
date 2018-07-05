from six.moves import urllib
import idna
import uritools
import urlextract
import re


# myString = "This is my tweet check it out https://www.tinyurl.com/blah"

# print re.search("(?P<url>https?://[^\s]+)", myString).group("url")

# # print  re.findall('((http|ftp)s?://.*?', myString)

def link(links,nbr_links,total_posts,linksA,nbr_linksA,total_postsA):
    score=0
    # check phishing

    # check frequency
    userLink=nbr_links/float(total_posts)
    anomalyLink=nbr_linksA/float(total_postsA)

    if(userLink<anomalyLink):
        score+=userLink
    return score 
