# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 00:02:41 2018

@author: dadangewp
"""
import requests
import re

tweetText = "RT @bnixole: bitch shut the fuck up you're fucking your best friends dad https://t.co/1YR6ydZMgc"
urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweetText)
    #print(urls)
flag = 0
for url in urls:
    link = ""
    try:
        link = requests.get(url, verify=False, timeout=10).url
    except :
        #print ("connection refused")
        link = "error"     
    link = str(link)
    print (link)
    structure = link.split("/")
print(structure)