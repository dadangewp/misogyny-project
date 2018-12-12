# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:39:02 2018

@author: dadangewp
"""

import emoji
import codecs
from emojiSentiment import EmojiSentiment

emosent = EmojiSentiment()

text = "I swear, any stupid ass walking around with a mask today is getting a swift kick in the dick 💀"
#text = codecs.encode(text, encoding="utf-8")
#print(text)
text = str(text)
parsed = emoji.demojize(text)
print (parsed)
#score = emosent.get_emoji_sentiment(parsed)
#print (score)