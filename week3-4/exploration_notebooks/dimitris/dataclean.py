# -*- coding: utf-8 -*-

# %%
# Note - The paths here imply you started your kernel on the week3-4 folder.
import sys
lib_dir = "libs/"
if lib_dir not in sys.path:
    sys.path.append(lib_dir)

import preprocessor
import pandas as pd
import re
from nltk import FreqDist, tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt

PATH_FILE = 'data/cp_tweets.pkl'

# %%
tweets = pd.read_pickle(PATH_FILE)
tweets.info()

#%% Preprocess data
tweets['text'] = preprocessor.preprocess_tweets(tweets['text'])

#%%
# Experiment - let's remove trump & hillary words.
tweets['text'] = tweets['text'].str.replace(r'donald', '', flags=re.MULTILINE)
tweets['text'] = tweets['text'].str.replace(r'trump', '', flags=re.MULTILINE)
tweets['text'] = tweets['text'].str.replace(r'hillary', '', flags=re.MULTILINE)
tweets['text'] = tweets['text'].str.replace(r'clinton', '', flags=re.MULTILINE)

#%% 
print("Number of tweets that won't be needed after text pre-processing: ", sum(tweets['text'] == ""))

#%%
# Keep only the tweets that have meaningful text
tweets = tweets[tweets['text'] != ""]
len(tweets)

#%% Generate word cloud
wordlist = ' '.join(tweets['text'])
wordcloud = WordCloud().generate(wordlist)

#%%
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#%%
text = tweets['text'].str.lower().str.cat(sep=' ')
words = tokenize.word_tokenize(text)
#%%
word_dist = FreqDist(words)
word_dist.plot(30, cumulative = False)