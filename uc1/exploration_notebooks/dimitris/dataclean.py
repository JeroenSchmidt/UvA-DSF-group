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
import matplotlib.pyplot as plt

PATH_FILE = 'data/bp_tweets.pkl'

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

#%% Remove most common words
text = tweets['text'].str.lower().str.cat(sep=' ')
tokens = tokenize.word_tokenize(text)
word_dist = FreqDist(tokens)
most_common_words = [word for word, freq in word_dist.most_common(10)]
tweets['text'] = tweets['text'].apply(lambda x: " ".join(x for x in x.split() if x not in most_common_words))

#%% 
print("Number of tweets that won't be needed after text pre-processing: ", sum(tweets['text'] == ""))
tweets = tweets[tweets['text'] != ""]

#%% Save to pickle
#tweets.to_pickle('data/ap_tweets.pkl')