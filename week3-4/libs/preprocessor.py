# -*- coding: utf-8 -*-
import re
import string
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))

def remove_hashtags(text):
    text = re.sub('#\S+', '', text, flags=re.MULTILINE)
    return text


def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop_words]
    return " ".join(text)

def preprocess_tweets(tweets):
    # Remove hyperlinks
    tweets = tweets.str.replace(r'http\S+', '', flags=re.MULTILINE)
    # Remove @mentions
    tweets = tweets.str.replace(r'@\S+', '', flags=re.MULTILINE)
    # Remove #hashtags 
    tweets = tweets.apply(remove_hashtags)
    # Remove stop words
    tweets = tweets.apply(remove_stopwords)
    # Remove punctuation
    tweets = tweets.str.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    tweets = tweets.str.replace(r'^\d+\s|\s\d+\s|\s\d+$', '', flags=re.MULTILINE)
    # Remove words that start with a number. To eliminate cases of 000000in etc.
    tweets = tweets.str.replace(r'\w*\d\w*', '', flags = re.MULTILINE)
    ## Remove ampersant (will need to remove other symbols as well, see wordcloud)
    tweets = tweets.str.replace(r'amp', '', flags=re.MULTILINE)
    
    return tweets
    