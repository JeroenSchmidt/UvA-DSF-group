# -*- coding: utf-8 -*-
import re
import string
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))

def remove_hashtags(text):
    text = re.sub('#\S+', '', text, flags = re.MULTILINE)
    return text


def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop_words]
    return " ".join(text)

def remove_hyperlinks(text):
    text = re.sub(r'http\S+', '', text, flags = re.MULTILINE)
    return text

def remove_mentions(text):
    text = re.sub(r'@\S+', '', text, flags = re.MULTILINE)
    return text

def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def remove_numbers(text):
    text = re.sub(r'^\d+\s|\s\d+\s|\s\d+$', '', text, flags = re.MULTILINE)
    return text

def remove_words_with_numbers(text):
    text = re.sub(r'\w*\d\w*', '', text, flags = re.MULTILINE)
    return text

def remove_ampersand(text):
    text = re.sub(r'amp', '', text, flags=re.MULTILINE)
    return text

def preprocess_tweets(tweets):
    # Remove hyperlinks
    tweets = tweets.apply(remove_hyperlinks)
    # Remove @mentions
    tweets = tweets.apply(remove_mentions)
    # Remove #hashtags 
    tweets = tweets.apply(remove_hashtags)
    # Remove stop words
    tweets = tweets.apply(remove_stopwords)
    # Remove punctuation
    tweets = tweets.apply(remove_punctuation)
    # Remove numbers
    tweets = tweets.apply(remove_numbers)
    # Remove words that start with a number. To eliminate cases of 000000in etc.
    tweets = tweets.apply(remove_words_with_numbers)
    ## Remove ampersand
    tweets = tweets.apply(remove_ampersand)
    
    return tweets
    