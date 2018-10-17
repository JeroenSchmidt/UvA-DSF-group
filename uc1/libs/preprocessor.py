# -*- coding: utf-8 -*-
import re
import string
from nltk.corpus import stopwords
from textblob import TextBlob


stop_words = set(stopwords.words('english'))

def remove_hashtags(text):
    text = re.sub(r'#\S+', '', text, flags = re.MULTILINE)
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
    
    # dont remove # and @, let remove hashtags and remove mentions remove these symbols for us
    punctuation = string.punctuation.replace("#","")\
                                       .replace("@","")
    
    text = text.translate(str.maketrans('', '', punctuation))
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

def remove_single_character_words(text):
    text = re.sub(r'((@|#)\W)|( (@|#) )','', text, flags=re.MULTILINE)
    return text

def remove_more_then_one_space(text):
    text = re.sub(r'\s\s+','', text, flags=re.MULTILINE)
    return text

def preprocess_tweets(tweets, remove_hashtags_arg=True, remove_mentions_arg=True):
    # Remove hyperlinks
    tweets = tweets.apply(remove_hyperlinks)
    
    # Remove @mentions
    if remove_hashtags_arg:
        tweets = tweets.apply(remove_mentions)

    # Remove #hashtags     
    if remove_mentions_arg:
        tweets = tweets.apply(remove_hashtags)
    
    # Remove stop words
    tweets = tweets.apply(remove_stopwords)
    # Remove punctuation
    tweets = tweets.apply(remove_punctuation)
    # Remove numbers
    tweets = tweets.apply(remove_numbers)
    # Remove words that start with a number. To eliminate cases of 000000in etc.
    tweets = tweets.apply(remove_words_with_numbers)
    # Remove ampersand
    tweets = tweets.apply(remove_ampersand)
    # Remove single letter words
    tweets = tweets.apply(remove_single_character_words)
    
    # Remove more then one spaceing
    tweets = tweets.apply(remove_more_then_one_space)
   
    # Correct spelling
    # print('correcting spelling')
    # tweets = tweets.apply(lambda x: str(TextBlob(x).correct()))
    
    return tweets
    