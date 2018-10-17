import pandas as pd
import matplotlib.pyplot as plt

PATH_FILE = 'data/ap_tweets_classified.pkl'
tweets = pd.read_pickle(PATH_FILE)

tweets['created_at'] =  pd.to_datetime(tweets['created_at'])
tweets.index = tweets['created_at']

tweets_trump = tweets[tweets['about'] == 0]
tweets_clinton = tweets[tweets['about'] == 1]

tweets_trump = tweets_trump[['created_at', 'tweet_id']]
tweets_trump = tweets_trump.drop(columns = 'created_at')

tweets_clinton = tweets_clinton[['created_at', 'tweet_id']]
tweets_clinton = tweets_clinton.drop(columns = 'created_at')

clinton_freq = tweets_clinton['tweet_id'].resample('D').count()
clinton_freq = pd.DataFrame(clinton_freq)
clinton_freq = clinton_freq.rename(columns = {'tweet_id': 'clinton_count'})

trump_freq = tweets_trump['tweet_id'].resample('D').count()
trump_freq = pd.DataFrame(trump_freq)
trump_freq = trump_freq.rename(columns = {'tweet_id': 'trump_count'})

freqs = clinton_freq.merge(trump_freq, how='inner', on='created_at')

freqs.plot()
