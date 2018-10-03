import sys
lib_dir = "libs/"
if lib_dir not in sys.path:
    sys.path.append(lib_dir)

import re
import preprocessor
import pandas as pd
import DSF_helpers
import geopandas as gpd
import seaborn as sns

PATH_FILE = 'data/ap_tweets_classified.pkl'

tweets = pd.read_pickle(PATH_FILE)
tweets = tweets[tweets['about'].notnull()]

states = gpd.read_file('data/states_21basic/states.shp')

# Only look for USA tweets
logic = (tweets.country == 'United States') & \
        (tweets.place_place_type != 'poi') & \
        (tweets.place_place_type != 'country') & \
        (tweets.place_place_type != 'neighborhood')

# Keep only USA tweets
tweets_USA = tweets[logic]
tweets_USA['state_abrv'] = tweets_USA.place_full_name.apply(DSF_helpers.get_state_ABRV)

# %% We Need to plot who do the most loud states in the US tweet about.
# Find the states who tweet the most
most_loud_states = tweets_USA.groupby('state_abrv')[['state_abrv']].count()
# Note tweet_id here is the count of all tweets - didn't bother to rename
most_loud_states = most_loud_states.rename(columns={'state_abrv': 'tweetcount'})
most_loud_states = most_loud_states.sort_values('tweetcount', ascending = False).head(15)

tweets_per_state = tweets_USA[['state_abrv', 'about']].groupby(['state_abrv'])['about'].value_counts()
tweets_per_state = pd.DataFrame(tweets_per_state)
tweets_per_state = tweets_per_state.rename(columns = {'about':'classcount'})
tweets_per_state = tweets_per_state.unstack(level = 1)['classcount'].reset_index()
tweets_per_state = tweets_per_state.rename(columns={0: 'trumpcount', 1: 'clintoncount', 2: 'bothcount'})

most_loud_states = most_loud_states.merge(tweets_per_state, how='inner', on='state_abrv')


df = pd.melt(most_loud_states, id_vars="state_abrv", var_name="category", value_name="totalcount")
sns.catplot(x='state_abrv', y='totalcount', hue='category', data=df, kind='bar')
