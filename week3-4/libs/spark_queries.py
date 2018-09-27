# -*- coding: utf-8 -*-

#%%
import pandas as pd
pd.set_option("max_colwidth",10000)

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, desc, col
from pyspark.sql.types import ArrayType, StringType, MapType, FloatType

#%% Start Spark Session
spark = SparkSession.builder\
                        .master("local[*]")\
                        .config('spark.executor.memory', '5g')\
                        .config('spark.driver.memory', '5g')\
                        .config("spark.sql.session.timeZone", "UTC")\
                        .config("spark.sql.execution.arrow.enabled","false")\
                    .appName("Sentiment Analysis")\
                    .getOrCreate()
spark

#%% Load Data into Spark
data=spark.read.json("data/geotagged_tweets_20160812-0912.jsons")
data.count()

#%% Define relative data schema
data_rel = data.selectExpr(
    "id as tweet_id",
    "created_at",
    "entities.hashtags as entities_hashtags",
    "entities.user_mentions as entities_mentions",
    "place.bounding_box as place_bounding_box",
    "place.country",
    "place.country_code as place_country_code",
    "place.full_name as place_full_name",
    "place.id as place_id",
    "place.name as place_name",
    "place.place_type as place_place_type",
    "place.url as place_url",
    "favorite_count",
    "geo.coordinates as geo_coordinates",
    "geo.type as geo_type",
    "text",
    "lang",
    "retweet_count",
    "retweeted",
    "source",
    "timestamp_ms",
    "user.id as user_id",
    "user.created_at as user_created_at",
    "user.favourites_count as user_favourites_count",
    "user.followers_count as user_followers_count",
    "user.friends_count as user_friends_count",
    "user.location as user_location",
    "user.screen_name as user_screen_name",
    "user.statuses_count as user_statuses_count",
    "user.time_zone as user_time_zone",
    "user.url as user_url",
    "user.verified as user_verified"
      )

# Ids of spammy bots that we won't need in the dataset.
blacklist = [115110145] 
data_rel = data_rel.filter(col("user_id").isin(*blacklist) == False)

data_rel.toPandas().to_pickle('data/bp_tweets.pkl')