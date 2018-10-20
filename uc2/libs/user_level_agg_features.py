import pandas as __pd

import image_level_agg_features as __img_f
import numpy as np


__data_dir = "../../data/Visual_well_being/"
__expensive_dir = "../../data/expensive_features/"

def age():
    '''Returns age of individual when they filled in the survay'''
    survey = __pd.read_pickle(__data_dir + "survey.pickle")
    survey["age"] = __pd.to_datetime(survey.start_q).dt.year - survey.born
    survey = survey[["insta_user_id","age"]]
    return survey

def instagram_account_stats():
    '''
    Returns statistics on the instagram user. 
    * Number of followers
    * Number of people following
    * Number of photos
    '''
    
    image_date = __pd.read_pickle(__data_dir + "image_data.pickle")
    instagram_account_info = image_date[["user_id","user_followed_by","user_follows","user_posted_photos"]]\
                            .drop_duplicates()
    
    return instagram_account_info


def ratio_of_topics():
    '''
    NOTE: This is an expensive operation.
    
    Returns a matrix of the percentages that shows you what percentage that object appears in the users photos.
    '''

    image_date = __pd.read_pickle("../../data/Visual_well_being/image_data.pickle")
    user_img = image_date[["image_id","user_id"]].drop_duplicates()

    ob = __img_f.binary_object_matrix()
    photo_counts = instagram_account_stats()[["user_id","user_posted_photos"]]

    counts_per_user = ob.merge(user_img,how="inner",on="image_id")\
                .groupby("user_id").sum()\
                .merge(photo_counts,on="user_id",how="inner")

    df = counts_per_user.iloc[:,1:-1]\
                        .divide(
                                counts_per_user.user_posted_photos,
                                axis=0
                                )

    df["user_id"] = counts_per_user.user_id

    return df

def average_number_of_faces_from_photos_with_faces():
    '''Returns the average number of faces in photos with faces'''
    image_date = __pd.read_pickle("../../data/Visual_well_being/image_data.pickle")
    num_faces = __img_f.number_of_faces()
    
    out = image_date[["image_id","user_id"]]\
    .merge(num_faces,on="image_id",how="inner")\
    .fillna(0)\
    .groupby("user_id")\
    .mean()
    
    return out

def average_number_of_faces_over_all_photos():
    '''Returns the average number of faces accross all photos of the user'''
    image_date = __pd.read_pickle("../../data/Visual_well_being/image_data.pickle")
    num_faces = __img_f.number_of_faces()
    
    #of the photos that have faces, what is the average
    out = image_date[["image_id","user_id"]]\
    .merge(num_faces,on="image_id",how="outer")\
    .fillna(0)\
    .groupby("user_id")\
    .mean()
    
    return out

def average_engagement():
    '''
    Returns the average number of likes and comments per user.
    '''
    image_data = __pd.read_pickle('../../data/Visual_well_being/image_data.pickle')
    updated_metrics = __img_f.final_like_and_comments()
    image_data = image_data.merge(updated_metrics, how='left', on='image_id')
    avg_engagement = image_data[['user_id', 'likes', 'comments']].groupby('user_id').mean().reset_index()
    avg_engagement = avg_engagement.rename(columns = {'likes': 'avg_likes', 'comments': 'avg_comments'})

    return avg_engagement

def filter_features():
    '''
    Returns percentage of happy/depressed filters, ratio of happy over depressed filters
    '''
    image_data = __pd.read_pickle('../../data/Visual_well_being/image_data.pickle')
    # Keep only filters, remove Unknown and Normal entries
    filter_data = image_data[~image_data.image_filter.isin(['Normal', 'Unknown'])][['image_id', 'image_filter']]
    # Load the filter categories
    filter_categories = __pd.read_csv('../../data/Visual_well_being/filter_categories.csv', sep=';')
    filter_categories = filter_categories.rename(columns={'class':'happiness_class'})
    # Remove images whose filter is not associated with a category in filter_categories (only 23 of them, no big deal)
    filter_data = filter_data[filter_data.image_filter.isin(filter_categories['filter'])]
    # Add filter category information to the dataFrame
    filter_data = filter_data.merge(filter_categories, how='left', left_on='image_filter', right_on='filter').drop('filter', axis=1)
    # Create Dummies that will help to summarize happy filters and depressed filters later on.
    filter_dummies = __pd.get_dummies(filter_data['happiness_class']).rename(columns= {0: 'depressed_filter', 1: 'happy_filter'})
    filter_data['happy_filter'] = filter_dummies['happy_filter']
    filter_data['depressed_filter'] = filter_dummies['depressed_filter']
    filter_data = filter_data.drop(['image_filter', 'happiness_class'], axis=1)
    # Merge with original image data
    image_data = image_data.merge(filter_data, 'left', 'image_id')
    # Create Filter features dataframe
    filter_features = image_data[['user_id', 'happy_filter', 'depressed_filter']].groupby('user_id').sum().reset_index()
    filter_features['total_photos'] = image_data[['user_id', 'user_posted_photos']].groupby('user_id').max().reset_index()['user_posted_photos']
    # Build the features
    filter_features['happy_flt_pct'] = filter_features['happy_filter'] / filter_features['total_photos']
    filter_features['depressed_flt_pct'] = filter_features['depressed_filter'] / filter_features['total_photos']
    filter_features['happy_to_depressed_flt_ratio'] = filter_features['happy_filter'] / filter_features['depressed_filter']
    filter_features = filter_features.replace([np.inf, -np.inf], np.nan).head()
    # Drop not neeed columns
    filter_features = filter_features.drop(['happy_filter', 'depressed_filter', 'total_photos'], axis=1).head()

    return filter_features
    