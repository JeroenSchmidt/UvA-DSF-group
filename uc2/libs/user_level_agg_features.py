import pandas as __pd
import numpy as __np
import image_level_agg_features as __img_f


__data_dir = "../../data/Visual_well_being/"

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


def avg_posts_per_day():
    '''
    Returns the average number of posts per day that the person posted. Returns 7 averages:
    `early day`: 8:00-12:00
    `late_day`: 12:00-20:00
    `early_night`: 20:00-00:00
    `late_night`: 00:00-8:00
    
    `day`: 8:00-20:00
    `night`: 20:00-8:00
    `whole_day`: the average for the whole date
    '''
    
    image_date = __pd.read_pickle(__data_dir + "image_data.pickle")
    image_date.image_posted_time = __pd.to_datetime(image_date.image_posted_time)
    
    x = image_date[["image_posted_time","image_id","user_id"]].drop_duplicates()    
    early_day_b = x.image_posted_time.isin(x.set_index("image_posted_time").between_time('8:00', '12:00').index)
    late_day_b = x.image_posted_time.isin(x.set_index("image_posted_time").between_time('12:00', '20:00').index)
    early_night_b = x.image_posted_time.isin(x.set_index("image_posted_time").between_time('20:00', '00:00').index)
    late_night_b = x.image_posted_time.isin(x.set_index("image_posted_time").between_time('00:00', '8:00').index)

    day_b = x.image_posted_time.isin(x.set_index("image_posted_time").between_time('8:00', '20:00').index)
    night_b = x.image_posted_time.isin(x.set_index("image_posted_time").between_time('20:00', '8:00').index)
    
    x["early_day"] = __np.where(early_day_b, 1, 0)
    x["late_day"] = __np.where(late_day_b, 1, 0)
    x["early_night"] = __np.where(early_night_b, 1, 0)
    x["late_night"] = __np.where(late_night_b, 1, 0)

    x["day"] = __np.where(day_b, 1, 0)
    x["night"] = __np.where(night_b, 1, 0)
    x["whole_date"] = 1
    
    xx = x.drop(columns="image_id",axis=0)\
    .groupby([x.user_id,x.image_posted_time.dt.date])\
    .sum()
    
    out = xx.groupby("user_id").mean()
    
    return out

    