import pandas as __pd

import image_level_agg_features as __img_f


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