import pandas as pd


def age():
    '''Returns age of individual when they filled in the survay'''
    survey = pd.read_pickle("data/Visual_well_being/survey.pickle")
    age = pd.to_datetime(survey.start_q).dt.year - survey.born
    return age

def instagram_account_stats():
    '''
    Returns statistics on the instagram user. 
    * Number of followers
    * Number of people following
    * Number of photos
    '''
    
    image_date = pd.read_pickle("data/Visual_well_being/image_data.pickle")
    instagram_account_info = image_date[["user_id","user_followed_by","user_follows","user_posted_photos"]]\
                            .drop_duplicates()
    
    return instagram_account_info



    