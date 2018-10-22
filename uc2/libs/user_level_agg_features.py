import pandas as __pd
import numpy as __np
import image_level_agg_features as __img_f
import numpy as np
import load_clean_data as __data


__data_dir = "../../data/Visual_well_being/"

def age():
    '''Returns age of individual when they filled in the survay'''
    survey = __data.load_survey()
    #__pd.read_pickle(__data_dir + "survey.pickle")
    
    survey["age"] = __pd.to_datetime(survey.start_q).dt.year - survey.born
    survey = survey[["insta_user_id","age"]]
    survey.columns = ['user_id', 'age']
    return survey

def instagram_account_stats():
    '''
    Returns statistics on the instagram user. 
    * Number of followers
    * Number of people following
    * Number of photos
    '''
    
    image_data = __data.load_image_data()
    
    instagram_account_info = image_data[["user_id","user_followed_by","user_follows","user_posted_photos"]]\
                            .drop_duplicates()
    
    return instagram_account_info


def ratio_of_topics(confidence = 90, subset=True, months=12):
    '''
    NOTE: This is an expensive operation.
    
    Returns a matrix of the percentages that shows you what percentage that object appears in the users photos.
    
    Arg:
    confidence: confidence of topic being in image
    subset: returns subset of topics for the user that we selected in advance as indicators of lifestyle 
            and which were not sparse.
    '''

    image_data = __data.load_image_data(months)
    #__pd.read_pickle("../../data/Visual_well_being/image_data.pickle")
    
    user_img = image_data[["image_id","user_id"]].drop_duplicates()

    ob = __img_f.binary_object_matrix(confidence)
    photo_counts = instagram_account_stats()[["user_id","user_posted_photos"]]

    counts_per_user = ob.merge(user_img,how="right",on="image_id")\
                .groupby("user_id").sum()\
                .merge(photo_counts,on="user_id",how="inner")

    df = counts_per_user.iloc[:,1:-1]\
                        .divide(
                                counts_per_user.user_posted_photos,
                                axis=0
                                )

    df["user_id"] = counts_per_user.user_id
    
    if subset == True:
        topics_considered = ["user_id","Person","Plant","Food","Collage","Animal","Outdoors","Pet","Book","Dog","Canine","Sky","Alcohol","Crowd","Toy","Cat","Coast","Tree","Beach","Sport","Teddy Bear","Sunlight","Light","Drawing","Sea Life","TV","Dusk","Bikini","Sunrise","Sunset","Swimwear","Selfie","Beard","Woman","Cocktail","Pool","Performer","Coffee Cup","Tattoo","Downtown","Musical Instrument","Festival","City","Laptop","Pizza","Cloud","Beer Bottle","Money","Club","Airplane","Sketch","Sandwich","Cafeteria","Breakfast","Child"]
        df = df[topics_considered]

    return df

def avg_number_of_faces_from_photos_with_faces(months=12):
    '''Returns the average number of faces in photos with faces'''
    image_data = __data.load_image_data(months)
    
    #__pd.read_pickle("../../data/Visual_well_being/image_data.pickle")
    
    num_faces = __img_f.number_of_faces()

    out = image_data[["image_id","user_id"]]\
    .merge(num_faces,on="image_id",how="left")\
    .fillna(0)\
    .groupby("user_id")\
    .mean()\
    .rename(columns = {'number_of_face': 'avg_number_of_faces_over_images_with_faces'})\
    .reset_index()

    return out

def avg_number_of_faces_over_all_photos(months=12):
    '''Returns the average number of faces accross all photos of the user'''
    image_data = __data.load_image_data(months)
    
    #__pd.read_pickle("../../data/Visual_well_being/image_data.pickle")
    
    num_faces = __img_f.number_of_faces()

    #of the photos that have faces, what is the average
    out = image_data[["image_id","user_id"]]\
    .merge(num_faces,on="image_id",how="left")\
    .fillna(0)\
    .groupby("user_id")\
    .mean()\
    .rename(columns = {'number_of_face': 'avg_number_of_faces_over_all_images'})\
    .reset_index()

    return out

def avg_engagement(months=12):
    '''
    Returns the average number of likes and comments per user.
    '''
    image_data = __data.load_image_data(months)
    
    #__pd.read_pickle('../../data/Visual_well_being/image_data.pickle')
    
    updated_metrics = __img_f.final_like_and_comments(months)
    image_data = image_data.merge(updated_metrics, how='left', on='image_id')
    avg_engagement = image_data[['user_id', 'likes', 'comments']].groupby('user_id').mean().reset_index()
    avg_engagement = avg_engagement.rename(columns = {'likes': 'avg_likes', 'comments': 'avg_comments'})

    return avg_engagement

def filter_features(months=12):
    '''
    Returns percentage of happy/depressed filters, ratio of happy over depressed filters
    '''
    image_data = __data.load_image_data(months)
    
    #__pd.read_pickle('../../data/Visual_well_being/image_data.pickle')
    
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
    filter_features = filter_features.replace([np.inf, -np.inf], np.nan)
    # Drop not neeed columns
    filter_features = filter_features.drop(['happy_filter', 'depressed_filter', 'total_photos'], axis=1)

    return filter_features
    

def avg_posts_per_day(months=12):
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

    image_data = __data.load_image_data(months)

    #__pd.read_pickle(__data_dir + "image_data.pickle")
    #image_date.image_posted_time = __pd.to_datetime(image_date.image_posted_time)

    x = image_data[["image_posted_time","image_id","user_id"]].drop_duplicates()

    early_day_b = x.image_posted_time.isin(x.set_index("image_posted_time").between_time('8:00', '12:00').index)
    late_day_b = x.image_posted_time.isin(x.set_index("image_posted_time").between_time('12:00', '20:00').index)
    early_night_b = x.image_posted_time.isin(x.set_index("image_posted_time").between_time('20:00', '00:00').index)
    late_night_b = x.image_posted_time.isin(x.set_index("image_posted_time").between_time('00:00', '8:00').index)

    day_b = x.image_posted_time.isin(x.set_index("image_posted_time").between_time('8:00', '20:00').index)
    night_b = x.image_posted_time.isin(x.set_index("image_posted_time").between_time('20:00', '8:00').index)

    x["avg_posts_early_day"] = __np.where(early_day_b, 1, 0)
    x["avg_posts_late_day"] = __np.where(late_day_b, 1, 0)
    x["avg_posts_early_night"] = __np.where(early_night_b, 1, 0)
    x["avg_posts_late_night"] = __np.where(late_night_b, 1, 0)

    x["avg_posts_day"] = __np.where(day_b, 1, 0)
    x["avg_posts_night"] = __np.where(night_b, 1, 0)
    x["avg_posts_whole_date"] = 1

    xx = x.drop(columns="image_id",axis=0)\
    .groupby([x.user_id,x.image_posted_time.dt.date])\
    .sum().drop("user_id",axis=1).reset_index()

    out = xx.groupby("user_id").mean().reset_index()

    return out

def percentage_animals(months=12):
    '''
    Returns a column called 'percentage_animals' that informs us what is the percentage of animals that the user has
    '''
    topics = ratio_of_topics(months)
    animals = topics[['Animal', 'Pet', 'Dog', 'Cat', 'Canine']]
    animals = animals.assign(percentage_animals=animals.sum(axis=1))
    animals['user_id'] = topics.user_id
    animals = animals[['user_id', 'percentage_animals']]
    return animals

def average_num_faces_per_image_and_emotion(months=12):
    '''
    Returns the average of faces per emotion that the user has per image
    '''
    image_data = __data.load_image_data(months)
    
    #__pd.read_pickle(__data_dir + "image_data.pickle")
    num_faces_df = __img_f.number_of_faces_per_emotion()
    num_faces_df = __pd.merge(num_faces_df, image_data[['user_id', 'image_id']], on='image_id', how='right')
    num_faces_df.fillna(0, inplace=True)
    return num_faces_df.groupby('user_id').mean().reset_index()

def avg_ratio_gender(confidence=90,months=12):
    '''
    Returns the average of the gender ratio that the user has per image
    '''
    image_data = __data.load_image_data(months)
    
    #__pd.read_pickle(__data_dir + "image_data.pickle")
    ratio_gender = __img_f.ratio_gender(confidence)
    avg_ratio_gender = __pd.merge(ratio_gender, image_data[['user_id', 'image_id']], on='image_id', how='right')
    avg_ratio_gender.fillna(0, inplace=True)
    return avg_ratio_gender.groupby('user_id').mean().reset_index()

def proportion_image_cluster(months=12):
    
    anp_cg = __img_f.anp_cluster_groups()
    u = instagram_account_stats()[["user_id","user_posted_photos"]]

    image_data = __data.load_image_data(months)
    image_user = image_data[["image_id","user_id"]]
    
    user_clusters = image_user.merge(anp_cg,on="image_id",how="left").fillna(0).drop("image_id",axis=1)
    
    user_clusters = user_clusters.groupby("user_id").sum()\
                                        .reset_index()\
                                        .merge(u,on="user_id",how="inner")\
                                        .reset_index()
    
    cluster_proportions = __pd.concat([user_clusters.user_id,
                        user_clusters.iloc[:,2:-1]\
                                    .divide(user_clusters.user_posted_photos,
                                            axis=0)
                       ],axis=1)
    
    return cluster_proportions
