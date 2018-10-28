import pandas as __pd
__data_dir = "../../data/Visual_well_being/"

def load_anp():
    df = __pd.read_pickle(__data_dir + 'anp.pickle')
    
    df = __clean_anp(df)
    
    return df


def __clean_anp(df):
    df = df.drop_duplicates()
    
    # removes the multiple instances of anp_label
    anp_counts = df.groupby(["image_id","anp_label"]).count()
    
    anp_morethen1 = anp_counts[anp_counts.anp_sentiment > 1].reset_index()\
                                            .drop(["anp_sentiment","emotion_score","emotion_label"],axis=1)
    
    l = df.image_id.isin(anp_morethen1.image_id) & df.anp_label.isin(anp_morethen1.anp_label)

    df_clean = df[~l]
    
    return df_clean
    
    

def load_celebrity():
    df = __pd.read_pickle(__data_dir + 'celebrity.pickle')
    
       
    return df


def load_faces():
    df = __pd.read_pickle(__data_dir + 'face.pickle')
    
    return df


def load_image_data(months=12):
    df = __pd.read_pickle(__data_dir + 'image_data.pickle')
    df.user_id = __pd.to_numeric(df.user_id)
    df.image_posted_time = __pd.to_datetime(df.image_posted_time)    
    
    if months != None:
        df = __filter_months(df,months)
    
    return df



def __filter_months(image_data,months=12):
    '''
    Private function that filters out the images that don't fall within x months of the date of survay.
    '''
    
    img = image_data
    surv = load_survey()[["start_q","insta_user_id"]]
    
    
    df = img.merge(surv,left_on="user_id",right_on="insta_user_id",how="inner").drop("insta_user_id",axis=1)
    
    lower_date_bound = df.start_q - __pd.to_timedelta(months, unit='M')
    l = (df.image_posted_time < df.start_q) & (lower_date_bound < df.image_posted_time)
    
    df = df[l]
    
    return df

def load_image_metrics():
    df = __pd.read_pickle(__data_dir + 'image_metrics.pickle')
    
    
    
    return df



def load_objects():
    df = __pd.read_pickle(__data_dir + 'object_labels.pickle')
    
    
    
    return df



def load_survey():
    df = __pd.read_pickle(__data_dir + 'survey.pickle')
    df.start_q = __pd.to_datetime(df.start_q)

    
    
    return df