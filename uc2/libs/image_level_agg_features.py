import pandas as __pd
import numpy as __np


__data_dir = "../../data/Visual_well_being/"

def number_of_faces():
    '''Returns a count of unique faces in a picture'''
    face = __pd.read_pickle(__data_dir + 'face.pickle')
    number_of_face = face[["image_id","face_id"]].drop_duplicates().groupby(by=["image_id"]).count()
    number_of_face = number_of_face.rename(columns={"face_id":"number_of_face"}).reset_index()
    
    return number_of_face


def number_of_faces_per_emotion(confidence = 75):
    '''Returns the number of faces per emotion within an image.'''
    face = __pd.read_pickle(__data_dir + 'face.pickle')
    
    num_faces = face[["image_id","face_emo","emo_confidence"]][face.emo_confidence >= confidence]\
            .groupby(by=["image_id","face_emo"]).count()\
            .reset_index()\
            .pivot_table(index="image_id",values="emo_confidence",columns="face_emo",fill_value=0)\
            .reset_index()\
            .rename_axis('',axis=1)
           
    return num_faces


def final_like_and_comments():
    '''Returns the number of final likes and comments per image'''
    image_metrics = __pd.read_pickle(__data_dir + "image_metrics.pickle")

    # Sort by id & date, drop_duplicates per image_id by keeping the last.
    final_likes = image_metrics.sort_values(by=['image_id', 'like_count_time_created'])[['image_id', 'like_count_time_created', 'like_count']]
    final_likes = final_likes.drop_duplicates(subset=['image_id'], keep='last')[['image_id', 'like_count']]
    final_likes = final_likes.rename(columns = {'like_count': 'final_like_count'})

    final_comments = image_metrics.sort_values(by=['image_id', 'comment_count_time_created'])[['image_id', 'comment_count_time_created', 'comment_count']]
    final_comments = final_comments.drop_duplicates(subset=['image_id'], keep='last')[['image_id','comment_count']]
    final_comments = final_comments.rename(columns = {'comment_count': 'final_comment_count'})

    image_metrics_updated = image_metrics.merge(final_likes, 'inner', 'image_id')
    image_metrics_updated = image_metrics_updated.drop_duplicates('image_id')

    image_metrics_updated = image_metrics_updated.merge(final_comments, 'inner', 'image_id')
    image_metrics_updated = image_metrics_updated.drop(['comment_count', 'comment_count_time_created', 'like_count', 'like_count_time_created'],axis = 1)
    image_metrics_updated = image_metrics_updated.rename(columns = {'final_like_count': 'likes', 'final_comment_count': 'comments'})

    # final_image_stats = image_metrics[["image_id","comment_count","like_count"]].groupby(by="image_id").max().reset_index()

    return image_metrics_updated
    
def average_emotions_per_image():
    '''
    Returns the mean of each emotion within an image. 
    Note: if an image has 6 faces and only 2 faces are sad, the mean will only be of the emotions assocaited to those 2 faces.
    '''
    face = __pd.read_pickle(__data_dir + 'face.pickle')

    face = face.pivot_table(aggfunc="mean",index=["image_id"],columns="face_emo",values="emo_confidence")\
                    .reset_index()\
                    .rename_axis('',axis=1)
    
    return face


def number_of_gender_faces(confidence = 0):
    '''
    Returns the number of male and female faces in an image. 
    
    You can specify the confidence from 0 to 100 of the gender identification. 
    '''
    
    conf_l = face.face_gender_confidence > confidence
    face_c = face[conf_l]

    number_of_gender = face_c[["image_id","face_id","face_gender"]]\
                            .drop_duplicates()\
                            .groupby(by=["image_id","face_gender"])\
                            .count()

    number_of_gender = number_of_gender.rename(columns={"face_id":"number_of_faces"}).reset_index()
    number_of_gender = number_of_gender.pivot(columns="face_gender",values="number_of_faces",index="image_id")\
                                        .reset_index()\
                                        .rename_axis('',axis=1)\
                                        .rename(columns={"Female":"Num_Female_Faces",
                                                         "Male":"Num_Male_Faces"})\
                                        .fillna(0)
    
    return number_of_gender

def binary_object_matrix(confidence = 0):
    '''
    Returns a binary matrix (with +-2500) cols - each coresponding to a detected object
    
    Arg:
        confidence: set the min object detection confidence from 0 to 100. anything bellow it will be removed. 
        The raw data has a confidence from 70% and above
    '''
    
    object_labels = __pd.read_pickle(__data_dir + "object_labels.pickle")
    
    object_labels_l = object_labels.data_amz_label_confidence > confidence
    object_labels_c = object_labels[object_labels_l]
    
    obj_counts = object_labels_c.groupby(by=["image_id","data_amz_label"])\
                                .count()
    
    obj_counts_p = obj_counts.reset_index()\
                                .pivot(index="image_id",columns="data_amz_label",values="data_amz_label_confidence")\
                                .reset_index()\
                                .rename_axis('',axis=1)\
                                .fillna(0)    
    return obj_counts_p
    
def anp_avg_emotional_scores():
    '''
    Returns the average anp emotion score attached to an emoation label.
    '''
    anp = __pd.read_pickle(_data_dir + "anp.pickle")
    
    avg_emo_scores = anp[["image_id","emotion_label","emotion_score"]]\
                        .groupby(by=["image_id","emotion_label"])\
                        .mean()\
                        .reset_index()\
                        .pivot(index="image_id",columns="emotion_label",values="emotion_score")\
                        .fillna(0)\
                        .reset_index()\
                        .rename_axis('',axis=1)

    return avg_emo_scores
    
def ratio_gender(confidence=90):
    face = __pd.read_pickle(__data_dir + "face.pickle")
    face_filter = face[face.face_gender_confidence >= confidence][['image_id', 'face_id', 'face_gender']]
    face_filter = face_filter.drop_duplicates()[['image_id', 'face_gender']]
    p = __pd.pivot_table(face_filter, index=['image_id'], columns=['face_gender'], aggfunc='size', fill_value=0).reset_index()
    p = p.assign(ratio_male=p.Male / (p.Male + p.Female))
    p = p.assign(ratio_female=p.Female / (p.Male + p.Female))
    return p[['image_id', 'ratio_male', 'ratio_female']]