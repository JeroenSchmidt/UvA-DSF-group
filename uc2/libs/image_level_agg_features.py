import pandas as __pd


__data_dir = "../../data/Visual_well_being/"

def number_of_faces():
    '''Returns a count of unique faces in a picture'''
    face = __pd.read_pickle(__data_dir + 'face.pickle')
    number_of_face = face[["image_id","face_id"]].drop_duplicates().groupby(by=["image_id"]).count()
    number_of_face = number_of_face.rename(columns={"face_id":"number_of_face"}).reset_index()
    
    return number_of_face


def number_of_faces_per_emotion():
    '''Returns the number of faces per emotion within an image.'''
    face = __pd.read_pickle(__data_dir + 'face.pickle')
    
    num_faces = face[["image_id","face_emo","emo_confidence"]]\
            .groupby(by=["image_id","face_emo"]).count()\
            .reset_index()\
            .pivot(index="image_id",values="emo_confidence",columns="face_emo")\
            .reset_index()\
            .rename_axis('',axis=1)
           
    return num_faces


def final_like_and_comments():
    '''Returns the number of final likes and comments per image'''
    image_metrics = pd.read_pickle(__data_dir + "image_metrics.pickle")    
    
    final_image_stats = image_metrics[["image_id","comment_count","like_count"]].groupby(by="image_id").max().reset_index()
    final_image_stats.head()