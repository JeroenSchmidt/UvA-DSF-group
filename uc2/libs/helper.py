# helper functions
import user_level_agg_features as u_features
import pandas as pd
from functools import reduce

def aggregate():
    dfs = []
    for name, func in u_features.__dict__.items():
        if callable(func):
            df = func()
            df['user_id'] = df['user_id'].astype(int)
            dfs.append(df)
    return reduce(lambda left,right: pd.merge(left,right,on='user_id'), dfs)