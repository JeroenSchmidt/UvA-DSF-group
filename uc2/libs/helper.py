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
    return reduce(lambda left,right: pd.merge(left,right,on='user_id', how='outer').fillna(0), dfs)

def clean_data_after_merging_features(df):
    drop_columns = ['A_2', 'N_1', 'P_1', 'E_1', 'A_1', 'H_1', 'M_1', 'R_1', 'M_2', 'E_2',
                    'H_2', 'P_2', 'N_2', 'A_3', 'N_3', 'E_3', 'H_3', 'R_2', 'M_3', 'R_3',
                    'P_3', 'N_EMO', 'P_EMO', 'HAP', 'LON',
                    'index', 'id', 'user_id', 'insta_user_id', 'network_id', 'start_q',
                    'end_q', 'private_account', 'completed', 'HAP']
    df = df.drop(drop_columns, axis=1)

    for col in df.columns:
        if (df[col].dtype.__str__() == 'category'):
            df[col] = df[col].cat.codes

    df = df.dropna(axis=0) # Dropping rows with nan
    return df
