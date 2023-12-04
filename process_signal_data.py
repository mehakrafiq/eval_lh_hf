import numpy as np
import pandas as pd
import itertools
import ast as ast
from scipy import stats



class dataset():
    def __init__(self, name):
        self.name_of_file =  name
        return

    def load_csv(self):
        self.dataframe = pd.read_csv(self.name_of_file)
        return 

    def extract_activities(self):
        self.activities = self.dataframe.section.unique()
        return

    def extract_users(self):
        self.users = self.dataframe.user.unique()
        return

    def len_data(dataframe: pd.core.frame.DataFrame) -> int:
        self.len_dataframe = len(dataframe)
        return 

    def convert_rr_array(data: pd.core.frame.DataFrame, val) -> np.ndarray:
        arr = ast.literal_eval(data.rr[val])
        return np.array(arr,dtype=float)
    


def get_age_summary(dataframe: pd.core.frame.DataFrame) ->  np.ndarray:
    data_reduced = dataframe.iloc[data['user'].drop_duplicates().index]
    age_unique =  data_reduced.age.to_numpy()
    return age_unique

def convert_rr_array(data: pd.core.frame.DataFrame, val) -> np.ndarray:
    arr = ast.literal_eval(data.rr[val])
    return np.array(arr,dtype=float)

def create_dict_users(data: pd.core.frame.DataFrame) -> dict:
    keys = data.user.unique()
    user_dict = {key: None for key in keys}
    return user_dict

def convert_rr_total(data: pd.core.frame.DataFrame) -> list:
    vals = []
    for i in data.rr:
        arr = ast.literal_eval(i)
        arr = np.array(arr,dtype=float)
        vals.append(arr)
    data['rr_array'] =  vals
    return 


def combine_signals(data: pd.core.frame.DataFrame) -> np.ndarray:
    arr = np.array([])
    for i in range(14):
        new = convert_rr_array(data,i)
        arr = np.concatenate((arr,new))
    return arr

def sliding_windows(data: pd.core.frame.DataFrame, seq_length: int) -> np.ndarray:
    x = []
    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        x.append(_x)

    return np.array(x)

def extract_valence_or_arousal(dataframe: pd.core.frame.DataFrame , value: str,  true_val: int) -> np.ndarray:
    val_or_arousal =  dataframe[dataframe.section==value].valence.to_numpy()
    list_anomaly = np.array(['Anom']* len(val_or_arousal))
    sign_of_vals = np.sign(val_or_arousal)
    idx_true = np.where(sign_of_vals==true_val)[0]
    idx_0 = np.where(sign_of_vals==0)[0]
    idx_total = np.concatenate((idx_true, idx_0))
    list_anomaly[idx_total] = 'NonAnom'
    return list_anomaly



def extract_true_valence(dataframe):
    sections = dataframe.section.unique()
    true_vals = []
    for i, j in enumerate(sections):
        signs = np.sign(dataframe[dataframe.section==j].valence.to_numpy())
        sign_pos = len(np.where(signs==1)[0])
        sign_neg =  len(np.where(signs==-1)[0])
        sign_0 =  len(np.where(signs==0)[0])
        if(sign_pos>sign_neg):
            true_vals.append(int(1))
        else:
            true_vals.append(int(-1))

    return true_vals


def extract_true_arousal(dataframe):
    sections = dataframe.section.unique()
    true_vals = []
    for i, j in enumerate(sections):
        signs = np.sign(dataframe[dataframe.section==j].arousal.to_numpy())
        sign_pos = len(np.where(signs==1)[0])
        sign_neg =  len(np.where(signs==-1)[0])
        sign_0 =  len(np.where(signs==0)[0])
        if(sign_pos>sign_neg):
            true_vals.append(int(1))
        else:
            true_vals.append(int(-1))

    return true_vals





