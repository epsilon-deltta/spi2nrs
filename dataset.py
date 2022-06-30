import torch
from torch.nn import functional as F
import pandas as pd


def nrs2class(x,pain_type=3):
    pain = ''
    
    if pain_type == 3:
        if x in [0,1,2,3]:
            pain = 0
        elif x in [4,5]:
            pain = 1
        else:
            pain = 2
    elif pain_type == 2:
        if x in [0,1,2,3]:
            pain = 0
        else:
            pain = 1
    return pain

class Spi10_2nrs(torch.utils.data.Dataset):
    def __init__(self,file_path='../data/spi10_2nrs.csv'):
        df = pd.read_csv(file_path)
        del df['id']
        df['10min'] = df['10min'].map(float)
        df['20min'] = df['20min'].map(float)
        df['30min'] = df['30min'].map(float)
        df['40min'] = df['40min'].map(float)
        df['50min'] = df['50min'].map(float)
        df['pacu_nrs'] = df.pacu_nrs.map(lambda x: nrs2class(x,2))
        self.df     = df

    def __getitem__(self,idx):
        x = torch.tensor(self.df.iloc[idx][:-1].to_numpy())
        y = torch.tensor([int(self.df.iloc[idx][-1])])
        y = F.one_hot(y,num_classes=2)
        return x,y
    
    def __len__(self):
        return len(self.df)
    
    
  