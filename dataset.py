import torch
from torch.nn import functional as F
import pandas as pd

class Spi10_2nrs(torch.utils.data.Dataset):
    def __init__(self,file_path='../data/spi10_2nrs.csv'):
        df = pd.read_csv(file_path)
        del df['id']
        df['10min'] = df['10min'].map(float)
        df['20min'] = df['20min'].map(float)
        df['30min'] = df['30min'].map(float)
        df['40min'] = df['40min'].map(float)
        df['50min'] = df['50min'].map(float)
        self.df     = df

    def __getitem__(self,idx):
        x = torch.tensor(df.iloc[idx][:-1].to_numpy())
        y = torch.tensor([int(df.iloc[idx][-1])])
        y = F.one_hot(y,num_classes=2)
        return x,y
    
    def __len__(self):
        return len(self.df)