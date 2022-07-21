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
        x = x.to(torch.float32)
        
        y = torch.tensor([int(self.df.iloc[idx][-1])])
        y = F.one_hot(y,num_classes=2)
        return x,y
    
    def __len__(self):
        return len(self.df)

import torch
from torch.nn import functional as F
import pandas as pd
import vitaldb
import numpy as np
import os

class PPGDT(torch.utils.data.Dataset):
    def __init__(self,vital_dir='../data/vital2/',op_path='../data/vital2/dd_all_simple.csv'):
        
        flist_ = os.listdir(vital_dir)
        flist_ = [ x for x in flist_ if x.endswith('.vital')]

        op = pd.read_csv(op_path)
        op = op[['key','pacu_nrs']]

        flist = []
        for key in op.key:

            for f in flist_:
                if f.startswith(key):
                    flist.append(f)

        self.fpaths = [os.path.join(vital_dir,path) for path in flist]
        self.labels = op.pacu_nrs.map(lambda x: nrs2class(x,2))
        
    def __getitem__(self,idx):

        cols = ['Intellivue/PLETH']
        vf = vitaldb.VitalFile(self.fpaths[idx],cols)

        x = vf.get_track_samples(cols[0],1/30)[:30*60*50] # 30*60*50
        x = self.interpolate(x)
        x = torch.tensor(x)
        
        y = torch.tensor([self.labels[idx]])
        y = F.one_hot(y,num_classes=2)
        return x,y
    
    def __len__(self):
        return len(self.fpaths)

    def interpolate(self,x :np.ndarray,mode='nearest')-> np.ndarray:
        
        if mode == 'nearest':
            x = pd.DataFrame(x,columns=['x'])
            x = x.fillna(method='ffill').fillna(method='bfill')
            x = x['x'].to_numpy()
        return x
    
    
    
    
    
    
# ===================================

import os
import pandas as pd
import numpy as np
import torch
import vitaldb
import neurokit2 as nk

from torch.nn import functional as F


class DynamicPPGDT(torch.utils.data.Dataset):
    def __init__(self,vital_dir='../data/vital2/', op_path='../data/vital2/dd_all_simple.csv', sample_rate=1/30, duration=30*60*50):
        super(DynamicPPGDT,self).__init__()
        dir_path = vital_dir
        self.sample_rate = sample_rate
        self.cols = ['Intellivue/PLETH','X002/PLETH']
        self.fs = int(1/sample_rate)
        self.duration = duration
        
        flist_ = os.listdir(dir_path)
        flist_ = [ x for x in flist_ if x.endswith('.vital')]

        # fpaths = [os.path.join(dir_path, x) for x in flist]

        
        op = pd.read_csv(os.path.join(dir_path,'dd_all_simple.csv'))
        op = op[['key','pacu_nrs']]

        flist = []
        for key in op.key:

            for f in flist_:
                if f.startswith(key):
                    flist.append(f)

        self.fpaths = [os.path.join(dir_path,path) for path in flist]

        self.labels = op.pacu_nrs.map(lambda x: nrs2class(x,2))
        
    def __getitem__(self,idx):
        
        fpath = self.fpaths[idx]
        if os.path.basename(fpath)=='JTOR3_211229_115807.vital':
            vf = vitaldb.VitalFile(fpath,self.cols[1])
            x = vf.get_track_samples(self.cols[1],interval=self.sample_rate)
        else:
            vf = vitaldb.VitalFile(fpath,self.cols[0])
            x = vf.get_track_samples(self.cols[0],interval=self.sample_rate)
            
        y = torch.tensor([self.labels[idx]])
        y = F.one_hot(y,num_classes=2)
        
        # cut nan head
        x = self.cut_nan_head(x)
    
        # right-zero-padding (if x.shape < duration)
        if x.shape[0] < self.duration:
            diff = self.duration - x.shape[0]
            x = np.pad(x,(0,diff),mode='constant')
        
        # interpolate missing values
        x = self.interpolate(x)
        
        # get peak indices
        sys_idx, dia_idx = self.get_peak_indices(x)
        
        # get dynamic value (ACV/ACA)
        x = self.get_dynamic(x,sys_idx,dia_idx)
        
        # zero-padding (4700 size)
        diff = 4700 - len(x)
        x = np.pad(x,(0,diff),mode='constant')
        
        x = torch.tensor(x)
        
        return x,y
    
    def __len__(self):
        return len(self.fpaths)

    def interpolate(self,x :np.ndarray,mode='nearest')-> np.ndarray:
        if mode == 'nearest':
            x = pd.DataFrame(x,columns=['x'])
            x = x.fillna(method='ffill').fillna(method='bfill')
            x = x['x'].to_numpy()
        return x
    
    def cut_nan_head(self,x):
        first_idx = 0
        for i, value in enumerate(x):
            if pd.isna(value):
                continue
            else:
                first_idx = i
                break
        x = x[first_idx:first_idx+self.duration] #fs*60*50
        return x
    
    def get_peak_indices(self,x):
        signals, info = nk.ppg_process(x, sampling_rate=self.fs)
        sys_idx = info['PPG_Peaks']
        
        bottom_dia_idx = []
        for i,idx in enumerate(sys_idx):
            if i == 0: # not sure if a diastolic peak exists
                continue
            pre_idx = sys_idx[i-1]
            min_idx = pre_idx + np.argmin(x[pre_idx:idx])
            bottom_dia_idx.append(min_idx)
        return sys_idx, bottom_dia_idx
    
    def get_dynamic(self,x,sys_idx,dia_idx):
        # acv_sys & aca_dia
        acv_sys = []
        aca_dia = []

        for i,idx in enumerate(sys_idx):
            if i == 0: # not sure if a diastolic peak exists
                continue
            pre_idx = sys_idx[i-1]

            pre_sys = x[pre_idx]
            sys_val = x[idx]
            dia_val = x[dia_idx[i-1]]

            acv_sys.append(sys_val-pre_sys)
            aca_dia.append(sys_val-dia_val)


        acv_sys = np.array(acv_sys)
        aca_dia = np.array(aca_dia)

        dynamic = acv_sys/aca_dia
        return dynamic
    
