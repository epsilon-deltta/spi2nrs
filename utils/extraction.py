# data
import os
path21 = 'e:/2021_1_1_10_14/충남대병원/마취통증의학과/OTHER'

flist = [] 
for path, dirs, files in os.walk(path21):
    for f in files:
        flist.append(os.path.join(path,f))

import os
path21_0 = 'e:/2021_1015_1231/충남대병원/마취통증의학과/OTHER'

flist0 = [] 
for path, dirs, files in os.walk(path21_0):
    for f in files:
        flist0.append(os.path.join(path,f))

path22 = 'e:\\2022\\충남대병원\\마취통증의학과\\OTHER'
flist1 = [] 
for path, dirs, files in os.walk(path22):
    for f in files:
        flist1.append(os.path.join(path,f))

allf = []
allf.extend(flist)
allf.extend(flist0)
allf.extend(flist1)

# remove non-vital files
allf = [x for x in allf if os.path.splitext(allf[0])[-1] == '.vital']

allf_n = [os.path.splitext(os.path.basename(x))[0] for x in allf]

allf_s = [x.split('_') for x in allf_n]

# len(flist),len(flist0),len(flist1)
# (22634, 9046, 9291)

###################

# label
label_path0 = './회복간호_21상.xlsx'
label_path1 = './회복간호_21하.xlsx'
label_path2= './회복간호_22상.xlsx'

# remove nrs-non-existing rows
# select columns
def get_data(path,filter_='simple'):
    df = pd.read_excel(path,header=1)
    df = df.iloc[df[['최대 NRS']].dropna().index]
    if filter_ == 'simple':
        df = df[['등록번호','회복실퇴실일시','최대 NRS']]
    elif filter_ == 'all':
        pass
    return df

df0 = get_data(label_path0)
df1 = get_data(label_path1)
df2 = get_data(label_path2)

df = pd.concat([df0,df1,df2])


df = df.drop_duplicates()

df = df.reset_index(drop=True)

def get_date(date:str,type_='ymd'):
    
    date = date.split()

    y,m,d = date[0].split('-')
    h,min_   = date[1].split(':')
    
    out = ''
    if type_ == 'ymd':
        ymd = y[-2:]+m+d
        out = ymd
    elif type_ == 'hm':
        hm = h+min_
        out = hm
    return out

df['time'] = df['회복실퇴실일시'].map(lambda x: get_date(x,'hm') )

df['date'] = df['회복실퇴실일시'].map(lambda x: get_date(x,'ymd') )

df = df[['등록번호','date','time','최대 NRS']]

# len(df0),len(df1),len(df2)
# (6338, 6855, 5508)
# len(df) 18701 (w/ duplicates)

###########################################

idx = []
for sample in df.values: # sample: ['등록번호','date','time','최대 NRS']
    sample[0] = str(sample[0]) 
    for i, data in enumerate(allf_s): # data: ['00000824', 'ORC14', '210611', '133500', '1']
        if data[0].endswith(sample[0]): # 등록번호
            if data[2].startswith(sample[1]): # 210611
                idx.append([i,sample,allf_n[i],allf[i]])

dfi = pd.DataFrame(idx,columns = ['idx','label','vital','path'])
# len(dfi) 33198