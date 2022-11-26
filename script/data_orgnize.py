#%%
import pandas as pd
import csv
import numpy as np
#%%
#C1_P1
#.\Desktop\Condition1\P1_C1desktop.csv
#.\Empatica\Condition1\P1_C1_EM\ACC.csv
#.\Empatica\Condition1\P1_C1_EM\BVP.csv
#.\Empatica\Condition1\P1_C1_EM\EDA.csv
#.\Empatica\Condition1\P1_C1_EM\HR.csv
#.\Empatica\Condition1\P1_C1_EM\IBI.csv
#.\Empatica\Condition1\P1_C1_EM\TEMP.csv
#.\Hololens\Condition1\P1_C1__High__Neutral_Audios_.wav
#.\Hololens\Condition1\P1_C1__High__Neutral_Faces_.csv
#.\Hololens\Condition1\P1_C1__High__Neutral_IMUs_.csv
#.\Hololens\Condition1\P1_C1__High__Neutral_Posture_.csv
#.\Laptop\Condition1\P1_C1laptop.csv
#.\Mobile\Condition1\P1_C1phone.csv
# %%
fp_desktop = ".\\Desktop\\Condition1\\P1_C1desktop.csv"
fp_e4acc = ".\\Empatica\\Condition1\\P1_C1_EM\\ACC.csv"
fp_e4bvp = ".\\Empatica\\Condition1\\P1_C1_EM\\BVP.csv"
fp_e4eda = ".\\Empatica\\Condition1\\P1_C1_EM\\EDA.csv"
fp_e4hr = ".\\Empatica\\Condition1\\P1_C1_EM\\HR.csv"
fp_e4temp = ".\\Empatica\\Condition1\\P1_C1_EM\\TEMP.csv"
fp_holoaudio = ".\\Hololens\\Condition1\\P1_C1__High__Neutral_Audios_.wav"
fp_holoface = ".\\Hololens\\Condition1\\P1_C1__High__Neutral_Faces_.csv"
fp_holoimu = ".\\Hololens\\Condition1\\P1_C1__High__Neutral_IMUs_.csv"
fp_holoposture = ".\\Hololens\\Condition1\\P1_C1__High__Neutral_Posture_.csv"
fp_laptop = ".\\Laptop\\Condition1\\P1_C1laptop.csv"
fp_phone = ".\\Mobile\\Condition1\\P1_C1phone.csv"
# %%
def empatica_timeseries(path):
    f = open(path, 'r')
    wr = csv.reader(f)
    t = 0.0
    tmp = []
    for idx, line in enumerate(wr):
        if idx ==  0:
            pass
        elif idx ==1 :
            freq = float(line[0])
        else:
            t += 1/freq
            line = np.array(line, dtype=np.float32)
            line = np.insert(line, 0, t, axis=0)
            print(line)
            tmp.append(line)
            if idx > 10:
                break
    return tmp
# %%
e4acc_time = empatica_timeseries(fp_e4acc)
e4bvp_time = empatica_timeseries(fp_e4bvp)
e4eda_time = empatica_timeseries(fp_e4eda)
e4hr_time = empatica_timeseries(fp_e4hr)
e4temp_time = empatica_timeseries(fp_e4temp)
# %%
e4acc_col = ['time', 'acc.x', 'acc.y', 'acc.z']
e4bvp_col = ['time', 'bvp']
e4eda_col = ['time', 'eda']
e4hr_col = ['time', 'hr']
e4temp_col = ['time', 'temp']

e4acc_df = pd.DataFrame(columns=e4acc_col, data=e4acc_time)
e4bvp_df = pd.DataFrame(columns=e4bvp_col, data=e4bvp_time)
e4eda_df = pd.DataFrame(columns=e4eda_col, data=e4eda_time)
e4hr_df = pd.DataFrame(columns=e4hr_col, data=e4hr_time)
e4temp_df = pd.DataFrame(columns=e4temp_col, data=e4temp_time)
# %%
e4acc_df.set_index("time", inplace=True)
e4bvp_df.set_index("time", inplace=True)
e4eda_df.set_index("time", inplace=True)
e4hr_df.set_index("time", inplace=True)
e4temp_df.set_index("time", inplace=True)
#%%
print(e4acc_df.head())
print(e4bvp_df.head())
# %%
temp = pd.merge(e4acc_df, e4bvp_df, how='outer', left_index=True, right_index=True)
print(temp.head())
# %%
temp['pid'] = "1"
#%%
print(temp)
# %%
temp = pd.merge(temp, e4eda_df, how='outer', left_index=True, right_index=True)
print(temp.head())
# %%
temp = pd.merge(temp, e4hr_df, how='outer', left_index=True, right_index=True)
print(temp.head())
# %%
temp = pd.merge(temp, e4temp_df, how='outer', left_index=True, right_index=True)
print(temp.head())
# %%
emptatica_df = temp.interpolate(method='values')

# %%
def csv_reader(path):
    f = open(path, 'r')
    reader = csv.reader(f)
    tmp = []
    for idx, line in enumerate(reader):
        if idx == 0:
            col = line
            col[0] = "time"
        else:
            list = [None if i == '' else i for i in line]
            temp = np.array(list, dtype=np.float32)
            tmp.append(temp)
    return col, tmp
# %%
# fp_holoface = ".\\Hololens\\Condition1\\P1_C1__High__Neutral_Faces_.csv"
# fp_holoimu = ".\\Hololens\\Condition1\\P1_C1__High__Neutral_IMUs_.csv"
# fp_holoposture = ".\\Hololens\\Condition1\\P1_C1__High__Neutral_Posture_.csv"
holoface_col, holoface_data = csv_reader(fp_holoface)
holoimu_col, holoimu_data = csv_reader(fp_holoimu)
holoposture_col, holoposture_data = csv_reader(fp_holoposture)
#%%
holoface_df = pd.DataFrame(columns=holoface_col, data=holoface_data)
print(holoface_df.head())
holoimu_df = pd.DataFrame(columns=holoimu_col, data=holoimu_data)
print(holoimu_df.head())
holoposture_df = pd.DataFrame(columns=holoposture_col, data=holoposture_data)
holoposture_df.dropna(axis = 1, how='all', inplace=True)
print(holoposture_df.head())
#%%
holoface_df.set_index('time', inplace=True)
holoimu_df.set_index('time', inplace=True)
holoposture_df.set_index('time', inplace=True)
# %%
temp = pd.merge(holoimu_df, holoface_df, how='outer', left_index=True, right_index=True)
# %%
print(temp.head())
# %%
temp = pd.merge(temp, holoposture_df, how='outer', left_index=True, right_index=True)
# %%
temp.dropna(axis=0, how='all', inplace=True)
print()
# %%
holo_df = temp.interpolate(method='values')
# %%
print(holo_df.head())
# %%
# fp_desktop = ".\\Desktop\\Condition1\\P1_C1desktop.csv"
# fp_laptop = ".\\Laptop\\Condition1\\P1_C1laptop.csv"
# fp_phone = ".\\Mobile\\Condition1\\P1_C1phone.csv"
def csv_reader_md(path):
    f = open(path, 'r')
    reader = csv.reader(f)
    tmp = []
    for idx, line in enumerate(reader):
        if idx == 0:
            pass
        else:
            temp = np.array(line, dtype=np.float32)
            tmp.append(temp)
    return tmp
# %%
desktop_data = csv_reader_md(fp_desktop)
laptop_data = csv_reader_md(fp_laptop)
phone_data = csv_reader_md(fp_phone)
print(desktop_data)
# %%
desktop_col = ['time', 'desktop']
laptop_col = ['time', 'laptop']
phone_col = ['time', 'phone']

# %%
desktop_df = pd.DataFrame(columns=desktop_col, data=desktop_data)
print(desktop_df.head())
laptop_df = pd.DataFrame(columns=laptop_col, data=laptop_data)
print(laptop_df.head())
phone_df = pd.DataFrame(columns=phone_col, data=phone_data)
print(phone_df.head())
# %%
desktop_df.set_index('time', inplace=True)
laptop_df.set_index('time', inplace=True)
phone_df.set_index('time', inplace=True)
# %%
temp = pd.merge(desktop_df, laptop_df, how='outer', left_index=True, right_index=True)
print(temp.head())
# %%
temp = pd.merge(temp, phone_df, how='outer', left_index=True, right_index=True)
print(temp.head())
# %%
print(temp.tail())
# %%
temp.dropna(axis=0, how='all', inplace=True)
temp.dropna(axis=1, how='all', inplace=True)

# %%
print(temp)
# %%
import os
import glob
# %%
base_path = 'C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\Hololens'
for i in range(1, 7):
    file_path = os.path.join(base_path, "Condition%d"%i)
    for pid in range(1, 8):
        flist = glob.glob(os.path.join(file_path, "P%d_C%d_*"%(pid, i)))
        for file in flist:
            basename = os.path.basename(file)
            _, ext = os.path.splitext(basename)
            print(ext)
            last_idx = basename.rfind("_")
            second_idx = basename[:last_idx].rfind("_")
            data_type = basename[second_idx+1:last_idx]
            print("P%d_C%d_%s%s"%(pid, i, data_type, ext))
            os.rename(file, os.path.join(file_path, "P%d_C%d_%s%s"%(pid, i, data_type, ext)))
# %%
from scipy.io import wavfile
# %%
wav_path = "C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\Hololens\\Condition1\\P1_C1_Audios.wav"
fs, datas = wavfile.read(wav_path)
t = 0.0
audio_data = []
for idx, data in enumerate(datas):
    t += 1/fs
    audio_data.append(np.array([t, data], dtype=np.float32))
for idx, data in enumerate(datas):
    t += 1/fs
    audio_data.append(np.array([t, data], dtype=np.float32))
col = ['time', 'audio']
audio_df = pd.DataFrame(columns=col, data=audio_data)
# %%
audio_df.head()
# %%
audio_df.tail()
#%%
print(fs)
print(len(datas))
# %%
f = open('C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\dataframe_test.csv', 'r')
reader = csv.reader(f)
for idx, line in enumerate(reader):
    print(line)
#%%
import pandas as pd
import csv
import numpy as np
import os
from scipy.io import wavfile
import librosa

def empatica_reader(path):
    f = open(path, 'r')
    wr = csv.reader(f)
    t = 0.0
    tmp = []
    for idx, line in enumerate(wr):
        if idx ==  0:
            pass
        elif idx ==1 :
            freq = float(line[0])
        else:
            t += 1/freq
            line = np.array(line, dtype=np.float32)
            line = np.insert(line, 0, t, axis=0)
            tmp.append(line)
    return tmp

def holo_reader(path):
    f = open(path, 'r')
    reader = csv.reader(f)
    tmp = []
    for idx, line in enumerate(reader):
        if idx == 0:
            col = line
            col[0] = "time"
        else:
            list = [None if i == '' else i for i in line]
            temp = np.array(list, dtype=np.float32)
            tmp.append(temp)
    return col, tmp

def device_reader(path):
    f = open(path, 'r')
    reader = csv.reader(f)
    tmp = []
    for idx, line in enumerate(reader):
        if idx == 0:
            pass
        else:
            temp = np.array(line, dtype=np.float32)
            tmp.append(temp)
    return tmp

def path_load(pid, condition):
    fp_dict = {}
    fp_dict['fp_e4acc'] = ".\\Empatica\\Condition%d\\P%d_C%d_EM\\ACC.csv"%(condition, pid, condition)
    fp_dict['fp_e4bvp'] = ".\\Empatica\\Condition%d\\P%d_C%d_EM\\BVP.csv"%(condition, pid, condition)
    fp_dict['fp_e4eda'] = ".\\Empatica\\Condition%d\\P%d_C%d_EM\\EDA.csv"%(condition, pid, condition)
    fp_dict['fp_e4hr'] = ".\\Empatica\\Condition%d\\P%d_C%d_EM\\HR.csv"%(condition, pid, condition)
    fp_dict['fp_e4temp'] = ".\\Empatica\\Condition%d\\P%d_C%d_EM\\TEMP.csv"%(condition, pid, condition)
    fp_dict['fp_holoaudio'] = ".\\Hololens\\Condition%d\\P%d_C%d_Audios.wav"%(condition, pid, condition)
    fp_dict['fp_holoface'] = ".\\Hololens\\Condition%d\\P%d_C%d_Faces.csv"%(condition, pid, condition)
    fp_dict['fp_holoimu'] = ".\\Hololens\\Condition%d\\P%d_C%d_IMUs.csv"%(condition, pid, condition)
    fp_dict['fp_holoposture'] = ".\\Hololens\\Condition%d\\P%d_C%d_Posture.csv"%(condition, pid, condition)
    fp_dict['fp_desktop'] = ".\\Desktop\\Condition%d\\P%d_C%ddesktop.csv"%(condition, pid, condition)
    fp_dict['fp_laptop'] = ".\\Laptop\\Condition%d\\P%d_C%dlaptop.csv"%(condition, pid, condition)
    fp_dict['fp_phone'] = ".\\Mobile\\Condition%d\\P%d_C%dphone.csv"%(condition, pid, condition)
    return fp_dict
# %%
path_dict = path_load(1, 1)
# %%
#Empatica E4 dataframe
e4acc_data = empatica_reader(path_dict['fp_e4acc'])
e4bvp_data = empatica_reader(path_dict['fp_e4bvp'])
e4eda_data = empatica_reader(path_dict['fp_e4eda'])
e4hr_data = empatica_reader(path_dict['fp_e4hr'])
e4temp_data = empatica_reader(path_dict['fp_e4temp'])
e4acc_col = ['time', 'acc.x', 'acc.y', 'acc.z']
e4bvp_col = ['time', 'bvp']
e4eda_col = ['time', 'eda']
e4hr_col = ['time', 'hr']
e4temp_col = ['time', 'temp']
e4acc_df = pd.DataFrame(columns=e4acc_col, data=e4acc_data)
e4bvp_df = pd.DataFrame(columns=e4bvp_col, data=e4bvp_data)
e4eda_df = pd.DataFrame(columns=e4eda_col, data=e4eda_data)
e4hr_df = pd.DataFrame(columns=e4hr_col, data=e4hr_data)
e4temp_df = pd.DataFrame(columns=e4temp_col, data=e4temp_data)
e4acc_df.set_index("time", inplace=True)
e4bvp_df.set_index("time", inplace=True)
e4eda_df.set_index("time", inplace=True)
e4hr_df.set_index("time", inplace=True)
e4temp_df.set_index("time", inplace=True)
temp = pd.merge(e4acc_df, e4bvp_df, how='outer', left_index=True, right_index=True)
temp = pd.merge(temp, e4eda_df, how='outer', left_index=True, right_index=True)
temp = pd.merge(temp, e4hr_df, how='outer', left_index=True, right_index=True)
temp = pd.merge(temp, e4temp_df, how='outer', left_index=True, right_index=True)
temp.dropna(axis=0, how="all", inplace=True)
temp.dropna(axis=1, how="all", inplace=True)
empatica_df = temp.copy()
#%%
print(empatica_df.tail())
#%%
empatica_df.interpolate(method='linear', inplace=True, axis=0)
#%%
print(empatica_df.tail())
# %%
import plotly.graph_objects as go

fig = go.Figure()
for var in empatica_df.columns:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=empatica_df.index,
            y=empatica_df.loc[:, var],
            name=var
        )
    )
    fig.update_layout(
        title = dict(
            text = var)
        )
    fig.show()
# %%
empatica_df
# %%
#holoens dataframe
holoface_col, holoface_data = holo_reader(path_dict['fp_holoface'])
holoimu_col, holoimu_data = holo_reader(path_dict['fp_holoimu'])
holoposture_col, holoposture_data = holo_reader(path_dict['fp_holoposture'])
holoface_df = pd.DataFrame(columns=holoface_col, data=holoface_data)
holoimu_df = pd.DataFrame(columns=holoimu_col, data=holoimu_data)
holoposture_df = pd.DataFrame(columns=holoposture_col, data=holoposture_data)
holoposture_df.dropna(axis = 1, how='all', inplace=True)
holoface_df.set_index('time', inplace=True)
holoimu_df.set_index('time', inplace=True)
holoposture_df.set_index('time', inplace=True)
temp = pd.merge(holoimu_df, holoface_df, how='outer', left_index=True, right_index=True)
temp = pd.merge(temp, holoposture_df, how='outer', left_index=True, right_index=True)
temp.dropna(axis=0, how='all', inplace=True)
temp.dropna(axis=1, how='all', inplace=True)
holo_df = temp.copy()
print(holo_df.tail())
holo_df['Face'].interpolate(method='linear', inplace=True, axis=0)
# %%
print(holo_df.tail())
# %%
import plotly.graph_objects as go

fig = go.Figure()
for var in holo_df.columns:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=holo_df.index,
            y=holo_df.loc[:, var],
            name=var
        )
    )
    fig.update_layout(
        title = dict(
            text = var)
        )
    fig.show()
# %%
print(holoface_col)
print(holoimu_col)
print(holoposture_col)
# %%
import pandas as pd
import csv
import numpy as np
import os
from scipy.io import wavfile
import librosa

def empatica_reader(path):
    f = open(path, 'r')
    wr = csv.reader(f)
    t = 0.0
    tmp = []
    for idx, line in enumerate(wr):
        if idx ==  0:
            pass
        elif idx ==1 :
            freq = float(line[0])
        else:
            t += 1/freq
            line = np.array(line, dtype=np.float32)
            line = np.insert(line, 0, t, axis=0)
            tmp.append(line)
    return tmp

def holo_reader(path):
    f = open(path, 'r')
    reader = csv.reader(f)
    tmp = []
    for idx, line in enumerate(reader):
        if idx == 0:
            col = line
            col[0] = "time"
        else:
            list = [None if i == '' else i for i in line]
            temp = np.array(list, dtype=np.float32)
            tmp.append(temp)
    return col, tmp

def device_reader(path):
    f = open(path, 'r')
    reader = csv.reader(f)
    tmp = []
    for idx, line in enumerate(reader):
        if idx == 0:
            pass
        else:
            temp = np.array(line, dtype=np.float32)
            tmp.append(temp)
    return tmp

def path_load(pid, condition):
    fp_dict = {}
    fp_dict['fp_e4acc'] = ".\\Empatica\\Condition%d\\P%d_C%d_EM\\ACC.csv"%(condition, pid, condition)
    fp_dict['fp_e4bvp'] = ".\\Empatica\\Condition%d\\P%d_C%d_EM\\BVP.csv"%(condition, pid, condition)
    fp_dict['fp_e4eda'] = ".\\Empatica\\Condition%d\\P%d_C%d_EM\\EDA.csv"%(condition, pid, condition)
    fp_dict['fp_e4hr'] = ".\\Empatica\\Condition%d\\P%d_C%d_EM\\HR.csv"%(condition, pid, condition)
    fp_dict['fp_e4temp'] = ".\\Empatica\\Condition%d\\P%d_C%d_EM\\TEMP.csv"%(condition, pid, condition)
    fp_dict['fp_holoaudio'] = ".\\Hololens\\Condition%d\\P%d_C%d_Audios.wav"%(condition, pid, condition)
    fp_dict['fp_holoface'] = ".\\Hololens\\Condition%d\\P%d_C%d_Faces.csv"%(condition, pid, condition)
    fp_dict['fp_holoimu'] = ".\\Hololens\\Condition%d\\P%d_C%d_IMUs.csv"%(condition, pid, condition)
    fp_dict['fp_holoposture'] = ".\\Hololens\\Condition%d\\P%d_C%d_Posture.csv"%(condition, pid, condition)
    fp_dict['fp_desktop'] = ".\\Desktop\\Condition%d\\P%d_C%ddesktop.csv"%(condition, pid, condition)
    fp_dict['fp_laptop'] = ".\\Laptop\\Condition%d\\P%d_C%dlaptop.csv"%(condition, pid, condition)
    fp_dict['fp_phone'] = ".\\Mobile\\Condition%d\\P%d_C%dphone.csv"%(condition, pid, condition)
    return fp_dict

#%%
path_dict = path_load(1, 1)
isFile = True
#Empatica E4 dataframe
e4acc_data = empatica_reader(path_dict['fp_e4acc'])
e4bvp_data = empatica_reader(path_dict['fp_e4bvp'])
e4eda_data = empatica_reader(path_dict['fp_e4eda'])
e4hr_data = empatica_reader(path_dict['fp_e4hr'])
e4temp_data = empatica_reader(path_dict['fp_e4temp'])
e4acc_col = ['time', 'acc.x', 'acc.y', 'acc.z']
e4bvp_col = ['time', 'bvp']
e4eda_col = ['time', 'eda']
e4hr_col = ['time', 'hr']
e4temp_col = ['time', 'temp']
e4acc_df = pd.DataFrame(columns=e4acc_col, data=e4acc_data)
e4bvp_df = pd.DataFrame(columns=e4bvp_col, data=e4bvp_data)
e4eda_df = pd.DataFrame(columns=e4eda_col, data=e4eda_data)
e4hr_df = pd.DataFrame(columns=e4hr_col, data=e4hr_data)
e4temp_df = pd.DataFrame(columns=e4temp_col, data=e4temp_data)
e4acc_df.set_index("time", inplace=True)
e4bvp_df.set_index("time", inplace=True)
e4eda_df.set_index("time", inplace=True)
e4hr_df.set_index("time", inplace=True)
e4temp_df.set_index("time", inplace=True)
temp = pd.merge(e4acc_df, e4bvp_df, how='outer', left_index=True, right_index=True)
temp = pd.merge(temp, e4eda_df, how='outer', left_index=True, right_index=True)
temp = pd.merge(temp, e4hr_df, how='outer', left_index=True, right_index=True)
temp = pd.merge(temp, e4temp_df, how='outer', left_index=True, right_index=True)
temp.dropna(axis=0, how="all", inplace=True)
temp.dropna(axis=1, how="all", inplace=True)
empatica_df = temp.copy()
empatica_df.interpolate(method='linear', inplace=True, axis=0)
#holoens dataframe
holoface_col, holoface_data = holo_reader(path_dict['fp_holoface'])
holoimu_col, holoimu_data = holo_reader(path_dict['fp_holoimu'])
holoposture_col, holoposture_data = holo_reader(path_dict['fp_holoposture'])
holoface_df = pd.DataFrame(columns=holoface_col, data=holoface_data)
holoimu_df = pd.DataFrame(columns=holoimu_col, data=holoimu_data)
holoposture_df = pd.DataFrame(columns=holoposture_col, data=holoposture_data)
holoposture_df.dropna(axis = 1, how='all', inplace=True)
holoface_df.set_index('time', inplace=True)
holoimu_df.set_index('time', inplace=True)
holoposture_df.set_index('time', inplace=True)
temp = pd.merge(holoimu_df, holoface_df, how='outer', left_index=True, right_index=True)
temp = pd.merge(temp, holoposture_df, how='outer', left_index=True, right_index=True)
temp.dropna(axis=0, how='all', inplace=True)
temp.dropna(axis=1, how='all', inplace=True)
holo_df = temp.copy()
holo_df['Face'].fillna(method='pad', inplace=True)
holo_df['Posture'].fillna(method='pad', inplace=True)
holo_df.interpolate(method='linear', inplace=True, axis=0)
#Multi-device dataframe
desktop_data = device_reader(path_dict['fp_desktop'])
laptop_data = device_reader(path_dict['fp_laptop'])
phone_data = device_reader(path_dict['fp_phone'])
desktop_col = ['time', 'desktop']
laptop_col = ['time', 'laptop']
phone_col = ['time', 'phone']
desktop_df = pd.DataFrame(columns=desktop_col, data=desktop_data)
laptop_df = pd.DataFrame(columns=laptop_col, data=laptop_data)
phone_df = pd.DataFrame(columns=phone_col, data=phone_data)
desktop_df.set_index('time', inplace=True)
laptop_df.set_index('time', inplace=True)
phone_df.set_index('time', inplace=True)
temp = pd.merge(desktop_df, laptop_df, how='outer', left_index=True, right_index=True)
temp = pd.merge(temp, phone_df, how='outer', left_index=True, right_index=True)
temp.dropna(axis=0, how='all', inplace=True)
temp.dropna(axis=1, how='all', inplace=True)
device_df = temp.copy()
device_df.fillna(method='pad', inplace=True)
#wav dataframe
x, sr = librosa.load(path_dict['fp_holoaudio'], sr=None)
x = np.append(x, x)
mfcc = librosa.feature.mfcc(y=x, sr=sr)
rmse = librosa.feature.rms(y=x)
times = librosa.times_like(rmse, sr=sr)
data_frame_dict = {"time" : times}
i = 0
for each in mfcc:
    data_frame_dict["mfcc%d"%(i+1)] = mfcc[i]
    i += 1
audio_df = pd.DataFrame(data_frame_dict)
audio_df.set_index('time', inplace=True)
#merge all -> interpolate
merge_df = pd.merge(empatica_df, holo_df, how='outer', left_index=True, right_index=True)
merge_df = pd.merge(merge_df, device_df, how='outer', left_index=True, right_index=True)
merge_df = pd.merge(merge_df, audio_df, how='outer', left_index=True, right_index=True)
# %%
merge_df['desktop'][25:26]
# %%
merge_df['desktop'].fillna(method='pad', inplace=True)
# %%
merge_df['desktop'][25:26]
# %%
df = pd.read_pickle('C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\P1_C1.pkl')
import plotly.graph_objects as go

fig = go.Figure()
for var in df.columns:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df.loc[:, var],
            name=var
        )
    )
    fig.update_layout(
        title = dict(
            text = var)
        )
    fig.show()
# %%
import pandas
import numpy
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

df = pd.read_pickle("C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\P7_C1.pkl")

for var in df.columns:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df.loc[:, var],
            name=var
        )
    )
    fig.update_layout(
        title = dict(
            text = var)
        )
    fig.show()
# %%
