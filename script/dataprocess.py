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
    
if __name__ == '__main__':
    for condition in range(1, 7): #7
        for pid in range(1, 8): #8
            print("P%d_C%d"%(pid, condition))
            path_dict = path_load(pid, condition)
            isFile = True
            for fp in path_dict.values():
                print(fp)
                if (not os.path.exists(fp)):
                    print("no file")
                    isFile = False
            if (not isFile):
                continue
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
            merge_df['Face'].fillna(method='pad', inplace=True)
            merge_df['Posture'].fillna(method='pad', inplace=True)
            merge_df['desktop'].fillna(method='pad', inplace=True)
            merge_df['laptop'].fillna(method='pad', inplace=True)
            merge_df['phone'].fillna(method='pad', inplace=True)
            merge_df.interpolate(method='linear', inplace=True, axis=0)
            merge_df['pid'] = pid
            merge_df['condition'] = condition
            merge_df.to_pickle('P%d_C%d.pkl'%(pid, condition)) 





