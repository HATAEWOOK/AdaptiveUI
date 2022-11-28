import numpy as np
import pandas as pd
import os

def window_mean():
    for condition in range(1, 7): #7
        for pid in range(1, 10): #8
            WIN_SIZE = 2.5
            W = np.arange(72.5, 400, WIN_SIZE * 0.5)
            isFile = True
            fp = 'C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\P%d_C%d.pkl'%(pid, condition)
            if (not os.path.exists(fp)):
                print("no file")
                continue
            df = pd.read_pickle(fp)
            df = df[70:400]
            sampled_data = []
            for w in W:
                sub = df.loc[lambda x: (x.index >= w - WIN_SIZE) & (x.index < w), df.columns]
                for col in sub.columns:
                    if col == 'Face' or col == 'Posture' or col == 'desktop' or col == 'latop' or col == 'phone':
                        sampled_data.append((w, col, sub[col].max()))
                    else:
                        sampled_data.append((w, col, sub[col].mean()))

            sampled_data = pd.DataFrame(sampled_data, columns = ['time', 'feature', 'value'])
            sampled_data = sampled_data.pivot(index='time', columns='feature', values='value')
            sampled_data['phone'].fillna(0, inplace=True)
            sampled_data['desktop'].fillna(0, inplace=True)
            sampled_data['laptop'].fillna(0, inplace=True)
            if condition == 1:
                if pid == 1:
                    sampled_data['arousal'] = 2
                    sampled_data['valence'] = 1
                elif pid == 2:
                    sampled_data['arousal'] = 2
                    sampled_data['valence'] = 1
                elif pid == 3:
                    sampled_data['arousal'] = 1
                    sampled_data['valence'] = 1
                elif pid == 4:
                    sampled_data['arousal'] = 1
                    sampled_data['valence'] = 1
                elif pid == 5:
                    sampled_data['arousal'] = 2
                    sampled_data['valence'] = 1
                elif pid == 6:
                    sampled_data['arousal'] = 1
                    sampled_data['valence'] = 0
                elif pid == 7:
                    sampled_data['arousal'] = 1
                    sampled_data['valence'] = 1
                elif pid == 8:
                    sampled_data['arousal'] = 1
                    sampled_data['valence'] = 1
                elif pid == 9:
                    sampled_data['arousal'] = 2
                    sampled_data['valence'] = 1
            elif condition == 2:
                if pid == 1:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
                elif pid == 2:
                    sampled_data['arousal'] = 1
                    sampled_data['valence'] = 1
                elif pid == 3:
                    sampled_data['arousal'] = 1
                    sampled_data['valence'] = 1
                elif pid == 4:
                    sampled_data['arousal'] = 1
                    sampled_data['valence'] = 1
                elif pid == 5:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 2
                elif pid == 6:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
                elif pid == 7:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 2
                elif pid == 8:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 2
                elif pid == 9:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 2
            elif condition == 3:
                if pid == 1:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
                elif pid == 2:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
                elif pid == 3:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
                elif pid == 4:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
                elif pid == 5:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
                elif pid == 6:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
                elif pid == 7:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
                elif pid == 8:
                    sampled_data['arousal'] = 1
                    sampled_data['valence'] = 1
                elif pid == 9:
                    sampled_data['arousal'] = 2
                    sampled_data['valence'] = 1
            elif condition == 4:
                if pid == 1:
                    sampled_data['arousal'] = 1
                    sampled_data['valence'] = 1
                elif pid == 2:
                    sampled_data['arousal'] = 2
                    sampled_data['valence'] = 1
                elif pid == 3:
                    sampled_data['arousal'] = 2
                    sampled_data['valence'] = 1
                elif pid == 4:
                    sampled_data['arousal'] = 1
                    sampled_data['valence'] = 1
                elif pid == 5:
                    sampled_data['arousal'] = 2
                    sampled_data['valence'] = 1
                elif pid == 6:
                    sampled_data['arousal'] = 2
                    sampled_data['valence'] = 1
                elif pid == 7:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
                elif pid == 8:
                    sampled_data['arousal'] = 2
                    sampled_data['valence'] = 1
            elif condition == 5:
                if pid == 1:
                    sampled_data['arousal'] = 1
                    sampled_data['valence'] = 1
                elif pid == 2:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 2
                elif pid == 3:
                    sampled_data['arousal'] = 2
                    sampled_data['valence'] = 2
                elif pid == 4:
                    sampled_data['arousal'] = 2
                    sampled_data['valence'] = 2
                elif pid == 5:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 2
                elif pid == 6:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 2
                elif pid == 7:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 2
            elif condition == 6:
                if pid == 1:
                    sampled_data['arousal'] = 2
                    sampled_data['valence'] = 2
                elif pid == 2:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
                elif pid == 3:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 2
                elif pid == 4:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
                elif pid == 5:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
                elif pid == 6:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
                elif pid == 7:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
                elif pid == 8:
                    sampled_data['arousal'] = 0
                    sampled_data['valence'] = 1
            sampled_data.to_pickle('.\\Data\\sampled_P%d_C%d.pkl'%(pid, condition)) 

def window_conv1d():
    window_dataset = np.empty((1, 262, 473, 39))
    min = np.inf
    for condition in range(1, 7): #7
        for pid in range(1, 8): #8
            fp = 'C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\P%d_C%d.pkl'%(pid, condition)
            if (not os.path.exists(fp)):
                print("no file")
                continue
            WIN_SIZE = 2.5
            W = np.arange(72.5, 400, WIN_SIZE * 0.5)
            isFile = True
            df = pd.read_pickle(fp)
            df = df[70:400]
            sw = False
            for w in W:
                sub = df.loc[lambda x: (x.index >= w - WIN_SIZE) & (x.index < w), df.columns]
                condition_list = sub["condition"].to_numpy()
                condition_list = np.expand_dims(condition_list, axis=1)
                sub.drop(['acc.x', 'acc.y', 'acc.z', 'pid', 'condition'], axis=1, inplace=True)
                tmp = sub.to_numpy()
                while(tmp.shape[0] > 473):
                    tmp = np.delete(tmp, -1, 0)
                    condition_list = np.delete(condition_list, -1, 0)
                sampled_data = np.concatenate([tmp, condition_list], axis=1)
                sampled_data = sampled_data.reshape((1, sampled_data.shape[0], sampled_data.shape[1]))
                if not sw:
                    merged_data_np = sampled_data
                    sw = True
                    continue
                merged_data_np = np.concatenate([merged_data_np, sampled_data], axis=0)
            merged_data_np = np.expand_dims(merged_data_np, axis=0)
            window_dataset = np.concatenate([window_dataset, merged_data_np], axis = 0)

    np.save('window_data.pkl', window_dataset, allow_pickle=True)
            # sampled_data = pd.DataFrame(sampled_data, columns = ['time', 'feature', 'value'])
            # sampled_data = sampled_data.pivot(index='time', columns='feature', values='value')
            # sampled_data['phone'].fillna(0, inplace=True)
            # sampled_data['desktop'].fillna(0, inplace=True)
            # sampled_data['laptop'].fillna(0, inplace=True)

if __name__ =="__main__":
    window_mean()