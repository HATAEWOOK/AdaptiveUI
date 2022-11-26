#%%
import pandas as pd
import csv
import numpy as np
import os
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
import librosa.display


for condition in range(1, 7): #7
    for pid in range(1, 8): #8
        fig, ax = plt.subplots()
        fp = ".\\Hololens\\Condition%d\\P%d_C%d_Audios.wav"%(condition, pid, condition)
        if not os.path.exists(fp):
            continue
        x, sr = librosa.load(fp, sr=None)
        x = np.append(x, x)
        librosa.display.waveshow(x, sr=sr, alpha=0.4)
        print(len(x))
        print(sr)
        plt.savefig(".\\audio\\audio_%d_%d.png"%(pid, condition))
        plt.close
#%%
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()

hl = 512 # number of samples per time-step in spectrogram
hi = 128 # Height of image
wi = 384 # Width of image

# Loading demo track
y, sr = librosa.load(librosa.ex('trumpet'))
window = y[0:wi*hl]

S = librosa.feature.melspectrogram(y=window, sr=sr, n_mels=hi, fmax=8000,
hop_length=hl)
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)

plt.savefig("out.png")
plt.show()
#%%
chroma_stft = librosa.feature.chroma_stft(y=x, sr=sr)
rmse = librosa.feature.rms(y=x)
spec_cent = librosa.feature.spectral_centroid(y=x, sr=sr)
spec_bw = librosa.feature.spectral_bandwidth(y=x, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
zcr = librosa.feature.zero_crossing_rate(x)
mfcc = librosa.feature.mfcc(y=x, sr=sr)
#%%

    
# %%
chroma_stft_mean = np.mean(chroma_stft, axis=0)
rmse_mean = np.mean(rmse, axis=0)
spec_cent_mean = np.mean(spec_cent, axis=0)
spec_bw_mean = np.mean(spec_bw, axis=0)
rolloff_mean = np.mean(rolloff, axis=0)
zcr_mean = np.mean(zcr, axis=0)
# %%
spec_cent_mean.shape
#%%
times = librosa.times_like(rmse, sr=sr)
#%%
data_frame_dict = {}
data_frame_dict = {"time" : times, "chroma_stft" : chroma_stft_mean, "rmse_mean" : rmse_mean, "spec_cent" : spec_cent_mean, "spec_bw" : spec_bw_mean, "rolloff" : rolloff_mean, "zcr" : zcr_mean}
i = 0
for each in mfcc:
    data_frame_dict["mfcc%d"%(i+1)] = mfcc[i]
    i += 1
#%%
print(chroma_stft.shape)
#%%
from IPython.display import display
display(pd.DataFrame(data_frame_dict))
#%%
#display Spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
#If to pring log of frequencies  
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
#%%
print(Xdb[1024].shape)
# %%
# Zooming in
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()
# %%
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))
# %%
#spectral centroid -- centre of mass -- weighted mean of the frequencies present in the sound
import sklearn
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
print(spectral_centroids.shape)
# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames, sr=sr)
print(frames)
print(t.shape)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveshow(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')
#%%
print(t[:30])
#%%
spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
librosa.display.waveshow(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
# %%
mfccs = librosa.feature.mfcc(x, sr=sr)
print(mfccs.shape)
#Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
# %%
# Get RMS value from each frame's magnitude value
S, phase = librosa.magphase(librosa.stft(x))
rms = librosa.feature.rms(S=S)
# Plot the RMS energy
fig, ax = plt.subplots(figsize=(15, 6), nrows=2, sharex=True)
times = librosa.times_like(rms, sr=sr)
ax[0].semilogy(times, rms[0], label='RMS Energy')
ax[0].set(xticks=[])
ax[0].legend()
ax[0].label_outer()
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr,
                         y_axis='log', x_axis='time', ax=ax[1])
ax[1].set(title='log Power spectrogram')
# %%
print(times.shape)
# %%
print(times[-1])
# %%
import pandas as pd
df = pd.read_pickle('C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\data_merged.pkl')
print(df.columns)
# %%
import plotly.graph_objects as go

fig = go.Figure()
for var in df.columns:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df.loc[:, var],
            name=var
        )
    )
fig.show()

# %%
import numpy as np
WIN_SIZE = 5
W = np.arange(WIN_SIZE, 600.0, WIN_SIZE * 0.5)
print(df.columns)

# %%
for w in W:
    sub = df.loc[lambda x: (x.index >= w - WIN_SIZE) & (x.index < w), df.columns]
# %%
print(sub.columns[0])
print(sub["acc.x"].mean())
#%%
import numpy as np
import pandas as pd
WIN_SIZE = 2.56
W = np.arange(WIN_SIZE, 600.0, WIN_SIZE * 0.5)
df = pd.read_pickle('C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\P1_C1.pkl')
sampled_data = []
for w in W:
    sub = df.loc[lambda x: (x.index >= w - WIN_SIZE) & (x.index < w), df.columns]
    for col in sub.columns:
        if col == 'Face' or col == 'Posture' or col == 'desktop' or col == 'latop' or col == 'phone':
            sampled_data.append((w, col, sub[col].max()))
        else:
            sampled_data.append((w, col, sub[col].mean()))

print(sub.columns)
sampled_data = pd.DataFrame(sampled_data, columns = ['time', 'feature', 'value'])
sampled_data = sampled_data.pivot(index='time', columns='feature', values='value')
print(sampled_data)
# %%
print(sampled_data.columns[:-2])
len(sampled_data.columns[:-2])
# %%
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd

fig = make_subplots(rows=len(sampled_data.columns[:3]), cols=1, shared_xaxes=True)

for idx, var in enumerate(sampled_data.columns[:3], start=1):
    fig.add_trace(
        go.Scatter(
            x=sampled_data.index,
            y=sampled_data.loc[:, var],
            name=var
        ), row = idx, col =1
    )
    fig.update_layout(
        title = dict(
            text = var)
        )

fig.show()
#%%
df = pd.read_pickle('C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\P1_C1.pkl')
df[50:51]
#%%
df = pd.read_pickle('C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\P1_C1.pkl')
sub.columns
#%%
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd
sampled_data = pd.read_pickle('C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\sampled_P1_C1.pkl')
#%%
print(sampled_data)
#%%
WIN_SIZE = 2.5
W = np.arange(10, 400, WIN_SIZE * 0.5)
w = W[0]
print(tmp.loc[lambda x: (x.index >= w - WIN_SIZE) & (x.index < w), tmp.columns])
# %%
print(sampled_data.columns)
#%%
fig = make_subplots(rows=len(sampled_data.columns), cols=1, shared_xaxes=True)

for idx, var in enumerate(sampled_data.columns, start=1):
    fig.add_trace(
        go.Scatter(
            x=sampled_data.index,
            y=sampled_data.loc[:, var],
            name=var
        ), row = idx, col =1
    )
    fig.update_layout(
        title = dict(
            text = var)
        )

fig.show()
#%%
fig = make_subplots(rows=len(sampled_data.columns[4:7]), cols=1, shared_xaxes=True)

for idx, var in enumerate(sampled_data.columns[4:7], start=1):
    fig.add_trace(
        go.Scatter(
            x=sampled_data.index,
            y=sampled_data.loc[:, var],
            name=var
        ), row = idx, col =1
    )
    fig.update_layout(
        title = dict(
            text = var)
        )

fig.show()
#%%
for i in range(0,40, 3):
    fig = make_subplots(rows=len(sampled_data.columns[i:i+3]), cols=1, shared_xaxes=True)

    for idx, var in enumerate(sampled_data.columns[i:i+3], start=1):
        fig.add_trace(
            go.Scatter(
                x=sampled_data.index,
                y=sampled_data.loc[:, var],
                name=var
            ), row = idx, col =1
        )
        fig.update_layout(
            title = dict(
                text = var)
            )

    fig.show()
    
#%%
fig = make_subplots(rows=len(sampled_data.columns[7:10]), cols=1, shared_xaxes=True)

for idx, var in enumerate(sampled_data.columns[7:10], start=1):
    fig.add_trace(
        go.Scatter(
            x=sampled_data.index,
            y=sampled_data.loc[:, var],
            name=var
        ), row = idx, col =1
    )
    fig.update_layout(
        title = dict(
            text = var)
        )

fig.show()
#%%
import numpy as np
import pandas as pd
dfs = []
for condition in range(1, 7): #7
    for pid in range(1, 7): #8
        WIN_SIZE = 2.5
        W = np.arange(12.5, 400, WIN_SIZE * 0.5)
        isFile = True
        fp = 'C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\sampled_P%d_C%d.pkl'%(pid, condition)
        if (not os.path.exists(fp)):
            print("no file")
            continue
        df = pd.read_pickle(fp)
        df.reset_index(inplace=True)
        df.drop(['time'], axis=1, inplace=True)
        dfs.append(df)
# %%
dfs[1]
# %%
tmp = pd.concat([dfs[0], dfs[1]])
#%%
len(dfs)

# %%
for i in range(2, len(dfs)):
    tmp = pd.concat([tmp, dfs[i]])
# %%
tmp.to_pickle("data_merged(pid1to6).pkl")
# %%
#%%
i = 0
for df in dfs:
    pid = df['pid'][0]
    condition = df['condition'][0]
    num = 0
    for i in range(0,40, 4):
        num += 1
        fig = make_subplots(rows=len(df.columns[i:i+4]), cols=1, shared_xaxes=True)

        for idx, var in enumerate(df.columns[i:i+4], start=1):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df.loc[:, var],
                    name=var
                ), row = idx, col =1
            )
            fig.update_layout(
                title = dict(
                    text = "%d_%d_plot%d"%(pid, condition, num))
                )
        fig.write_image(".\\plot\\%d_%d_plot%d.jpeg"%(pid, condition, num))
# %%
