import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

dfs = []
for condition in range(1, 7): #7
    for pid in range(7, 10): #8
        WIN_SIZE = 2.5
        W = np.arange(12.5, 400, WIN_SIZE * 0.5)
        isFile = True
        fp = 'C:\\Users\\uvrlab\\Downloads\\Data for test\\Test\\sampled_P%d_C%d.pkl'%(pid, condition)
        if (not os.path.exists(fp)):
            print("no file")
            continue
        print(pid)
        print(condition)
        df = pd.read_pickle(fp)
        df.reset_index(inplace=True)
        df.drop(['time'], axis=1, inplace=True)
        dfs.append(df)

tmp = pd.concat([dfs[0], dfs[1]])
for i in range(2, len(dfs)):
    tmp = pd.concat([tmp, dfs[i]])
tmp.to_pickle("data_merged_test.pkl")

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