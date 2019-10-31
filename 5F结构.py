# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 23:09:54 2019

@author: bjlij
"""

from MetaTrader5 import *
import datetime
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np

#matplotlib.use('Qt5Agg')
matplotlib.use('Agg')
plt.style.use('dark_background')

MT5Initialize()
MT5WaitForTerminal()
#eurusd = MT5CopyRatesFromPos('EURUSD', MT5_TIMEFRAME_M5, 0, 10000)
eurusd = MT5CopyRatesRange('EURUSD', MT5_TIMEFRAME_M5, datetime.datetime(2019, 1, 1), datetime.datetime(2019, 10, 29))
MT5Shutdown()

delta = datetime.timedelta(hours=-8)
df = pd.DataFrame(eurusd,columns=['time','open','low','high','close','tick_volum','spread','real_volume'])
df.time = df.time + delta
df['date'] = df['time'].apply(lambda x:x.date())
data = []
txt = []
days = 2
for name,value in df.groupby(df['date']):
    data.append({'name':name,'value':value.close})
for i in range(1,len(data)-days):
    filename = data[i]['name'].strftime('%y{y}%m{m}%d{d}').format(y='年',m='月',d='日')
    path = r'..\2019\%s.jpg'%filename
    if (data[i+days]['value'].iloc[-1] - data[i+days-1]['value'].iloc[-1])/data[i+days-1]['value'].iloc[-1]*100 > 0.2:
        label = 1 
    elif (data[i+days]['value'].iloc[-1] - data[i+days-1]['value'].iloc[-1])/data[i+days-1]['value'].iloc[-1]*100 < -0.2:
        label = 2
    else:
        label = 0
    data[i]['label'] = label
    for j in range(days):
        ax = data[i+j]['value'].plot(color='w',figsize=(20,20),linewidth=3)
    ax.axis('off')
    ax.get_figure().savefig(path,dpi=20)
    txt.append(path + ' ' + str(label) + '\n')
    ax.clear()
    '''
    plt.axis('off')
    plt.plot(data[i]['value'])
    plt.savefig(r'..\2019\%s.png'%filename)
    plt.close()
    '''
with open(r'..\2019\2019.txt','w') as f:
    f.writelines(txt)
