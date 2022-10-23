import os
import pandas as pd
import numpy as np




COLUMNS_CHART_DATA = ['num', 'open', 'high', 'low', 'close', 'Volume']  #바꿨음
COLUMNS_TRAIN_DATA = ['open', 'high', 'low', 'close',  'ma5', 'ma10', 'ma20','ma60','ma120','Volume','rsi','macd']


def load_data(path):

    header = 0
    df = pd.read_csv(path)


    # 날짜 오름차순 정렬
    df = df.sort_values(by='num').reset_index(drop=True)
    chart_data = df[COLUMNS_CHART_DATA]

    tmin =1000000000000000
    tmax =-1
    for i in ['open', 'high', 'low', 'close',  'ma5', 'ma10', 'ma20','ma60','ma120']:
        tmin = min(tmin, df[i].min())
        tmax = max(tmax, df[i].max())
    for i in ['open', 'high', 'low', 'close', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120']:
        df[i] = (df[i]-tmin)/(tmax - tmin)
    df['Volume'] = (df['Volume']-df['Volume'].min()) / (df['Volume'].max()-df['Volume'].min())
    df['rsi'] =(df['rsi']-df['rsi'].min()) / (df['rsi'].max()-df['rsi'].min())
    df['macd'] =(df['macd']-df['macd'].mean()) / (df['macd'].std())

    training_data = df[COLUMNS_TRAIN_DATA]

    return chart_data, training_data

