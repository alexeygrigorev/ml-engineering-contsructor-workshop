#!/usr/bin/env python
# coding: utf-8

import pickle
from datetime import datetime

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error



def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


def train(train_month: datetime, val_month: datetime, model_output_path: str) -> None:
    url_template = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    url_train = url_template.format(year=train_month.year, month=train_month.month)
    url_val = url_template.format(year=val_month.year, month=val_month.month)
    
    df_train = read_dataframe(url_train)
    df_val = read_dataframe(url_val)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')

    pipeline = make_pipeline(
        DictVectorizer(),
        LinearRegression()
    )

    pipeline.fit(train_dicts, y_train)

    y_pred = pipeline.predict(val_dicts)

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f'rmse={rmse}')

    with open(model_output_path, 'wb') as f_out:
        pickle.dump(pipeline, f_out)

