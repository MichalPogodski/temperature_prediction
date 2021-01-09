import argparse
import json
from pathlib import Path
import pandas as pd
from processing.utils import perform_processing
from sklearn import ensemble, metrics



def read_temp_mid():
    with open('data/additional_info.json') as f:
        additional_data = json.load(f)

    devices = additional_data['offices']['office_1']['devices']
    sn_temp_mid = [d['serialNumber'] for d in devices if d['description'] == 'temperature_middle'][0]
    return sn_temp_mid


def data_collect():
    sn_temp_mid = read_temp_mid()

    df_temp = pd.read_csv('data/office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv')
    df_temp.rename(columns={'Unnamed: 0': 'time', 'value': 'temp'}, inplace=True)
    df_temp['time'] = pd.to_datetime(df_temp['time'])
    df_temp.drop(columns=['unit'], inplace=True)
    df_temp.set_index('time', inplace=True)
    df_temp = df_temp[df_temp['serialNumber'] == sn_temp_mid]


    df_temp_target = pd.read_csv('data/office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv')
    df_temp_target.rename(columns={'Unnamed: 0': 'time', 'value': 'target_temp'}, inplace=True)
    df_temp_target['time'] = pd.to_datetime(df_temp_target['time'])
    df_temp_target.drop(columns=['unit'], inplace=True)
    df_temp_target.set_index('time', inplace=True)

    df_combined =pd.concat([df_temp, df_temp_target])
    df_combined_resampled = df_combined.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')
    df_combined_resampled['temp_gt'] = df_combined_resampled['temp'].shift(-1, fill_value=
    df_combined_resampled['temp'].tail(1).values[0])
    return df_combined_resampled



def train_predict(df_combined_resampled):

    mask_train = df_combined_resampled.index < '2020-10-27'
    df_train = df_combined_resampled.loc[mask_train]
    X_train = df_train['temp'].to_numpy()[1:-1].reshape(-1, 1)
    y_train = df_train['temp_gt'].to_numpy()[1:-1]

    reg_rf = ensemble.RandomForestRegressor(random_state=42)
    reg_rf.fit(X_train, y_train)

    mask_test = (df_combined_resampled.index > '2020-10-27') & (df_combined_resampled.index <= '2020-10-28')
    df_test = df_combined_resampled.loc[mask_test]
    X_test = df_test['temp'].to_numpy()[1:-1].reshape(-1, 1)
    y_test = df_test['temp_gt'].to_numpy()[1:-1]

    y_predicted = reg_rf.predict(X_test)

    print('MAE: ', metrics.mean_absolute_error(y_test, y_predicted))




if __name__ == '__main__':
    df_combined_resampled = data_collect()
    train_predict(df_combined_resampled)









