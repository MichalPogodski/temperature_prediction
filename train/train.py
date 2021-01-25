import argparse
import json
from pathlib import Path
import pandas as pd
from processing.utils import perform_processing
from sklearn import ensemble, metrics
from sklearn.svm import SVR
import pickle


def read_temp_mid():
    with open('../../data/additional_info.json') as f:
        additional_data = json.load(f)
    devices = additional_data['offices']['office_1']['devices']
    sn_temp_mid = [d['serialNumber'] for d in devices if d['description'] == 'temperature_middle'][0]
    return sn_temp_mid

def read_temp_wall():
    with open('../../data/additional_info.json') as f:
        additional_data = json.load(f)
    devices = additional_data['offices']['office_1']['devices']
    sn_temp_wall = [d['serialNumber'] for d in devices if d['description'] == 'temperature_wall'][0]
    return sn_temp_wall

def read_temp_window():
    with open('../../data/additional_info.json') as f:
        additional_data = json.load(f)
    devices = additional_data['offices']['office_1']['devices']
    sn_temp_window = [d['serialNumber'] for d in devices if d['description'] == 'temperature_window'][0]
    return sn_temp_window



def data_collect():
    sn_temp_mid = read_temp_mid()
    sn_temp_wall = read_temp_wall()
    sn_temp_window = read_temp_window()

    df_temp_01 = pd.read_csv('../../data/office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv')
    df_temp_02 = pd.read_csv('../../data/office_1_temperature_supply_points_data_2020-03-05_2020-03-19.csv')
    df_temp = pd.concat([df_temp_01, df_temp_02], axis=0, ignore_index=True)
    df_temp.rename(columns={'Unnamed: 0': 'time', 'value': 'temp'}, inplace=True)
    df_temp['time'] = pd.to_datetime(df_temp['time'])
    df_temp.drop(columns=['unit'], inplace=True)
    df_temp.set_index('time', inplace=True)
    # df_temp = df_temp[df_temp['serialNumber'] == sn_temp_mid]
    df_temp.sort_index(inplace=True)

    df_temp_mid = df_temp[df_temp['serialNumber'] == sn_temp_mid]
    df_temp_mid.rename(columns={'temp':'temp_mid'}, inplace=True)
    df_temp_wall = df_temp[df_temp['serialNumber'] == sn_temp_wall]
    df_temp_wall.rename(columns={'temp': 'temp_wall'}, inplace=True)
    df_temp_window = df_temp[df_temp['serialNumber'] == sn_temp_window]
    df_temp_window.rename(columns={'temp': 'temp_window'}, inplace=True)
    # print(df_temp_window)
    # print(df_temp)


    df_temp_target_01 = pd.read_csv('../../data/office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv')
    df_temp_target_02 = pd.read_csv('../../data/office_1_targetTemperature_supply_points_data_2020-03-05_2020-03-19.csv')
    df_temp_target = pd.concat([df_temp_target_01, df_temp_target_02], axis=0, ignore_index=True)
    df_temp_target.rename(columns={'Unnamed: 0': 'time', 'value': 'target_temp'}, inplace=True)
    df_temp_target['time'] = pd.to_datetime(df_temp_target['time'])
    df_temp_target.drop(columns=['unit'], inplace=True)
    df_temp_target.set_index('time', inplace=True)


    df_valve_01 = pd.read_csv('../../data/office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv', index_col=0, parse_dates=True)
    df_valve_02 = pd.read_csv('../../data/office_1_valveLevel_supply_points_data_2020-03-05_2020-03-19.csv', index_col=0, parse_dates=True)
    df_valve = pd.concat([df_valve_01, df_valve_02], axis=0, ignore_index=False)
    df_valve.rename(columns={'value': 'opened[%]'}, inplace=True)
    df_valve.drop(columns=['unit'], inplace=True)
    df_valve.index.name = 'time'


    df_combined =pd.concat([df_valve, df_temp_target, df_temp_mid, df_temp_wall, df_temp_window])
    df_combined_resampled = df_combined.resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')
    df_combined_resampled['temp_gt'] = df_combined_resampled['temp_mid'].shift(-1, fill_value=
    df_combined_resampled['temp_mid'].tail(1).values[0])
    df_combined_resampled['valve_gt'] = df_combined_resampled['opened[%]'].shift(-1, fill_value=
    df_combined_resampled['opened[%]'].tail(1).values[0])
    return df_combined_resampled



def train_predict(df_combined_resampled):
    data_list_temp = ['temp_mid', 'target_temp', 'temp_wall', 'temp_window']
    data_list_valve = ['opened[%]', 'target_temp', 'temp_mid']


    mask_train = (df_combined_resampled.index < '2020-10-27') & (df_combined_resampled.index.hour >= 4)\
                & (df_combined_resampled.index.hour < 16)
    df_train = df_combined_resampled.loc[mask_train]
    X_train_temp = df_train[data_list_temp].to_numpy()[1:-1]
    y_train_temp = df_train['temp_gt'].to_numpy()[1:-1]
    X_train_valve = df_train[data_list_valve].to_numpy()[1:-1]
    y_train_valve = df_train['valve_gt'].to_numpy()[1:-1]


    # reg_temp = ensemble.RandomForestRegressor(n_estimators=300, random_state=42)
    # reg_temp = ensemble.BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=42)
    # reg_temp = ensemble.ExtraTreesRegressor(n_estimators=950, random_state=42, criterion='mae')
    reg_temp = ensemble.ExtraTreesRegressor(n_estimators=13, random_state=42)
    # reg_temp = ensemble.GradientBoostingRegressor(n_estimators=120, random_state=42)
    reg_temp.fit(X_train_temp, y_train_temp)

    reg_valve = ensemble.ExtraTreesRegressor(n_estimators=13, random_state=42)
    reg_valve.fit(X_train_valve, y_train_valve)


    mask_test = (df_combined_resampled.index > '2020-10-27') & (df_combined_resampled.index <= '2020-10-28') \
                & (df_combined_resampled.index.hour >= 4) & (df_combined_resampled.index.hour < 16)
    df_test_temp = df_combined_resampled.loc[mask_test]
    X_test_temp = df_test_temp[data_list_temp].to_numpy()[1:-1]
    y_test_temp = df_test_temp['temp_gt'].to_numpy()[1:-1]
    X_test_valve = df_test_temp[data_list_valve].to_numpy()[1:-1]
    y_test_valve = df_test_temp['valve_gt'].to_numpy()[1:-1]


    ###############
    # model = pickle.load(open('model.p', 'rb'))
    # y_predicted_temp = model.predict(X_test_temp)
    ##################

    y_predicted_temp = reg_temp.predict(X_test_temp)
    y_predicted_valve = reg_valve.predict(X_test_valve)

    print('MAE temp: ', metrics.mean_absolute_error(y_test_temp, y_predicted_temp))
    print('MAE valve: ', metrics.mean_absolute_error(y_test_valve, y_predicted_valve))

    pickle.dump(reg_valve, open('model_valve.p', 'wb'))
    pickle.dump(reg_temp, open('model.p', 'wb'))


if __name__ == '__main__':
    df_combined_resampled = data_collect()
    train_predict(df_combined_resampled)









