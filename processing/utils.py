from typing import Tuple
import pickle
import pandas as pd

model_temp = pickle.load(open('processing/model.p', 'rb'))
model_valve = pickle.load(open('processing/model_valve.p', 'rb'))

def perform_processing(
        temperature: pd.DataFrame,
        target_temperature: pd.DataFrame,
        valve_level: pd.DataFrame,
        serial_number_for_prediction: str,

) -> Tuple[float, float]:

    global model_valve, model_temp

    df_valve = valve_level
    df_valve.rename(columns={'value': 'valve_level'}, inplace=True)
    df_valve.drop(columns=['unit', 'serialNumber'], inplace=True)

    df_temp_mid = temperature[temperature['serialNumber'] == "0015BC0035001299"]
    df_temp_mid.drop(columns=['unit', 'serialNumber'], inplace=True)
    df_temp_mid.rename(columns={'value': 'temp_mid'}, inplace=True)

    df_temp_wall = temperature[temperature['serialNumber'] == "0015BC00350010A1"]
    df_temp_wall.drop(columns=['unit', 'serialNumber'], inplace=True)
    df_temp_wall.rename(columns={'value': 'temp_wall'}, inplace=True)

    df_temp_window = temperature[temperature['serialNumber'] == "0015BC0035001050"]
    df_temp_window.drop(columns=['unit', 'serialNumber'], inplace=True)
    df_temp_window.rename(columns={'value': 'temp_window'}, inplace=True)

    df_target_temp = target_temperature
    df_target_temp.drop(columns=['unit', 'serialNumber'], inplace=True)
    df_target_temp.rename(columns={'value': 'target_temp'}, inplace=True)

    df_combined = pd.concat([df_valve, df_target_temp, df_temp_mid, df_temp_wall, df_temp_window])
    df_combined_resampled = df_combined.resample(pd.Timedelta(minutes=15), label='right').mean().fillna(method='ffill')

    data_list_temp = ['temp_mid', 'target_temp', 'temp_wall', 'temp_window']
    data_list_valve = ['valve_level', 'target_temp', 'temp_mid']

    X_test_temp = df_combined_resampled[data_list_temp]
    X_test_valve = df_combined_resampled[data_list_valve]

    y_predicted_temp = model_temp.predict(X_test_temp)
    y_predicted_valve = model_valve.predict(X_test_valve)


    return y_predicted_temp[-1], y_predicted_valve[-1]
