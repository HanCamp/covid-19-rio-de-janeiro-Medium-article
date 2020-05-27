import pandas as pd
import numpy as np
from os import path
import json
import matplotlib.pyplot as plt
from lmfit import Model, Parameters, minimize
from lmfit.printfuncs import report_fit
from scipy.optimize import curve_fit
from sird_models import *

def get_data(path_to_pop_data, path_to_covid_data):
    """
    Get all necessary data and return a 2D Numpy array with the number of
    susceptible, active infected, recovered and dead individuals in each column
    """
    populations_df = pd.read_csv(path_to_pop_data)
    parser = lambda x : pd.to_datetime(x, format="%Y-%m-%d")
    data_df = pd.read_csv(path_to_covid_data, parse_dates=["dt_sintoma", "dt_coleta_dt_notif", "dt_obito"], date_parser=parser)
    data_df["dt_coleta_dt_notif"] = pd.to_datetime(data_df["dt_coleta_dt_notif"], infer_datetime_format=True, dayfirst=True)
    data_df["dt_sintoma"] = pd.to_datetime(data_df["dt_sintoma"], infer_datetime_format=True, dayfirst=True)
    data_df["dt_obito"] = pd.to_datetime(data_df["dt_obito"], infer_datetime_format=True, dayfirst=True)
    start_date = data_df["dt_sintoma"][0]
    n_days = max(data_df["dias"]) + 1
    data_array_change = np.zeros((n_days, 4))

    for i in range(max(data_df["dias"]) + 1):
        tmp_df = data_df.loc[data_df["dias"] == i]
        if not tmp_df.empty:
            for index, row in tmp_df.iterrows():
                if row["municipio_res"] == "RIO DE JANEIRO":
                    if pd.notna(row["dt_coleta_dt_notif"]) and not pd.notna(row["dt_sintoma"]):
                        day_of_infection = (row["dt_coleta_dt_notif"] - start_date).days
                    else:
                        day_of_infection = (row["dt_sintoma"] - start_date).days
                    data_array_change[day_of_infection, 1] += 1
                    data_array_change[day_of_infection, 0] -= 1
                    if row["evolucao"] == "OBITO" or row["evolucao"] == "obito":
                        if pd.notna(row["dt_obito"]):
                            day_of_death = (row["dt_obito"] - start_date).days
                        else:
                            day_of_death = i
                        data_array_change[day_of_death, 0] -= 1
                        data_array_change[day_of_death, 1] -= 1
                        data_array_change[day_of_death, 3] += 1
                    elif row["evolucao"] == "RECUPERADO" or row["evolucao"] == "recuperado" or ( not (row["evolucao"] == "INTERNADO" or row["evolucao"] == "internado")):
                        if day_of_infection + 14 < n_days:
                            data_array_change[day_of_infection + 14, 1] -= 1
                            data_array_change[day_of_infection + 14, 2] += 1

    data_array = np.cumsum(data_array_change, axis=0)
    for index, row in populations_df.iterrows():
        if row["municipio"] == "RIO DE JANEIRO":
            data_array[:, 0] += row["populacao"]
    return data_array

if __name__ == '__main__':
    np.random.seed(42)
    path_to_covid_data = "RJ_data.csv"
    path_to_pop_data = "RJ_populations.csv"

    data_array = get_data(path_to_pop_data, path_to_covid_data)
    T = np.arange(0, data_array.shape[0], 1)
    X0 = data_array[0]

    params_kinetic_seasonal = Parameters()
    params_kinetic_seasonal.add("beta0", value=0.1, min=math.pow(10.0, -6.0), max=1.0, vary=True)
    params_kinetic_seasonal.add("beta0_mag", value=0.1, min=math.pow(10.0, -6.0), max=1.0, vary=True)
    params_kinetic_seasonal.add("tau_beta", value=100.0, min=1.0, max=365.0, vary=True)
    params_kinetic_seasonal.add("gamma_sum", value=0.07, min=math.pow(10.0, -6.0), max=0.2, vary=True)
    params_kinetic_seasonal.add("gamma_partition", value=0.5, min=math.pow(10.0, -6.0), max=1.0, vary=True)
    params_kinetic_seasonal.add("tau_gamma", value=100.0, min=1.0, max=365.0, vary=True)
    params_kinetic_seasonal.add("mu_sum", value=0.01, min=math.pow(10.0, -6.0), max=0.5, vary=True)
    params_kinetic_seasonal.add("mu_partition", value=0.5, min=math.pow(10.0, -6.0), max=1.0, vary=True)
    params_kinetic_seasonal.add("tau_mu", value=100.0, min=1.0, max=365.0, vary=True)

    kinetic_seasonal_model= Model(SIRD_kinetic_seasonal, independent_vars=['t', 'x0'])
    optimization_method = "differential_evolution" # change to another method if desired

    out_kinetic_seasonal = minimize(loss_function, params_kinetic_seasonal, args=(T, X0,), kws={"model" : kinetic_seasonal_model, "data" : data_array}, method=optimization_method)
    report_fit(out_kinetic_seasonal, show_correl=False)
    params_json = out_kinetic_seasonal.params.dumps()
    with open(optimization_method + "_params.json", mode='w') as fname:
        json.dump(params_json, fname)
