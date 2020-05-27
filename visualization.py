import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import json
from lmfit import Parameters, Model
from sird_models import SIRD_kinetic_seasonal, SIRD_kinetic_seasonal_reactive, SIRD_kinetic_seasonal_no_SD
from covid_rio_city_fit import get_data

def seasonal_beta(t, beta_prime, mag):
    return beta_prime * (1 - (mag / 2) * (1 - np.cos(2 * np.pi * t / 365)))

def kinetic_beta(t, beta0, tau_beta):
    return beta0 * np.exp(- t / tau_beta)

def kinetic_gamma(t, gamma0, gamma1, tau_gamma):
    return gamma0 + gamma1 / (1 + np.exp(-t + tau_gamma))

def kinetic_mu(t, mu0, mu1, tau_mu):
    return mu0 * np.exp(-t / tau_mu) + mu1

if __name__ == '__main__':
    """ Make all plots used in the article """
    
    LBFGSb_params = Parameters()
    with open("lbfgsb_params.json", mode='r') as fname:
        params = json.load(fname)
        LBFGSb_params.loads(params)

    NelderMead_params = Parameters()
    with open("nelder_params.json", mode='r') as fname:
        params = json.load(fname)
        NelderMead_params.loads(params)

    DE_params = Parameters()
    with open("differential_evolution_params.json", mode='r') as fname:
        params = json.load(fname)
        DE_params.loads(params)

    BH_params = Parameters()
    with open("basinhopping_params.json", mode='r') as fname:
        params = json.load(fname)
        BH_params.loads(params)

    DA_params = Parameters()
    with open("dual_annealing_params.json", mode='r') as fname:
        params = json.load(fname)
        DA_params.loads(params)

    path_to_covid_data = "RJ_data.csv"
    path_to_pop_data = "RJ_populations.csv"
    DATA = get_data(path_to_pop_data, path_to_covid_data)
    X0 = DATA[0]
    time_frame = np.arange(0, 365, 1)

    kinetic_seasonal_model = Model(SIRD_kinetic_seasonal, independent_vars=['t', 'x0'])
    no_social_distancing_model = Model(SIRD_kinetic_seasonal_no_SD, independent_vars=['t', 'x0'])

    DE_pred = kinetic_seasonal_model.eval(DE_params, t=time_frame, x0=X0)
    BH_pred = kinetic_seasonal_model.eval(BH_params, t=time_frame, x0=X0)
    DA_pred = kinetic_seasonal_model.eval(DA_params, t=time_frame, x0=X0)

    print "Differential Evolution:"
    DE_params.pretty_print()
    print "gamma0 : %f" % (DE_params["gamma_sum"] * DE_params["gamma_partition"])
    print "gamma1 : %f" % (DE_params["gamma_sum"] * (1 - DE_params["gamma_partition"]))
    print "mu0 : %f" % (DE_params["mu_sum"] * DE_params["mu_partition"])
    print "mu1 : %f" % (DE_params["mu_sum"] * (1 - DE_params["mu_partition"]))

    print "Total  Number of Infected DE:\t\t%.1f" % (DE_pred[-1,2] + DE_pred[-1,3])
    print "Total Number of Dead DE:\t\t\t%.1f\n" % DE_pred[-1,3]

    print "Basin Hopping:"
    BH_params.pretty_print()
    print "gamma0 : %f" % (BH_params["gamma_sum"] * BH_params["gamma_partition"])
    print "gamma1 : %f" % (BH_params["gamma_sum"] * (1 - BH_params["gamma_partition"]))
    print "mu0 : %f" % (BH_params["mu_sum"] * BH_params["mu_partition"])
    print "mu1 : %f" % (BH_params["mu_sum"] * (1 - BH_params["mu_partition"]))

    print "Total  Number of Infected BH:\t\t%.1f" % (BH_pred[-1,2] + BH_pred[-1,3])
    print "Total Number of Dead BH:\t\t\t%.1f\n" % BH_pred[-1,3]

    print "Dual Annealing:"
    DA_params.pretty_print()
    print "gamma0 : %f" % (DA_params["gamma_sum"] * DA_params["gamma_partition"])
    print "gamma1 : %f" % (DA_params["gamma_sum"] * (1 - DA_params["gamma_partition"]))
    print "mu0 : %f" % (DA_params["mu_sum"] * DA_params["mu_partition"])
    print "mu1 : %f" % (DA_params["mu_sum"] * (1 - DA_params["mu_partition"]))

    print "Total  Number of Infected DA:\t\t%.1f" % (DA_pred[-1,2] + DA_pred[-1,3])
    print "Total Number of Dead DA:\t\t\t%.1f\n" % DA_pred[-1,3]

    R0_DE = seasonal_beta(time_frame, kinetic_beta(time_frame, DE_params["beta0"], DE_params["tau_beta"]), DE_params["beta0_mag"]) / ( kinetic_gamma(time_frame, DE_params["gamma_sum"] * DE_params["gamma_partition"], DE_params["gamma_sum"] * (1 - DE_params["gamma_partition"]), DE_params["tau_gamma"]) + kinetic_mu(time_frame, DE_params["mu_sum"] * DE_params["mu_partition"], DE_params["mu_sum"] * (1 - DE_params["mu_partition"]), DE_params["tau_mu"]) )
    R0_BH = seasonal_beta(time_frame, kinetic_beta(time_frame, BH_params["beta0"], BH_params["tau_beta"]), BH_params["beta0_mag"]) / ( kinetic_gamma(time_frame, BH_params["gamma_sum"] * BH_params["gamma_partition"], BH_params["gamma_sum"] * (1 - BH_params["gamma_partition"]), BH_params["tau_gamma"]) + kinetic_mu(time_frame, BH_params["mu_sum"] * BH_params["mu_partition"], BH_params["mu_sum"] * (1 - BH_params["mu_partition"]), BH_params["tau_mu"]) )
    R0_DA = seasonal_beta(time_frame, kinetic_beta(time_frame, DA_params["beta0"], DA_params["tau_beta"]), DA_params["beta0_mag"]) / ( kinetic_gamma(time_frame, DA_params["gamma_sum"] * DA_params["gamma_partition"], DA_params["gamma_sum"] * (1 - DA_params["gamma_partition"]), DA_params["tau_gamma"]) + kinetic_mu(time_frame, DA_params["mu_sum"] * DA_params["mu_partition"], DA_params["mu_sum"] * (1 - DA_params["mu_partition"]), DA_params["tau_mu"]) )

    percent_suceptible_DE = DE_pred[:,0] / (DE_pred[:,0] + DE_pred[:,1] + DE_pred[:,2] - DE_pred[:,3])
    percent_suceptible_BH = BH_pred[:,0] / (BH_pred[:,0] + BH_pred[:,1] + BH_pred[:,2] - BH_pred[:,3])
    percent_suceptible_DA = DA_pred[:,0] / (DA_pred[:,0] + DA_pred[:,1] + DA_pred[:,2] - DA_pred[:,3])


    containment_R0_sup_DE = 1 / percent_suceptible_DE
    containment_R0_sup_BH = 1 / percent_suceptible_BH
    containment_R0_sup_DA = 1 / percent_suceptible_DA

    last_data_day = DATA.shape[0] - 1

    activity_increase_sup_DE = (containment_R0_sup_DE[1:] - R0_DE[1:]) / (np.max(R0_DE) - R0_DE[1:])
    activity_increase_sup_BH = (containment_R0_sup_BH[1:] - R0_BH[1:]) / (np.max(R0_BH) - R0_BH[1:])
    activity_increase_sup_DA = (containment_R0_sup_DA[1:] - R0_DA[1:]) / (np.max(R0_DA) - R0_DA[1:])


    DE_pred_no_social_distancing = no_social_distancing_model.eval(DE_params, t=time_frame[last_data_day:], x0=DATA[-1])
    BH_pred_no_social_distancing = no_social_distancing_model.eval(BH_params, t=time_frame[last_data_day:], x0=DATA[-1])
    DA_pred_no_social_distancing = no_social_distancing_model.eval(DA_params, t=time_frame[last_data_day:], x0=DATA[-1])

    print "Total  Number of Infected DE without SD:\t\t%.1f" % (DE_pred_no_social_distancing[-1,2] + DE_pred_no_social_distancing[-1,3])
    print "Total Number of Dead DE without SD:\t\t\t%.1f\n" % DE_pred_no_social_distancing[-1,3]

    print "Total  Number of Infected BH without SD:\t\t%.1f" % (BH_pred_no_social_distancing[-1,2] + BH_pred_no_social_distancing[-1,3])
    print "Total Number of Dead BH without SD:\t\t\t%.1f\n" % BH_pred_no_social_distancing[-1,3]

    print "Total  Number of Infected DA without SD:\t\t%.1f" % (DA_pred_no_social_distancing[-1,2] + DA_pred_no_social_distancing[-1,3])
    print "Total Number of Dead DA without SD:\t\t\t%.1f\n" % DA_pred_no_social_distancing[-1,3]

    delta_range = np.linspace(0.0, 1.0, 100)
    DE_pandemic_end_time = np.empty(delta_range.shape)
    DE_total_deaths = np.empty(delta_range.shape)
    DE_total_infections = np.empty(delta_range.shape)
    BH_pandemic_end_time = np.empty(delta_range.shape)
    BH_total_deaths = np.empty(delta_range.shape)
    BH_total_infections = np.empty(delta_range.shape)
    DA_pandemic_end_time = np.empty(delta_range.shape)
    DA_total_deaths = np.empty(delta_range.shape)
    DA_total_infections = np.empty(delta_range.shape)
    for i in range(delta_range.shape[0]):
        DE_reactive_model = Model(SIRD_kinetic_seasonal_reactive, independent_vars=['t', 'x0'])
        BH_reactive_model = Model(SIRD_kinetic_seasonal_reactive, independent_vars=['t', 'x0'])
        DA_reactive_model = Model(SIRD_kinetic_seasonal_reactive, independent_vars=['t', 'x0'])
        DE_reactive_params = Parameters()
        BH_reactive_params = Parameters()
        DA_reactive_params = Parameters()
        DE_reactive_params.add_many(("delta", delta_range[i]),
                                    ("gamma_sum", DE_params["gamma_sum"]),
                                    ("gamma_partition", DE_params["gamma_partition"]),
                                    ("tau_gamma", DE_params["tau_gamma"]),
                                    ("mu_sum", DE_params["mu_sum"]),
                                    ("mu_partition", DE_params["mu_partition"]),
                                    ("tau_mu", DE_params["tau_mu"]))
        BH_reactive_params.add_many(("delta", delta_range[i]),
                                    ("gamma_sum", BH_params["gamma_sum"]),
                                    ("gamma_partition", BH_params["gamma_partition"]),
                                    ("tau_gamma", BH_params["tau_gamma"]),
                                    ("mu_sum", BH_params["mu_sum"]),
                                    ("mu_partition", BH_params["mu_partition"]),
                                    ("tau_mu", BH_params["tau_mu"]))
        DA_reactive_params.add_many(("delta", delta_range[i]),
                                    ("gamma_sum", DA_params["gamma_sum"]),
                                    ("gamma_partition", DA_params["gamma_partition"]),
                                    ("tau_gamma", DA_params["tau_gamma"]),
                                    ("mu_sum", DA_params["mu_sum"]),
                                    ("mu_partition", DA_params["mu_partition"]),
                                    ("tau_mu", DA_params["tau_mu"]))
        DE_reactive_pred = DE_reactive_model.eval(DE_reactive_params, t=time_frame[last_data_day:], x0=DATA[-1])
        DE_end_time = np.where(np.abs(DE_reactive_pred[:,1]) < 1.0)[0]
        DE_total_deaths[i] = DE_reactive_pred[-1,3]
        DE_total_infections[i] = DE_reactive_pred[-1,2] + DE_reactive_pred[-1,3]
        if DE_end_time.size > 0:
            DE_pandemic_end_time[i] = DE_end_time[0] + 1 + last_data_day
        else:
            DE_pandemic_end_time[i] = np.nan
        BH_reactive_pred = BH_reactive_model.eval(BH_reactive_params, t=time_frame[last_data_day:], x0=DATA[-1])
        BH_end_time = np.where(np.abs(BH_reactive_pred[:,1]) < 1.0)[0]
        BH_total_deaths[i] = BH_reactive_pred[-1,3]
        BH_total_infections[i] = BH_reactive_pred[-1,2] + BH_reactive_pred[-1,3]
        if BH_end_time.size > 0:
            BH_pandemic_end_time[i] = BH_end_time[0] + 1 + last_data_day
        else:
            BH_pandemic_end_time[i] = np.nan
        DA_reactive_pred = DA_reactive_model.eval(DA_reactive_params, t=time_frame[last_data_day:], x0=DATA[-1])
        DA_end_time = np.where(np.abs(DA_reactive_pred[:,1]) < 1.0)[0]
        DA_total_deaths[i] = DA_reactive_pred[-1,3]
        DA_total_infections[i] = DA_reactive_pred[-1,2] + DA_reactive_pred[-1,3]
        if DA_end_time.size > 0:
            DA_pandemic_end_time[i] = DA_end_time[0] + 1 + last_data_day
        else:
            DA_pandemic_end_time[i] = np.nan

    date_list = [pd.to_datetime("01/01/20", format="%m/%d/%y").date() + pd.Timedelta(days=int(i)) for i in time_frame]

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b")
    fig0, axs0 = plt.subplots(figsize=(5.5, 4.8), dpi=340)
    axs0.plot(date_list[:DATA.shape[0]], DE_pred[:DATA.shape[0],1], label="Differential Evolution")
    axs0.plot(date_list[:DATA.shape[0]], BH_pred[:DATA.shape[0],1], label="Basin Hopping")
    axs0.plot(date_list[:DATA.shape[0]], DA_pred[:DATA.shape[0],1], label="Dual Annealing")
    axs0.bar(date_list[:DATA.shape[0]], DATA[:,1], label="DATA", color='m', alpha=0.5)
    axs0.set_title("Confirmed COVID-19 Active Cases in Rio de Janeiro")
    axs0.set_ylabel("Absolute Number of Active Cases")
    axs0.grid(True, linestyle=":")
    axs0.legend(loc="best")
    axs0.xaxis.set_major_locator(locator)
    axs0.xaxis.set_major_formatter(fmt)

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b")
    fig1, axs1 = plt.subplots(figsize=(5.5, 4.8), dpi=340)
    axs1.plot(date_list[:DATA.shape[0]], DE_pred[:DATA.shape[0],2], label="Differential Evolution")
    axs1.plot(date_list[:DATA.shape[0]], BH_pred[:DATA.shape[0],2], label="Basin Hopping")
    axs1.plot(date_list[:DATA.shape[0]], DA_pred[:DATA.shape[0],2], label="Dual Annealing")
    axs1.bar(date_list[:DATA.shape[0]], DATA[:,2], label="DATA", color='m', alpha=0.5)
    axs1.set_title("Confirmed COVID-19 Recoveries in Rio de Janeiro")
    axs1.set_ylabel("Absolute Number of Recoveries")
    axs1.grid(True, linestyle=":")
    axs1.legend(loc="best")
    axs1.xaxis.set_major_locator(locator)
    axs1.xaxis.set_major_formatter(fmt)

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b")
    fig2, axs2 = plt.subplots(figsize=(5.5, 4.8), dpi=340)
    axs2.plot(date_list[:DATA.shape[0]], DE_pred[:DATA.shape[0],3], label="Differential Evolution")
    axs2.plot(date_list[:DATA.shape[0]], BH_pred[:DATA.shape[0],3], label="Basin Hopping")
    axs2.plot(date_list[:DATA.shape[0]], DA_pred[:DATA.shape[0],3], label="Dual Annealing")
    axs2.bar(date_list[:DATA.shape[0]], DATA[:,3], label="DATA", color='m', alpha=0.5)
    axs2.set_title("Confirmed COVID-19 Deaths in Rio de Janeiro")
    axs2.set_ylabel("Absolute Number of Deaths")
    axs2.grid(True, linestyle=":")
    axs2.legend(loc="best")
    axs2.xaxis.set_major_locator(locator)
    axs2.xaxis.set_major_formatter(fmt)

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b")
    fig3, axs3 = plt.subplots(figsize=(5.5, 4.8), dpi=340)
    axs3.plot(date_list, DE_pred[:,1], label="Differential Evolution")
    axs3.plot(date_list, BH_pred[:,1], label="Basin Hopping")
    axs3.plot(date_list, DA_pred[:,1], label="Dual Annealing")
    axs3.set_title("Predicted COVID-19 Active Cases in Rio de Janeiro")
    axs3.set_ylabel("Absolute Number of Active Cases")
    axs3.grid(True, linestyle=":")
    axs3.legend(loc="best")
    axs3.xaxis.set_major_locator(locator)
    axs3.xaxis.set_major_formatter(fmt)

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b")
    fig4, axs4 = plt.subplots(figsize=(5.5, 4.8), dpi=340)
    axs4.plot(date_list, DE_pred[:,3], label="Differential Evolution")
    axs4.plot(date_list, BH_pred[:,3], label="Basin Hopping")
    axs4.plot(date_list, DA_pred[:,3], label="Dual Annealing")
    axs4.set_title("Predicted COVID-19 Deaths in Rio de Janeiro")
    axs4.set_ylabel("Absolute Number of Deaths")
    axs4.grid(True, linestyle=":")
    axs4.legend(loc="best")
    axs4.xaxis.set_major_locator(locator)
    axs4.xaxis.set_major_formatter(fmt)


    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b")
    fig5, axs5 = plt.subplots(figsize=(5.5, 4.8), dpi=340)
    axs5.plot(date_list, seasonal_beta(time_frame, kinetic_beta(time_frame, DE_params["beta0"], DE_params["tau_beta"]), DE_params["beta0_mag"]), label="Differential Evolution")
    axs5.plot(date_list, seasonal_beta(time_frame, kinetic_beta(time_frame, BH_params["beta0"], BH_params["tau_beta"]), BH_params["beta0_mag"]), label="Basin Hopping")
    axs5.plot(date_list, seasonal_beta(time_frame, kinetic_beta(time_frame, DA_params["beta0"], DA_params["tau_beta"]), DA_params["beta0_mag"]), label="Dual Annealing")
    axs5.set_title("Infection Rate $\\beta$ Over Time Frame")
    axs5.set_ylabel("$\\beta$")
    axs5.grid(True, linestyle=":")
    axs5.legend(loc="best")
    axs5.xaxis.set_major_locator(locator)
    axs5.xaxis.set_major_formatter(fmt)

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b")
    fig6, axs6 = plt.subplots(figsize=(5.5, 4.8), dpi=340)
    axs6.plot(date_list, kinetic_gamma(time_frame, DE_params["gamma_sum"] * DE_params["gamma_partition"], DE_params["gamma_sum"] * (1 - DE_params["gamma_partition"]), DE_params["tau_gamma"]), label="Differential Evolution")
    axs6.plot(date_list, kinetic_gamma(time_frame, BH_params["gamma_sum"] * BH_params["gamma_partition"], BH_params["gamma_sum"] * (1 - BH_params["gamma_partition"]), BH_params["tau_gamma"]), label="Basin Hopping")
    axs6.plot(date_list, kinetic_gamma(time_frame, DA_params["gamma_sum"] * DA_params["gamma_partition"], DA_params["gamma_sum"] * (1 - DA_params["gamma_partition"]), DA_params["tau_gamma"]), label="Dual Annealing")
    axs6.set_title("Recovery Rate $\gamma$ Over Time Frame")
    axs6.set_ylabel("$\gamma$")
    axs6.grid(True, linestyle=":")
    axs6.legend(loc="best")
    axs6.xaxis.set_major_locator(locator)
    axs6.xaxis.set_major_formatter(fmt)

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b")
    fig7, axs7 = plt.subplots(figsize=(5.5, 4.8), dpi=340)
    axs7.plot(date_list, kinetic_mu(time_frame, DE_params["mu_sum"] * DE_params["mu_partition"], DE_params["mu_sum"] * (1 - DE_params["mu_partition"]), DE_params["tau_mu"]), label="Differential Evolution")
    axs7.plot(date_list, kinetic_mu(time_frame, BH_params["mu_sum"] * BH_params["mu_partition"], BH_params["mu_sum"] * (1 - BH_params["mu_partition"]), BH_params["tau_mu"]), label="Basin Hopping")
    axs7.plot(date_list, kinetic_mu(time_frame, DA_params["mu_sum"] * DA_params["mu_partition"], DA_params["mu_sum"] * (1 - DA_params["mu_partition"]), DA_params["tau_mu"]), label="Dual Annealing")
    axs7.set_title("Mortality Rate $\mu$ Over Time Frame")
    axs7.set_ylabel("$\mu$")
    axs7.grid(True, linestyle=":")
    axs7.legend(loc="best")
    axs7.xaxis.set_major_locator(locator)
    axs7.xaxis.set_major_formatter(fmt)

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b")
    fig8, axs8 = plt.subplots(figsize=(5.5, 4.8), dpi=340)
    axs8.plot(date_list, R0_DE, label="Differential Evolution")
    axs8.plot(date_list, R0_BH, label="Basin Hopping")
    axs8.plot(date_list, R0_DA, label="Dual Annealing")
    axs8.set_title("$R_0(t)$ Values Over Time Frame")
    axs8.xaxis.set_major_locator(locator)
    axs8.xaxis.set_major_formatter(fmt)
    axs8.set_ylabel("$R_0$")
    axs8.grid(True, linestyle=":")
    axs8.legend(loc="best")


    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b")
    fig9, axs9 = plt.subplots(figsize=(5.5, 4.8), dpi=340)
    axs9.plot(date_list[(last_data_day-1):np.where(DE_pred[:,1] < 1.0)[0][0]], activity_increase_sup_DE[(last_data_day-1):np.where(DE_pred[:,1] < 1.0)[0][0]], label="Differential Evolution")
    axs9.plot(date_list[(last_data_day-1):np.where(BH_pred[:,1] < 1.0)[0][0]], activity_increase_sup_BH[(last_data_day-1):np.where(BH_pred[:,1] < 1.0)[0][0]], label="Basin Hopping")
    axs9.plot(date_list[(last_data_day-1):np.where(DA_pred[:,1] < 1.0)[0][0]], activity_increase_sup_DA[(last_data_day-1):np.where(DA_pred[:,1] < 1.0)[0][0]], label="Dual Annealing")
    axs9.set_title("Maximum Activity $\%$ Increase for Virus Suppression")
    axs9.xaxis.set_major_locator(locator)
    axs9.xaxis.set_major_formatter(fmt)
    axs9.set_ylabel("Activity $\%$ Increase")
    axs9.grid(True, linestyle=":")
    axs9.legend(loc="best")
    plt.tight_layout()

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b")
    DE_mask = np.isfinite(DE_pandemic_end_time)
    BH_mask = np.isfinite(BH_pandemic_end_time)
    DA_mask = np.isfinite(DA_pandemic_end_time)
    fig10, axs10 = plt.subplots(figsize=(5.5, 4.8), dpi=340)
    axs10.plot(DE_pandemic_end_time[DE_mask], delta_range[DE_mask], label="Differential Evolution")
    axs10.plot(BH_pandemic_end_time[BH_mask], delta_range[BH_mask], label="Basin Hopping")
    axs10.plot(DA_pandemic_end_time[DA_mask], delta_range[DA_mask], label="Dual Annealing")
    axs10.set_ylabel("$\delta$ Margin")
    axs10.set_xlabel("Ending Time")
    axs10.set_ylim(0.0, 1.0)
    axs10.xaxis.set_major_locator(locator)
    axs10.xaxis.set_major_formatter(fmt)
    axs10.set_title("Containment Margin According to Pandemic Ending Time")
    axs10.grid(True, linestyle=":")
    axs10.legend(loc="best")
    plt.tight_layout()

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b")
    fig11, axs11 = plt.subplots(figsize=(5.5, 5.0), dpi=340)
    axs11.plot(delta_range, DE_total_infections, label="Differential Evolution")
    axs11.plot(delta_range, BH_total_infections, label="Basin Hopping")
    axs11.plot(delta_range, DA_total_infections, label="Dual Annealing")
    axs11.set_title("Total Infections by Margin $\delta$")
    axs11.set_ylabel("Total Infections")
    axs11.set_xlabel("Containment Margin $\delta$")
    axs11.grid(True, linestyle=":")
    axs11.legend(loc="best")
    plt.tight_layout()

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b")
    fig12, axs12 = plt.subplots(figsize=(5.5, 5.0), dpi=340)
    axs12.plot(delta_range, DE_total_deaths, label="Differential Evolution")
    axs12.plot(delta_range, BH_total_deaths, label="Basin Hopping")
    axs12.plot(delta_range, DA_total_deaths, label="Dual Annealing")
    axs12.set_title("Total Deaths by Margin $\delta$")
    axs12.set_ylabel("Total Deaths")
    axs12.set_xlabel("Containment Margin $\delta$")
    axs12.grid(True, linestyle=":")
    axs12.legend(loc="best")
    plt.tight_layout()

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b")
    fig13, axs13 = plt.subplots(figsize=(5.5, 4.8), dpi=340)
    axs13.plot(date_list[last_data_day:], DE_pred_no_social_distancing[:,1], label="Differential Evolution")
    axs13.plot(date_list[last_data_day:], BH_pred_no_social_distancing[:,1], label="Basin Hopping")
    axs13.plot(date_list[last_data_day:], DA_pred_no_social_distancing[:,1], label="Dual Annealing")
    axs13.set_title("Predicted Active Cases No Social Distancing")
    axs13.set_ylabel("Absolute Number of Active Cases")
    axs13.xaxis.set_major_locator(locator)
    axs13.xaxis.set_major_formatter(fmt)
    axs13.grid(True, linestyle=":")
    axs13.legend(loc="best")
    plt.tight_layout()

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b")
    fig14, axs14 = plt.subplots(figsize=(5.5, 4.8), dpi=340)
    axs14.plot(date_list[last_data_day:], DE_pred_no_social_distancing[:,3], label="Differential Evolution")
    axs14.plot(date_list[last_data_day:], BH_pred_no_social_distancing[:,3], label="Basin Hopping")
    axs14.plot(date_list[last_data_day:], DA_pred_no_social_distancing[:,3], label="Dual Annealing")
    axs14.set_title("Predicted Deaths No Social Distancing")
    axs14.set_ylabel("Absolute Number of Deaths")
    axs14.xaxis.set_major_locator(locator)
    axs14.xaxis.set_major_formatter(fmt)
    axs14.grid(True, linestyle=":")
    axs14.legend(loc="best")


    plt.tight_layout()
    plt.show()
