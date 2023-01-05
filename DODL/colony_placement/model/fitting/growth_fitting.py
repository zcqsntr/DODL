import numpy as np

import sys
import os
import matplotlib as mpl
mpl.use('tkagg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import copy
import math
import fitting_functions as ff
from scipy.optimize import curve_fit
import os

def gompertz(t, A, um, lam):

    return A* np.exp(-np.exp((um*np.e)/A*(lam - t) +1))


if __name__ == '__main__':

    TH_file = os.path.join('data', '20220704_data-processed-ZG.csv')
    BP_file = os.path.join('data', '20220711data-processed-ZBD.csv')
    TH_data = ff.load_data(TH_file)
    BP_data = ff.load_data(BP_file)
    colours = ['b', 'red', 'g']



    normed_data = BP_data

    #growth_data_TH = load_spot_data(filepath_growth_TH, n_points)
    #growth_data_BP = load_spot_data(filepath_growth_BP, n_points)

    IPTG_concs = [0.03, 0.015, 0.0075, 0.00375, 0.00188, 0.0]
    distances = [13.5, 12.7, 6.4, 4.5, 14.2, 10.1, 9.0, 16.2]


    all_normed_data = []
    all_times = []
    for IPTG_conc in IPTG_concs:
        #plt.figure()
        #plt.title('Threshold: ' + str(IPTG_conc))
        for i, distance in enumerate(distances):

            #normalised_growth_data[IPTG_conc][distance] = np.array(growth_data_BP[IPTG_conc][distance]).T - np.array(growth_data_BP[IPTG_conc][distance])[:,0] # normalise the first set of characterisation data

            #all_normed_data.extend(normalised_growth_data[IPTG_conc][distance].T) # first set
            all_normed_data.extend(np.array(normed_data[IPTG_conc][distance]['absorbance']))



    #plt.show()
    all_normed_data = np.array(all_normed_data)
    print(all_normed_data.shape)
    mean_normed_data = np.mean(all_normed_data, axis = 0)
    #print(all_normed_data.shape)


    #time_points = np.linspace(0, n_points*20, n_points)# first set of data (mins)
    n_points = len(TH_data[IPTG_concs[0]][distances[0]]['GFP'][0])
    all_times = np.arange(0, n_points*20, 20) # times in minutes
    print(all_times)
    popt,pcov = curve_fit(gompertz, all_times, mean_normed_data, p0 = (1.34750462e-01, 1.90257947e-04, 3.33841052e+02 ))

    print(popt)
    #popt = [1.24927088e-01, 1.79595968e-04, 3.48051876e+02]
    fitted_data = gompertz(all_times, popt[0], popt[1], popt[2])

    plt.figure()
    plt.plot(all_times, fitted_data, label='fit')
    for i in range(len(all_normed_data)):
        plt.scatter(all_times, all_normed_data[i], marker = '.', c = 'g')
    plt.legend()
    plt.xlabel('time (mins)')
    plt.ylabel('absorbance')
    print(fitted_data[0])
    plt.show()