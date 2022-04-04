
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model'))
dir_path = os.path.dirname(os.path.realpath(__file__))
import fitting_functions as ff
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import itertools
import copy
from simulator import DigitalSimulator

import math
from multiprocessing import Pool

from itertools import repeat
import numpy as np
import time
import argparse
import json

from simulation_functions import get_shape_matrix, run_sim, get_node_coordinates


parser = argparse.ArgumentParser(description='Simulate and plot a given placement')
parser.add_argument('in_file', metavar='T', type=str, nargs=1, help='the path of the saved placement of colonies and inducers')
parser.add_argument('--outpath', type=str, help='the filepath to save output in, default is colony_placement/plot')


## 1536 well plate
environment_size = (39, 39)
plate_width = 35
w = plate_width / environment_size[0] # inter-well spacing in mm
w = 0.9
points_per_well = int(4.5/w)
conc = 5
dt = 10  # sample time of simulations in minutes
def laplace(x):

    return np.matmul(A,x.flatten()).reshape(x.shape)


def normalise_GFP(all_sims):
    all_processed_GFPs = []

    for i, sims in enumerate(all_sims):  # for each receiver
        processed_GFPs = []
        for j in range(2 ** n_inputs):
            sim_ivp = sims[j]

            GFP = sim_ivp[3]
            # plt.figure()
            # plt.imshow(sim_ivp[1, :, :, -1])
            t_series = []

            for t in range(GFP.shape[2]):
                receiver_GFP = GFP[:, :, t]
                r_c = receiver_coords[i]
                t_series.append(np.mean(np.sort(receiver_GFP[np.min(r_c[:, 0]): np.max(r_c[:, 0]) + 1,
                                                np.min(r_c[:, 1]): np.max(r_c[:, 1]) + 1].flatten())[-7:]))
            processed_GFPs.append(t_series)

        all_processed_GFPs.append(processed_GFPs)
    return all_processed_GFPs


def plot_timecourses(all_processed_GFPs):
    for i, processed_GFPs in enumerate(all_processed_GFPs):  # for each receiver

        timecourse_fig = plt.figure()
        timecourse_ax = timecourse_fig.add_subplot(111)


        for j in range(2 ** n_inputs):

            timecourse_ax.plot(np.arange(len(processed_GFPs[j])) / 600, processed_GFPs[j], colours[j],
                               label=str(ISs[j]))
            # plt.ylim(top = 0.6, bottom = -0.01)
            timecourse_ax.set_xlabel('Time (h)')
            timecourse_ax.set_ylabel('GFP mean pixel value per well')

        plt.legend()

    timecourse_fig.savefig('timecourse.png', dpi=300)

def plot_barchart(all_processed_GFPs):
    print(np.array(all_processed_GFPs).shape)


    for processed_GFPs in np.array(all_processed_GFPs):
        end_GFPs = processed_GFPs[:, -1]
        '''
        plt.figure()

        

        # plt.ylim(top=np.max(end_GFPs), bottom=0)
        # end_GFPs.reverse()
        plt.bar(range(2 ** n_inputs), end_GFPs)

        plt.xticks(range(2 ** n_inputs), ISs)

        plt.ylabel('GFP mean pixel value per well at {}hrs'.format(bar_chart_t))

        # plt.legend()
        plt.savefig('bar_chart.png', dpi=300)
        '''

        plt.figure()

        # plt.ylim(top=np.max(end_GFPs), bottom=0)
        # end_GFPs.reverse()
        plt.bar(range(2 ** n_inputs), np.array(end_GFPs) / end_GFPs[0])

        plt.xticks(range(2 ** n_inputs), ISs)

        plt.ylabel('GFP fold change {}hrs'.format(bar_chart_t))

        # plt.legend()
        plt.savefig('fold_change.png', dpi=300)

if __name__ == '__main__':


    args = parser.parse_args()

    in_file = args.in_file[0]
    outpath = args.outpath

    if outpath is None:
        outpath = os.path.join(dir_path, 'output')

    if in_file is None:
        in_file = os.path.join(os.path.join(os.path.join(os.path.dirname(dir_path), 'colony_placement'), 'output'),
                               'placement.json')

    os.makedirs(outpath, exist_ok=True)



    data = json.load(open(in_file))
    print(data)
    receiver_inds = np.array(data['receiver_inds'])
    IPTG_inds = np.array(data['IPTG_inds'])
    activations = data['activations']

    A, bound = get_shape_matrix(environment_size[0], environment_size[1], environment_size[0] // 2)
    simulator = DigitalSimulator(conc, environment_size, w, dt, laplace=laplace)
    simulator.bound = bound

    activations = data['activations']

    ind0, ind1, ind2 = np.array(data['IPTG_inds'])
    n_inputs = len(data['IPTG_inds'])
    start_coords = np.array([[7, 7]])

    inducer_coords = np.array(
        [[start_coords + ind0 * points_per_well], [start_coords + ind1 * points_per_well],
         [start_coords + ind2 * points_per_well]]).reshape(3, 2)


    #score, t, best_receiver_pos, all_sims = simulator.max_fitness_over_t(receiver_coords, coords,thresholds,logic_gates, activations,test_t=-1, plot = False)



    receiver_radius = 2

    rec_coords = start_coords + receiver_inds * points_per_well
    receiver_pos = rec_coords * w
    receiver_coords = [
        get_node_coordinates(np.array([rp]), receiver_radius, environment_size[0], environment_size[1], w) for rp in
        receiver_pos]

    # plot the inducer and receiver locations
    grid = np.zeros(environment_size)

    grid[np.where(bound == -1)] = -1

    for i, coord in enumerate(inducer_coords):
        grid[coord[0], coord[1]] = i + 1

    for j, rc in enumerate(receiver_coords):
        grid[rc[:, 0], rc[:, 1]] = i + j + 2

    im = plt.imshow(grid)
    plt.colorbar(im)



    all_sims = []
    for i in range(len(activations)):
        sims = simulator.run_sims( inducer_coords, receiver_coords[i], activations[i] == 'BP', t_final = 20*60, growth_delay=0*60)

        all_sims.append(sims)


    if n_inputs == 2:
        ISs = ['00', '01', '10', '11']
    elif n_inputs == 3:
        ISs = ['000', '001', '010', '011', '100', '101', '110', '111']
    colours = ['purple', 'green', 'red', 'steelblue', 'orange', 'blue', 'black', 'yellow']

    bar_chart_t = 20  # hours
    
    all_processed_GFPs = normalise_GFP(all_sims)

    plot_timecourses(all_processed_GFPs)

    plot_barchart(all_processed_GFPs)

    plt.show()


