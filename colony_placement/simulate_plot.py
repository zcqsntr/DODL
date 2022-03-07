
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model'))
dir_path = os.path.dirname(os.path.realpath(__file__))


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


    #score, t, best_receiver_pos, all_sims = simulator.max_fitness_over_t(receiver_coords, coords,thresholds,logic_gates, activations,test_t=-1, plot = False)



    receiver_radius = 2
    ISs = ['000', '001', '010', '011', '100', '101', '110', '111']
    for i, rec_ind in enumerate(rec_inds):
        rec_coords = start_coords + rec_ind * points_per_well
        #receiver_coords = get_node_coordinates(rec_coords, receiver_radius, environment_size[0],environment_size[1], w)
        #receiver_coords = [receiver_coords] * len(activations)
        all_sims = simulator.run_sims(coords, receiver_coords[i], bandpass=activations[i] == 'BP', plot = False)

        logic_area = simulator.get_logic_area(logic_gates[i], all_sims, thresholds[i], -1)

        max_r, best_position = simulator.get_max_r(logic_area, inducer_coords=coords)

        receiver_coords = get_node_coordinates(rec_coords, receiver_radius, environment_size[0],environment_size[1], w)
        receiver_coords = receiver_coords
        for i, coord in enumerate(coords):
            logic_area[coord[0], coord[1]] = i + 2

        logic_area[rec_coords[:, 0], rec_coords[:, 1]] = i + 3

        plt.figure()
        logic_area[np.where(bound == -1)] = -1
        plt.imshow(logic_area)



    plt.show()
