#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model'))

print(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model'))
dir_path = os.path.dirname(os.path.realpath(__file__))


import itertools
import copy
from simulator import DigitalSimulator

import math
from multiprocessing import Pool
import multiprocessing as mp

from itertools import repeat
import numpy as np
import time
import argparse
import json

from simulation_functions import get_shape_matrix, get_node_coordinates
import fitting_functions as ff
import matplotlib.pyplot as plt

## 1536 well plate
environment_size = (39, 39)
plate_width = 35
w = plate_width / environment_size[0] # inter-well spacing in mm
w = 0.9
points_per_well = int(4.5/w)
conc = 7.5/1000


#activations = [['TH']]
#thresholds = [[3,3]]


A, bound = get_shape_matrix(environment_size[0], environment_size[1], environment_size[0]//2)

def laplace(x):

    return np.matmul(A,x.flatten()).reshape(x.shape)

def run_sim(inducer_inds, r_activations, r_logic_gates, r_thresholds):
    '''
    Wrapper to run sims in parrallel
    :param IPTG_inds:
    :param receiver_info:
    :return:
    '''



    A, bound = get_shape_matrix(environment_size[0], environment_size[1], environment_size[0] // 2)


    def laplace(x):
        return np.matmul(A, x.flatten()).reshape(x.shape)

    #receiver_coords = simulator.get_colony_coords(opentron_inds = [[3,3] for i in range(len(receiver_info['activations']))])  #these are lawns anyway so position doesnt matter
    receiver_coords = simulator.get_colony_coords([3,3]) * len(r_activations)  #these are lawns anyway so position doesnt matter

    inducer_coords = simulator.opentron_to_coords(inducer_inds)

    # make a plate for every reciever-input combination, need different plates for each receiver for lawn simulations
    ''' the make plate function needs to be written for each specific system '''

    plates = ff.make_plates(receiver_coords, r_activations, inducer_coords, conc, environment_size, w, laplace)


    # some positions might have been mutated to have an index on the corner so need to check
    corners = [[0, 0], [0, max_ind-1], [max_ind-1, 0],
               [max_ind-1, max_ind-1]]  # cant put IPTG in the corners of the allowable square


    on_corner = np.any(list(np.all(i == j) for i in corners for j in inducer_inds))


    if not on_corner:
        inducer_coords = np.array([[simulator.opentron_to_coords(ind)] for ind in inducer_inds]).reshape(-1, 2)
        print(r_logic_gates)
        all_sims = simulator.run_sims(plates)

        score, t, best_receiver_coords = simulator.max_fitness_over_t(all_sims, inducer_coords, r_logic_gates, r_thresholds, test_t=-1)
    else:
        coords = -1
        score = -1
        best_receiver_pos = -1
        t = -1
    print({'score':score, 'receiver_coords': best_receiver_coords, 'inducer_coords': inducer_coords, 't':t})
    return {'score':score, 'receiver_coords': best_receiver_coords, 'inducer_coords': inducer_coords, 't':t}



parser = argparse.ArgumentParser(description='Run the colony placement algorithm')
parser.add_argument('in_path', metavar='in_path', type=str, nargs=1, help='the path of the saved output from macchiato')
parser.add_argument('--outpath', type=str, help='the filepath to save output in, default is colony_placement/output')

if __name__ == '__main__':

    '''------- Simulation setup--------'''
    args = parser.parse_args()

    in_path = args.in_path[0]

    out_path = args.outpath
    if out_path is None:
        out_path = os.path.join(dir_path, 'output')
    os.makedirs(out_path, exist_ok=True)

    n_inputs = 3
    min_distance = 4.5
    conc = 7.5 / 1000

    macchiato_results = json.load(open(in_path))

    r_logic_gates = []
    r_activations = []
    r_thresholds = [] #TODO:: let user specify these, currently these are fold change thresholds
    for act in macchiato_results['logic_gates'].keys():

        for lg in macchiato_results['logic_gates'][act]:
            r_activations.append(act)
            ''' ---- SET THRESHOLDS HERE-----'''
            if act == 'BP':
                r_thresholds.append([5,5])
            elif act == 'TH':
                r_thresholds.append([5,5])
            r_logic_gates.append(lg)

    print(r_logic_gates, r_activations, r_thresholds)

    all_outputs = list(map(np.array, list(itertools.product([0, 1], repeat=n_inputs))))

    max_score = 0
    max_coords = np.array([[0, 0]])
    max_t = -1


    dt = 60 * 20  # sample time of simulations in minutes

    simulator = DigitalSimulator(conc, environment_size, w, dt, environment_bound=bound, laplace=laplace,
                                 colony_radius=1000)

    all_indices = simulator.get_opentron_indices()

    pop_size = 10
    n_gens = 3
    plot = False
    max_ind = 6

    IPTG_is = np.random.choice(len(all_indices), size=(pop_size, 3)) # three inputs for each
    IPTG_inds = np.array(all_indices)[IPTG_is] # (pop_size, 3, 2), the opentron indeices


    simulated = np.zeros((pop_size)) # keep track of grids that have been simulated so we dont do them again
    scores = np.zeros((pop_size))
    max_receiver_pos = np.array(np.ones((len(r_activations), 2 ))*7) #TODO: set this to 0,0 after testing

    for gen in range(n_gens):

        # 0 IPTG
        # 1 Nutrients
        # 2 Receiver
        # 3 GFP

        n_cores = int(mp.cpu_count())
        ti = time.time()

        print(r_logic_gates)
        pool = Pool(5)
        #results = pool.starmap(run_sim, zip(IPTG_inds, repeat(receiver_info)))
        results = [run_sim(IPTG_inds[i], r_activations, r_logic_gates, r_thresholds) for i in range(len(IPTG_inds))] # for debugging
        pool.close()
        for i, result in enumerate(results):
            score = result['score']
            best_receiver_pos = result['receiver_coords']
            coords = result['inducer_coords']
            t = result['t']
            scores[i] = score
            if score > max_score:
                max_coords = coords
                max_score = score
                max_t = t
                max_receiver_pos = best_receiver_pos

        print()
        print('Generation:', gen)
        print('simulation:', time.time() - ti)

        ind = np.argsort(scores)[::-1]
        scores = scores[ind]
        IPTG_inds = IPTG_inds[ind]
        simulated = simulated[ind]
        new_IPTG_inds = copy.deepcopy(IPTG_inds)


        print(scores)
        #mutate top 45% and ad them to bottom 45%
        for i in range(0, int(pop_size*0.45)):
            scores[int(pop_size*0.45) + i] = scores[i] # so these grids dont get reinitilised if parent doesnt have 0 score
            simulated[int(pop_size*0.45) + i] = 0
            new_IPTG_inds[int(pop_size*0.45) + i] = IPTG_inds[i] + np.random.randint(-1, 1, size = ( n_inputs, 2))
            new_IPTG_inds[int(pop_size*0.45) + i] = np.clip(new_IPTG_inds[i], 0, max_ind)

        for i in range(pop_size): # reinitilise grids with 0 score
            if scores[i] <=0:
                simulated[i] = 0
                IPTG_is = np.random.choice(len(all_indices), size=(1, 3))  # three inputs for each
                new_IPTG_inds[i] = np.array(all_indices)[IPTG_is]  # (pop_size, 3, 2), the opentron indeices


        # add random grids for bottom 10%
        for i in range(int(pop_size*0.9), pop_size):
            simulated[i] = 0
            IPTG_is = np.random.choice(len(all_indices), size=(1, 3))  # three inputs for each
            new_IPTG_inds[i] = np.array(all_indices)[IPTG_is]  # (pop_size, 3, 2), the opentron indeice

        IPTG_inds = copy.deepcopy(new_IPTG_inds)

        print("gen : " + str(gen) + ", av score: ", np.mean(scores), " top score: ", np.max(scores))
        print('IPTG inds:', IPTG_inds[0])
        print('max receiver ind:', np.array(max_receiver_pos-simulator.opentron_origin)/points_per_well)
        print(scores)

    receiver_inds = (np.array(max_receiver_pos-simulator.opentron_origin)/int(points_per_well)).astype(np.int32)
    IPTG_inds = np.array(IPTG_inds[0], dtype = np.int32)
    print(max_receiver_pos)
    print(receiver_inds)
    print(IPTG_inds)


    save_data = {'receiver_inds': receiver_inds.tolist(), 'IPTG_inds': IPTG_inds.tolist(), 'activations': r_activations, 'logic_gates': r_logic_gates, 'thresholds':r_thresholds, 'score': np.max(scores)}

    json.dump(save_data, open(os.path.join(out_path, 'placement.json'), 'w+'))











