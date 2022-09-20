#!/usr/bin/env python

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model'))

print(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model'))
dir_path = os.path.dirname(os.path.realpath(__file__))

import itertools
import copy
from model.simulator import DigitalSimulator

import math
from multiprocessing import Pool

from itertools import repeat

import numpy as np
import time
import argparse
import json

from model.simulation_functions import get_shape_matrix, get_node_coordinates

## 1536 well plate
environment_size = (39, 39)
plate_width = 35
w = plate_width / environment_size[0]  # inter-well spacing in mm
w = 0.9
points_per_well = int(4.5 / w)
conc = 7.5 / 1000

# activations = [['TH']]
# thresholds = [[3,3]]


A, bound = get_shape_matrix(environment_size[0], environment_size[1], environment_size[0] // 2)


def laplace(x):
    return np.matmul(A, x.flatten()).reshape(x.shape)


def run_sim(indices, receiver_coords, thresholds, logic_gates, activations):
    ind0, ind1, ind2 = indices
    simulator = DigitalSimulator(conc, environment_size, w, dt, laplace=laplace)
    simulator.bound = bound
    inputs_diff = np.any(ind0 != ind1) and np.any(ind1 != ind2) and np.any(ind0 != ind2)
    corners = [[0, 0], [0, max_ind - 1], [max_ind - 1, 0],
               [max_ind - 1, max_ind - 1]]  # cant put IPTG in the corners of the allowable square

    on_corner = np.any(list(np.all(i == j) for i in corners for j in indices))
    ti = time.time()
    if inputs_diff and not on_corner:
        coords = np.array(
            [[start_coords + ind * points_per_well] for ind in IPTG_inds]).reshape(-1, 2)

        score, t, best_receiver_pos, all_sims = simulator.max_fitness_over_t(receiver_coords, coords, thresholds,
                                                                             logic_gates, activations, test_t=-1)
    else:
        coords = -1
        score = -1
        best_receiver_pos = -1
        t = -1

    return {'score': score, 'receiver_pos': best_receiver_pos, 'coords': coords, 't': t}


parser = argparse.ArgumentParser(description='Run the colony placement algorithm')
parser.add_argument('in_path', metavar='in_path', type=str, nargs=1, help='the path of the saved output from macchiato')
parser.add_argument('--outpath', type=str, help='the filepath to save output in, default is colony_placement/output')

# receiver_pos = [[int(environment_size[0] / 2), int(environment_size[1] / 2)]]
# print(receiver_pos)
if __name__ == '__main__':
    args = parser.parse_args()

    in_path = args.in_path[0]

    out_path = args.outpath
    if out_path is None:
        out_path = os.path.join(dir_path, 'output')
    os.makedirs(out_path, exist_ok=True)

    macchiato_results = json.load(open(in_path))

    logic_gates = []

    activations = []
    thresholds = []  # TODO:: let user specify these
    for act in macchiato_results['logic_gates'].keys():

        for lg in macchiato_results['logic_gates'][act]:
            activations.append(act)
            if act == 'BP':
                thresholds.append([2, 2])
            elif act == 'TH':
                thresholds.append([5, 5])
            logic_gates.append(lg)

    print(activations, logic_gates, thresholds)
    receiver_pos = [[int(environment_size[0] / 2), int(environment_size[1] / 2)]]
    # receiver_pos = [[19, 19]]
    receiver_radius = 1000
    receiver_coords = get_node_coordinates(receiver_pos, receiver_radius, environment_size[0], environment_size[1],
                                           w)  # lawn of receivers
    receiver_coords = [receiver_coords] * len(activations)
    # print(receiver_coords)
    n_inputs = 3
    min_distance = 4.5
    conc = 7.5 / 1000

    all_outputs = list(map(np.array, list(itertools.product([0, 1], repeat=n_inputs))))

    max_score = 0
    max_coords = np.array([[0, 0]])

    max_t = -1
    start_coords = np.array([[2, 7]])

    all_indices = []
    max_ind = 8
    dt = 60 * 20  # sample time of simulations in minutes

#     for i in range(max_ind):
#         for j in range(max_ind):
#             all_indices.append(np.array([i, j]))
    """
    only coordinates inside circle are included
    """
    for i in range(int(max_ind / 2)):
        for j in range(int(max_ind/2)):
            distance = ((i - max_ind / 2) ** 2 + (j - max_ind / 2) ** 2) ** (1 / 2)
            # if re_range[i][1] <= j <= re_range[i][2] or (re_range[i][1]-j)<=0.5 or (j-re_range[i][2])<=0.5 :
            if distance <= (max_ind / 2):
                all_indices.append(np.array([i, j]))
                all_indices.append(np.array([max_ind - 1 - i, j]))
                all_indices.append(np.array([i, max_ind - 1 - j]))
                all_indices.append(np.array([max_ind - 1 - i, max_ind - 1 - j]))



            elif distance - max_ind / 2 <= (2 ** (1 / 2)) / 2:
                if ((i - max_ind / 2) ** 2 + (j + 1 - max_ind / 2) ** 2) ** (1 / 2) - max_ind / 2 <= (2 ** (1 / 2)) / 2 or (
                        (i - max_ind / 2) ** 2 + (j + 1 - max_ind / 2) ** 2) ** (1 / 2) <= (max_ind / 2):
                    all_indices.append(np.array([i, j]))
                    all_indices.append(np.array([max_ind - 1 - i, j]))
                    all_indices.append(np.array([i, max_ind - 1 - j]))
                    all_indices.append(np.array([max_ind - 1 - i, max_ind - 1 - j]))


    pop_size = 1
    n_gens = 10
    plot = False

    IPTG_is = np.random.choice(len(all_indices), size=(pop_size, 3))  # three inputs for each
    IPTG_inds = np.array(all_indices)[IPTG_is]  # (pop_size, 3, 2), the opentron indeices

    simulated = np.zeros((pop_size))  # keep track of grids that have been simulated so we dont do them again
    scores = np.zeros((pop_size))
    max_receiver_pos = np.array(np.ones((len(activations), 2)) * 7)  # TODO: set this to 0,0 after testing

    for gen in range(n_gens):

        # 0 IPTG
        # 1 Nutrients
        # 2 Receiver
        # 3 GFP

        ti = time.time()

        indices = IPTG_inds

        pool = Pool(5)
        results = pool.starmap(run_sim, zip(indices, repeat(receiver_coords), repeat(thresholds), repeat(logic_gates),
                                            repeat(activations)))
        pool.close()
        for i, result in enumerate(results):
            score = result['score']
            best_receiver_pos = result['receiver_pos']
            coords = result['coords']
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
        # mutate top 45% and ad them to bottom 45%
        for i in range(0, int(pop_size * 0.45)):
            scores[int(pop_size * 0.45) + i] = scores[
                i]  # so these grids dont get reinitilised if parent doesnt have 0 score
            simulated[int(pop_size * 0.45) + i] = 0
            new_IPTG_inds[int(pop_size * 0.45) + i] = IPTG_inds[i] + np.random.randint(-1, 1, size=(n_inputs, 2))
            new_IPTG_inds[int(pop_size * 0.45) + i] = np.clip(new_IPTG_inds[i], 0, max_ind)

        for i in range(pop_size):  # reinitilise grids with 0 score

            if scores[i] <= 0:
                simulated[i] = 0
                IPTG_is = np.random.choice(len(all_indices), size=(1, 3))  # three inputs for each
                new_IPTG_inds[i] = np.array(all_indices)[IPTG_is]  # (pop_size, 3, 2), the opentron indeices

        # add random grids for bottom 10%
        for i in range(int(pop_size * 0.9), pop_size):
            simulated[i] = 0
            IPTG_is = np.random.choice(len(all_indices), size=(1, 3))  # three inputs for each
            new_IPTG_inds[i] = np.array(all_indices)[IPTG_is]  # (pop_size, 3, 2), the opentron indeices

        IPTG_inds = copy.deepcopy(new_IPTG_inds)

        print("gen : " + str(gen) + ", av score: ", np.mean(scores), " top score: ", np.max(scores))
        print('IPTG inds:', IPTG_inds[0])
        print('max receiver ind:', np.array(max_receiver_pos - start_coords) / points_per_well)
        print(scores)

    receiver_inds = (np.array(max_receiver_pos - start_coords) / int(points_per_well)).astype(np.int32)
    IPTG_inds = np.array(IPTG_inds[0], dtype=np.int32)
    print(max_receiver_pos)
    print(receiver_inds)
    print(IPTG_inds)

    save_data = {'receiver_inds': receiver_inds.tolist(), 'IPTG_inds': IPTG_inds.tolist(), 'activations': activations,
                 'logic_gates': logic_gates, 'thresholds': thresholds, 'score': np.max(scores)}

    json.dump(save_data, open(os.path.join(out_path, 'placement.json'), 'w+'))
