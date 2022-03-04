
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

## 1536 well plate
environment_size = (39, 39)
plate_width = 35
w = plate_width / environment_size[0] # inter-well spacing in mm
w = 0.9
points_per_well = int(4.5/w)
conc = 5


#activations = [['TH']]
#thresholds = [[3,3]]


def get_vertex_coordinates(vertex_numbers, n_rows, n_cols):
    '''
    use to get grid coordinates of vertices

    args:
        vertex_numbers: the numbers of the vertices you want coordinates for 0 <= vertex_number < n_rows * n_cols
        n_rows, n_cols: number of rows and columns in the finite difference simulation, a total of n-rows*n_cols vertices

    returns:
        vertex_coordinates: the coordinates on the finite difference grid of the supplied vertex number: [[r0, c0]; [r1,c1]; ... [rn,cn]]
            these use matrix indexing, in the format (row, col) starting from the top left of the grid
    '''

    vertex_coordinates = np.hstack((vertex_numbers // n_cols, vertex_numbers % n_rows))

    return vertex_coordinates

def get_vertex_positions(vertex_numbers, n_rows, n_cols, w):
    '''
    use to get the positions (in mm) of vertices on the real grid

    args:
        vertex_numbers: the numbers of the vertices you want coordinates for 0 <= vertex_number < n_rows * n_cols
        n_rows, n_cols: number of rows and columns in the finite difference simulation, a total of n-rows*n_cols vertices
        w: the distance between finite difference vertices
    returns:
        vertex_positions: the positions on the finite difference grid of the supplied vertex number (in mm from the top left of the grid):
            [[r0, c0]; [r1,c1]; ... [rn,cn]]
    '''


    vertex_coordinates = get_vertex_coordinates(vertex_numbers, n_rows, n_cols)

    vertex_positions = vertex_coordinates * w

    return vertex_positions


def assign_vertices(vertex_positions, node_positions, node_radius):
    '''
    assigns vertices to be part of nodes in node_positions with radius: node radius.


    args:
        vertex_positions: the positions of the vertices to be tested
        node_positions, node_radius: positions and radius of the nodes we want vertices for
    returns:
        vertex_numbers: the numbers of the vertices that are within on of the nodes
        indicators: vector with an index for each vertex indicating whether it is inside a node (value = 1) or outside all nodes (value = 0)

     NOTE: this assigns position based on real life position, not the grid coordinates i.e the distance in mm
    '''


    indicators = np.zeros(len(vertex_positions))

    if node_positions == []:
        return [], indicators

    if node_positions[0] is not None:
        node_positions = np.array(node_positions)
        differences = vertex_positions - node_positions[:, None]

        vertex_numbers = np.where(np.linalg.norm(differences, axis=2) < node_radius)[1].reshape(-1, 1)

        indicators[vertex_numbers] = 1

    indicators = np.array(indicators, dtype=np.int32)



    return vertex_numbers, indicators

# this is the only one you really need to use
def get_node_coordinates(node_positions, node_radius, n_rows, n_cols, w):
    '''
       gets the coordinates of the vertices inside the nodes with position node_positions with radius: node radius.

       args:
           vertex_positions: the positions of the vertices to be tested
           node_positions, node_radius: positions and radius of the nodes we want vertices for
           n_rows, n_cols: the number of rows and cols on the finite difference grid
       returns:
           coordinates: the coordinates of the vertices that are within on of the nodes

        NOTE: this assigns position based on real life position, not the grid coordinates i.e the distance in mm
       '''


    # use the individual functions if repeating these two lines for each node type is too slow
    all_vertex_numbers = np.arange(n_rows * n_cols).reshape(-1, 1)  # reshpae to colum vector
    all_vertex_positions = get_vertex_positions(all_vertex_numbers, n_rows, n_cols, w)

    vertex_numbers, vertex_indicators = assign_vertices(all_vertex_positions, node_positions, node_radius)
    coordinates = get_vertex_coordinates(vertex_numbers, n_rows, n_cols)

    return coordinates

def make_stencil(n, m, r):
    grid = np.zeros((n, m))

    centre = np.array([n // 2, m // 2])

    c = 0
    for i in range(n):
        for j in range(m):
            if np.linalg.norm(centre - np.array([i, j])) > r:
                grid[i, j] = -1
            else:
                grid[i, j] = c
                c += 1

    return grid

def get_shape_matrix(n, m, r):
    '''
    constructs the shape matrix representing the laplacian for a circle with radius r centred within an n x m square

    '''

    sten = make_stencil(n, m, r)
    stencil = np.pad(sten, ((1, 1), (1, 1)), constant_values=(-1,))  # pad the array so that every boundary is a -1

    A = np.zeros((n * m, n * m))

    for i in range(n):
        for j in range(m):

            if stencil[i + 1, j + 1] != -1:  # +1 for the padding
                current_ind = n * i + j

                adjacent = np.array([stencil[i, j + 1], stencil[i + 2, j + 1], stencil[i + 1, j],
                                     stencil[i + 1, j + 2]])  # plus one for padding
                adjacent_inds = np.array([[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]])

                # remove the boundaries as they are insulating and dont contrbute to the dreivitive
                adjacent_inds = adjacent_inds[adjacent != -1]
                adjacent = adjacent[adjacent != -1]

                A[current_ind, current_ind] = len(adjacent) * -1  # stuff can flow out through this many sides

                # stuff can flow into the adjacent squares
                for ai in adjacent_inds:
                    A[current_ind, ai[0] * n + ai[1]] = 1

    return A, sten

A, bound = get_shape_matrix(environment_size[0], environment_size[1], environment_size[0]//2)

def laplace(x):

    return np.matmul(A,x.flatten()).reshape(x.shape)

def run_sim(indices, receiver_coords, thresholds, logic_gates, activations):


    ind0, ind1, ind2 = indices
    simulator = DigitalSimulator(conc, environment_size, w, dt, laplace=laplace)
    simulator.bound = bound
    inputs_diff = np.any(ind0 != ind1) and np.any(ind1 != ind2) and np.any(ind0 != ind2)
    corners = [[0, 0], [0, max_ind-1], [max_ind-1, 0],
               [max_ind-1, max_ind-1]]  # cant put IPTG in the corners of the allowable square


    on_corner = np.any(list(np.all(i == j) for i in corners for j in indices))
    ti = time.time()
    if inputs_diff and not on_corner:
        coords = np.array(
            [[start_coords + ind0 * points_per_well], [start_coords + ind1 * points_per_well], [start_coords + ind2 * points_per_well]]).reshape(3, 2)

        score, t, best_receiver_pos, all_sims = simulator.max_fitness_over_t(receiver_coords, coords, thresholds,
                                                                             logic_gates, activations, test_t=-1)
    else:
        coords = -1
        score = -1
        best_receiver_pos = -1
        t = -1

    return {'score':score, 'receiver_pos': best_receiver_pos, 'coords': coords, 't':t}



parser = argparse.ArgumentParser(description='Run the colony placement algorithm')
parser.add_argument('in_path', metavar='T', type=str, nargs=1, help='the path of the saved output from macchiato')
parser.add_argument('--outpath', type=str, help='the filepath to save output in, default is colony_placement/output')

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
    thresholds = [] #TODO:: let user specify these
    for act in macchiato_results['logic_gates'].keys():

        for lg in macchiato_results['logic_gates'][act]:
            activations.append(act)
            if act == 'BP':
                thresholds.append([1,1])
            elif act == 'TH':
                thresholds.append([3,3])
            logic_gates.append(lg)

    print(activations, logic_gates, thresholds)
    receiver_pos = [[int(environment_size[0] / 2), int(environment_size[1] / 2)]]
    receiver_radius = 1000
    receiver_coords = get_node_coordinates(receiver_pos, receiver_radius, environment_size[0], environment_size[1],
                                           w)  # lawn of receivers
    receiver_coords = [receiver_coords] * len(activations)
    n_inputs = 3
    min_distance = 4.5
    conc = 5

    all_outputs = list(map(np.array, list(itertools.product([0, 1], repeat=n_inputs))))

    max_score = 0
    max_coords = np.array([[0, 0]])

    max_t = -1
    start_coords = np.array([[2, 7]])

    all_indices = []
    max_ind = 6
    dt = 60 * 20  # sample time of simulations in minutes

    for i in range(max_ind):
        for j in range(max_ind):
            all_indices.append(np.array([i, j]))

    pop_size = 1
    n_gens = 0
    plot = False

    if plot:

        simulator = DigitalSimulator(conc, environment_size, w, dt, laplace=laplace)
        simulator.bound = bound
        ind0, ind1, ind2 = np.array([[1,1], [2, 0], [3, 1]])

        rec_inds = np.array([[2, 1]])

        coords = np.array(
            [[start_coords + ind0 * points_per_well], [start_coords + ind1 * points_per_well],
             [start_coords + ind2 * points_per_well]]).reshape(3, 2)


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
            print(max_r, best_position)
            receiver_coords = get_node_coordinates(rec_coords, receiver_radius, environment_size[0],environment_size[1], w)
            receiver_coords = receiver_coords
            for i, coord in enumerate(coords):
                logic_area[coord[0], coord[1]] = i + 2

            logic_area[rec_coords[:, 0], rec_coords[:, 1]] = i + 3

            plt.figure()
            logic_area[np.where(bound == -1)] = -1
            plt.imshow(logic_area)



        plt.show()
        sys.exit()



    IPTG_is = np.random.choice(len(all_indices), size=(pop_size, 3)) # three inputs for each
    IPTG_inds = np.array(all_indices)[IPTG_is] # (pop_size, 3, 2), the opentron indeices


    simulated = np.zeros((pop_size)) # keep track of grids that have been simulated so we dont do them again
    scores = np.zeros((pop_size))
    max_receiver_pos = np.array(np.ones((len(activations), 2 ))*7) #TODO: set this to 0,0 after testing

    for gen in range(n_gens):

        # 0 IPTG
        # 1 Nutrients
        # 2 Receiver
        # 3 GFP

        ti = time.time()



        indices = IPTG_inds

        pool = Pool(5)
        results = pool.starmap(run_sim, zip(indices, repeat(receiver_coords), repeat(thresholds), repeat(logic_gates), repeat(activations)))
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
            new_IPTG_inds[i] = np.array(all_indices)[IPTG_is]  # (pop_size, 3, 2), the opentron indeices







        IPTG_inds = copy.deepcopy(new_IPTG_inds)

        print("gen : " + str(gen) + ", av score: ", np.mean(scores), " top score: ", np.max(scores))
        print('IPTG inds:', IPTG_inds[0])
        print('max receiver ind:', np.array(max_receiver_pos-start_coords)/points_per_well)
        print(scores)

    receiver_inds = (np.array(max_receiver_pos-start_coords)/int(points_per_well)).astype(np.int32)
    IPTG_inds = np.array(IPTG_inds[0], dtype = np.int32)
    print(max_receiver_pos)
    print(receiver_inds)
    print(IPTG_inds)


    save_data = {'receiver_inds': receiver_inds.tolist(), 'IPTG_inds': IPTG_inds.tolist(), 'activations': activations}

    json.dump(save_data, open(os.path.join(out_path, 'placement.json'), 'w+'))











