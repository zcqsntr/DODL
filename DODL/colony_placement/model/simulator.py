import os
import sys


import itertools
import fitting_functions as ff
import numpy as np
import scipy.ndimage as sn
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt

from skimage import feature
import fitting_functions as ff

from simulation_functions import get_shape_matrix, get_node_coordinates

class DigitalSimulator():

    def __init__(self, inducer_conc, environment_size, w, dt,environment_bound = None, laplace=False, make_plate = ff.make_plate, gfp_index = 3, colony_radius = 2):
        '''
        Initialises a simulator for digital optimisation

        :param inducer_conc: concentration of the inducer inputs
        :param environment_size:
        :param w:
        :param dt:
        :param laplace:
        :param make_plate:
        :param gfp_index:
        '''
        self.inducer_conc = inducer_conc
        self.environment_size = environment_size
        self.w = w
        self.laplace = laplace
        self.dt = dt #minutes
        self.make_plate = make_plate
        self.gfp_index = gfp_index
        self.opentron_origin = np.array([[2, 7]])
        self.points_per_well = 5
        self.colony_radius = colony_radius
        self.bound = environment_bound

    def opentron_to_coords(self, opentron_inds):
        '''
        converts an opentron index to the simulation grid coordinate in the centre of the well

        :param opentron_ind:
        :return:
        '''
        coords = self.opentron_origin + opentron_inds * self.points_per_well

        return coords

    def get_colony_coords(self, opentron_inds, colony_radius = None):
        if colony_radius is None:
            colony_radius = self.colony_radius
        colony_centre_coords = self.opentron_to_coords(np.array(opentron_inds))

        pos = colony_centre_coords * self.w


        colony_coords = get_node_coordinates(pos, colony_radius, self.environment_size[0], self.environment_size[1], self.w)

        return colony_coords


    def run_sims(self, plates, t_final = 20*60):
        '''
        runs the simulations for all input states for a given set of inducer and receiver coords and receiver activation

        :param plates: runs all simulations for a 2d list of plates (intended dims (n_receivers, 2**n_inputs))
        :return:
        '''

        all_sims = []

        for receiver in range(len(plates)):

            r_sims = []
            params = ff.get_fitted_params(opt='TH')[0]
            for input_state in range(len(plates[receiver])):
                sim = plates[receiver][input_state].run(t_final=t_final, dt=self.dt, params=params)

                r_sims.append(sim)

            all_sims.append(r_sims)


        return all_sims #shape = (n_receivers, n_input_states, n_species, x,y,t)

    def get_logic_area(self, logic_gate, all_GFPs, thresholds):
        '''
        given a logic gate, set of simulations, thresholds and time returns the area in which the logic gate is encoded
        '''

        inactive_thr, active_thr = thresholds

        logic_area = np.ones(self.environment_size)  # start off as all ones and eliminate

        if self.bound is not None:
            logic_area[np.where(self.bound == -1)] = 0 # only look in the circular plate


        for i in range(len(logic_gate)):

            GFP = all_GFPs[i]



            if logic_gate[i] == 0:
                logic_area[GFP/all_GFPs[0] > inactive_thr] = 0
            elif logic_gate[i] == 1:
                logic_area[GFP/all_GFPs[0] < active_thr] = 0

        return logic_area

    def get_opentron_indices(self):
        '''
        generates all allowable opentron indices in a six well
        '''
        indices= []

        for i in range(6):
            for j in range(6):

                if [i,j] not in [[0,0], [0,5], [5,0], [5,5]]: # if not on the corner
                    indices.append([i, j])
        return indices

    def get_opentron_positions(self):
        '''
        generates all allowable opentron positions
        '''
        positions = []
        indices = self.get_opentron_indices()
        for index in indices:
            positions.append(np.around([index[0] * self.points_per_well + self.opentron_origin[0][0], index[1] * self.points_per_well+ self.opentron_origin[0][1]] , decimals=0).astype(dtype=np.int32))
        return positions

    def get_opentron_pos_within_area(self, labels, area, inducer_coords):
        '''
        gets opentron positions inside the area
        '''

        #revmove receiver positions taken up by the inducers
        receiver_positions = self.get_opentron_positions()
        inducer_coords = inducer_coords.tolist()
        receiver_positions = np.array(receiver_positions).tolist()
        receiver_positions = [rp for rp in receiver_positions if rp not in list(inducer_coords)]


        indices = np.argwhere(labels == area)
        # up to here debugged
        label = np.zeros_like(labels)
        label[indices[:, 0], indices[:, 1]] = 1
        edges = feature.canny(label, low_threshold=0.5, high_threshold=0.5, sigma=0.01)
        positions = set(map(tuple, receiver_positions))
        indices = set(map(tuple, indices))
        pos_inside = list(set.intersection(indices, positions))
        edge_indices = np.argwhere(edges == 1)

        return pos_inside, edge_indices

    def get_max_r(self, logic_area, inducer_coords):
        '''
        gets the maximum (over each area) of the radius of a circle, centred at an opentron position, that fits within each area
        where the logic gate is encoded
        '''

        labels, n = sn.label(logic_area)  # find and label blobs
        best_pos = [0, 0]
        maxmin_r = 0
        for i in range(n):  # for each area

            pos_inside, edge_indices = self.get_opentron_pos_within_area(labels, i+1, inducer_coords)
            if len(pos_inside) > 0:

                for pos in pos_inside:  # maximum over positions
                    # get max radius
                    pos = np.array(pos)
                    min_r = 9999

                    if len(edge_indices) > 0:
                        for ei in edge_indices:  # minimum over the edge
                            ei = np.array(ei)

                            d = np.linalg.norm(pos - ei)
                            if d < min_r:
                                min_r = d
                        if min_r > maxmin_r:
                            maxmin_r = min_r
                            best_pos = pos

        return maxmin_r, best_pos

    def get_fitness(self, GFPs, r_logic_gate, r_threshold, inducer_coords):
        '''
        default fitness function, overwrite if you want to use another metric e.g. ON OFF ratio. This is supposed to
        match the lab experiments where we have set thresholds for each reciever function. Fitness is scored by the
        minimum distance of an opentron position (where a receiver will be placed) to the boundary at which the function
        encoded.
        :param sims:
        :param receiver_info:
        :param inducer_coords:
        :return:
        '''

        logic_area = self.get_logic_area(r_logic_gate, GFPs, r_threshold)
        max_r, best_pos = self.get_max_r(logic_area, inducer_coords)

        return max_r, best_pos

    def fitness_over_t(self, GFPs, r_logic_gate, r_threshold, inducer_coords):

        #plt.show()
        n_t = GFPs[0].shape[-1]

        all_t = list(range(n_t))


        all_scores = []
        all_receiver_pos = []

        for t in all_t:


            max_r, best_pos = self.get_fitness(GFPs[:,:,:,t], r_logic_gate, r_threshold, inducer_coords)
            all_scores.append(max_r)
            all_receiver_pos.append(best_pos)


        return all_scores, all_receiver_pos

    def max_fitness_over_t(self, sims, inducer_coords, r_logic_gates, r_thresholds, test_t = False):

        # each logic gate has a corresponding activation and set of thresholds

        all_scores = []
        all_receiver_pos = []

        for r_ind in range(len(sims)):
            # sims_shape = (n_receivers, n_input_states, n_species, x,y,t)

            GFPs = np.array([sims[r_ind][i][self.gfp_index, :,:,:] for i in range(len(sims[0]))])

            scores, receiver_pos = self.fitness_over_t(GFPs, r_logic_gates[r_ind], r_thresholds[r_ind], inducer_coords)

            all_scores.append(scores)

            all_receiver_pos.append(receiver_pos)

        all_scores = np.array(all_scores)


        # find the time which max(min(all_scores))
        max_min_score = 0
        best_t = -1
        best_receiver_pos = -1

        if not test_t: # search over all t
            for t in range(len(sims[0,0,0,0,0,:])):
                scores = all_scores[:,t]
                receiver_pos = [all_receiver_pos[j][t] for j in range(len(sims))]

                min_score = min(scores)
                if min_score > max_min_score:
                    max_min_score = min_score
                    best_t = t
                    best_receiver_pos = receiver_pos
        else: # just check at test_t
            #test_t -= 1

            scores = all_scores[:, test_t]
            best_receiver_pos = [all_receiver_pos[j][test_t] for j in range(len(sims))]
            # TODO:: check all recievers are at different positions
            max_min_score = min(scores)



        return max_min_score, best_t, best_receiver_pos





