import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'IPTG_characterisation'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'Alex_model'))
import itertools
import fitting_functions as ff
import numpy as np
import scipy.ndimage as sn
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt

from skimage import feature
import fitting_functions as ff



class DigitalSimulator():

    def __init__(self, inducer_conc, environment_size, w, dt, laplace=False):
        self.inducer_conc = inducer_conc
        self.environment_size = environment_size
        self.w = w
        self.laplace = laplace
        self.dt = dt #minutes


    def run_sims(self, inducer_coords, receiver_coords, bandpass = False, plot = False):
        '''
        runs the simulations for all input states for a given set of inducer and receiver coords and receiver activation

        '''

        params, gompertz_ps = ff.get_fitted_params(bandpass=bandpass)

        dx = lambda t, y: ff.dgompertz(t, *gompertz_ps)

        n_inputs = len(inducer_coords)
        all_inputs = list(map(np.array, list(itertools.product([0, 1], repeat=n_inputs))))
        all_sims = []



        # muliple recievers can have multiple activations
        for i in range(2 ** n_inputs):

            plate = ff.make_plate(receiver_coords, inducer_coords[all_inputs[i] == 1], params, self.inducer_conc,
                                  self.environment_size, self.w, dx, laplace = self.laplace, bandpass=bandpass)

            sim_ivp = plate.run(t_final=20 * 60, dt=self.dt, params=params)
            if plot:
                plate.plot_simulation(sim_ivp, 1, time_points = [-1], scale='linear', cols=2)
            all_sims.append(sim_ivp)

        return all_sims


    def get_logic_area(self, logic_gate, all_sims, thresholds, t):
        '''
        given a logic gate, set of simulations, threhsolds and time returns the area in which the logic gate is encoded
        '''


        inactive_thr, active_thr = thresholds





        logic_area = np.ones(self.environment_size)  # start off as all ones and eliminate
        logic_area[np.where(self.bound == -1)] = 0 # only look in the circular plate


        for i in range(len(logic_gate)):
            sim_ivp = all_sims[i]
            GFP = sim_ivp[3]
            end_GFP = GFP[:, :, t]
            if logic_gate[i] == 0:
                logic_area[end_GFP > inactive_thr] = 0
            elif logic_gate[i] == 1:
                logic_area[end_GFP < active_thr] = 0

        return logic_area


    def get_opentron_positions(self):
        '''
        generates all allowable opentron positions
        '''
        positions = []
        start_coords = [2, 7]
        points_per_well = 5
        for i in range(6):
            for j in range(6):

                if [i,j] not in [[0,0], [0,5], [5,0], [5,5]]: # if not on the corner


                    positions.append(np.around([i * points_per_well + start_coords[0], j * points_per_well+ start_coords[1]] , decimals=0).astype(dtype=np.int32))
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
        max_r = 0
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

    def score_over_t(self,  receiver_coords, inducer_coords, thresholds, logic_gate, bandpass = False, plot = False):

        all_sims = self.run_sims(inducer_coords, receiver_coords, bandpass, plot = plot)
        #plt.show()
        n_t = all_sims[0].shape[3]

        all_t = list(range(n_t))


        all_scores = []
        all_receiver_pos = []
        for t in all_t:

            logic_area = self.get_logic_area(logic_gate, all_sims, thresholds, t)

            max_r, best_pos = self.get_max_r(logic_area, inducer_coords)

            all_scores.append(max_r)
            all_receiver_pos.append(best_pos)


        return all_scores,all_receiver_pos, all_t, all_sims

    def max_fitness_over_t(self, all_receiver_coords, inducer_coords, all_thresholds, logic_gates, activations, test_t = False, plot = False):

        # each logic gate has a corresponding activation and set of thresholds

        all_scores = []
        all_sims = []
        all_receiver_pos = []
        for i in range(len(all_thresholds)):

            thresholds = all_thresholds[i]
            logic_gate = logic_gates[i]
            activation = activations[i]
            receiver_coords = all_receiver_coords[i]

            scores, receiver_pos, t, sims = self.score_over_t(receiver_coords, inducer_coords, thresholds, logic_gate, bandpass = activation == 'BP', plot=plot)

            all_scores.append(scores)
            all_sims.append(sims)
            all_receiver_pos.append(receiver_pos)

        all_scores = np.array(all_scores)

        # find the time which max(min(all_scores))
        max_min_score = 0
        best_t = -1
        best_receiver_pos = -1

        if not test_t: # search over all t
            for i in range(len(t)):
                scores = all_scores[:,i]
                receiver_pos = [all_receiver_pos[j][i] for j in range(len(all_thresholds))]

                min_score = min(scores)
                if min_score > max_min_score:
                    max_min_score = min_score
                    best_t = t[i]
                    best_receiver_pos = receiver_pos
        else: # just check at test_t
            #test_t -= 1

            scores = all_scores[:, test_t]
            best_receiver_pos = [all_receiver_pos[j][test_t] for j in range(len(all_thresholds))]

            # TODO:: check all recievers are at different positions
            max_min_score = min(scores)



        return max_min_score, best_t, best_receiver_pos, all_sims





