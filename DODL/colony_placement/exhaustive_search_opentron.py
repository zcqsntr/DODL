
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'IPTG_characterisation'))





import itertools

from simulator import DigitalSimulator


from multiprocessing import Pool






## 1536 well plate
environment_size = (39, 39)
plate_width = 35
w = plate_width / environment_size[0] # inter-well spacing in mm
w = 0.9

points_per_well = int(4.5/w)

conc = 5
activations = ['TH']
thresholds = [[3,3]]
#thresholds = [[3,3], [1,1]]


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

receiver_pos = [[int(environment_size[0] / 2), int(environment_size[1] / 2)]]
receiver_radius = 1000
receiver_coords = get_node_coordinates(receiver_pos, receiver_radius, environment_size[0], environment_size[1],w) # lawn of receivers
receiver_coords = [receiver_coords]*len(activations)
n_inputs = 3
min_distance = 4.5
#logic_gates = [[0,0,1,1,1,1,0,0], [0,0,0,0,0,0,0,1]]
#logic_gates = [[0,0,0,1,1,0,0,0], [0,0,0,0,0,0,1,1]] #BP, TH, multiplexer
logic_gates = [[0,0,0,0,0,0,1,1]]

conc = 7.5/1000


all_outputs = list(map(np.array, list(itertools.product([0, 1], repeat=n_inputs))))

max_score = 0
max_coords = np.array([[0,0]])
max_receiver_pos = -1
max_t = -1
start_coords = np.array([[7,7]])

all_indices = []
max_ind = 6
dt = 60*20# sample time of simulations in minutes

for i in range(max_ind):
    for j in range(max_ind):
        all_indices.append(np.array([i,j]))





if __name__ == '__main__':

    #simulator = DigitalSimulator(conc, environment_size, w, laplace=laplace)




    tc = time.time()
    for ind0 in all_indices:

        for ind1 in all_indices:
            ti = time.time()

            indices =[[ind0, ind1, ind2] for ind2 in all_indices]
            pool = Pool(5)
            results = pool.map(run_sim, indices)
            pool.close()
            for result in results:
                score = result['score']
                best_receiver_pos = result['receiver_pos']
                coords = result['coords']
                t = result['t']

                if score > max_score:
                    max_coords = coords
                    max_score = score
                    max_t = t
                    max_receiver_pos = best_receiver_pos
            print('time:', time.time() - ti)

            print(ind0, ind1)
            print('best score; ', max_score, 'at', max_t/10/60, 'hrs')
            print('coords', max_coords)
            print('receiver_pos', max_receiver_pos)

    print()
    print('FINAL RESULTS')
    print('time:', time.time() - tc)

    print('best score; ', max_score, 'at', max_t / 10 / 60, 'hrs')
    print('coords', max_coords)
    print('receiver_pos', max_receiver_pos)



