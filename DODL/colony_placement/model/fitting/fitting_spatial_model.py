import sys
import os
import matplotlib as mpl
mpl.use('tkagg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))





from bayes_opt import SequentialDomainReductionTransformer
from plate import Plate
from species import Species
import numpy as np
import math

import matplotlib.pyplot as plt


import pyswarms as ps
from multiprocessing import Pool

import multiprocessing as mp

from bayes_opt import BayesianOptimization
import fitting_functions as ff
import helper_functions as hf
import simulation_functions as sf


if len(sys.argv) == 3: # for cluster

    if int(sys.argv[2]) == 1:
        save_file = '/home/ntreloar/colony-com/IPTG_logic_gates/IPTG_characterisation/bayes_out.csv'
        bayes_f = 1
        swarm_f = 0
    else:
        save_file = '/home/ntreloar/colony-com/IPTG_logic_gates/IPTG_characterisation/swarm_out.csv'
        bayes_f = 0
        swarm_f = 0
    # for parameter scan
    '''
    exp = int(sys.argv[2]) - 1
    # 3 learning rates
    # 4 hl sizes
    # 3 repeats per combination
    n_repeats = 3
    comb = exp // n_repeats
    pol_learning_rate = pol_learning_rates[comb//len(hidden_layer_sizes)]
    hidden_layer_size = hidden_layer_sizes[comb%len(hidden_layer_sizes)]
    '''

    save_path = sys.argv[1] + sys.argv[2] + '/'

    os.makedirs(save_path, exist_ok=True)
elif len(sys.argv) == 2: # custom save path
    save_path = sys.argv[1] + '/'
    os.makedirs(save_path, exist_ok=True)
else: # default
    save_path = './working_results'



swarm_f = 1
evolve_f = 0
plot_f = 0
threshold = 0
bandpass = 1


'''if 'threshold' will fit threshold gene circuit params. If 'bandpass'
fit all gene circuit params on the bandpass data'''

if bandpass:
    opt = 'bandpass'
elif threshold:
    opt = 'threshold'


# we need to gompertx parameters which are fitted to the growth rate
params, gompertz_ps = ff.get_fitted_params(opt)

n_cores = int(mp.cpu_count())
print('cores:', n_cores)
## 1536 well plate
environment_size = (39, 39)
plate_width = 35

w = 0.9
A, stencil = sf.get_shape_matrix(environment_size[0], environment_size[1], environment_size[0]//2)
laplace  = lambda x: np.matmul(A,x.flatten()).reshape(x.shape)


inducer_coords = np.array([[environment_size[0]//2+1,environment_size[1]//2], [environment_size[0]//2,environment_size[1]//2+1], [environment_size[0]//2,environment_size[1]//2] ,[environment_size[0]//2+1,environment_size[1]//2+1]])  # positions specified on 384 well plate [[row], [col]]
dist = 4.5  # mm
centre = plate_width / 2
receiver_radius = 2.1


'''receiver positions for the new characterisation experiment'''
coords_from_centre = np.array([
    [-3, 0],
    [-2, -2],
    [-1, 1],
    [0, -1],
    [1, -3],
    [1, 2],
    [2, 0],
    [3, -2]
]) # the position of each receiver in terms of number of wells away from the centre (matrix coords)

offset = np.array([-1,+1]) # offset as inducer is not exactly centred in the six wells
receiver_pos = coords_from_centre*4.5 + centre + offset
inducer_coords += offset
receiver_coords = sf.get_node_coordinates(receiver_pos, receiver_radius, environment_size[0], environment_size[1],w)


def plot_layout():
    grid = np.zeros(environment_size)

    grid[np.where(stencil == -1)] = 3


    plt.figure()
    colony_coords = coords_from_centre + np.array([3, 3])
    n_rows, n_cols = environment_size
    distances = []
    for c in coords_from_centre:
        distances.append(round(np.sqrt(np.sum(c ** 2)) * 4.5, 1))
    pixels = np.zeros(environment_size)
    colony_width = math.ceil(n_rows / 8)
    print('width', colony_width)
    for i, coords in enumerate(colony_coords):
        distance = distances[i]
        # arm = arms[i]
        row, col = coords
        grid[colony_width * row +2:colony_width * (row + 1)+1,
             colony_width * col +4:colony_width * (col + 1) +3] = 4


    for i, coord in enumerate(inducer_coords):
        grid[coord[0], coord[1]] =  5

    #
    #grid[receiver_coords[:, 0], receiver_coords[:, 1]] = 2


    im = plt.imshow(grid)
    plt.colorbar(im)
    plt.show()

#plot_layout() use this to check the layout
plt.close('all')


dx = lambda t, y: ff.dgompertz(t,*gompertz_ps)

print(ff.gompertz(0, *gompertz_ps))

def measure_flourescence(U):
    # each well split into 8x8 squares and these used to measure flourescence
    # only the following coordinates (in terms of the measurement grid have colonies
    meas_time = 20 # time between measurements in mins
    #colony_coords = [(0, 3), (1, 3), (2, 3), (4, 3), (5, 3), (6, 3), (3,0), (3, 1), (3, 2), (3, 4), (4, 5), (3, 6)]

    colony_coords = coords_from_centre + np.array([3,3])

    distances =[]
    for c in coords_from_centre:
        distances.append(round(np.sqrt(np.sum(c**2))*4.5 , 1))



    #arms = [0,0,0,1,1,1,2,2,2,3,3,3]
    n_rows, n_cols = U[3,:,:,0].shape
    colony_width = math.ceil(n_rows/8)

    simulated_data = {}

    #for t in range(0,U.shape[-1], int(meas_time/dt)): # first characterisation data

    pixels = np.zeros(environment_size)
    for t in range(0,U.shape[-1]):
        u = U[:,:,:,t]
        gfp = u[3]

        for i, coords in enumerate(colony_coords):
            distance = distances[i]
            #arm = arms[i]
            row, col = coords



            pixels = gfp[colony_width * row +2:colony_width * (row + 1)+1,
                            colony_width * col +4:colony_width * (col + 1) +3]

            pixels = np.sort(pixels.flatten())[4:12] # take middle 50% of 16 pixels

            try:
                simulated_data[distance].append(np.mean(pixels))
            except:

                simulated_data[distance] = []
                simulated_data[distance].append(np.mean(pixels))

    return simulated_data

def run_all_experiments(params, plot=False):

    all_data = {}
    if plot:
        fig, axs = plt.subplots(6,8, figsize = (50,50), dpi = 100)

        fig.text(0.5, 0.05, 'Time (min)', ha='center',fontsize=20)
        fig.text(0.08, 0.5, 'GFP mean pixel value per well', va='center', rotation='vertical',fontsize=20)

        fig.text(0.5, 0.92, 'Distance (mm)', ha='center', fontsize=20)
        fig.text(0.93, 0.5, 'IPTG concentration (mM)', va='center', rotation=90, fontsize=20)

    for i, conc in enumerate(IPTG_concs):

        plate = ff.make_plate(receiver_coords, inducer_coords, params, conc, environment_size, w, dx, laplace = laplace, bandpass=bandpass, fitting = True)
        #sol = plate.run(t_final = 20 * 62, dt = .1, params=params) #first set of data
        sol = plate.run(t_final = time_points[-1], dt = .1, params=params, t_eval = time_points)
        simulated_data = measure_flourescence(sol)
        if plot:

            for j, distance in enumerate(sorted(distances)):
                axs[i][j].plot( time_points, np.array(simulated_data[distance]).T , label = 'simulated')

            #axs[i][j].set_prop_cycle(None)

            for j, distance in enumerate(sorted(distances)):
                axs[i][j].plot(time_points, np.mean(np.array(all_lab_data[conc][distance]['GFP']), axis = 0), '--' , label = 'data')

                #axs[i][j].set_ylim(bottom=1)
                if bandpass:
                    axs[i][j].set_ylim(top=3)
                else:
                    axs[i][j].set_ylim(top=25)


                if j == len(distances) - 1:
                    axs[i][j].set_ylabel(str(conc*1000), fontsize=15)
                    axs[i][j].yaxis.set_label_position("right")

                if i == 0:
                    axs[i][j].set_title(str(distance), fontsize=15)

                axs[i][j].legend()
                axs[i][j].tick_params(labelsize = 12)
                #axs[i][j].set_yscale('log')
                axs[i][j].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

                if i != len(IPTG_concs) - 1:
                    axs[i][j].set_xticklabels([])

                if j !=  0:
                    axs[i][j].set_yticklabels([])




            if threshold:
                save_path = './characterisation_sims/threshold'
            elif bandpass:
                save_path = './characterisation_sims/bandpass'


        all_data[conc] = simulated_data

    plt.savefig('characterisation.pdf')
    plt.show()

    return all_data

def objective(params):
    all_params =  ff.get_fitted_params(opt)[0]

    # threshold
    if threshold:
        all_params[4:11] = params[0:7]
        all_params[17:21] = params[7:11]
        all_params[-1] = params[11]

    #bandpass
    if bandpass:
        all_params[-14:-8] = params[-8:-2]
        all_params[-4:-2] = params[-2:]


    all_sim_data = run_all_experiments(all_params, plot=False)
    error = 0

    for IPTG_conc in IPTG_concs:
        for distance in distances:
            lab_data = np.array(all_lab_data[IPTG_conc][distance]['GFP'])

            sim_data = np.array(all_sim_data[IPTG_conc][distance])


            #diff = np.log10(lab_data+ -np.min(np.min(lab_data), 0) + 0.00001) - np.log10(sim_data+0.00001)
            diff = lab_data - sim_data

            #error += np.sum(((diff)/(lab_data+0.00001))**2)/len(sim_data)
            #remove dodgy points for threshold
            if threshold and not bandpass:
                if not (distance == 4.5 and IPTG_conc == 0.015) and not (distance == 4.5 and IPTG_conc == 0.03) and not (distance == 6.4 and IPTG_conc == 0.03):
                    error += np.sum(np.abs(diff)**2) / len(sim_data)

    #print(error, params)
    return error

def vector_objective_par(params, *args):
    '''
    objective in vector form for the PSO library
    :param params:
    :param args:
    :return:
    '''
    with Pool(n_cores) as p:
        errors = p.map(objective, params)



    ind = np.argmin(errors)
    min_error = errors[ind]
    best_params = params[ind]

    global global_best_cost
    global global_best_params

    if min_error < global_best_cost:
        global_best_cost = min_error

        global_best_params = best_params
        save_file.write(str(min_error) + ',' + str(best_params) + '\n')

    print()
    print('best params:', global_best_params, 'loss', global_best_cost)
    print()

    return np.array(errors)

def vector_objective(params, *args):
    '''
    objective in vector form for the PSO library
    :param params:
    :param args:
    :return:
    '''



    errors = []
    #print(params)


    for i in range(len(params)):

        error = objective(params[i])

        if i == 0:
            best_params = params[i]
            min_error = error
        elif error< min_error:
            min_error = error
            best_params = params[i]
            print()
            print('new best params:', best_params, 'loss', min_error)
            print()


        errors.append(error)

    global global_best_cost
    global global_best_params
    if min_error < global_best_cost:

        global_best_cost = min_error

        global_best_params = best_params
        #save_file = open(save_file, 'w+')
        #save_file.write(str(min_error) + ',' + str(best_params)  + '\n')
        #save_file.close()
    print()
    print('current best params:', global_best_params, 'loss', global_best_cost)
    print()

    return np.array(errors)

def clip_params(params):
    params = np.clip(params, a_min=0, a_max=None)
    mx = 20

    if threshold:
        params[params[:,9] > mx, 9] = mx
        params[params[:, 4] > mx, 4] = mx

    if bandpass:
        params[params[:, -2] > mx, -2] = mx
        params[params[:, -5] > mx, -5] = mx
    return params

def evolve(params, n_gens, pop_size):

    fine_tune = True

    p = np.array(params)

    n=len(params)

    if fine_tune:
        d = 100
    else:
        d = 10
    params = np.random.normal(loc = p, scale = p/d, size =(pop_size, n) )


    # D_A, alpha_T, beta_T, K_IT, n_IT, K_lacT, T7_0, alpha_G, beta_G, n_A, K_A, G_s
    params = clip_params(params)
    params[0] = p

    scores = vector_objective(params)

    print(scores)
    for i in range(n_gens):
        print('GENERATION', i)
        order = np.argsort(scores)
        params = params[order]
        scores = scores[order]

        print('mean:', np.mean(scores))
        print('min:', np.min(scores), 'best params', params[0])

        p = params[0]

        if fine_tune:
            params = np.random.normal(loc=p, scale=p / 100, size=(pop_size, n))
            params[0] = p
        else:
            params[-int(len(params)/2):] = np.random.normal(loc = params[:int(len(params)/2)], scale = params[:int(len(params)/2)]/10)

        print('p', p.shape)

        params = clip_params(params)
        if fine_tune:
            scores[1:] = vector_objective(params[1:])  # only simulate the params that have changed
        else:
            scores[-int(len(params)/2):]  = vector_objective(params[-int(len(params)/2):] ) # only simulate the params that have changed
        #
        print(scores)

def average_over_repeats(data, time_points, IPTG_concs):
    # extracts the average GFP measurement for eah time point in order
    averaged_data = {}
    for IPTG_conc in IPTG_concs:

        for distance in [4.5, 9, 13.5]:
            points = data[IPTG_conc][distance]

            for t in time_points:
                repeats = []
                for p in points:
                    if p[0] == t:
                        repeats.append(p[2])

                av = np.mean(repeats)

                try:
                    averaged_data[IPTG_conc][distance].append(av)
                except:
                    try:

                        averaged_data[IPTG_conc][distance] = []
                        averaged_data[IPTG_conc][distance].append(av)
                    except:

                        averaged_data[IPTG_conc] = {}
                        averaged_data[IPTG_conc][distance] = []
                        averaged_data[IPTG_conc][distance].append(av)
    return averaged_data



if __name__ == '__main__':

    TH_file = os.path.join('data', '20220704_data-processed-ZG.csv')
    BP_file = os.path.join('data', '20220711data-processed-ZBD.csv')
    TH_data = ff.load_data(TH_file)
    BP_data = ff.load_data(BP_file)

    if bandpass:
        all_lab_data = BP_data
    elif threshold:
        all_lab_data = TH_data



    #sys.exit()
    dt = 0.1
    ## run the experiment
    distances = [13.5, 12.7, 6.4, 4.5, 14.2, 10.1, 9.0, 16.2]
    IPTG_concs = [0.0, 0.00188, 0.00375, 0.0075, 0.015, 0.03]



    best_params = None
    best_error = 99999999

    n_points = len(TH_data[IPTG_concs[0]][distances[0]]['GFP'][0])
    time_points = np.arange(0, n_points * 20, 20)  # times in minutes


    global_best_cost = 9999999
    global_best_params = None

    params, _ = ff.get_fitted_params(opt)

    if evolve_f: #evolve
        evolve(params, n_gens = 1000, pop_size = 100)
    elif plot_f: #plot
        run_all_experiments(params, plot = True)
    elif swarm_f: #particle swarm

        if threshold:
            # particle swarm optimisation
            bounds = ([0.01, 1e3, 1e-6, 1e-5, 0.5, 1e3, 1e-7, 5, 0.05, 0.5, 1, 1e-3],
                      [0.3, 5e4, 1e-4, 1e-3, 20, 1e5, 1e-5, 450, 7.5, 20, 20,
                       3])  # D_A, alpha_T, beta_T, K_IT, n_IT, K_lacT, T7_0, alpha_G, beta_G, n_A, K_A, G_s
        elif bandpass:
            bounds = (
            [0.01, 1e3, 1e-6, 1e-5, 0.5, 1e3, 1e-7, 5, 0.05, 0.5, 1, 1e-3, 1e2, 1e-6, 1e-7, 0.1, 1e1, 1e-5, 0.1, 0.1],
            [0.3, 5e4, 1e-4, 1e-3, 20, 1e5, 1e-5, 450, 7.5, 20, 20, 3, 1e5, 1e-4, 1e-2, 20, 1e5, 1, 20,
             100])  # D_A, alpha_T, beta_T, K_IT, n_IT, K_lacT, T7_0, alpha_G, beta_G, n_A, K_A, G_s, alpha_R, beta_R, K_IR, n_IR, K_lacR, R_0, n_R, K_R


        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 10}


        # Call instance of PSO
        if threshold:
            n_dims = 12
        elif bandpass:
            n_dims = 20
        optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=n_dims, options=options, bounds=bounds)

        # Perform optimization
        cost, pos = optimizer.optimize(vector_objective, iters=1000, verbose=True)

