import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Alex_model'))
import helper_functions as hf
from plate import Plate
from species import Species
import math
import itertools

def gompertz(t, A, um, lam):
    return A* np.exp(-np.exp((um*np.e)/A*(lam - t) +1))


def dgompertz(t, A, um, lam):
    return um * np.exp(um*(np.e*lam -np.e*t)/A - np.exp(um*(np.e*lam-np.e*t)/A +1) + 2) #1e8 to convert from OD to cells per grid point

def load_data(filepath):
    # dictionary structure: time_courses = dict[spot_concentration][distance]
    observed_wells = []  # to keep track of when we start a new repeat

    data = {}
    with open(filepath) as file:
        file.readline()
        for i, line in enumerate(file):
            line = line.split(',')

            time_point = int(line[21]) #* 20.0  # each timepoint is 20mins
            flouresence = float(line[36])
            IPTG_conc = float(line[18])
            distance = round(float(line[16]), 1)
            absorbance = float(line[20]) # for growth
            replicate = int(line[19]) -1





            try:
                if time_point <= 69:
                    data[IPTG_conc][distance]['GFP'][replicate].append(flouresence)
                    data[IPTG_conc][distance]['absorbance'][replicate].append(absorbance)
            except Exception as e:

                try:

                    data[IPTG_conc][distance]['GFP'] = [[], [], []] #three replicates
                    data[IPTG_conc][distance]['absorbance'] = [[], [], []]
                    data[IPTG_conc][distance]['GFP'][replicate].append(flouresence)
                    data[IPTG_conc][distance]['absorbance'][replicate].append(absorbance)
                except Exception as e:


                    try:
                        data[IPTG_conc][distance] = {}
                        data[IPTG_conc][distance]['GFP'] = [[], [], []]  # three replicates
                        data[IPTG_conc][distance]['absorbance'] = [[], [], []]
                        data[IPTG_conc][distance]['GFP'][replicate].append(flouresence)
                        data[IPTG_conc][distance]['absorbance'][replicate].append(absorbance)
                    except:
                        data[IPTG_conc] = {}
                        data[IPTG_conc][distance] = {}
                        data[IPTG_conc][distance]['GFP'] = [[], [], []]  # three replicates
                        data[IPTG_conc][distance]['absorbance'] = [[], [], []]
                        data[IPTG_conc][distance]['GFP'][replicate].append(flouresence)
                        data[IPTG_conc][distance]['absorbance'][replicate].append(absorbance)

            #print(time_point, flouresence, IPTG_conc, distance, absorbance, replicate, data[IPTG_conc][distance]['GFP'][replicate])

            if time_point <= 69 and time_point != len(data[IPTG_conc][distance]['GFP'][replicate]):

                #print(time_point, len(data[IPTG_conc][distance]['GFP'][replicate]), data[IPTG_conc][distance]['GFP'][replicate])
                raise Exception("Time point and time index out of sync")
    return data

def make_plate(receiver_coords, inducer_coords, params, inducer_conc, environment_size, w, dx, laplace = False, bandpass = False, fitting = False):
    #amount = inducer_conc * 1e-3 #micro moles
    if inducer_conc > 0.1:
        raise Exception('CONC IS VERY HIGH, CHECK THAT IT IS IN MOLAR, NOT mM')

    amount = inducer_conc

    agar_thickness = 3.12  # mm

    init_conc = amount / (w ** 2 * agar_thickness) #mol/mm^3

    if fitting: # in fitting the input is spread over a few grid points
        init_conc /= len(inducer_coords)

    init_conc *= 1e3  # convert to  mM
    A_0 = init_conc

    D_N, mu_max, K_mu, gamma, D_A, \
    alpha_T, beta_T, K_IT, n_IT, K_lacT, T7_0, \
    alpha_R, beta_R, K_IR, n_IR, K_lacR, R_0, \
    alpha_G, beta_G, n_A, K_A, n_R, K_R, X_0, G_s = params

    ## Create our environment
    plate = Plate(environment_size)

    ## add one strain to the plate

    rows = receiver_coords[:, 0]
    cols = receiver_coords[:, 1]

    receiver_flags = np.zeros(environment_size)
    receiver_flags[rows,cols] = 1
    U_X = np.zeros(environment_size)
    U_X[rows, cols] = X_0

    strain = Species("X", U_X)
    def X_behaviour(t, species, params):
        ## unpack params

        #mu = mu_max * np.maximum(0, species['N']) / (K_mu + np.maximum(0, species['N'])) * np.maximum(0,species['X'])
        mu  = dx(t, species)*receiver_flags

        return mu
    strain.set_behaviour(X_behaviour)
    plate.add_species(strain)

    ## add IPTG to plate
    #inducer_position = [[int(j * (4.5/w)) for j in i] for i in inducer_positions  # convert position to specified dims

    U_A = np.zeros(environment_size)
    rows = inducer_coords[:, 0]
    cols = inducer_coords[:, 1]

    U_A[rows, cols] = A_0

    A = Species("A", U_A)
    def A_behaviour(t, species, params):


        a = D_A * hf.ficks(np.maximum(0,species['A']), w, laplace = laplace)
        return a
    A.set_behaviour(A_behaviour)
    plate.add_species(A)

    inducer_flags = np.zeros(environment_size)
    inducer_flags[rows, cols] = 2

    #plt.imshow(inducer_flags + receiver_flags)
    #plt.show()

    #add T7 to the plate
    U_T7 = np.ones(environment_size) * T7_0
    T7 = Species("T7", U_T7)
    def T7_behaviour(t, species, params):

        mu = dx(t, species)*receiver_flags

        dT7 = (alpha_T*mu*(1 + (np.maximum(0,species['A'])/K_IT)**n_IT))/(1 + (np.maximum(0,species['A'])/K_IT)**n_IT + K_lacT) + beta_T*mu - mu*np.maximum(0,species['T7'])


        return dT7
    T7.set_behaviour(T7_behaviour)
    plate.add_species(T7)

    ## add GFP to plate
    U_G = np.zeros(environment_size)
    G = Species("G", U_G)
    def G_behaviour(t, species, params):


        mu = dx(t, species)*receiver_flags
        T7 = np.maximum(0, species['T7'])

        if bandpass:
            R = np.maximum(0, species['R'])

        #R = 0  # produces treshold
        if bandpass:
            dGFP = alpha_G * mu * T7**n_A / (K_A**n_A + T7**n_A) * K_R**n_R / (K_R**n_R + R**n_R) + beta_G * mu - np.maximum(0, species['G']) * mu*G_s
        else:
            dGFP = alpha_G * mu * T7**n_A / (K_A**n_A + T7**n_A) + beta_G * mu - np.maximum(0, species['G']) * mu * G_s


        return dGFP
    G.set_behaviour(G_behaviour)
    plate.add_species(G)

    # add R to the plate
    U_R = np.ones(environment_size) * R_0
    R = Species("R", U_R)

    def R_behaviour(t, species, params):


        mu = dx(t, species) * receiver_flags
        #print('nir', n_IR)
        dR = (alpha_R * mu * (1 + (np.maximum(0, species['A']) / K_IR) ** n_IR)) / (
                1 + (np.maximum(0, species['A']) / K_IR) ** n_IR + K_lacR) + beta_R * mu - mu * np.maximum(0,
                                                                                                           species[
                                                                                                               'R'])
        return dR

    R.set_behaviour(R_behaviour)
    if bandpass:
        plate.add_species(R)

    return plate


def make_plates(all_receiver_coords, receiver_acts, inducer_coords, inducer_conc, environment_size, w, laplace = False, fitting = False):
    plates = []
    print(laplace)

    n_inputs = len(inducer_coords)

    all_inputs = list(map(np.array, list(itertools.product([0, 1], repeat=n_inputs))))
    for i, rc in enumerate(all_receiver_coords):
        print('mp:', i)
        r_plates = []
        activation = receiver_acts[i]
        params, gompertz_ps = get_fitted_params(activation)
        dx = lambda t, y: dgompertz(t, *gompertz_ps)

        for j in range(2 ** n_inputs):
            plate = make_plate(rc, inducer_coords[all_inputs[j] == 1], params, inducer_conc,
                                  environment_size, w, dx, laplace = laplace, bandpass=activation=='BP', fitting = fitting)

            r_plates.append(plate)
        plates.append(r_plates)
    return plates

def get_default_params():
    ## experimental parameters
    D_A = 1e-4  # / w ** 2  # mm^2 per min ***** IPTG DIFFUSION RATE
    T7_0 = 0  # ***** a.u. initial T7RNAP concentration per cell
    R_0 = 0  # ***** a.u. initial REPRESSOR concentration per cell
    GFP_0 = 0  # a.u. ***** initial GFP concentration per cell
    environment_size = (35, 35)
    X_0 = 0.3 * 10 / (environment_size[0] * environment_size[
        1])  # ***** initial cell count per grid position - rescaled to be in OD or 1e8 cells per grid pos
    ## growth parameters (currently Monod growth but can be replaced with fitted growth curves)
    D_N = 1e-4  # / w**2  # mm^2 per min  ***** nutrient diffusion rate
    mu_max = 0.02  # per min  *****  max growth rate
    K_mu = 1  # g  ***** growth Michaelis-Menten coeffecient
    gamma = 1E12  # cells per g  ***** yield
    gamma = 1E4  # OD or 1e8 cells per g  ***** yield rescaled

    ## From Zong paper
    alpha_T = 6223  #
    beta_T = 12.8  #
    K_IT = 1400  # 1.4e-6 M @ 1 molecule per nM
    K_IT = 1.4e-3  # mM
    n_IT = 2.3  #
    K_lacT = 15719  #
    alpha_R = 8025
    beta_R = 30.6
    K_IR = 1200  # 1.2e-6 M @ 1 molecule per nM
    K_IR = 1.2e-3  # mM
    n_IR = 2.2
    K_lacR = 14088
    alpha_G = 16462
    beta_G = 19
    n_A = 1.34
    K_A = 2532  # scaled
    n_R = 3.9
    K_R = 987
    G_s = 1

    params = [D_N, mu_max, K_mu, gamma, D_A,
              alpha_T, beta_T, K_IT, n_IT, K_lacT, T7_0,
              alpha_R, beta_R, K_IR, n_IR, K_lacR, R_0,
              alpha_G, beta_G, n_A, K_A, n_R, K_R, X_0, G_s]

    return params

def get_fitted_params(opt):
    '''
    returns the fitted gompertz growth and gene circuit/diffusion parameters for BP and TH characterisation
    :param opt: if 'threshold' will return threshold gompertz params and thrrehsold gene circuit params. If 'bandpass'
     will return the bandpass gompertz params with the bandpass gene circuit params. If 'both' will return gompertz
     growth params and the bandpass gene circuit params fitted on top of the threshold params
    :return:
    '''
    if opt in ['bandpass', 'both', 'BP']:
        gompertz_ps = [2.53791252e-01, 2.95660906e-04, 3.54576795e+02]  # bandpass second characterisation data
        X_0 = 5.970081581135449e-05  # from gompertz, bandpasss
    else:
        gompertz_ps = [2.61562834e-01, 3.21504837e-04 ,4.04301820e+02]  # threshold new characterisation data
        X_0 = 7.2412638409233995e-06 # from gompertz, threshold, new characterisation
    params = get_default_params()

    # min: 43.94537506931077, MSE after second evolution with problematic points removed
    TH_params = [3.68033023e-02,
                     8.21667283e+04,
                     1.00235768e-04,
                     4.85833372e-04,
                     1.02155766e+00,
                     1.47406373e+04,
                     7.58449901e-06,
                     1.03026478e+02,
                     2.37032357e+00,
                     3.01979916e+00,
                     7.53503109e+00,
                     3.23521712e-01]

    params[4:11] = TH_params[0:7]
    params[17:21] = TH_params[7:11]
    params[-1] = TH_params[11]


    # loss = 3.0985708889858046 BP on top of threshold after second evolve with new characterisation
    BP_params = [2.25538992e+04,
    2.77648415e-05,
    5.65224390e-03,
    4.43375276e+00,
    2.30406041e+05,
    8.36906261e-01,
    1.94664816e+01,
    6.81117529e-01]

    if opt == 'bandpass' or 'BP':
        # min:  1.1516729125694962 BP all params after second evolution with new characterisation
        BP_params = [4.35462879e-02, 4.88625617e+04, 1.83905487e-05, 3.95081261e-05,
         4.47402392e-01, 1.24947521e+04, 1.10207308e-05, 1.40349891e+01,
         1.01251668e+00, 8.56144749e+00, 3.70436050e+00, 2.13140568e-01,
         1.04554814e+05, 9.67789421e-06, 7.18971464e-03, 1.08965612e+01,
         2.45219227e+04, 2.18075133e-01, 4.49477997e+00, 1.87324583e+01]

        params[4:11] = BP_params[0:7]
        params[17:21] = BP_params[7:11]
        params[-1] = BP_params[11]




    params[-14:-8] = BP_params[-8:-2]
    params[-4:-2] = BP_params[-2:]
    params[-2] = X_0



    return params, gompertz_ps


