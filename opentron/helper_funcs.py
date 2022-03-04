def get_wells(receiver_coords, inducer_coords, activations):
    '''
    converts the configuration specified by the coords and activations and creates a list of wells of a 6 well plate where each well will implement each input state

    all coords are 1 indexed
    :param receiver_coords:
    :param inducer_coords:
    :param activations:
    :return:
    '''
    n_inputs = len(inducer_coords)
    all_inputs = list(map(np.array, list(itertools.product([0, 1], repeat=n_inputs))))


    # get relative (to well origin) opentron coords for the inducers

    wells = []

    print('rc', receiver_coords)
    bp = receiver_coords[np.array(activations) == 'BP']
    th = receiver_coords[np.array(activations) == 'TH']
    for i in range(2**n_inputs): # a well in the six well plate for each input state


        ics = inducer_coords[all_inputs[i] == 1]

        well = {'BP':bp, 'TH': th, 'IPTG': ics}
        wells.append(well)
    return wells

def convert_to_opentron(coordinates):
    '''
    converts a np array of coordinates into opentron well with format A-P, 1-24
    :param coordinates:
    :return:
    '''

    return [chr(ord('@') + coordinate[0]) + str(coordinate[1]) for coordinate in coordinates]


def get_plates(wells):
    '''
    takes a list of wells an assigns them to opentron positions on a plate

    all coords are 1 indexed
    :param receiver_coords:
    :param inducer_coords:
    :param activations:
    :return:
    '''
    origins = np.array([[2,1], [2,10], [2, 19], [10, 1], [10, 10], [10, 19]])
    plates = []

    n_wells = len(wells)
    n_plates = math.ceil(n_wells/6)


    positions = ['8', '5', '2'] #TODO:: ask ke yan about this
    for p in range(n_plates):

        bandpass_wells = []
        threshold_wells = []
        IPTG_wells = []

        try:
            wells_for_plate = wells[p*6:(p+1)*6]
        except IndexError:
            wells_for_plate = wells[p * 6:]
        for w, well in enumerate(wells_for_plate):

            if not well['BP'].size == 0:
                bandpass_wells.append(well['BP'] + origins[w] - np.array([1,1])) #- np.array([1,1]) to acocunt for 1 indexing
            if not well['TH'].size == 0:
                threshold_wells.append(well['TH'] + origins[w]- np.array([1,1]))

            if not well['IPTG'].size == 0:
                IPTG_wells.append(well['IPTG'] + origins[w]- np.array([1,1]))

        bandpass_wells, threshold_wells, IPTG_wells = map(np.vstack, [bandpass_wells, threshold_wells, IPTG_wells])

        bandpass_wells, threshold_wells, IPTG_wells = map(convert_to_opentron, [bandpass_wells, threshold_wells, IPTG_wells])

        plate = {'position': positions[p], 'BP':bandpass_wells, 'TH':threshold_wells, 'IPTG': IPTG_wells}

        plates.append(plate)
    return plates
