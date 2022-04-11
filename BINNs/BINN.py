import os
import numpy as np


def plates_from_file(filepath, exposure = '20000', intensity = '3'):
    plates = []
    with open(filepath) as file:
        file.readline()  # skip headers

        for i, line in enumerate(file):

            line = line.split(',')

            if i == 0:
                time = float(line[31])
                current_id = int(line[4])
                input_plate = np.zeros((16, 24)) # the input wells, i.e. blank, input or receiver
                output_plate = np.zeros((16, 24)) # the output, flourescence in each well

            if int(line[4]) != current_id:
                plates.append((time, input_plate, output_plate))
                current_id = int(line[4])
                time = float(line[31])
                input_plate = np.zeros((16, 24))
                output_plate = np.zeros((16, 24))



            exp = line[5]
            inten = line[6]

            if exp == exposure and inten == intensity:
                well_c = int(line[0])
                well_r = int(line[1])

                time = float(line[31]) # in hours
                flouresence = float(line[12]) # the median
                IPTG_conc = 0 if line[20] == 'NA' else float(line[20]) #uM
                receiver = 0 if line[19] == 'NA' else float(line[20]) #uM


                input_plate[well_r, well_c] =
                output_plate[well_r, well_c] = flouresence

def create_dataset(data_dir, activation_func = 'ZG', exposure = '20000', intensity = '3'):
    '''
    creates the numpy array dataset of the wells from the files containing plates
    :param data_dir:
    :return:
    '''

    # plate divided B-G = 2-7, J-O=10-15; 1-6, 10-15, 19-24 inclusive
    # B-G, 1-6 is well 1 etc
    plates = []
    for filename in os.listdir(data_dir):
        if activation_func in filename:
            plates.extend(plates_from_file(os.path.join(data_dir, filename), activation_func = 'ZG', exposure = '20000', intensity = '3'))



if __name__=='__main__':
    create_dataset('./data')



