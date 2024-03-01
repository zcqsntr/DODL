#!/usr/bin/env python

import argparse
import os
import sys
import json
import numpy as np
import itertools
import math
from PIL import Image, ImageDraw
from copy import deepcopy
import csv
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_logic_combinations(json_data, n_replicates):
    '''
    converts the configuration specified by the coords and activations and creates a list of wells of a 6 well plate where each well will implement each input state

    all coords are 1 indexed
    :param receiver_coords:
    :param inducer_coords:
    :param activations:
    :return:
    '''

    n_gates = len(json_data)

    all_combinations = []
    for gate_idx in range(n_gates):
        json_gate = json_data[gate_idx]

        inducer_coords = np.array(json_gate['IPTG'])
        n_inputs = len(inducer_coords)
        gate_input_combinations = list(map(np.array, list(itertools.product([0, 1], repeat=n_inputs))))
        gate_input_combinations = gate_input_combinations[1:]  # drop first combination as we don't want to create 0,0 for all gates

        bp = np.array(json_gate['ZBD'])
        th = np.array(json_gate['ZG'])

        for i in range(len(gate_input_combinations)):  # a well in the six well plate for each input state
            ics = inducer_coords[gate_input_combinations[i] == 1]

            well = {'gate': json_gate['logic_gate'], 'input_configuration': gate_input_combinations[i], 'BP': bp.__copy__(), 'TH': th.__copy__(), 'IPTG': ics.__copy__()}
            all_combinations.append(well)

    # repeat the combinations n_replicates times
    all_combinations = all_combinations * n_replicates
    return all_combinations


def coord_to_well(coordinate):
    '''
    converts a np array of coordinates into opentron well with format A-P, 1-24
    :param coordinates:
    :return:
    '''
    return chr(ord('@') + coordinate[0,0]) + str(coordinate[1,0])


def get_plates(positions):
    '''
    takes a list of wells an assigns them to opentron positions on a plate

    all coords are 1 indexed
    :param receiver_coords:
    :param inducer_coords:
    :param activations:
    :return:
    '''

    wells_per_plate = 5  # leave final well for blanks
    plates = []
    ot_plate_positions = ['2', '3', '5', '6', '8', '9', '11']

    n_wells = len(positions)
    n_plates_total = math.ceil(n_wells / wells_per_plate)
    n_ot_positions = len(ot_plate_positions)
    n_sets = math.ceil(n_plates_total / n_ot_positions)

    metadata = []
    for plate_set in range(n_sets):
        n_plates = min(n_ot_positions, n_plates_total - n_ot_positions * plate_set)

        for p in range(n_plates):
            bandpass_wells = ['L22']
            # threshold_wells = ['N21']

            metadata_plate = {'plate_set': plate_set + 1, 'plate_number': p+1,
                              'well': 'B3',
                              'sample_well': 'L22', 'well_row': 12, 'well_column': 22,
                              'name': 'ZBD', 'logic_function': '', 'pattern_position': ''}
            metadata.append(metadata_plate)

            threshold_wells = []
            IPTG_wells = []
            first_well = plate_set * n_ot_positions * wells_per_plate + p * wells_per_plate
            last_well = first_well + wells_per_plate
            pos_copy = deepcopy(positions)
            wells_for_plate = pos_copy[first_well:min(last_well, n_wells)]

            for w, well in enumerate(wells_for_plate):
                # converted well position is given by:
                #   col = col + 9 * (six_well_col - 1)
                #   if six_well_row == 2:
                #       row = 17 - row
                six_well_row = (w // 3) + 1
                six_well_col = (w % 3) + 1

                well['six_well_coord'] = [six_well_row, six_well_col]
                well['plate'] = p+1

                well_types = ['BP', 'TH', 'IPTG']
                for wtype in well_types:
                    if well[wtype].size > 1:   #TODO: for some reason empty well are given a size == 1
                        for bp_well in well[wtype]:
                            new_well = bp_well.__copy__()
                            new_well[1] = new_well[1] + 9 * (six_well_col - 1)
                            if six_well_row == 2:
                                new_well[0] = 17 - new_well[0]

                            ot_well = coord_to_well(new_well)
                            if wtype == 'BP':
                                bandpass_wells.append(ot_well)
                            elif wtype == 'TH':
                                threshold_wells.append(ot_well)
                            elif wtype == 'IPTG':
                                IPTG_wells.append(ot_well)

                            metadata_plate = {'plate_set': plate_set+1, 'plate_number': well['plate'],
                                              'well': coord_to_well(np.array([[six_well_row], [six_well_col]])),
                                              'sample_well': ot_well, 'well_row': new_well[0][0], 'well_column': new_well[1][0],
                                              'name': wtype, 'logic_function': well['gate'], 'pattern_position': coord_to_well(bp_well),
                                              'inputs_present': sum(well['input_configuration']),
                                              'input_configuration': ''.join([str(x) for x in well['input_configuration']])}
                            metadata.append(metadata_plate)

            plate = {'set': plate_set, 'position': ot_plate_positions[p],
                     'BP': bandpass_wells, 'TH': threshold_wells, 'IPTG': IPTG_wells}

            plates.append(plate)

    metadata_pd = pd.DataFrame(metadata)
    return plates, metadata_pd


def draw_blank_plate(scale):
    plate_dim = np.array([24, 16])

    im = Image.new('RGB', tuple((plate_dim + np.array([3,3])) * scale), color=(255, 255, 255))

    draw = ImageDraw.Draw(im)

    # draw the wells
    for x in range(plate_dim[0]):
        for y in range(plate_dim[1]):
            if y == 0:
                draw.text(((x + 1.25) * scale, (y + 0.5) * scale), str(x + 1), fill='black')
            if x == 0:
                draw.text(((x + 0.5) * scale, (y + 1.25) * scale), chr(ord('@') + y + 1), fill='black')
            draw.rectangle((((x + 1) * scale + scale/10, (y + 1) * scale + scale/10), ((x + 2) * scale -scale/10, (y + 2) * scale -scale/10)),
                           fill=(255, 255, 255), outline=(0, 0, 0) , width=1)
    # draw the circles on

    circle_diam = scale * 7.77777
    for x in range(3):
        for y in range(2):
            start = (scale*0.35 + (circle_diam + scale)*x, scale + (circle_diam + 0.5*scale)*y)
            end = (start[0] + circle_diam, start[1] + circle_diam)

            draw.arc((start, end), start=0, end=360, fill=(0,0,0))

    return im


def draw_wells(plate_draw, wells, colour, scale):

    for well in wells:
        x = int(well[1:])
        y = int(ord(well[0]) - ord('@'))
        plate_draw.rectangle(((x * scale +scale/10, y * scale +scale/10), ((x + 1) * scale-scale/10, (y + 1) * scale-scale/10)),
                       fill=colour, outline=(0, 0, 0), width=2)


def draw_plates(plates, out_dir):
    scale = 20
    for p,plate in enumerate(plates):
        im = draw_blank_plate(scale)
        draw = ImageDraw.Draw(im)

        draw_wells(draw, plate['BP'], (187, 161, 205), 20)
        draw_wells(draw, plate['TH'], (59, 140, 53), 20)
        draw_wells(draw, plate['IPTG'], (55, 73, 153), 20)

        im.save(os.path.join(out_dir, 'plate{}.png'.format(p)))


parser = argparse.ArgumentParser(description='Produce plate configurations for the opentron from colony placements ')
parser.add_argument('--in_file',  type=str, help='the input data from colony_placement, default is colony_placement/output/placement.json')
parser.add_argument('--out_file', type=str, help='the filepath to save output in, default is opentron/output/plate_config.json')
parser.add_argument('--plot', type=str, help='1 to plot dest_plates, 0 to not')

if __name__ == '__main__':
    args = parser.parse_args()
    n_replicates = 3

    in_file = args.in_file
    out_file = args.out_file

    if out_file is None:
        out_file = os.path.join(dir_path, 'output')

    if in_file is None:
        in_file = os.path.join(os.path.join(os.path.join(os.path.dirname(dir_path), 'colony_placement'), 'output'), 'placement.json')

    os.makedirs(out_file, exist_ok=True)

    data = json.load(open(in_file))

    combinations = get_logic_combinations(data, n_replicates)
    print(combinations)
    plates, metadata = get_plates(combinations)

    if args.plot:
        draw_plates(plates,  out_file)

    out_file = os.path.join(out_file, 'plate_config.json')
    json.dump(plates, open(out_file, 'w+'))

    metadata.to_csv('plate_layout.csv', index=False)
