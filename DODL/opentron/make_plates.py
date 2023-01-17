#!/usr/bin/env python

import argparse
import os
import sys
import json
import numpy as np
import itertools
import math
from PIL import Image, ImageDraw
from PIL import ImageFont, ImageDraw
dir_path = os.path.dirname(os.path.realpath(__file__))


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
    for i in range(2**n_inputs - 1,-1, -1): # a well in the six well plate for each input state


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

def vstack(array):

    if len(array) > 0:
        return np.vstack(array)
    else:
        return np.array([])

def get_plates(wells):
    '''
    takes a list of wells an assigns them to opentron positions on a plate

    all coords are 1 indexed
    :param receiver_coords:
    :param inducer_coords:
    :param activations:
    :return:
    '''
    origins = np.array([[1,1], [1,10], [1, 19], [9, 1], [9, 10], [9, 19]])
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
                bandpass_wells.append(well['BP'] + origins[w])# - np.array([1,1])) #- np.array([1,1]) to acocunt for 1 indexing
            if not well['TH'].size == 0:
                threshold_wells.append(well['TH'] + origins[w])#- np.array([1,1]))

            if not well['IPTG'].size == 0:
                IPTG_wells.append(well['IPTG'] + origins[w])#- np.array([1,1]))



        bandpass_wells, threshold_wells, IPTG_wells = map(vstack, [bandpass_wells, threshold_wells, IPTG_wells]) #TODO:: this crashes if one of these is empty

        bandpass_wells, threshold_wells, IPTG_wells = map(convert_to_opentron, [bandpass_wells, threshold_wells, IPTG_wells])

        plate = {'position': positions[p], 'BP':bandpass_wells, 'TH':threshold_wells, 'IPTG': IPTG_wells}

        plates.append(plate)
    return plates

def draw_blank_plate(scale):
    plate_dim = np.array([24, 16])

    im = Image.new('RGB', tuple((plate_dim + np.array([3,3])) * scale), color=(255, 255, 255))

    draw = ImageDraw.Draw(im)
    fnt = ImageFont.truetype('/Library/Fonts/Arial unicode.ttf', int(scale/2))
    # draw the wells
    for x in range(plate_dim[0]):
        for y in range(plate_dim[1]):
            if y == 0:
                draw.text(((x + 1.25) * scale, (y + 0.5) * scale), str(x + 1), fill='black', font = fnt)
            if x == 0:
                draw.text(((x + 0.5) * scale, (y + 1.25) * scale), chr(ord('@') + y + 1), fill='black', font = fnt)
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
    scale = 100
    for p,plate in enumerate(plates):
        im = draw_blank_plate(scale)
        draw = ImageDraw.Draw(im)

        draw_wells(draw, plate['BP'], (187, 161, 205), scale)
        draw_wells(draw, plate['TH'], (59, 140, 53), scale)
        draw_wells(draw, plate['IPTG'], (55, 73, 153), scale)



        im.save(os.path.join(out_dir, 'plate{}.pdf'.format(p)))


parser = argparse.ArgumentParser(description='Produce plate configurations for the opentron from colony placements ')
parser.add_argument('--in_file',  type=str, help='the input data from colony_placement, default is colony_placement/output/placement.json')
parser.add_argument('--out_file', type=str, help='the filepath to save output in, default is opentron/output/plate_config.json')
parser.add_argument('--plot', type=str, help='1 to plot plates, 0 to not')

if __name__ == '__main__':
    args = parser.parse_args()

    in_file = args.in_file
    out_file = args.out_file

    if out_file is None:
        out_file = os.path.join(dir_path, 'output')

    if in_file is None:
        in_file = os.path.join(os.path.join(os.path.join(os.path.dirname(dir_path), 'colony_placement'), 'output'), 'placement.json')

    os.makedirs(out_file, exist_ok=True)



    data = json.load(open(in_file))
    receiver_inds = np.array(data['receiver_inds'])
    IPTG_inds = np.array(data['IPTG_inds'])
    activations = data['activations']

    print(receiver_inds, IPTG_inds, activations)
    wells = get_wells(receiver_inds, IPTG_inds, activations)
    print(wells)
    plates = get_plates(wells)

    if args.plot:
        draw_plates(plates,  out_file)

    out_file = os.path.join(out_file, 'plate_config.json')
    json.dump(plates, open(out_file, 'w+'))