import json
import os
metadata = {'apiLevel': '2.0'}

def pipette(plates, typ, liq, pip):
    # atm using pipette with max volume 10uL and pipetting 1uL each time, will need to change if these change
    max_vol = 10
    vol_disp = 1

    pip.pick_up_tip()

    for plate in plates:

        wells = plate[typ]

        agar = plate['agar']

        n_wells_remaining = len(wells)
        vol = min(max_vol, n_wells_remaining*vol_disp)
        n_wells_remaining -= vol/vol_disp

        pip.aspirate(vol, liq, rate=0.5)

        for well in wells:
            if vol < vol_disp:
                vol = min(max_vol, n_wells_remaining * vol_disp)

                n_wells_remaining -= vol / vol_disp
                pip.aspirate(vol, liq, rate=0.5)

            pip.dispense(1, agar[well].bottom(), rate=0.5)

    pip.drop_tip()


def run(protocol):

    ''' WHEN PUTTING ON OPENTRON COPY AND PASTE THE CONTENTS OF THE CONFIG FILE HERE AS A PYTHON DICT BECAUSE YOU CANT UPLOAD A SEPERATE FILE TO THE OPENTRON'''
    plates = json.load(open(os.path.join('output', 'plate_config.json')))
    deep96 = protocol.load_labware('nest_96_wellplate_2ml_deep', '4')

    tip10 = protocol.load_labware('opentrons_96_filtertiprack_10ul', '1')

    p10 = protocol.load_instrument('p10_single',
                                   mount='left',
                                   tip_racks=[tip10])

    R1 = deep96['D1']  # bandpass
    R2 = deep96['D2']  # threshold
    I1 = deep96['D3']  # 5mM


    for plate in plates:
        plate['agar'] = protocol.load_labware('corning_384_wellplate_112ul_flat', plate['position'])


    # do the bandpass receivers
    pipette(plates, 'BP', R1, p10)

    # do the threshold receivers
    pipette(plates, 'TH', R2, p10)

    # do the IPTG
    pipette(plates, 'IPTG', I1, p10)






