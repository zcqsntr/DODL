import json
import os
metadata = {'apiLevel': '2.0'}

waste_destination = [  # COMMENT OUT THE BLOCKS NOT IN USE
    # 'A3', 'A4',                                 # A1
    # 'B2', 'B3', 'B4', 'B5',
    # 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
    # 'D1', 'D2', 'D3', 'D4', 'D5', 'D6',
    # 'E1', 'E2', 'E3', 'E4', 'E5', 'E6',
    # 'F1', 'F2', 'F3', 'F4', 'F5', 'F6',
    # 'G2', 'G3', 'G4', 'G5',
    'A12', 'A13',                               # A2
    'B11', 'B12', 'B13', 'B14',
    'C10', 'C11', 'C12', 'C13', 'C14', 'C15',
    'D10', 'D11', 'D12', 'D13', 'D14', 'D15',
    'E10', 'E11', 'E12', 'E13', 'E14', 'E15',
    'F10', 'F11', 'F12', 'F13', 'F14', 'F15',
    'G11', 'G12', 'G13', 'G14',
    # 'A21', 'A22',                               # A3
    # 'B20', 'B21', 'B22', 'B23',
    # 'C19', 'C20', 'C21', 'C22', 'C23', 'C24',
    # 'D19', 'D20', 'D21', 'D22', 'D23', 'D24',
    # 'E19', 'E20', 'E21', 'E22', 'E23', 'E24',
    # 'F19', 'F20', 'F21', 'F22', 'F23', 'F24',
    # 'G20', 'G21', 'G22', 'G23',
    # 'J2', 'J3', 'J4', 'J5',                     # B1
    # 'K1', 'K2', 'K3', 'K4', 'K5', 'K6',
    # 'L1', 'L2', 'L3', 'L4', 'L5', 'L6',
    # 'M1', 'M2', 'M3', 'M4', 'M5', 'M6',
    # 'N1', 'N2', 'N3', 'N4', 'N5', 'N6',
    # 'O2', 'O3', 'O4', 'O5',
    # 'P3', 'P4',
    # 'J11', 'J12', 'J13', 'J14',                 # B2
    # 'K10', 'K11', 'K12', 'K13', 'K14', 'K15',
    # 'L10', 'L11', 'L12', 'L13', 'L14', 'L15',
    # 'M10', 'M11', 'M12', 'M13', 'M14', 'M15',
    # 'N10', 'N11', 'N12', 'N13', 'N14', 'N15',
    # 'O11', 'O12', 'O13', 'O14',
    # 'P12', 'P13',
    # 'J20', 'J21', 'J22', 'J23',                 # B3
    # 'K19', 'K20', 'K21', 'K22', 'K23', 'K24',
    # 'L19', 'L20', 'L21', 'L22', 'L23', 'L24',
    # 'M19', 'M20', 'M21', 'M22', 'M23', 'M24',
    # 'N19', 'N20', 'N21', 'N22', 'N23', 'N24',
    # 'O20', 'O21', 'O22', 'O23',
    # 'P21', 'P22'
]


def pipette(dest_plates, liquid_type, liquid_source, pip, waste_plate, waste_counter, plate_set):
    # atm using pipette with max volume 10uL and pipetting 1uL each time, will need to change if these change
    max_vol = 10
    vol_disp = 1

    dispense_count = 0

    pip.pick_up_tip()
    pip.aspirate(max_vol, liquid_source.bottom(), rate=0.5)
    pip.touch_tip()

    for plate in dest_plates:
        if plate['set'] == plate_set:
            wells = plate[liquid_type]
            agar = plate['agar']

            for well in wells:
                dispense_count += 1

                # discard the last few drops and re-aspirate
                if dispense_count > 7:
                    pip.blow_out(liquid_source)
                    pip.aspirate(max_vol, liquid_source.bottom(), rate=0.5)
                    pip.touch_tip()
                    dispense_count = 1

                # discard first drop to waste agar
                if dispense_count == 1:
                    pip.dispense(vol_disp, waste_plate[waste_destination[waste_counter[0]]].bottom(3), rate=0.5)
                    waste_counter[0] += 1  # increment waste position counter
                    dispense_count += 1

                pip.dispense(1, agar[well].bottom(3), rate=0.5)

    pip.drop_tip()


def run(protocol):

    # destination_plates = json.load(open(os.path.join('output/plate_config.json')))
    destination_plates = [{"set": 0, "position": "2", "BP": ["L22", "D4", "D13", "D22", "M4", "M13"], "TH": ["N21"], "IPTG": ["E5", "C12", "C21", "E23", "L5", "L12"]},
                          {"set": 0, "position": "3", "BP": ["L22", "D4", "D13", "D22", "M4", "M13"], "TH": ["N21"], "IPTG": ["E3", "E5", "C12", "C21", "E23", "N3", "L3", "N12", "L12", "L14"]},
                          {"set": 0, "position": "5", "BP": ["L22"], "TH": ["N21", "D4", "D13", "D22", "M4", "M13"], "IPTG": ["E4", "C13", "C22", "E22", "K2", "M14"]},
                          {"set": 0, "position": "6", "BP": ["L22"], "TH": ["N21", "D4", "D13", "D22", "M4", "M13"], "IPTG": ["D5", "F2", "B11", "B20", "F20", "O2", "M5", "O11", "M14", "K11"]},
                          {"set": 1, "position": "2", "BP": ["L22"], "TH": ["N21", "D4", "D13", "D22", "M4", "M13"], "IPTG": ["F6", "B11", "B20", "F24", "L4", "M12"]},
                          {"set": 1, "position": "3", "BP": ["L22"], "TH": ["N21", "D4", "D13", "D22", "M4", "M13"], "IPTG": ["D3", "E4", "C13", "C22", "E22", "N4", "M3", "N13", "M12", "L13"]},
                          {"set": 1, "position": "5", "BP": ["L22", "D4", "D13", "D22"], "TH": ["N21", "M4", "M13"], "IPTG": ["F4", "B13", "B22", "F22", "K6", "K11"]},
                          {"set": 1, "position": "6", "BP": ["L22"], "TH": ["N21", "D4", "D13", "D22", "M4", "M13"], "IPTG": ["F2", "F6", "B11", "B20", "F24", "O2", "K2", "O11", "K11", "K15"]},
                          {"set": 2, "position": "2", "BP": ["L22", "D4", "D13", "D22", "N4", "N13"], "TH": ["N21", "L2", "L11"], "IPTG": ["F5", "B12", "B21", "F23", "K5", "O12"]},
                          {"set": 2, "position": "3", "BP": ["L22", "C4", "C13", "C22", "N4", "N13"], "TH": ["N21", "E2", "E11", "E20", "L2", "L11"], "IPTG": ["B3", "F5", "D14", "D23", "F23", "M5", "O3", "M14", "O12", "K14"]},
                          {"set": 2, "position": "5", "BP": ["L22", "D4", "D13", "D22", "L5", "L14"], "TH": ["N21", "O3", "O12"], "IPTG": ["E4", "C13", "C22", "E22", "M5", "M10"]},
                          {"set": 2, "position": "6", "BP": ["L22", "E5", "E14", "E23", "L5", "L14"], "TH": ["N21", "B3", "B12", "B21", "O3", "O12"], "IPTG": ["D1", "D5", "F15", "F24", "D23", "K6", "M1", "K15", "M10", "M14"]}]
    deep96 = protocol.load_labware('nest_96_wellplate_2ml_deep', '4')
    waste_agar = protocol.load_labware('corning_384_wellplate_112ul_flat', '8')
    tip10 = protocol.load_labware('opentrons_96_filtertiprack_10ul', '1')

    p10 = protocol.load_instrument('p10_single',
                                   mount='left',
                                   tip_racks=[tip10])

    # THINGS TO CHANGE AT THE BEGINNING OF AN EXPERIMENT
    bp_source = deep96['A1']  # bandpass
    th_source = deep96['B1']  # threshold
    iptg_source = deep96['C1']  # 7.5mM IPTG
    p10.starting_tip = tip10.well('B10')
    plate_set = 1

    waste_position_counter = [0]  # this is a list to allow pass by reference

    for plate in destination_plates:
        if plate['set'] == plate_set:
            plate['agar'] = protocol.load_labware('corning_384_wellplate_112ul_flat', plate['position'])

    # do the bandpass receivers
    pipette(destination_plates, 'BP', bp_source, p10, waste_agar, waste_position_counter, plate_set)

    # do the threshold receivers
    pipette(destination_plates, 'TH', th_source, p10, waste_agar, waste_position_counter, plate_set)

    # do the IPTG
    pipette(destination_plates, 'IPTG', iptg_source, p10, waste_agar, waste_position_counter, plate_set)

