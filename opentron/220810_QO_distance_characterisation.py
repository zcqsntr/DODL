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
    # 'A12', 'A13',                               # A2
    # 'B11', 'B12', 'B13', 'B14',
    # 'C10', 'C11', 'C12', 'C13', 'C14', 'C15',
    # 'D10', 'D11', 'D12', 'D13', 'D14', 'D15',
    # 'E10', 'E11', 'E12', 'E13', 'E14', 'E15',
    # 'F10', 'F11', 'F12', 'F13', 'F14', 'F15',
    # 'G11', 'G12', 'G13', 'G14',
    'A21', 'A22',                               # A3
    'B20', 'B21', 'B22', 'B23',
    'C19', 'C20', 'C21', 'C22', 'C23', 'C24',
    'D19', 'D20', 'D21', 'D22', 'D23', 'D24',
    'E19', 'E20', 'E21', 'E22', 'E23', 'E24',
    'F19', 'F20', 'F21', 'F22', 'F23', 'F24',
    'G20', 'G21', 'G22', 'G23',
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


def pipette(dest_wells, source_well, dest_location, pip, waste_plate, waste_counter):
    # atm using pipette with max volume 10uL and pipetting 1uL each time, will need to change if these change
    max_vol = 10
    vol_disp = 1

    dispense_count = 0

    pip.pick_up_tip()
    pip.aspirate(max_vol, source_well.bottom(), rate=0.5)
    pip.touch_tip()

    for well in dest_wells:
        dispense_count += 1
        # discard the last few drops and re-aspirate
        if dispense_count > 7:
            pip.blow_out(source_well)
            pip.aspirate(max_vol, source_well.bottom(), rate=0.5)
            pip.touch_tip()
            dispense_count = 1

        # discard first drop to waste agar
        if dispense_count == 1:
            pip.dispense(vol_disp, waste_plate[waste_destination[waste_counter[0]]].bottom(3), rate=0.5)
            waste_counter[0] += 1  # increment waste position counter
            dispense_count += 1

        pip.dispense(1, dest_location[well].bottom(3), rate=0.5)
    pip.drop_tip()
    

def run(protocol):
    agar = protocol.load_labware('corning_384_wellplate_112ul_flat', '2')
    waste_agar = protocol.load_labware('corning_384_wellplate_112ul_flat', '5')
    deep96 = protocol.load_labware('nest_96_wellplate_2ml_deep', '4')
    tip10 = protocol.load_labware('opentrons_96_filtertiprack_10ul', '1')
    p10 = protocol.load_instrument('p10_single', mount='left', tip_racks=[tip10])

    # CHANGE ME
    p10.starting_tip = tip10.well('D1')

    # CHANGE ME
    strain_1_source = deep96['A1']
    strain_2_source = deep96['B1']
    strain_3_source = deep96['C1']
    strain_4_source = deep96['D1']
    strain_5_source = deep96['E1']
    strain_6_source = deep96['F1']

    # CHANGE ME
    inducer_1_source = deep96['A2']
    inducer_2_source = deep96['B2']
    inducer_3_source = deep96['C2']
    inducer_4_source = deep96['D2']
    inducer_5_source = deep96['E2']

    inducer_1_destination = ['D4']
    inducer_2_destination = ['D13']
    inducer_3_destination = ['D22']
    inducer_4_destination = ['M4']
    inducer_5_destination = ['M13']

    strain_1_destination = ['A4', 'B2', 'C5', 'D3', 'E1', 'E6', 'F4', 'G2', 'K19']
    strain_2_destination = ['A13', 'B11', 'C14', 'D12', 'E10', 'E15', 'F13', 'G11', 'K21']
    strain_3_destination = ['A22', 'B20', 'C23', 'D21', 'E19', 'E24', 'F22', 'G20', 'K23']
    strain_4_destination = ['J2', 'K4', 'L1', 'L6', 'M3', 'N5', 'O2', 'P4', 'N22']
    strain_5_destination = ['J11', 'K13', 'L10', 'L15', 'M12', 'N14', 'O11', 'P13', 'N24']
    strain_6_destination = ['N20']

    waste_position_counter = [0]

    # dispense strains
    pipette(strain_1_destination, strain_1_source, agar, p10, waste_agar, waste_position_counter)
    pipette(strain_2_destination, strain_2_source, agar, p10, waste_agar, waste_position_counter)
    pipette(strain_3_destination, strain_3_source, agar, p10, waste_agar, waste_position_counter)
    pipette(strain_4_destination, strain_4_source, agar, p10, waste_agar, waste_position_counter)
    pipette(strain_5_destination, strain_5_source, agar, p10, waste_agar, waste_position_counter)
    pipette(strain_6_destination, strain_6_source, agar, p10, waste_agar, waste_position_counter)

    pipette(inducer_1_destination, inducer_1_source, agar, p10, waste_agar, waste_position_counter)
    pipette(inducer_2_destination, inducer_2_source, agar, p10, waste_agar, waste_position_counter)
    pipette(inducer_3_destination, inducer_3_source, agar, p10, waste_agar, waste_position_counter)
    pipette(inducer_4_destination, inducer_4_source, agar, p10, waste_agar, waste_position_counter)
    pipette(inducer_5_destination, inducer_5_source, agar, p10, waste_agar, waste_position_counter)
