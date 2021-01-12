import itertools
import math
import numpy as np
import copy
from utilities import *
# three input combinitorics
def remove_degen_perms(perms, barrier_pos):
    #slice at barrier pos
    for i, p in enumerate(perms):
        perms[i] = [set(p[0:bar_pos]), set(p[bar_pos:])]


    logic_gates = []


    while len(perms) > 0:

        for j, p1 in enumerate(perms):


            logic_gates.append(p1)

            new_perms = []

            for i, p2 in enumerate(perms): #generate new perms with all p1 clones removed
                if not (p1[0] == p2[0] and p1[1] == p2[1]): #if degenerate

                    new_perms.append(p2)

            perms = new_perms

            break #force back to start with updated perms list

    return logic_gates

def get_missed_gates(ahl_logic_gates, logic_gates):

    missed_gates = []
    for i in range(len(logic_gates)):
        missed = True

        for j in range(len(ahl_logic_gates)):
            if logic_gates[i][0] == ahl_logic_gates[j][0] and logic_gates[i][1] == ahl_logic_gates[j][1]:
                missed = False

        if missed:
            missed_gates.append(logic_gates[i])

    return missed_gates


def draw_karnaugh(gate):

    map = np.zeros((4, 2))

    for i in [0, 1]:
        for state in gate[i]:
            A = state[0]
            BC = state[1:]

            col = int(A)

            if BC == '00':
                row = 0
            elif BC == '10':
                row = 1
            elif BC == '11':
                row = 2
            elif BC == '01':
                row = 3

            map[row,col] = i

    return map



print('Lets first enumerate all three input logic gates as a sanity check: ')
#enumerate all three input possible logic gates
total = 0
for i in range(9): #all barrier pos
    term = math.factorial(8)/(math.factorial(8-i)*math.factorial(i))
    print('Number of logic gates with barrier in pos ' + str(i) + ': ', term)
    total += term

print('Total number of three input logic gates: ', total)
print('This is correct!')
print()

print('Find number of distinct gates we can represent by swapping middle states using AHL conc: ')
states = ['000', '001', '010', '011', '100', '101', '110', '111']

middle_states = states[1:-1]
print(middle_states)

middle_perms = list(itertools.permutations(middle_states))

print('Total number of ways of arranging middle states: ', len(middle_perms))

satistfies_first_constraint = []

for mp in middle_perms:
    if mp.index('001') < mp.index('011') and mp.index('001') < mp.index('101'): #if satisfies first constraint
        satistfies_first_constraint.append(mp)

print('Number of states satisfying first constraint: ', len(satistfies_first_constraint))


sat_first_and_second = []

for mp in satistfies_first_constraint:
    if mp.index('010') < mp.index('011') and mp.index('010') < mp.index('110'): #if satisfies second constraint
        sat_first_and_second.append(mp)

print('Number of states satisfying first and second constraint: ', len(sat_first_and_second))


sat_all_constraints = []

for mp in sat_first_and_second:
    if mp.index('100') < mp.index('101') and mp.index('100') < mp.index('110'): #if satisfies second constraint
        sat_all_constraints.append(mp)


print('Number of states satisfying all constraints: ', len(sat_all_constraints))
for i, perm in enumerate(sat_all_constraints): # add the '000' and '111' states
    sat_all_constraints[i] = ['000'] + list(perm) + ['111']



# get all circular permutations of these, equivalent to having a bandpass and inverse
circ_perms = []
n = 8
for perm in sat_all_constraints:
    # get the CP:
    CPS = [[perm[i - j] for i in range(n)] for j in range(n)]
    for cp in CPS:
        circ_perms.append(cp)
print(sat_all_constraints)

print('Number of circular permutations (from bandpass and its inverse) of swapped middle states per barrier position using AHL concentration: ', len(circ_perms))
print()

# now we need to check that for all positions of the barrier we can put any state into either box


print('Now investigate with the barrier in all the different positions, so all three input logic gates with four states mapped to OFF four to ON')
print('Derive the number of actual three input logic gates: ')

all_logic_gates = []
all_ahl_logic_gates = []
total_ahl_lgs = 0
for bar_pos in range(9):
    print('Barrier in position ' ,str(bar_pos))
    # get all three input logic gates by getting all perumtations and removing degenerate ones
    perms = list(itertools.permutations(states))

    # now remove permutations that represent the same logic gates (check set euqulity between box assigments)
    logic_gates = remove_degen_perms(perms, bar_pos)
    all_logic_gates.append(logic_gates)
    print('The number of actual three input logic gates: ')
    print(len(logic_gates))

    '''
    for gate in logic_gates:
        print(gate)
    '''


    print('The number of three input logic gates possible with the inverse and normal bandpass: ')


    #now get all the logic gates possible using ahl conc
    circ_perms_new = copy.deepcopy(circ_perms)
    ahl_logic_gates = remove_degen_perms(circ_perms, bar_pos)
    all_ahl_logic_gates.append(ahl_logic_gates)
    print(len(ahl_logic_gates))
    print()
    total_ahl_lgs += len(ahl_logic_gates)
    circ_perms = circ_perms_new

print(total_ahl_lgs)
print()
for i in range(len(all_ahl_logic_gates[2])):
    print(all_ahl_logic_gates[2][i])
print()
for i in range(len(all_logic_gates[2])):
    print(all_logic_gates[2][i])

missed = get_missed_gates(all_ahl_logic_gates[2], all_logic_gates[2])
print()

'''
for m in missed:
    print(m)

print()
for m in missed:
    print(draw_karnaugh(m))
    print()
'''
'''
for g in all_ahl_logic_gates[2]:
    print()
    print(draw_karnaugh(g))
'''

for N_inputs in range(5):
    print('Number of inputs: ', N_inputs)
    for gate in all_ahl_logic_gates[N_inputs]:
        print(gate)
    print()
