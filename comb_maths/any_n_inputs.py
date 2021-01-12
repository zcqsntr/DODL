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

def pad_binary(n, n_inputs):
    '''
    pads binary with zeros if not all bits assigned
    '''
    n_zeros = n_inputs - len(n)

    return '0' * n_zeros + n

def get_shared_ones(below, above):
    shared_ones = []
    for i in range(len(below)):
        if below[i] == '1' and above[i] == '1':
            shared_ones.append(i)
    return shared_ones

def apply_constraints(perms, n_inputs):

    # get all intermediate states
    intermediate_states = []

    for i in range(1, n_inputs):

        # create every permutation of state with i ones
        state = ['0'] * (n_inputs - i) + ['1']*i
        states = list(set(itertools.permutations(state)))

        intermediate_states.append(states)


    testing = perms
    # enumerate all constraint first to test against all at once, otherwise list fills up ram

    constraints = []
    for i in range(len(intermediate_states)-1):
        # check that order is conserved
        below = intermediate_states[i]
        for above in intermediate_states[i+1:]:
            for b in below:
                for a in above:

                    a = ''.join(a)
                    b = ''.join(b)

                    # if a and b have i ones in the same position then we know that b is constrained to be on the left of a
                    if len(get_shared_ones(b, a)) == i+1:
                        print(b,a)
                        constraints.append((b,a))
                        print(len(constraints))



    satisfies_constraints = []
    for i,p in enumerate(testing):

        if i% 100000 == 0:
            print(i)
        violated_constraint = False

        for (b,a) in constraints:

            if p.index(b) > p.index(a):
                violated_constraint = True
                #print(len(satisfies_constraints))
                break


        if not violated_constraint:

            satisfies_constraints.append(p)

    return satisfies_constraints

def expand_nested_tuple(tuple):

    if type(tuple[0]) is str:
        return list(tuple)

    return expand_nested_tuple(tuple[0]) + expand_nested_tuple(tuple[1])

def get_middle_perms(n_inputs):
    '''
    TODO: generalise beyond 3 inputs
    '''
    #create all sets of1-n_inputs on
    sets = []
    all_states = []
    for i in range(1, n_inputs):
        state = ['0'] * (n_inputs - i) + ['1']*i
        states = [''.join(s) for s in set(itertools.permutations(state))]

        all_states.append(states)
        perms = itertools.permutations(states)

        sets.append(perms)

    # get all possible orders of states with no mixing
    unmixed = list(itertools.product(sets[0], sets[1]))

    for i,p in enumerate(sets):
        if i > 1:
            unmixed = list(itertools.product(unmixed, sets[i]))


    #convert to list
    for i, u in enumerate(unmixed):

        unmixed[i] = expand_nested_tuple(u)

    print(all_states)
    #now get the mixed states

    '''
    MIGHT HAVE TO DO THIS RECURSIVELY

    '''
    all_mixed = []
    for i, s in enumerate(all_states): # for each possible mixing point
        for state0 in all_states[i]:

            for higher_states in all_states[i+1:]:
                for state1 in higher_states:

                    if len(get_shared_ones(state0, state1)) < i + 1:
                        temp0 = copy.deepcopy(all_states[i])
                        temp1 = copy.deepcopy(higher_states)

                        temp0.remove(state0)
                        temp1.remove(state1)

                        comb_remaining = itertools.product(itertools.permutations(temp0), itertools.permutations(temp1))
                        

                        mixed = [list(c[0]) + [state1] + [state0] + list(c[1]) for c in comb_remaining]
                        print('mixed: ',mixed)
                        all_mixed += mixed
    #print('mixed: ', all_mixed[-1])
    return unmixed + all_mixed


n_inputs = 3

print('Lets first enumerate all four input logic gates as a sanity check: ')
#enumerate all three input possible logic gates

total = 0
for i in range(2**n_inputs + 1): #all barrier pos
    term = math.factorial(2**n_inputs)/(math.factorial(2**n_inputs-i)*math.factorial(i))
    print('Number of logic gates with barrier in pos ' + str(i) + ': ', term)
    total += term

print('Total number of three input logic gates: ', total)
print('This is correct!')
print()

print('Find number of distinct gates we can represent by swapping middle states using AHL conc: ')
states = [pad_binary(str(bin(i))[2:], n_inputs) for i in range(2**n_inputs)]

middle_states = states[1:-1]

middle_perms = itertools.permutations(middle_states)
print('done middle perms')
#print('Total number of ways of arranging middle states: ', len(middle_perms))
sat_all_constraints = get_middle_perms(n_inputs)

print('Number of states satisfying all constraints: ', len(sat_all_constraints))
for i, perm in enumerate(sat_all_constraints): # add the '000' and '111' states
    sat_all_constraints[i] = ['000'] + list(perm) + ['111']
# get all circular permutations of these, equivalent to having a bandpass and inverse
circ_perms = []
n = 2**n_inputs
for perm in sat_all_constraints:

    # get the CP:
    CPS = [[perm[i - j] for i in range(n)] for j in range(n)]
    for cp in CPS:
        circ_perms.append(cp)


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
    print('Number of OFFs: ', N_inputs)
    for gate in all_ahl_logic_gates[N_inputs]:
        print(gate)
    print()
