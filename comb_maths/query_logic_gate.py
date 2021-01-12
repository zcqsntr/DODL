import itertools
import math
import numpy as np
import copy
from utilities import *
import sys


def get_shared_ones(below, above):
    shared_ones = []
    for i in range(len(below)):
        if below[i] == '1' and above[i] == '1':
            shared_ones.append(i)
    return shared_ones

def count_ones(state):

    sum = 0
    for s in state:
        if s =='1':
            sum += 1
    return sum

def apply_constraints(test_states):
    # tests whether a state ordering has violated any constraints

    n_inputs = math.log(math.log(len(logic(gate)), 2), 2)

    # get all intermediate states
    intermediate_states = []

    for i in range(1, n_inputs):

        # create every permutation of state with i ones
        state = ['0'] * (n_inputs - i) + ['1']*i
        states = list(set(itertools.permutations(state)))

        intermediate_states.append(states)

    violated_constraint = False

    # enumerate all constraint first to test against all at once, otherwise list fills up ram
    constraints = []
    for i in range(len(test_states)):
        # check that order is conserved
        below = test_states[i]
        for above in test_states[i+1:]:

            #above = ''.join(above)
            #below = ''.join(below)

            # if a and b have i ones in the same position then we know that b is constrained to be on the left of a
            if len(get_shared_ones(below, above)) == count_ones(below):

                violated_constraint = True
                        #print(len(satisfies_constraints))
                break

    return not violated_constraint

def pad_binary(n, n_inputs):
    '''
    pads binary with zeros if not all bits assigned
    '''
    n_zeros = n_inputs - len(n)

    return '0' * n_zeros + n


if __name__ == '__main__':

    logic_gate = sys.argv[1]

    states_in_off = []
    states_in_on = []


    for i in range(len(logic_gate)):


    n_inputs = math.log(math.log(len(logic(gate)), 2), 2)

    states = [pad_binary(str(bin(i))[2:], n_inputs) for i in range(2**n_inputs)]




    # take logic gate and map it to where the input
