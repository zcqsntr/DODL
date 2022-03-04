#!/usr/bin/env python
import sys
import os
import numpy as np
import copy
from collections import OrderedDict
import queue as qu
import math
import itertools
import json
import argparse


dir_path = os.path.dirname(os.path.realpath(__file__))

def get_blocks(truth_table):
    #returns the of blocks of 0s and 1s and the number of 1 blocks
    outputs = truth_table[:, -1]

    block = outputs[0]
    blocks = [block]
    block_sizes = [1]

    n_ones = 0

    for i in range(1,len(outputs)):
        if outputs[i] != block:

            block = outputs[i]
            n_ones += block #count the ones
            blocks.append(block)
            block_sizes.append(1)
        else:
            block_sizes[-1] += 1

    return np.vstack((blocks, block_sizes)).T, n_ones

def check_constraints(lower, upper):
    #checks whther putting lower and upper in this order is valid
    return np.sum(lower*upper) < np.sum(upper)

def create_truth_table(outputs):
    # create inputs table
    inputs_table = []
    n_inputs = int(np.log2(outputs.size))
    for n in range(outputs.size):
        b_string = "{0:b}".format(n)
        b_string = b_string.rjust(n_inputs, '0')
        b_list = list(map(int, list(b_string)))
        inputs_table.append(b_list)

    inputs_table = np.array(inputs_table)

    truth_table = np.hstack((inputs_table, outputs.reshape(-1, 2 ** n_inputs).T))
    return truth_table

def hash_table(truth_table):
    #hashes truth table into a unique string so we can keep track of which ones have been visited
    hashed = ''
    for input_states in truth_table[:, :-1]:
        for s in input_states:
            hashed += str(s)
    return hashed


def simplify(state_mapping, n_inputs):
    redenduant_inputs = []
    has_factored = False

    for input in range(n_inputs):  # test each input
        can_factor = True

        for s in range(len(state_mapping)):
            states = np.array(copy.deepcopy(state_mapping[s]))
            states[:, input] = -1  # remove one input and test for degeneracy

            distinct_states = map(tuple, states)
            distinct_states = set(distinct_states)



            if len(distinct_states) > len(states) / 2:
                can_factor = False
                break

        if can_factor:
            has_factored = True
            redenduant_inputs.append(input)

    if has_factored:
        new_state_mapping = []

        #build reduced state mapping
        for s in range(len(state_mapping)):
            states = copy.deepcopy(state_mapping[s])
            states[:, redenduant_inputs] = -1  # remove one input and test for degeneracy

            distinct_states = map(tuple, states)
            distinct_states = set(distinct_states)
            new_state_mapping.append(distinct_states)
    else:
        new_state_mapping = state_mapping

    return new_state_mapping

def graph_search(truth_table, target_n_nodes = -1):

    discovered_tables = {hash_table(truth_table)} # use a set for this

    blocks, n_ones = get_blocks(truth_table)
    truth_tables = qu.PriorityQueue()


    truth_tables.put((n_ones, len(discovered_tables), truth_table))
    best_table = truth_table


    while not truth_tables.empty():


        n_ones, _, truth_table = truth_tables.get()  # BFS or DFS depending on this line


        if n_ones < get_blocks(best_table)[1]:
            best_table = truth_table

        if n_ones <= target_n_nodes: #exit condition

            return best_table, discovered_tables

        blocks, _ = get_blocks(truth_table)

        # look at smallest block and see if we can move it to reduce the number of blocks
        indices = np.argsort(blocks[:, 1])

        for i in indices[indices != 0][indices[indices != 0] != max(indices)]:  #we can never move the end states, only move other states into their blocks

            block_start = np.sum(blocks[0:i, 1])
            block_size = blocks[i, 1]


            for s, state in enumerate(truth_table[block_start:block_start+block_size, :-1]): # for each state in the block


                lower = truth_table[block_start+s -1, :-1]
                higher = truth_table[block_start + s + 1, :-1]


                if check_constraints(state, lower): #check if we can put state below

                    new_truth_table = copy.deepcopy(truth_table)
                    new_truth_table[[s+block_start -1, s+block_start], :] = new_truth_table[[s+block_start, s+block_start-1], :]
                    new_blocks, new_n_ones = get_blocks(new_truth_table)
                    #if len(new_blocks) <= len(blocks) : #dont accept swaps that increase the number of blocks
                    if hash_table(new_truth_table) not in discovered_tables:
                        discovered_tables.add(hash_table(new_truth_table))
                        truth_tables.put((new_n_ones, len(discovered_tables), new_truth_table))


                if check_constraints(higher, state): #check if we can put state above

                    new_truth_table = copy.deepcopy(truth_table)
                    new_truth_table[[s+block_start, 1+s+block_start], :] = new_truth_table[[s+1+block_start, s+block_start], :]

                    new_blocks, new_n_ones = get_blocks(new_truth_table)

                    #if len(new_blocks) <= len(blocks) :  # dont accpt swaps that increase the number of blocks
                    if hash_table(new_truth_table) not in discovered_tables:
                        discovered_tables.add(hash_table(new_truth_table))
                        truth_tables.put((new_n_ones, len(discovered_tables), new_truth_table))


    return best_table, discovered_tables

def get_activations(best_table, n_inputs, allowed_acts = ['TH', 'IT', 'BP', 'IB']):

    blocks = get_blocks(best_table)[0]

    pos = 0

    activations = OrderedDict()

    if 'IB' in allowed_acts and len(blocks) == 3 and np.all(blocks[:, 0] == np.array([1, 0, 1])):  # only situation where IBP is admissable

        activations['IB'] = []
        this_IB = []
        for i in range(3):
            start = np.sum(blocks[0:pos + i, 1])
            end = np.sum(blocks[0:pos + i + 1, 1])
            this_IB.append(best_table[start:end, :n_inputs].tolist())
        activations['IB'].append(this_IB)
        pos = len(blocks) - 1


    while pos < len(blocks) - 1:

        if 'BP' in allowed_acts and pos < len(blocks) - 2 and np.all(blocks[pos:pos + 3, 0] == np.array([0, 1, 0])):

            # can have multiple bp
            this_BP = []

            # need to put all off states before and after the bandoass into its off set
            boundary = np.sum(blocks[0:pos + 1, 1])
            off_set = best_table[0: boundary, :n_inputs]
            # off_set = best_table[best_table[:, -1] == 0][0: boundary, :n_inputs]
            this_BP.append(off_set.tolist())
            boundary1 = np.sum(blocks[0:pos + 2, 1])

            on_set = best_table[boundary: boundary1, :n_inputs]
            this_BP.append(on_set.tolist())
            off_set = best_table[boundary1:, :n_inputs]
            # off_set = best_table[best_table[:, -1] == 0][boundary:, :n_inputs]
            this_BP.append(off_set.tolist())
            pos += 2

            if 'BP' in activations.keys():
                activations['BP'].append(this_BP)
            else:
                activations['BP'] = []
                activations['BP'].append(this_BP)


        elif 'IT' in allowed_acts and pos == 0 and np.all(blocks[0:2, 0] == np.array([1, 0])):  # inverse threshold can only be at beginning

            this_IT = []
            boundary = np.sum(blocks[0:pos + 1, 1])
            on_set = best_table[0: boundary, :n_inputs]
            this_IT.append(on_set.tolist())
            off_set = best_table[boundary:, :n_inputs]
            # off_set = best_table[best_table[:, -1] == 0][boundary:, :n_inputs]
            this_IT.append(off_set.tolist())
            pos += 1


            activations['IT'] = []
            activations['IT'].append(this_IT)  # can only have one inverse threhsold


        elif 'TH' in allowed_acts and  pos == len(blocks) - 2 and np.all(blocks[-2:, 0] == np.array([0, 1])):  # threshld can only be at end
            start = np.sum(blocks[0:pos, 1])
            activations['TH'] = []  # can only have one threshold

            # need to put all previous off states into thresholds off state
            boundary = np.sum(blocks[0:pos + 1, 1])
            this_TH = []
            # off_set = best_table[best_table[:, -1] == 0][0: boundary, :n_inputs]
            off_set = best_table[0: boundary, :n_inputs]
            this_TH.append(off_set.tolist())

            on_set = best_table[boundary:, :n_inputs]
            this_TH.append(on_set.tolist())
            activations['TH'].append(this_TH)
            pos += 1
        else: #failed to assign and so gate isnt possible
            return -1



    return activations


def covers_from_blocks(blocks):
    # returns the start position and size of each cover
    indices = np.where(blocks[:, 0] == 1)[0]

    sizes = blocks[indices, 1]

    start_pos = [np.sum(blocks[:i, 1]) for i in indices]

    return np.vstack((start_pos, sizes)).T


def can_move(truth_table, frm, to):
    # checks if a state can be moved from a to b based on the AHL concentration constraints

    # if frm < to then to be able to move the state must be able to swap above all states between frm and to
    can_swap = True


    if frm < to:
        for i in range(frm + 1, to + 1):
            can_swap = can_swap and check_constraints(truth_table[i, :-1], truth_table[frm, :-1])
    elif to < frm:
        for i in range(to, frm):
            can_swap = can_swap and check_constraints(truth_table[frm, :-1], truth_table[i, :-1])

    return can_swap


def move(truth_table, frm, to):
    # moves a state from a to b by movin gthe states and shifting all states between a and b down or up one as required

    new_table = copy.deepcopy(truth_table)

    if frm < to:
        lower = new_table[0:frm, :]
        from_state = new_table[frm, :]
        middle = new_table[frm+1:to+1, :]
        upper = new_table[to+1:, :]

        table = np.vstack((lower, middle, from_state, upper))


    elif to < frm:
        lower = new_table[0:to, :]
        from_state = new_table[frm, :]
        middle = new_table[to:frm, :]
        upper = new_table[frm + 1:, :]

        table = np.vstack((lower, from_state, middle,  upper))
    else:
        table = new_table

    return table


def modify_covers(covers, frm, to):
    # updates the list of covers based on what move has happened


    new_covers = []

    if frm < to: # moved forwards
        for cover in covers:
            if cover[0] <= frm <= cover[0] + cover [1]:
                cover[1] -= 1
            elif cover[0] <= to <= cover[0] + cover [1]:
                cover[0] -=1 #start moves down
                cover[1] += 1
            new_covers.append(cover)

    elif to < frm:
        for cover in covers:
            if cover[0] <= frm <= cover[0] + cover[1]:
                cover[0] += 1
                cover[1] -= 1
            elif cover[0] <= to <= cover[0] + cover[1]:
                cover[1] += 1
            new_covers.append(cover)

    return np.array(new_covers)

def sort_truth_table(truth_table):
    '''
    sort according to the number of inputs activated instead of binary orders
    '''
    n_inputs = int(np.log2(truth_table.shape[0]))
    sum_inputs = np.sum(truth_table[:, :n_inputs], axis = 1)
    indices = np.argsort(sum_inputs)

    return truth_table[indices]

def count_output_blocks(output_blocks):

    '''
    counts the number of blocks of ones

    '''

    current = output_blocks[0][0]
    n_blocks = 0
    if current == 1:
        n_blocks += 1

    for input_group in output_blocks:
        for block in input_group:
            if block == 1 and current == 0:
                n_blocks += 1
            current = block

    return n_blocks


def rough_optimisation(truth_table):
    '''
    optimises based on matching he 0s and ones between the different numbers of inputs, gives a really good starting point
    for earl grey to do the final bit of opt
    '''
    n_inputs = int(np.log2(truth_table.shape[0]))


    truth_table = sort_truth_table(truth_table)

    # split into groups based on how many inputs are activated

    input_groups = []
    counter = 0
    for i in range(n_inputs+1):
        n_states = int(math.factorial(n_inputs)/(math.factorial(i)*math.factorial(n_inputs-i)))
        group = truth_table[counter:counter+n_states ]
        sorted_group = group[np.argsort([group[:, -1]])]
        input_groups.append(sorted_group[0])
        counter += n_states


    output_blocks = []
    flips = [] # whether or not each input group has been flipped
    flippable = [] #which input groups have 0s and 1s and therofore can be flipped
    for i,group in enumerate(input_groups):

        active_outputs = np.sum(group[:,-1])

        if active_outputs == 0:
            output_blocks.append([0])
        elif active_outputs == group.shape[0]:
            output_blocks.append([1])
        else:
            flips.append(False)
            output_blocks.append([0,1])
            flippable.append(i)

    # no go through each flip combination and find the one with the smallest number of blocks

    min_blocks = count_output_blocks(output_blocks)
    best_blocks = copy.deepcopy(output_blocks)
    best_flip_comb = flips

    flip_combs = list(itertools.product([False, True], repeat=len(flippable)))

    for flip_comb in flip_combs:
        for i, flip in enumerate(flip_comb):
            if flip:
                output_blocks[flippable[i]] = [1,0]
            else:
                output_blocks[flippable[i]] = [0,1]

        n_blocks = count_output_blocks(output_blocks)
        if n_blocks < min_blocks:

            best_blocks = copy.deepcopy(output_blocks)
            best_flip_comb = list(copy.deepcopy(flip_comb))
            min_blocks = n_blocks


    # assemble the new truth table base don which input groups are flipped

    new_input_groups = copy.deepcopy(input_groups)

    for i, ind in enumerate(flippable):
        if best_flip_comb[i]:
            new_input_groups[ind] = np.flip(new_input_groups[ind], axis = 0)

    return np.vstack(new_input_groups), min_blocks

def macchiato(outputs):
    n_inputs = int(np.log2(outputs.size))
    truth_table = create_truth_table(outputs)
    truth_table,_ = rough_optimisation(truth_table)

    current_table = copy.deepcopy(truth_table)
    current_table = truth_table

    will_exit = True


    finished = False

    while not finished:

        finished = True
        blocks = get_blocks(current_table)[0]

        covers = covers_from_blocks(blocks)

        # each block of ones is a cover, try and maximise each cover in turn, starting from largest cover
        cov_sort = np.argsort(covers[:, 1])  # this will bias towards states in the middle, probably not what we want


        test_table = copy.deepcopy(current_table)

        for index in cov_sort:
            # get smallest cover
            smallest_cover = covers[index]


            small_start = smallest_cover[0]
            small_end = smallest_cover[0] + smallest_cover[1]


            # try and eliminate smallest cover by puttin gones into the covers on either side


            if index > 0:
                lower_cover = covers[index -1]
            else:
                lower_cover = None

            if index < len(covers) - 1:
                higher_cover = covers[index+ 1]
            else:
                higher_cover = None

            test_table = copy.deepcopy(current_table)


            if lower_cover is not None:

                # go through oes and put as many into the lower cover as possible
                for i, state in enumerate(range(small_start, small_end)):

                    if can_move(test_table, state, lower_cover[0] + lower_cover[1] + i):
                        test_table = move(test_table, state, lower_cover[0] + lower_cover[1] + i)
                        smallest_cover[0] += 1
                        smallest_cover[1] -=1

            small_start = smallest_cover[0]
            small_end = smallest_cover[0] + smallest_cover[1]

            if higher_cover is not None:
                for i, state in enumerate(range(small_start, small_end)[::-1]):
                    if can_move(test_table, state, higher_cover[0] - i - 1):
                        test_table = move(test_table, state, higher_cover[0] - i - 1)
                        smallest_cover[1] -= 1

            if smallest_cover[1] == 0:
                current_table = test_table
                finished = False
                break

    return current_table


parser = argparse.ArgumentParser(description='Run the Macchiato algorithm')
parser.add_argument('outputs', metavar='T', type=str, nargs=1, help='the output of the truth table to be encoded')
parser.add_argument('--outpath', type=str, help='the filepath to save output in, default is Macchiato/output')

if __name__ == '__main__':
    args = parser.parse_args()


    try:
        outputs = np.array(list(args.outputs[0]), dtype=np.int32)
        truth_table = create_truth_table(outputs)
    except ValueError as e:
        print()
        print('CHECK THE NUMBER AND TYPE OF OUTPUTS SUPPLIED, IT MUST BE 2^n NUMBERS FROM {0,1}, WHERE n IS THE NUMBER OF INPUTS TO THE DIGITAL FUNCTION!!')
        print('e.g. a valid three input function is encoded like: 01011100')
        print()
        print('The program chrashed with the following error:')
        print()
        raise e

    print('Truth table:')
    print(truth_table)
    print()

    out_path = args.outpath
    if out_path is None:
        out_path = os.path.join(dir_path, 'output')


    os.makedirs(out_path, exist_ok=True)

    n_inputs = int(np.log2(outputs.size))

    best_table = macchiato(outputs)
    print('Simplified truth table: ')
    print(best_table)
    print()

    colonies = get_activations(best_table, n_inputs)

    print('Colony mapping: ')
    new_colonies = {}
    for act in colonies.keys():
        new_activations = []
        for state_mapping in colonies[act]:


            simplified_state_mapping = simplify(state_mapping, n_inputs)
            new_activations.append(simplified_state_mapping)
            print(act)
            for s in simplified_state_mapping:
                print(s)
        new_colonies[act] = new_activations
    print()


    results_dict = {'truth_table': truth_table.tolist(), 'simplified_table': best_table.tolist(), 'colonies': colonies}



    json.dump(results_dict, open(os.path.join(out_path,'{}.json'.format(sys.argv[1])), 'w+'))

    print('Results saved in ',os.path.join(out_path,'{}.json'.format(sys.argv[1])))
















