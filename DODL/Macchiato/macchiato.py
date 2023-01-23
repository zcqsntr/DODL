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
    #TODO:: compatibility with dont care set
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

def get_conflicting_constraints(truth_table):
    inputs = truth_table[1:-1, :-1] # we can ignore the 000 and 111 ends


    relations = []
    violated = []
    count = 0
    for i in range(len(inputs)):
        for j in range(i+1, len(inputs)):
            AND = inputs[i, :] * inputs[j,:]
            if np.sum(inputs[i, :]-AND) != 0:
                relations.append([inputs[i, :]-AND, inputs[j,:] -AND])
                count += 1
                if not check_constraints(inputs[i, :], inputs[j,:]):
                    violated.append('CONC')

    for i in range(len(relations)):
        for j in range(len(relations)):
            r = relations[i]
            r2 = relations[j]
            if np.all(r[0] == r2[1]) and np.all(r2[0] == r[1]):

                violated.append([r, r2])

    return violated

def create_truth_table(outputs, string = False):
    # create inputs table
    inputs_table = []
    n_inputs = int(np.log2(outputs.size))
    for n in range(outputs.size):
        b_string = "{0:b}".format(n)
        b_string = b_string.rjust(n_inputs, '0')
        b_list = list(map(int, list(b_string)))

        if string:
            inputs_table.append(b_string)
        else:
            inputs_table.append(b_list)



    if not string:
        inputs_table = np.array(inputs_table)
        truth_table = np.hstack((inputs_table, outputs.reshape(-1, 2 ** n_inputs).T))
    else:
        truth_table = inputs_table
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




def covers_from_blocks(blocks):

    current_pos = 0
    in_cover = False
    start_pos = []
    sizes = []
    scores = []


    if np.sum(blocks[:, 0]) == 0: # if all zeros
        return np.vstack(([], [], [])).T

    for b, block in enumerate(blocks):

        if not in_cover and block[0] in [1,2]:
            start = current_pos
            in_cover = True
            size = block[1]
            if block[0] == 1:
                score = block[1]
            else:
                score = 0


        elif in_cover and block[0] in [1, 2]:

            size += block[1]

            if block[0] == 1:
                score += block[1]

        if (in_cover and block[0] == 0) or b == len(blocks)-1 :
            start_pos.append(start)
            sizes.append(size)
            scores.append(score)
            in_cover = False



        current_pos += block[1]

    return np.vstack((start_pos, sizes, scores)).T


def can_move(truth_table, frm, to, higher_order = True):
    # checks if a state can be moved from a to b based on the AHL concentration constraints

    # if frm < to then to be able to move the state must be able to swap above all states between frm and to
    can_swap = True


    if frm < to:
        for i in range(frm + 1, to + 1):
            can_swap = can_swap and check_constraints(truth_table[i, :-1], truth_table[frm, :-1])
    elif to < frm:
        for i in range(to, frm):
            can_swap = can_swap and check_constraints(truth_table[frm, :-1], truth_table[i, :-1])

    if can_swap and higher_order: # now check new table doesnt violate any higher order constraints
        test = move(truth_table, frm, to)
        conflicts = get_conflicting_constraints(test)
        can_swap = can_swap and len(conflicts) == 0


    return can_swap




def move(truth_table, frm, to):
    # moves a state from a to b by moving the states and shifting all states between a and b down or up one as required

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

def group_inputs(truth_table):
    n_inputs = int(np.log2(truth_table.shape[0]))
    # split into groups based on how many inputs are activateds
    input_groups = []
    counter = 0
    for i in range(n_inputs + 1):
        n_states = int(math.factorial(n_inputs) / (math.factorial(i) * math.factorial(n_inputs - i)))
        group = truth_table[counter:counter + n_states]
        sorted_group = group[np.argsort([group[:, -1]])]

        input_groups.append(sorted_group[0])
        counter += n_states
    return input_groups

def get_possible_flips(input_groups):
    output_blocks = []
    flips = []  # whether or not each input group has been flipped
    flippable = []  # which input groups have 0s and 1s and therofore can be flipped
    for i, group in enumerate(input_groups):

        active_outputs = np.sum(group[:, -1])

        if active_outputs == 0:
            output_blocks.append([0])
        elif active_outputs == group.shape[0]:
            output_blocks.append([1])
        else:
            flips.append(False)
            output_blocks.append([0, 1])
            flippable.append(i)
    return output_blocks, flips, flippable


def find_best_flips(input_groups, output_blocks, flips, flippable, higher_order = False):
    # no go through each flip combination and find the one with the smallest number of blocks

    min_blocks = count_output_blocks(output_blocks)

    best_flip_comb = flips

    flip_combs = list(itertools.product([False, True], repeat=len(flippable)))

    for flip_comb in flip_combs:
        for i, flip in enumerate(flip_comb):
            if flip:
                output_blocks[flippable[i]] = [1, 0]
            else:
                output_blocks[flippable[i]] = [0, 1]

        n_blocks = count_output_blocks(output_blocks)

        new_table = flip_table(input_groups, flip_comb, flippable)

        if n_blocks < min_blocks and (not higher_order or len(get_conflicting_constraints(new_table)) == 0):  # and no contraints violated

            best_flip_comb = list(copy.deepcopy(flip_comb))
            min_blocks = n_blocks

    return best_flip_comb, min_blocks

def flip_table(input_groups, best_flip_comb, flippable):
    # assemble the new truth table base don which input groups are flipped
    new_input_groups = copy.deepcopy(input_groups)

    for i, ind in enumerate(flippable):
        if best_flip_comb[i]:
            new_input_groups[ind] = np.flip(new_input_groups[ind], axis=0)
    new_table = np.vstack(new_input_groups)

    return new_table

def rough_optimisation(truth_table, higher_order = True):
    '''
    optimises based on matching the 0s and ones between the different numbers of inputs, gives a really good starting point
    for earl grey to do the final bit of opt
    '''

    truth_table = sort_truth_table(truth_table)

    input_groups = group_inputs(truth_table)

    output_blocks, flips, flippable = get_possible_flips(input_groups)

    best_flip_comb, min_blocks = find_best_flips(input_groups, output_blocks, flips, flippable, higher_order=higher_order)

    new_table = flip_table(input_groups, best_flip_comb, flippable)

    if len(get_conflicting_constraints(truth_table)) > 0:
        print('CONFLICTS')
    return new_table, min_blocks



def hash_table(truth_table):
    #hashes truth table into a unique string so we can keep track of which ones have been visited
    hashed = ''
    for input_states in truth_table[:, :-1]:
        for s in input_states:
            hashed += str(s)
    return hashed



def least_blocks_obj(truth_table):
    blocks, n_ones = get_blocks(truth_table)
    return n_ones

def greedy_obj(truth_table, allowed_acts = None):
    blocks, n_ones = get_blocks(truth_table)
    covers = covers_from_blocks(blocks)
    scores = [0]
    if len(covers) == 0:
        return 0

    if 'IB' in allowed_acts and truth_table[0, -1] in [1,2]:
        if truth_table[-1, -1] in [1, 2]:
            scores.append((covers[0, 2] + covers[-1, 2]))
        else:
            scores.append(covers[0, 2])
    if 'TH' in allowed_acts  and truth_table[-1, -1] in [1,2]:
        scores.append(covers[-1, 2])
    if 'IT' in allowed_acts and truth_table[0, -1] in [1,2]:
        scores.append(covers[0, 2])
    if 'BP' in allowed_acts:
        s = list(covers[:, 2])

        if truth_table[0, -1] in [1, 2]:
            s = s[1:]
        scores.extend(s)


    return -np.max(scores)


def get_covered(truth_table, allowed_acts = ['TH', 'IT', 'IB', 'BP']):
    blocks, n_ones = get_blocks(truth_table)
    covers = covers_from_blocks(blocks)
    if len(covers) == 0:
        return []

    allowed_covers = []
    covering_receivers = []


    if 'TH' in allowed_acts and truth_table[-1, -1] in [1, 2]:
        allowed_covers.append(covers[-1])
        covering_receivers.append('TH')

    if 'BP' in allowed_acts:
        if truth_table[0, -1] in [1, 2]:
            c = covers[1:]
        else:
            c = covers
        allowed_covers.extend(c)
        covering_receivers.extend(['BP']*len(c))

    if 'IT' in allowed_acts and truth_table[0, -1] in [1, 2]:
        allowed_covers.append(covers[0])
        covering_receivers.append('IT')


    if 'IB' in allowed_acts and truth_table[0, -1] in [1, 2]:

        if truth_table[0, -1] in [1] and truth_table[-1, -1] in [1] and (covers[0, 2] + covers[-1, 2]) > np.max(covers[:, 2]): # if inverse bandpass is the best option
            return [covers[0], covers[-1]], 'IB'
        else:
            allowed_covers.append(covers[0])
            covering_receivers.append('IB')

    if len(allowed_covers) == 0:
        return [], 'NA'

    allowed_covers = np.array(allowed_covers)
    covering_receivers = np.array(covering_receivers)

    covered = [allowed_covers[np.argmax(allowed_covers[:, 2])]]
    covering_receiver = covering_receivers[np.argmax(allowed_covers[:, 2])]

    return covered, covering_receiver


def check_top_move(truth_table, frm):
    # checks if the input state at index frm can be moved into the top block
    # TODO: test
    blocks, n_ones = get_blocks(truth_table)


    if blocks[-1][0] == 0:
        return False

    to = np.sum(blocks[:-1, 1])

    return can_move(truth_table, to, frm)


def check_bot_move(truth_table, frm):
    # checks if the input state at index frm can be moved into the top block
    #TODO: test
    blocks, n_ones = get_blocks(truth_table)


    if blocks[0][0] == 0:
        return False

    to = blocks[0, 1]

    return can_move(truth_table, to, frm)




def graph_search(truth_table, objective=least_blocks_obj, max_queue_size=0):
    '''
    :param outputs:
    :param max_queue_size:
    :param objective:  the function that is to be minimised by the graph search, takes a truth table and outputs a scalar
    :return:
    '''





    #truth_table, _ = rough_optimisation(truth_table, hgi)
    discovered_tables = {hash_table(truth_table)} # use a set for this


    obj = objective(truth_table)

    truth_tables = qu.PriorityQueue(maxsize=max_queue_size)

    truth_tables.put((obj, len(discovered_tables), truth_table), block = False)
    best_table = truth_table

    if -objective(best_table) == np.sum(best_table[:, -1][best_table[:,
                                                          -1] == 1]):  # early exit condition if all ones are contributing to the objective
        print('len:', len(discovered_tables))
        return best_table




    while not truth_tables.empty():

        #print(len(discovered_tables))
        #print(truth_tables.qsize())

        obj, _, truth_table = truth_tables.get(block = False)  # BFS or DFS depending on this line



        if obj < objective(best_table) and len(get_conflicting_constraints(truth_table)) == 0:
            best_table = truth_table


            if -objective(best_table) == np.sum(best_table[:, -1][best_table[:, -1] == 1]): # early exit condition if all ones are contributing to the objective

                return best_table

        blocks, _ = get_blocks(truth_table)


        indices = range(len(blocks))
        for i in indices:  #we can never move the end states, only move other states into their blocks

            block_start = np.sum(blocks[0:i, 1])
            block_size = blocks[i, 1]

            for s, state in enumerate(truth_table[block_start:block_start+block_size, :-1]): # for each state in the block

                if block_start + s + 1 < len(truth_table) - 1 and block_start + s - 1 > 0:

                    lower = truth_table[block_start+s -1, :-1]
                    higher = truth_table[block_start + s + 1, :-1]

                    if check_constraints(state, lower): #check if we can put state below

                        new_truth_table = move(truth_table, s+block_start, s+block_start -1)
                        new_obj = objective(new_truth_table)
                        #if len(new_blocks) <= len(blocks) : #dont accept swaps that increase the number of blocks
                        if hash_table(new_truth_table) not in discovered_tables:
                            discovered_tables.add(hash_table(new_truth_table))
                            try:
                                truth_tables.put((new_obj, len(discovered_tables), new_truth_table), block = False)
                            except Exception as e: # queue is full
                                pass


                    if check_constraints(higher, state): #check if we can put state above

                        new_truth_table = move(truth_table, s + block_start, s + block_start + 1)

                        new_obj = objective(new_truth_table)

                        #if len(new_blocks) <= len(blocks) :  # dont accpt swaps that increase the number of blocks
                        if hash_table(new_truth_table) not in discovered_tables:
                            discovered_tables.add(hash_table(new_truth_table))
                            try:
                                truth_tables.put((new_obj, len(discovered_tables), new_truth_table), block=False)
                            except Exception as e: # queue is full
                                pass
    print('len:', len(discovered_tables))

    return best_table

def heuristic_search(outputs, objective=least_blocks_obj, max_queue_size=0):
    '''
    :param outputs:
    :param max_queue_size:
    :param objective:  the function that is to be minimised by the graph search, takes a truth table and outputs a scalar
    :return:
    '''

    n_inputs = int(np.log2(outputs.size))
    truth_table = create_truth_table(outputs)

    #truth_table, _ = rough_optimisation(truth_table, hgi)
    discovered_tables = {hash_table(truth_table)} # use a set for this

    blocks, n_ones = get_blocks(truth_table)

    obj = objective(truth_table)

    truth_tables = qu.PriorityQueue(maxsize=max_queue_size)

    truth_tables.put((obj, len(discovered_tables), truth_table), block = False)
    best_table = truth_table

    if -objective(best_table) == np.sum(best_table[:, -1][best_table[:,
                                                          -1] == 1]):  # early exit condition if all ones are contributing to the objective
        print('len:', len(discovered_tables))
        return best_table




    while not truth_tables.empty():

        #print(len(discovered_tables))
        #print(truth_tables.qsize())

        obj, _, truth_table = truth_tables.get(block = False)  # BFS or DFS depending on this line



        if obj < objective(best_table) and len(get_conflicting_constraints(truth_table)) == 0:
            best_table = truth_table


            if -objective(best_table) == np.sum(best_table[:, -1][best_table[:, -1] == 1]): # early exit condition if all ones are contributing to the objective

                return best_table

        blocks, _ = get_blocks(truth_table)


        indices = range(len(blocks))
        for i in indices:  #we can never move the end states, only move other states into their blocks

            block_start = np.sum(blocks[0:i, 1])
            block_size = blocks[i, 1]

            for s, state in enumerate(truth_table[block_start:block_start+block_size, :-1]): # for each state in the block

                if block_start + s + 1 < int(outputs.size)-1 and block_start + s - 1 > 0:

                    lower = truth_table[block_start+s -1, :-1]
                    higher = truth_table[block_start + s + 1, :-1]

                    if check_constraints(state, lower): #check if we can put state below

                        new_truth_table = move(truth_table, s+block_start, s+block_start -1)
                        new_obj = objective(new_truth_table)
                        #if len(new_blocks) <= len(blocks) : #dont accept swaps that increase the number of blocks
                        if hash_table(new_truth_table) not in discovered_tables:
                            discovered_tables.add(hash_table(new_truth_table))
                            try:
                                truth_tables.put((new_obj, len(discovered_tables), new_truth_table), block = False)
                            except Exception as e: # queue is full
                                pass


                    if check_constraints(higher, state): #check if we can put state above

                        new_truth_table = move(truth_table, s + block_start, s + block_start + 1)

                        new_obj = objective(new_truth_table)

                        #if len(new_blocks) <= len(blocks) :  # dont accpt swaps that increase the number of blocks
                        if hash_table(new_truth_table) not in discovered_tables:
                            discovered_tables.add(hash_table(new_truth_table))
                            try:
                                truth_tables.put((new_obj, len(discovered_tables), new_truth_table), block=False)
                            except Exception as e: # queue is full
                                pass
    print('len:', len(discovered_tables))

    return best_table

def macchiato_v2(outputs,priorities = [], allowed_acts = ['TH', 'IT', 'IB', 'BP'], max_queue_size = 0):
    '''
    Does the iterative graph search to distribute over multiple colonies in different positions
    :param outputs:
    :param max_queue_size:
    :return:
    '''
    #priorities = ['top', 'bot'] #{-1: 0, 0: 1, 1: 94, 2: 158, 3: 3, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
    #priorities = [] #{-1: 0, 0: 1, 1: 94, 2: 149, 3: 12, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
    #priorities = ['end', 'top', 'bot'] #{-1: 0, 0: 1, 1: 151, 2: 104, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

    round = 0
    tables = []
    receivers = []

    truth_table = create_truth_table(outputs)

    while np.sum(outputs[outputs == 1]) > 0:

        if len(priorities) > 0:
            priority = priorities[0]
            obj = lambda x: greedy_obj(x, allowed_acts=[priority])
        else:
            obj = lambda x: greedy_obj(x, allowed_acts=allowed_acts)
            priority = None

        truth_table = graph_search(truth_table, objective=obj, max_queue_size=max_queue_size)


        if priority is not None:
            covered, receiver = get_covered(truth_table, allowed_acts=[priority])
        else: # if no priority
            covered, receiver = get_covered(truth_table, allowed_acts=allowed_acts)


        for cov in covered:

            truth_table[cov[0]: cov[0]+cov[1], -1] = 2

        # if something got covered this round
        if sum([c[2] for c in covered]) > 0:
            tables.append(copy.deepcopy(truth_table))
            receivers.append(receiver)
        else:
            # if there is a priority and nothing covered then move to next highest priority
            if priority is not None:
                priorities = priorities[1:]
            else: # if no priority and haven't covered anything then gate is not possible
                return [], []



        outputs = truth_table[:, -1]
        round += 1

    return tables, receivers


def get_colony_gates(best_tables, receivers):
    '''
    gets the logic gates that are implemented by each colony and returns those output mappings
    :param activations:
    :return:
    '''

    colony_gates = {}


    n_inputs = int(np.log2(len(best_tables[0])))


    inputs_table = create_truth_table(np.array([0] * 2**n_inputs), string = True)

    covered = np.zeros((2 ** n_inputs))


    for ind in range(len(best_tables)):

        receiver = receivers[ind]
        best_table = best_tables[ind]

        logic_gate = [0]*2**n_inputs

        for j in range(len(best_tables[0])):

            input_state = best_table[j, :-1]

            input_ind = inputs_table.index(''.join(map(str, input_state)))

            if best_table[j, -1] == 2 and covered[input_ind] != 1:

                logic_gate[input_ind] = 1
                covered[input_ind] = 1

        tmp = colony_gates.get(receiver, [])
        tmp.append(logic_gate)
        colony_gates[receiver] = tmp

    return colony_gates


parser = argparse.ArgumentParser(description='Run the Macchiato algorithm')
parser.add_argument('outputs', metavar='T', type=str, nargs=1, help='the output of the truth table to be encoded')
parser.add_argument('--outpath', type=str, help='the filepath to save output in, default is Macchiato/output')

# TODO:: add activation function choice as input
if __name__ == '__main__':
    args = parser.parse_args()
    try:
        outputs = np.array(list(args.outputs[0]), dtype=np.str)

        outputs[outputs == 'X'] = '2'
        outputs = outputs.astype(np.int32)
        print(outputs)
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

    #best_table = macchiato(outputs, max_queue_size=1)
    #best_table = graph_search(outputs, max_queue_size=0)
    best_tables, receivers = macchiato_v2(outputs)

    print('VIOLATED CONSTRAINTS')
    all_conflicts = [get_conflicting_constraints(np.array(best_table)) for best_table in best_tables]
    for conflicts in all_conflicts:
        print(len(conflicts))
        for c in conflicts:
            pass
            print(np.array(c).tolist()[0])
    print()


    print('Covered: ')
    for i,t in enumerate(best_tables):
        print(receivers[i],'\n', t, '\n')





    print('Logic gates:')
    colony_gates = get_colony_gates(best_tables, receivers)
    for act in colony_gates.keys():
        print(act)
        for lg in colony_gates[act]:
            print(lg)
    print(colony_gates)


    bt = [best_table.tolist() for best_table in best_tables]


    results_dict = {'truth_table': truth_table.tolist(), 'simplified_tables': bt, 'logic_gates': colony_gates}

    print(results_dict)

    json.dump(results_dict, open(os.path.join(out_path,'{}.json'.format(sys.argv[1])), 'w+'))

    print('Results saved in ',os.path.join(out_path,'{}.json'.format(sys.argv[1])))














