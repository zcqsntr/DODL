import numpy as np
import copy
from collections import OrderedDict
import sys
import queue as qu

from time import time

from exhaustive_search import *

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






outputs = np.array([[0,1,0,1,0,1,0,1]]) #dependant on one input
#outputs = np.array([[1,1,0,0,1,1,0,1]]) #0xCD


outputs = np.array([[0,0,1,1,0,1,1,1,0,0,1,1,0,1,1,1]]) #threshold

#priority encoder
#outputs = np.array([[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]])
#outputs = np.array([[0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1]])
#outputs = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])


n_inputs = int(np.log2(outputs.size))

def earl_gey(outputs):
    n_inputs = int(np.log2(outputs.size))
    truth_table = create_truth_table(outputs)
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


t = time()
best_table = earl_gey(outputs)
print('time:', time()-t)

print('result:', current_table)

valid = True
for i in range(2**n_inputs):
    for j in range(2**n_inputs):
        if i < j:
            valid = valid and check_constraints(best_table[i, :-1], best_table[j, :-1])
        if j < i:
            valid = valid and check_constraints(best_table[j, :-1], best_table[i, :-1])


print('valid:', valid)

activations = get_activations(best_table, n_inputs)

print('NODES: ')
for act in activations.keys():
    for state_mapping in activations[act]:


        simplified_state_mapping = simplify(state_mapping, n_inputs)
        print(act)
        for s in simplified_state_mapping:
            print(s)














