import numpy as np
import copy
from collections import OrderedDict
import sys
import queue as qu
from time import time

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
            states = copy.deepcopy(state_mapping[s])
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
            this_IB.append(best_table[start:end, :n_inputs])
        activations['IB'].append(this_IB)
        pos = len(blocks) - 1


    while pos < len(blocks) - 1:

        if 'BP' in allowed_acts and np.all(blocks[pos:pos + 3, 0] == np.array([0, 1, 0])):

            # can have multiple bp
            this_BP = []

            # need to put all off states before and after the bandoass into its off set
            boundary = np.sum(blocks[0:pos + 1, 1])
            off_set = best_table[0: boundary, :n_inputs]
            # off_set = best_table[best_table[:, -1] == 0][0: boundary, :n_inputs]
            this_BP.append(off_set)
            boundary1 = np.sum(blocks[0:pos + 2, 1])

            on_set = best_table[boundary: boundary1, :n_inputs]
            this_BP.append(on_set)
            off_set = best_table[boundary1:, :n_inputs]
            # off_set = best_table[best_table[:, -1] == 0][boundary:, :n_inputs]
            this_BP.append(off_set)
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
            this_IT.append(on_set)
            off_set = best_table[boundary:, :n_inputs]
            # off_set = best_table[best_table[:, -1] == 0][boundary:, :n_inputs]
            this_IT.append(off_set)
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
            this_TH.append(off_set)

            on_set = best_table[boundary:, :n_inputs]
            this_TH.append(on_set)
            activations['TH'].append(this_TH)
            pos += 1
        else: #failed to assign and so gate isnt possible
            return -1



    return activations


if __name__ == '__main__':


    n_inputs = 3
    outputs = np.array([[1,1,0,0,1,1,0,1]]) #0xCD
    #outputs = np.array([[0,0,0,0,1,0,1,1]]) #0x0B
    #outputs = np.array([[1,1,1,0,1,0,0,0]]) #0xE8
    #outputs = np.array([[1,1,0,0,1,0,0,0]]) #0xC8
    #outputs = np.array([[0,0,1,1,0,1,1,1]]) #0x37
    #outputs = np.array([[0,0,1,1,1,1,0,1]]) #0x3D
    #outputs = np.array([[1,0,1,0,1,1,1,0]]) #0xAE
    #outputs = np.array([[0,1,0,1,0,1,0,1]]) #dependant on one input

    n_inputs = 4
    outputs = np.array([[0,0,1,1,0,1,1,1,0,0,1,1,0,1,1,1]]) # threshold



    #priority encoder
    #outputs = np.array([[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]])
    #outputs = np.array([[0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1]])
    #outputs = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])



    if len(sys.argv) > 1: #if run from command line
        n_inputs = int(sys.argv[1])

        outputs_string = sys.argv[2]

        if len(outputs_string) != 2**n_inputs:
            print('WRONG NUMBER OF OUTPUTS')

        outputs = list(map(int, list(outputs_string)))
        outputs = np.array([outputs])





    truth_table = create_truth_table(outputs)

    print('INPUT TABLE: ')
    print(truth_table)
    print()

    on_set = truth_table[truth_table[:, -1] == 1][:, :n_inputs]
    off_set = truth_table[truth_table[:, -1] == 0][:, :n_inputs]

    state_mapping = [off_set, on_set]

    simplified_state_mapping = simplify(state_mapping, n_inputs)


    t = time()
    best_table, discovered_tables = graph_search(truth_table)
    print('time:', time()-t)
    print('BEST TABLE: ')
    print(best_table)
    print(print(get_blocks(best_table)[1]))

    #analyse best table to create a grid of bacterial populations
    # IT only allowed at low end, threshold only allowed at high end
    activations = get_activations(best_table, n_inputs)

    print('NODES: ')
    for act in activations.keys():
        for state_mapping in activations[act]:


            simplified_state_mapping = simplify(state_mapping, n_inputs)
            print(act)
            for s in simplified_state_mapping:
                print(s)



