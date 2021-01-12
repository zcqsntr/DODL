from earl_grey import *
import itertools
import numpy as np


n_inputs = 3
all_outputs = list(map(np.array,list(itertools.product([0, 1], [0, 1], repeat = 4))))

print(len(all_outputs))
# create inputs table
inputs_table = []
for n in range(2 ** n_inputs):
    b_string = "{0:b}".format(n)
    b_string = b_string.rjust(n_inputs, '0')
    b_list = list(map(int, list(b_string)))
    inputs_table.append(b_list)

inputs_table = np.array(inputs_table)

priority_detector = [
    [0,0,1,1,0,0,1,1],
    [0,0,0,0,1,0,0,0],
    [0,1,0,0,0,1,0,0]
]

# find gates possible with IPTG (bandpass and threshold)
possible_functions = []
for outputs in all_outputs:
    truth_table = np.hstack((inputs_table, outputs.reshape(-1, 2 ** n_inputs).T))

    on_set = truth_table[truth_table[:, -1] == 1][:, :n_inputs]
    off_set = truth_table[truth_table[:, -1] == 0][:, :n_inputs]

    state_mapping = [off_set, on_set]

    simplified_state_mapping = simplify(state_mapping, n_inputs)

    truth_tables = []
    best_table = truth_table

    _, best_table, truth_tables = earl_grey(truth_table, best_table, truth_tables)


    # analyse best table to create a grid of bacterial populations
    # IT only allowed at low end, threshold only allowed at high end
    activations = get_activations(best_table, n_inputs, allowed_acts = ['TH', 'BP'])

    if activations != -1:
        possible_functions.append(list(outputs))

    try:
        n_colonies = 0
        for k in activations.keys():
            n_colonies += len(activations[k])

        if list(outputs) in [[0,0,0,1,1,1,0,1]]:
            print(activations)
    except:
        pass


print(len(possible_functions))



GCDA = [
    [0,0,0,1,1,1,0,0],
    [1,1,1,1,0,1,1,0],
    [0,1,0,0,0,0,0,1],
    [0,1,1,0,0,0,0,0],
    [1,0,0,0,0,0,0,1],
    [0,1,0,0,1,1,0,1],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,1,0,0],
    [0,1,1,1,1,0,0,0],
    [1,0,0,0,0,1,1,1],
    [1,0,1,1,1,1,0,1],
    [1,1,0,0,0,1,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,1,1,1,0],
    [1,1,1,0,1,0,0,0],
    [1,1,0,0,1,1,0,1],
    [0,0,0,1,1,1,0,1],
    [0,0,0,0,1,0,1,1],
    [1,1,0,0,0,1,1,1],
    [1,1,1,1,1,0,1,1],
    [0,1,1,0,1,1,1,0],
    [0,0,1,1,1,0,1,1],
    [1,1,1,1,0,1,1,1],
    [0,1,1,1,1,1,1,1],
    [1,0,1,0,1,1,1,0],
    [1,1,0,0,1,0,0,0],
    [1,0,0,0,0,0,0,0],
    [1,1,1,0,1,0,1,0],
    [0,0,0,0,0,1,1,1],
    [0,1,1,1,0,0,0,0],
    [0,0,1,1,0,1,1,1],
    [0,0,1,1,1,1,0,1],
    [1,0,0,0,1,1,1,0],
    [0,0,0,1,0,1,1,1],
    [1,1,0,0,1,0,0,1]

]
# priority detector is possible
print(all([p in possible_functions for p in priority_detector]))

n = 0
for f in GCDA:
    n += f in possible_functions
print(n)
print(len(GCDA))
print(len(possible_functions))