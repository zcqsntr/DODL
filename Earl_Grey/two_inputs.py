from earl_grey import *
import itertools
import numpy as np


n_inputs = 2
all_outputs = list(map(np.array,list(itertools.product([0, 1], [0, 1], repeat = 2))))

print(len(all_outputs))
# create inputs table
inputs_table = []
for n in range(2 ** n_inputs):
    b_string = "{0:b}".format(n)
    b_string = b_string.rjust(n_inputs, '0')
    b_list = list(map(int, list(b_string)))
    inputs_table.append(b_list)

inputs_table = np.array(inputs_table)

count_dict = {-1: 0, 0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
single_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
singles = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}

for outputs in all_outputs:

    truth_table = np.hstack((inputs_table, outputs.reshape(-1, 2 ** n_inputs).T))

    on_set = truth_table[truth_table[:, -1] == 1][:, :n_inputs]
    off_set = truth_table[truth_table[:, -1] == 0][:, :n_inputs]

    state_mapping = [off_set, on_set]

    simplified_state_mapping = simplify(state_mapping, n_inputs)

    truth_tables = [hash_table(truth_table)]
    best_table = truth_table

    _, best_table, truth_tables = earl_grey(truth_table, best_table, truth_tables)


    # analyse best table to create a grid of bacterial populations
    # IT only allowed at low end, threshold only allowed at high end
    activations = get_activations(best_table, n_inputs)

    try:
        n_colonies = 0
        for k in activations.keys():
            n_colonies += len(activations[k])

        count_dict[n_colonies] += 1

        if n_colonies == 1:
            single_counts[ 8 - np.sum(outputs)] +=1
            singles[8-np.sum(outputs)].append(best_table[:,-1])
            print()
            print(best_table[:, 0:-1])
            print()
    except:
        count_dict[activations] += 1




print(count_dict)
print()
print(single_counts)
print()
print(singles[2])
#print(singles[6])


new = []
new.extend([tuple(singles[2][i]) for i in range(len(singles[2]))])
new.extend([tuple(singles[6][i]) for i in range(len(singles[6]))])

print()
print(new)
print()

for i in range(len(singles[2])):
    zs = np.where(singles[2][i] == 0)
    os = np.where(singles[2][i] == 1)

    singles[2][i][zs] = 1
    singles[2][i][os] = 0

new = []
new.extend([tuple(singles[2][i]) for i in range(len(singles[2]))])
new.extend([tuple(singles[6][i]) for i in range(len(singles[6]))])

print(len(set(new)))
#print(set(new))