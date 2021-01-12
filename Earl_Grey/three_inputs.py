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

count_dict = {-1: 0, 0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
single_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
singles = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[]}
sing = []
for outputs in all_outputs:
    truth_table = np.hstack((inputs_table, outputs.reshape(-1, 2 ** n_inputs).T))

    on_set = truth_table[truth_table[:, -1] == 1][:, :n_inputs]
    off_set = truth_table[truth_table[:, -1] == 0][:, :n_inputs]

    state_mapping = [off_set, on_set]

    simplified_state_mapping = simplify(state_mapping, n_inputs)

    truth_tables = []
    best_table = truth_table

    best_table, truth_tables = earl_grey(truth_table)


    # analyse best table to create a grid of bacterial populations
    # IT only allowed at low end, threshold only allowed at high end
    activations = get_activations(best_table, n_inputs, allowed_acts = ['TH', 'IT'])

    try:
        n_colonies = 0
        for k in activations.keys():
            n_colonies += len(activations[k])

        count_dict[n_colonies] += 1

        if n_colonies == 2:

            single_counts[ 8 - np.sum(outputs)] +=1
            singles[8-np.sum(outputs)].append(best_table[:,-1])
            sing.append(best_table[:, :3])

    except:
        count_dict[activations] += 1

string_sing = []

for s in sing:
    new_s = []
    for IS in s:
        string_IS = ''
        for i in range(3):
            string_IS += str(IS[i])
        new_s.append(string_IS)
    string_sing.append(new_s)


satistfies_first_constraint = []

for mp in string_sing:
    if mp.index('001') < mp.index('011') and mp.index('001') < mp.index('101'): #if satisfies first constraint
        satistfies_first_constraint.append(mp)
    else:
        print(mp.index('001') , mp.index('011') , mp.index('001') , mp.index('101'), mp)

print('Number of states satisfying first constraint: ', len(satistfies_first_constraint))


sat_first_and_second = []

for mp in satistfies_first_constraint:
    if mp.index('010') < mp.index('011') and mp.index('010') < mp.index('110'): #if satisfies second constraint
        sat_first_and_second.append(mp)
    else:
        print(mp)

print('Number of states satisfying first and second constraint: ', len(sat_first_and_second))


sat_all_constraints = []

for mp in sat_first_and_second:
    if mp.index('100') < mp.index('101') and mp.index('100') < mp.index('110'): #if satisfies second constraint
        sat_all_constraints.append(mp)
    else:
        print(mp)

print('Number of states satisfying all constraints: ', len(sat_all_constraints))

print(count_dict)
print(single_counts)


new = []
new.extend([tuple(singles[2][i]) for i in range(len(singles[2]))])
new.extend([tuple(singles[6][i]) for i in range(len(singles[6]))])

print()

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