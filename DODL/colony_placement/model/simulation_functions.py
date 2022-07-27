import numpy as np



def get_vertex_coordinates(vertex_numbers, n_rows, n_cols):
    '''
    use to get grid coordinates of vertices

    args:
        vertex_numbers: the numbers of the vertices you want coordinates for 0 <= vertex_number < n_rows * n_cols
        n_rows, n_cols: number of rows and columns in the finite difference simulation, a total of n-rows*n_cols vertices

    returns:
        vertex_coordinates: the coordinates on the finite difference grid of the supplied vertex number: [[r0, c0]; [r1,c1]; ... [rn,cn]]
            these use matrix indexing, in the format (row, col) starting from the top left of the grid
    '''

    vertex_coordinates = np.hstack((vertex_numbers // n_cols, vertex_numbers % n_rows))

    return vertex_coordinates

def get_vertex_positions(vertex_numbers, n_rows, n_cols, w):
    '''
    use to get the positions (in mm) of vertices on the real grid

    args:
        vertex_numbers: the numbers of the vertices you want coordinates for 0 <= vertex_number < n_rows * n_cols
        n_rows, n_cols: number of rows and columns in the finite difference simulation, a total of n-rows*n_cols vertices
        w: the distance between finite difference vertices
    returns:
        vertex_positions: the positions on the finite difference grid of the supplied vertex number (in mm from the top left of the grid):
            [[r0, c0]; [r1,c1]; ... [rn,cn]]
    '''


    vertex_coordinates = get_vertex_coordinates(vertex_numbers, n_rows, n_cols)

    vertex_positions = vertex_coordinates * w

    return vertex_positions


def assign_vertices(vertex_positions, node_positions, node_radius):
    '''
    assigns vertices to be part of nodes in node_positions with radius: node radius.


    args:
        vertex_positions: the positions of the vertices to be tested
        node_positions, node_radius: positions and radius of the nodes we want vertices for
    returns:
        vertex_numbers: the numbers of the vertices that are within on of the nodes
        indicators: vector with an index for each vertex indicating whether it is inside a node (value = 1) or outside all nodes (value = 0)

     NOTE: this assigns position based on real life position, not the grid coordinates i.e the distance in mm
    '''


    indicators = np.zeros(len(vertex_positions))

    if node_positions == []:
        return [], indicators

    if node_positions[0] is not None:
        node_positions = np.array(node_positions)
        differences = vertex_positions - node_positions[:, None]

        vertex_numbers = np.where(np.linalg.norm(differences, axis=2) < node_radius)[1].reshape(-1, 1)

        indicators[vertex_numbers] = 1

    indicators = np.array(indicators, dtype=np.int32)



    return vertex_numbers, indicators

# this is the only one you really need to use
def get_node_coordinates(node_positions, node_radius, n_rows, n_cols, w):
    '''
       gets the coordinates of the vertices inside the nodes with position node_positions with radius: node radius.

       args:
           vertex_positions: the positions of the vertices to be tested
           node_positions, node_radius: positions and radius of the nodes we want vertices for
           n_rows, n_cols: the number of rows and cols on the finite difference grid
       returns:
           coordinates: the coordinates of the vertices that are within on of the nodes

        NOTE: this assigns position based on real life position, not the grid coordinates i.e the distance in mm
       '''


    # use the individual functions if repeating these two lines for each node type is too slow
    all_vertex_numbers = np.arange(n_rows * n_cols).reshape(-1, 1)  # reshpae to colum vector
    all_vertex_positions = get_vertex_positions(all_vertex_numbers, n_rows, n_cols, w)

    vertex_numbers, vertex_indicators = assign_vertices(all_vertex_positions, node_positions, node_radius)
    coordinates = get_vertex_coordinates(vertex_numbers, n_rows, n_cols)

    return coordinates

def make_stencil(n, m, r):
    grid = np.zeros((n, m))

    centre = np.array([n // 2, m // 2])

    c = 0
    for i in range(n):
        for j in range(m):
            if np.linalg.norm(centre - np.array([i, j])) > r:
                grid[i, j] = -1
            else:
                grid[i, j] = c
                c += 1

    return grid

def get_shape_matrix(n, m, r):
    '''
    constructs the shape matrix representing the laplacian for a circle with radius r centred within an n x m square

    '''

    sten = make_stencil(n, m, r)
    stencil = np.pad(sten, ((1, 1), (1, 1)), constant_values=(-1,))  # pad the array so that every boundary is a -1

    A = np.zeros((n * m, n * m))

    for i in range(n):
        for j in range(m):

            if stencil[i + 1, j + 1] != -1:  # +1 for the padding
                current_ind = n * i + j

                adjacent = np.array([stencil[i, j + 1], stencil[i + 2, j + 1], stencil[i + 1, j],
                                     stencil[i + 1, j + 2]])  # plus one for padding
                adjacent_inds = np.array([[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]])

                # remove the boundaries as they are insulating and dont contrbute to the dreivitive
                adjacent_inds = adjacent_inds[adjacent != -1]
                adjacent = adjacent[adjacent != -1]

                A[current_ind, current_ind] = len(adjacent) * -1  # stuff can flow out through this many sides

                # stuff can flow into the adjacent squares
                for ai in adjacent_inds:
                    A[current_ind, ai[0] * n + ai[1]] = 1

    return A, sten

def run_sim(indices, receiver_coords, thresholds, logic_gates, activations):


    ind0, ind1, ind2 = indices
    simulator = DigitalSimulator(conc, environment_size, w, dt, laplace=laplace)
    simulator.bound = bound
    inputs_diff = np.any(ind0 != ind1) and np.any(ind1 != ind2) and np.any(ind0 != ind2)
    corners = [[0, 0], [0, max_ind-1], [max_ind-1, 0],
               [max_ind-1, max_ind-1]]  # cant put IPTG in the corners of the allowable square


    on_corner = np.any(list(np.all(i == j) for i in corners for j in indices))
    ti = time.time()
    if inputs_diff and not on_corner:
        coords = np.array(
            [[start_coords + ind0 * points_per_well], [start_coords + ind1 * points_per_well], [start_coords + ind2 * points_per_well]]).reshape(3, 2)

        score, t, best_receiver_pos, all_sims = simulator.max_fitness_over_t(receiver_coords, coords, thresholds,
                                                                             logic_gates, activations, test_t=-1)
    else:
        coords = -1
        score = -1
        best_receiver_pos = -1
        t = -1

    return {'score':score, 'receiver_pos': best_receiver_pos, 'coords': coords, 't':t}