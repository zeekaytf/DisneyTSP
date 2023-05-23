import itertools

import networkx as nx
from networkx.algorithms.matching import max_weight_matching
from networkx.algorithms.euler import eulerian_circuit
from pytsp.utils import minimal_spanning_tree
import numpy as np
from tspdb import TSPDatabase

from pytsp.data_structures.opt_case import OptCase


def get_attraction_by_name(name):
    for attr in TSPDatabase.attractions:
        if attr.name == name:
            return attr
def get_solution_cost_change(graph, route, case, i, j, k):
    """ Compare current solution with 7 possible 3-opt moves"""
    A, B, C, D, E, F = route[i - 1], route[i], route[j - 1], route[j], route[k - 1], route[k % len(route)]
    if case == OptCase.opt_case_1:
        # first case is the current solution ABC
        return 0
    elif case == OptCase.opt_case_2:
        # second case is the case A'BC
        return graph[A, B] + graph[E, F] - (graph[B, F] + graph[A, E])
    elif case == OptCase.opt_case_3:
        # ABC'
        return graph[C, D] + graph[E, F] - (graph[D, F] + graph[C, E])
    elif case == OptCase.opt_case_4:
        # A'BC'
        return graph[A, B] + graph[C, D] + graph[E, F] - (graph[A, D] + graph[B, F] + graph[E, C])
    elif case == OptCase.opt_case_5:
        # A'B'C
        return graph[A, B] + graph[C, D] + graph[E, F] - (graph[C, F] + graph[B, D] + graph[E, A])
    elif case == OptCase.opt_case_6:
        # AB'C
        return graph[B, A] + graph[D, C] - (graph[C, A] + graph[B, D])
    elif case == OptCase.opt_case_7:
        # AB'C'
        return graph[A, B] + graph[C, D] + graph[E, F] - (graph[B, E] + graph[D, F] + graph[C, A])
    elif case == OptCase.opt_case_8:
        # A'B'C
        return graph[A, B] + graph[C, D] + graph[E, F] - (graph[A, D] + graph[C, F] + graph[B, E])

def reverse_segments(route, case, i, j, k):
    """
    Create a new tour from the existing tour
    Args:
        route: existing tour
        case: which case of opt swaps should be used
        i:
        j:
        k:
    Returns:
        new route
    """
    if (i - 1) < (k % len(route)):
        first_segment = route[k% len(route):] + route[:i]
    else:
        first_segment = route[k % len(route):i]
    second_segment = route[i:j]
    third_segment = route[j:k]

    if case == OptCase.opt_case_1:
        # first case is the current solution ABC
        pass
    elif case == OptCase.opt_case_2:
        # A'BC
        solution = list(reversed(first_segment)) + second_segment + third_segment
    elif case == OptCase.opt_case_3:
        # ABC'
        solution = first_segment + second_segment + list(reversed(third_segment))
    elif case == OptCase.opt_case_4:
        # A'BC'
        solution = list(reversed(first_segment)) + second_segment + list(reversed(third_segment))
    elif case == OptCase.opt_case_5:
        # A'B'C
        solution = list(reversed(first_segment)) + list(reversed(second_segment)) + third_segment
    elif case == OptCase.opt_case_6:
        # AB'C
        solution = first_segment + list(reversed(second_segment)) + third_segment
    elif case == OptCase.opt_case_7:
        # AB'C'
        solution = first_segment + list(reversed(second_segment)) + list(reversed(third_segment))
    elif case == OptCase.opt_case_8:
        # A'B'C
        solution = list(reversed(first_segment)) + list(reversed(second_segment)) + list(reversed(third_segment))
    return solution
def tsp_3_opt(graph, route):
    """
    Approximate the optimal path of travelling salesman according to 3-opt algorithm
    Args:
        graph:  2d numpy array as graph
        route: route as ordered list of visited nodes. if no route is given, christofides algorithm is used to create one.
    Returns:
        optimal path according to 3-opt algorithm
    Examples:
    """
    # Deleted option for no route since there will always be a route


    moves_cost = {OptCase.opt_case_1: 0, OptCase.opt_case_2: 0,
                  OptCase.opt_case_3: 0, OptCase.opt_case_4: 0, OptCase.opt_case_5: 0,
                  OptCase.opt_case_6: 0, OptCase.opt_case_7: 0, OptCase.opt_case_8: 0}
    improved = True
    best_found_route = route
    while improved:
        improved = False
        for (i, j, k) in possible_segments(len(graph)):
            # we check all the possible moves and save the result into the dict
            for opt_case in OptCase:
                moves_cost[opt_case] = get_solution_cost_change(graph, best_found_route, opt_case, i, j, k)
            # we need the minimum value of substraction of old route - new route
            best_return = max(moves_cost, key=moves_cost.get)
            if moves_cost[best_return] > 0:
                best_found_route = reverse_segments(best_found_route, best_return, i, j, k)
                improved = True
                break
    # just to start with the same node -> we will need to cycle the results.
    # cycled = itertools.cycle(best_found_route)
    # skipped = itertools.dropwhile(lambda x: x != 0, cycled)
    # sliced = itertools.islice(skipped, None, len(best_found_route))
    # best_found_route = list(sliced)
    return best_found_route
def possible_segments(N):
    """ Generate the combination of segments """
    segments = ((i, j, k) for i in range(N) for j in range(i + 2, N-1) for k in range(j + 2, N - 1 + (i > 0)))
    return segments

# def choose_starting_location(user_input)
def run_christofides_algorithm(graph, starting_node=0):
    """
    Christofides TSP algorithm
    http://www.dtic.mil/dtic/tr/fulltext/u2/a025602.pdf
    Args:
        graph: 2d numpy array matrix
        starting_node: of the TSP
    Returns:
        tour given by christofides TSP algorithm

    Examples:
        #>>> import numpy as np
        #>>> graph = np.array([[  0, 300, 250, 190, 230],
        #>>>                   [300,   0, 230, 330, 150],
        #>>>                   [250, 230,   0, 240, 120],
        #>>>                   [190, 330, 240,   0, 220],
        #>>>                   [230, 150, 120, 220,   0]])
        #>>> christofides_tsp(graph)
    """

    # Minimal spanning tree (Connect all nodes to at least one other, based on distance)
    mst = minimal_spanning_tree(graph, 'Prim', starting_node=0)

    # Save nodes that have an odd number of connections
    odd_degree_nodes = list(_get_odd_degree_vertices(mst))
    odd_degree_nodes_ix = np.ix_(odd_degree_nodes, odd_degree_nodes)
    nx_graph = nx.from_numpy_array(-1 * graph[odd_degree_nodes_ix])

    # Minimal-weight perfect matching
    # (Create edges by matching all nodes to each other; edges have no common nodes)
    matching = max_weight_matching(nx_graph, maxcardinality=True)

    # Unite minimal-weight perfect matching and minimal spanning tree to create a graph where all nodes have an even
    # number of connections (graph does not loop back on itself)
    euler_multigraph = nx.MultiGraph(mst)
    for edge in matching:
        euler_multigraph.add_edge(odd_degree_nodes[edge[0]], odd_degree_nodes[edge[1]],
                                  weight=graph[odd_degree_nodes[edge[0]]][odd_degree_nodes[edge[1]]])

    # Create an initial path along the previously created shape; nodes may appear more than once
    euler_tour = list(eulerian_circuit(euler_multigraph, source=starting_node))
    path = list(itertools.chain.from_iterable(euler_tour))

    # Eliminate duplicate nodes in the path
    final_path = _remove_repeated_vertices(path, starting_node)[:-1]

    return final_path


def _get_odd_degree_vertices(graph):
    """
    Finds all the odd degree vertices in graph
    Args:
        graph: 2d np array as adj. matrix

    Returns:
    Set of vertices that have odd degree
    """
    odd_degree_vertices = set()
    for index, row in enumerate(graph):
        if len(np.nonzero(row)[0]) % 2 != 0:
            odd_degree_vertices.add(index)
    return odd_degree_vertices


def _remove_repeated_vertices(path, starting_node):
    path = list(dict.fromkeys(path).keys())
    path.append(starting_node)
    return path


if __name__ == "__main__":

    # Instantiate database
    db = TSPDatabase()

    # Toggle variable for while loop
    toggle = True

    while toggle:

        # User chooses starting attraction (program ignores apostrophes, leading/trailing whitespace, and
        # capitalization, but end-of-line punctuation should not be passed)
        first_attr_name = None
        name_to_search = input('Enter either a part or the entire name of the desired starting attraction.\n'
                               'Press ENTER to start at the front of the park.:\n ')
        stripped_string = ''.join(name_to_search.lower().strip().split("'"))

        # Make lists of attraction names
        attr_name_list = [db.get_attraction_name(attr.id) for attr in db.get_attractions_list()]
        attr_name_list_lower = [db.get_attraction_name(attr.id).lower() for attr in db.get_attractions_list()]

        # List storing all attraction names that match/include the entered string
        searched_attr_list = [attr_name.title() for attr_name in attr_name_list_lower if stripped_string in attr_name]

        # If at least one attraction is found, update the name of the first visited attraction
        if searched_attr_list and stripped_string != '':
            first_attr_name = searched_attr_list[0]

        # Check if attraction is valid
        if len(searched_attr_list) == 1:
            toggle = False

        elif stripped_string == '':
            first_attr_name = db.get_attraction_by_id(36)
            toggle = False

        elif first_attr_name is None:
            print(f'''
    ERROR: Invalid attraction name. Either the entered name includes unsupported punctuation, is misspelled, or there is \
    no attraction associated with the entered name.

    Enter either a part or the entire name of one of the following attraction names:
    {attr_name_list}
    ''')

        else:
            print(f'''
    ERROR: Multiple attractions found that include "{stripped_string}".

    Did you mean one of these?:
    {searched_attr_list}
    ''')

    # Get first attraction and id
    first_attr_id = get_attraction_by_name(first_attr_name).id


    # Pixel coordinates of each attraction: (x, y)
    attraction_coordinates = [80.43, 667.25,
                              1001.09, 656.34,
                              120.65, 415.39,
                              927.18, 680.33,
                              1033.70, 221.32,
                              252.17, 593.106,
                              988.05, 288.9,
                              706.52, 234.4,
                              766.3, 430.7,
                              379.35, 344.53,
                              508.69, 288.9,
                              297.83, 739.2,
                              395.65, 451.37,
                              859.79, 373.96,
                              881.52, 207.15,
                              603.26, 379.41,
                              865.22, 648.7,
                              532.61, 331.44,
                              976.1, 139.55,
                              176.09, 717.40,
                              663.698, 361.97,
                              690.22, 375.05,
                              770.65, 305.28,
                              1133.699, 587.66,
                              58.695, 482.99,
                              384.78, 654.16,
                              1075, 243.13,
                              469.57, 476.45,
                              291.31, 672.697,
                              781.52, 359.79,
                              159.78, 466.64,
                              936.96, 481.899,
                              977.177, 666.15,
                              701.09, 955.08,
                              1084.79, 165.72,
                              27.174, 443.74,
                              641.31, 1011.77,
                              843.481, 189.71,
                              989.13, 722.85,
                              254.35,  698.86]

    x_coordinates, y_coordinates = [], []

    # Split pixel coordinates into x and y
    for i in attraction_coordinates:
        if len(x_coordinates) <= len(y_coordinates):
            x_coordinates.append(i)
        else:
            y_coordinates.append(i)

    # Create symmetric array of attraction pixel distances
    all_pixel_distances = np.diag(np.zeros(40))

    for y in range(0, 40):
        for x in range(0, 40):

            if x == y:
                all_pixel_distances[x][y] = 0

            else:
                all_pixel_distances[x][y] = (
                    np.sqrt((x_coordinates[x] - x_coordinates[y]) ** 2 + (y_coordinates[x] - y_coordinates[y]) ** 2))

    # Run christofides algorithm (guaranteed to be no longer than 3/2 of the optimal path)
    recommended_path = run_christofides_algorithm(all_pixel_distances, first_attr_id)

    print(recommended_path)

    # improve upon old path
    improved_path = tsp_3_opt(all_pixel_distances, recommended_path)
    print(improved_path)
