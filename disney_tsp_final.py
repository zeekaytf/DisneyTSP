"""Find the optimum path through Disney World's Magic Kingdom

Done by chaining Christofides algorithm and a 3-opt algorithm.

Authors: Will Bradford, Zack Kreitzer, Alex Kreitzer
Version: 5-7-23

Assumptions: A .json file of all point-of-interest (POI) names and a .json file of distances between all
             POIs, as the crow flies, are given.

References: http://matejgazda.com/christofides-algorithm-in-python/
            http://matejgazda.com/tsp-algorithms-2-opt-3-opt-in-python/
"""
from colorama import Fore
from pytsp.data_structures.opt_case import OptCase
from networkx.algorithms.euler import eulerian_circuit
from networkx.algorithms.matching import max_weight_matching
from pytsp.utils import minimal_spanning_tree
from tspdb import TSPDatabase
import itertools
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def get_pixel_distances(attr_coordinates):
    """
    Get distances between attractions based off of pixel coordinates

    :param attr_coordinates: List of attraction pixel coordinates in image [x, y, x, y, x, y, ... ]
    :return: Hollow, symmetric, 2D numpy array of attraction pixel coordinates
    """
    # Split pixel coordinates into x and y
    global x_coordinates, y_coordinates
    x_coordinates, y_coordinates = [], []

    for coord in attr_coordinates:
        if len(x_coordinates) <= len(y_coordinates):
            x_coordinates.append(coord)
        else:
            y_coordinates.append(coord)

    # Create symmetric array of attraction pixel distances
    pixel_distances = np.diag(np.zeros(40))

    for y in range(0, 40):
        for x in range(0, 40):

            if x == y:
                pixel_distances[x][y] = 0

            else:
                pixel_distances[x][y] = (
                    np.sqrt((x_coordinates[x] - x_coordinates[y]) ** 2 + (y_coordinates[x] - y_coordinates[y]) ** 2))

    return pixel_distances


def run_christofides_algorithm(graph, starting_attr=0):
    """
    Christofides TSP algorithm

    :param graph: A 2D, hollow, symmetric numpy array matrix
    :param starting_attr: Starting attraction of the TSP
    :return: Path given by Christofides TSP algorithm
    """
    # Minimal spanning tree (Connect all attractions to at least one other, based on distance)
    mst = minimal_spanning_tree(graph, 'Prim', starting_node=0)

    # Find all odd degree attractions
    odd_degree_attrs = [index for index, row in enumerate(mst) if len(np.nonzero(row)[0]) % 2 != 0]
    odd_degree_attrs_ix = np.ix_(odd_degree_attrs, odd_degree_attrs)
    nx_graph = nx.from_numpy_array(-1 * graph[odd_degree_attrs_ix])

    # Minimal-weight perfect matching
    # (Create edges by matching all attractions to each other; edges have no common attractions)
    matching = max_weight_matching(nx_graph, maxcardinality=True)

    # Unite minimal-weight perfect matching and minimal spanning tree to create a graph where all attractions
    # have an even number of connections (graph does not loop back on itself)
    euler_multigraph = nx.MultiGraph(mst)
    for edge in matching:
        euler_multigraph.add_edge(odd_degree_attrs[edge[0]], odd_degree_attrs[edge[1]],
                                  weight=graph[odd_degree_attrs[edge[0]]][odd_degree_attrs[edge[1]]])

    # Create an initial path along the previously created shape; attractions may appear more than once
    euler_path = list(eulerian_circuit(euler_multigraph, source=starting_attr))
    path = list(itertools.chain.from_iterable(euler_path))

    # Eliminate duplicate nodes in the path
    path = list(dict.fromkeys(path).keys())
    path.append(starting_attr)

    # Slicing to remove beginning attraction from end of path. Remove slicing to create full loop
    final_path = path[:-1]

    return final_path


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
    Create a new path from an existing path

    :param route: Existing path
    :param case: Which case of opt swaps should be used
    :param i:
    :param j:
    :param k:
    :return: Improved path
    """
    if (i - 1) < (k % len(route)):
        first_segment = route[k% len(route):] + route[:i]

    else:
        first_segment = route[k % len(route):i]

    second_segment = route[i:j]
    third_segment = route[j:k]

    # first case is the current solution ABC
    if case == OptCase.opt_case_1:
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

    :param graph: 2D numpy array as graph
    :param route: Route as ordered list of visited attractions
    :return: Optimal path according to 3-opt algorithm
    """
    moves_cost = {OptCase.opt_case_1: 0, OptCase.opt_case_2: 0,
                  OptCase.opt_case_3: 0, OptCase.opt_case_4: 0, OptCase.opt_case_5: 0,
                  OptCase.opt_case_6: 0, OptCase.opt_case_7: 0, OptCase.opt_case_8: 0}

    improved = True
    best_found_route = route

    while improved:
        improved = False

        for (i, j, k) in possible_segments(len(graph)):

            # Check all the possible moves and save the result into the dict
            for opt_case in OptCase:
                moves_cost[opt_case] = get_solution_cost_change(graph, best_found_route, opt_case, i, j, k)

            # Minimum value of subtraction of old route - new route
            best_return = max(moves_cost, key=moves_cost.get)
            if moves_cost[best_return] > 0:
                best_found_route = reverse_segments(best_found_route, best_return, i, j, k)
                improved = True
                break

    return best_found_route


def possible_segments(N):
    """ Generate the combination of segments """
    segments = ((i, j, k) for i in range(N) for j in range(i + 2, N-1) for k in range(j + 2, N - 1 + (i > 0)))

    return segments


def plot_fastest_route(path_list, image):
    """
    Plots the recommended path to take between different attractions using each attraction's ID

    :param path_list: List of attraction IDs
    :param image: Image name to be used
    :return: Plot of inputted image with path overlay
    """
    # Display inputted image
    img = mpimg.imread(image)
    plt.imshow(img)

    # Create counter for plotting
    counter = 0

    # Variables for holding point coordinates
    x = 0
    y = 0

    # Graph and annotate each point on image
    for _ in x_coordinates:

        plt.plot(x_coordinates[path_list[counter]], y_coordinates[path_list[counter]],
                 marker='*', color='black', markersize=7)

        # Annotate each point with the attraction number
        plt.annotate(counter + 1, (x_coordinates[path_list[counter]],
                                   y_coordinates[path_list[counter]]), color='blue', weight='bold', size=11)

        # Skip first point
        if x != 0:

            # Draw arrows that connect attractions
            plt.arrow(x_coordinates[path_list[counter]], y_coordinates[path_list[counter]],
                      (x - x_coordinates[path_list[counter]]), y - y_coordinates[path_list[counter]])

        # Get previous point for drawing arrows
        x = x_coordinates[path_list[counter]]
        y = y_coordinates[path_list[counter]]

        counter += 1

    plt.show()


def find_total_distance(path_list, distance_array):
    """
    Function which takes a list of points and a symmetrical distance array between points and
    finds total distance of the path taken

    :param path_list: list of integers which is the series of destinations through the park
    :param distance_array: symmetrical array of distances between each point and every other point
    :return: single total distance value between all points
    """

    # Create list to hold each distance
    distance_list = []

    # Create variable to store previous point
    point = -1

    for i in path_list:
        if point != -1:

            # Find distance between points
            distance = attr_pixel_distances[i, point]

            # Multiplied by factor which converts pixel distance to feet
            distance_list.append(distance * 2.272)

        # Update previous point
        point = i

    # Find total distance
    total_distance = sum(distance_list)

    print('The total distance walked would be', int(total_distance), 'feet')


if __name__ == "__main__":

    # Instantiate database
    db = TSPDatabase()

    # Make lists of attraction names
    attr_name_list = [attr.get_attraction_name() for attr in db.get_attractions_list()]
    attr_name_list_lower = [attr.get_attraction_name().lower() for attr in db.get_attractions_list()]

    # Toggle variable for while loop
    toggle = True
    print_counter = 0

    # User decides which attraction to start at
    while toggle:

        first_attr_name = None

        # Print IDs and names for user to choose from
        if print_counter == 0:

            print(Fore.LIGHTBLUE_EX + 'ID: Name' + Fore.RESET)
            for attr in db.get_attractions_list():
                print(f'{attr.id:02d}: {attr.get_attraction_name()}')

            # Only print on first loop
            print_counter += 1

        # User chooses starting attraction (program ignores apostrophes, leading/trailing whitespace, and
        # capitalization, but end-of-line punctuation should not be passed)
        name_to_search = input(Fore.RESET + '\nEnter one of the above attraction IDs or at least a portion '
                                            'of the attraction\'s name that you would like to start at. '
                                            'Press ENTER to start at the front of the park.:\n')
        stripped_input = ''.join(name_to_search.lower().strip().split("'"))

        # List storing all attraction names that match/include the entered string
        searched_attr_list = [attr_name.title() for attr_name in attr_name_list_lower if stripped_input in attr_name]

        # Choose starting location from inputted id, must be valid id
        if stripped_input.isdigit() and 0 <= int(stripped_input) <= 39:
            first_attr_name = db.get_attraction_name_by_id(int(stripped_input))
            toggle = False

        # Start at front of park
        elif stripped_input == '':
            first_attr_name = db.get_attraction_name_by_id(36)
            toggle = False

        # Check number of attractions associated with input
        elif len(searched_attr_list) == 1:
            first_attr_name = searched_attr_list[0]
            toggle = False

        # Multiple attractions associated with input
        elif len(searched_attr_list) > 1:
            print(f'{Fore.LIGHTRED_EX}\nERROR: Multiple attractions found that include "{stripped_input}".'
                  f'\n{Fore.RESET}Did you mean one of these?:\n{searched_attr_list}\n')

        # Input is invalid
        else:
            # If input is an id
            if stripped_input.strip('-.').isnumeric() or \
                    (''.join(stripped_input.lower().strip('-').split(".")).isnumeric() and '.' in stripped_input):
                print(Fore.LIGHTRED_EX + '\nERROR: Invalid attraction ID. Valid IDs are integers from 0-39.\n')

            # If input is a name
            else:
                print(Fore.LIGHTRED_EX + '\nERROR: Invalid attraction name. Either the entered name includes '
                                         'unsupported punctuation, is misspelled, or there is no '
                                         'attraction associated with the entered name.\n')

    # Get first attraction object and id
    first_attr = db.get_attraction_by_name(first_attr_name)
    first_attr_id = first_attr.id

    # Image name
    image_name = 'Magic Kingdom Updated.png'

    # Manually found pixel coordinates of each attraction on the image: (x, y)
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

    # Get distances between attractions based off of pixel coordinates
    attr_pixel_distances = get_pixel_distances(attraction_coordinates)

    # Run christofides algorithm (guaranteed to be no longer than 3/2 of the optimal path)
    chris_path = run_christofides_algorithm(attr_pixel_distances, first_attr_id)

    '''# Print and plot Christofides path
    print(f'\nIt is recommended that you visit attractions in the following order:\n{chris_path}')
    find_total_distance(chris_path, attr_pixel_distances)
    plot_fastest_route(chris_path, image_name)'''

    # Improve path using 3-opt algorithm
    improved_path = tsp_3_opt(attr_pixel_distances, chris_path)

    # Shift improved path to start at the correct attraction
    shift_val = improved_path.index(chris_path[0])
    path_array = np.array(improved_path)
    recommended_path = list(np.roll(path_array, - shift_val))

    # Print and plot 3-opt path
    print(f'\nIt is recommended that you visit attractions in the following order:\n{recommended_path}')
    find_total_distance(recommended_path, attr_pixel_distances)
    plot_fastest_route(recommended_path, image_name)
