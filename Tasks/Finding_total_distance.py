from Array_Pixel_Data import all_pixel_distances
from Plotting_Attraction_List import best_path

def find_total_distance(path_list,distance_array):
    '''
    Function which takes a list of points and a symmetrical distance array between points and
    finds total distance of the path taken

    :param path_list: list of integers which is the series of destinations through the park
    :param distance_array: symmetrical array of distances between each point and every other point
    :return: single total distance value between all points
    '''

    #create list to hold each distance
    distance_list = []

    #create variable to store previous point
    point = -1

    for i in path_list:
        if point != -1:

            #find distance between points
            distance = all_pixel_distances[i,point]

            #multiplied by factor which converts pixel distance to feet
            distance_list.append(distance * 2.272)

        #update previous point
        point = i

    #find total distance
    total_distance = sum(distance_list)



    print('The total distance walked would be',int(total_distance),'feet')


path = [36, 33, 16, 32, 38, 23, 1, 3, 31, 13, 29, 22, 7, 37, 14, 18, 4, 34, 26, 6, 8, 21, 20, 15, 17, 10, 9, 12, 27, 25, 28, 39, 5, 30, 2, 24, 35, 0, 19, 11]
find_total_distance(path,all_pixel_distances)




