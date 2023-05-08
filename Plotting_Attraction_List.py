import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def plot_fastest_route(path_list,image):
    '''
    Function which takes a list of ride id's from Magic Kingdom and plots them on the image with arrows illustrating
    the path to take
    :param path_list: list of attraction id's
    :param image: image of the park
    :return: plot of the route through the attractions
    '''
    img = mpimg.imread(image)

    #display magic kingdom image
    implot = plt.imshow(img)

    #create counter for plotting
    counter = 0


    #variables for holding point coordinates
    x = 0
    y = 0

    #graph each point on magic kingdom map and annotate them
    for i in x_coordinates:

        plt.plot(x_coordinates[path_list[counter]], y_coordinates[path_list[counter]],
                 marker = '*', color = 'black', markersize = 7)

        #annotate each point with the destination number
        plt.annotate(counter + 1, (x_coordinates[path_list[counter]],
                        y_coordinates[path_list[counter]]), color ='blue', weight ='bold', size = 11)


        #skipping first point
        if x != 0:
                #drawing arrows which connect attractions
                plt.arrow(x_coordinates[path_list[counter]],y_coordinates[path_list[counter]],
                    (x-x_coordinates[path_list[counter]]),y - y_coordinates[path_list[counter]])

        #getting previous point for drawing arrows

        x = x_coordinates[path_list[counter]]
        y = y_coordinates[path_list[counter]]



        counter += 1



    plt.show()


#best path from traveling salesman algorithm

best_path = [36, 33, 16, 3, 38, 32, 1, 23, 31, 13, 6, 4, 26, 34, 18, 14, 37, 7, 22, 29, 8, 21, 20, 15, 17, 10, 9, 12, 27, 25, 5, 30, 2, 35, 24, 0, 19, 39, 28, 11]



#plot fastest path around magic kingdom park
plot_fastest_route(best_path, 'Magic Kingdom Updated.png')




