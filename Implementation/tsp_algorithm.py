import tspdb as tb
import numpy as np
import json
import matplotlib.pyplot as plt


# Load in database
db = tb.TSPDatabase()


# Load Disney image

x = plt.imread('Magic Kingdom Updated.png')
disney = x[:, 0:1150]  # Removes the legend from map

# plot each point onto the picture

Attraction_Coordinates = [80.43,667.25,1001.09,656.34,120.65,415.39,927.18,680.33,1033.70,221.32,252.17,593.106,988.05,288.9,706.52,234.4,
        766.3,430.7,379.35,344.53,508.69,288.9,297.83,739.2,395.65,451.37,859.79,373.96,881.52,207.15,603.26,379.41,865.22,648.7,
        532.61,331.44,976.1,139.55,176.09,717.40,663.698,361.97,690.22,375.05,770.65,305.28,1133.699,587.66,58.695,482.99,
        384.78,654.16,1075,243.13,469.57,476.45,291.31,672.697,781.52,359.79,159.78,466.64,936.96,481.899,977.177,666.15,
        701.09,955.08,1084.79,165.72,27.174,443.74,641.31,1011.77,843.481,189.71,989.13,722.85,254.35,
        698.86]

x_coordinates = []
y_coordinates = []

for i in Attraction_Coordinates:
    if len(x_coordinates) <= len(y_coordinates):
        x_coordinates.append(i)
    else:
        y_coordinates.append(i)
all_pixel_distances = np.diag(np.zeros(40))
for y in range(0, 40):
    for x in range(0, 40):
        if x == y:
            all_pixel_distances[x][y] = 0
        else:
            all_pixel_distances[x][y] = np.sqrt((x_coordinates[x] - x_coordinates[y]) ** 2 + (y_coordinates[x] - y_coordinates[y]) ** 2)

plt.imshow(disney)
plt.show()


# Load Distances File
file = open('distances.json')
distances = json.load(file)
file.close()


# Make Numpy Array for ALL distances
all_distances = np.diag(np.zeros(40))
for y in range(0, 40):
    for x in range(0, 40):
        if x == y:
            all_distances[x][y] = 0
        else:
            all_distances[x][y] = db.get_attraction_distance(id1=x, id2=y)

# Make Numpy Array for ALL pixel distances
all_pixel_distances = np.diag(np.zeros(40))
for y in range(0, 40):
    for x in range(0, 40):
        if x == y:
            all_pixel_distances[x][y] = 0
        else:
            all_pixel_distances[x][y] = (np.sqrt((x_coordinates[x] - x_coordinates[y]) ** 2 + (y_coordinates[x] - y_coordinates[y]) ** 2))

# Collect user input about start
start_point = int(input('Enter the id of the ride you would like to start at.'))

# Determine Path Based only on shortest distance
path = []
path.append(start_point)
count = 0

for i in range(0, 40):
    the_id = path[-1]
    distance_options = all_distances[:, the_id]
    for item in path:
        distance_options[item] = 100000000
    minimum = 10000
    for value in distance_options:
        if value == 0:
            count += 1
            continue
        elif value < minimum:
            minimum = value
            second_id = count
            all_mins = [count]
            count += 1
        elif value == minimum:
            all_mins.append(count)
            count += 1
        else:
            count += 1
    all_mins_copy = all_mins
    real_min_options = []
    for item in all_mins: # For all options, find the distance on the map from the last point
        real_min_options.append(np.sqrt((x_coordinates[item] - x_coordinates[the_id]) ** 2 + (y_coordinates[item] - y_coordinates[the_id]) ** 2))
    real_min = all_mins_copy[np.argmin(real_min_options)]
    path.append(real_min)
    count = 0
final_list = []
[final_list.append(x) for x in path if x not in final_list]
print(final_list)

# Go to nearest cluster

# Apply Algorithm
done = True