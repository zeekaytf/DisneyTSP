# DisneyTSP
 dependencies: numpy, matplotlib, json, os, netowrkx, pytsp
 
 attractions.json contains a dictionary of attractions where each attraction has a key for its id and name. The id was assigned to each ride in alphabetical order based on the first letter of the attraction.
 
 distances.json contains a dictionary of distances (in feet) from each ride to all other rides. The keys in the dictionary follow the patter "x,y" where x and y are nonidentical integers representing ride IDs. The distance output is the distance from x to y.

 Magic Kingdom Updated.png contains an image used throughout the project to determine optimal pathing as well as for visualization
 
 tspdb.py contains a database of functions used in some of the other projects. The database is typically imported as tb.

 tsp_algorithm.py implements the nearest neighbor algorithm where each sequencial point is the nearest point to the previous point.
 
 Finding_total_distance.py calculates the total distance traveled in feet of any given path.
 
 Plotting_Attraction_List.py plots the coordinates of each ride and labels with a black star.
 
 disney_tsp_final.py compiles all algorithms and visualizations into a final script. In this script, the Christofides algorithm is used to find an initial path, then the 3-opt algorithm is applied to the found path and further optimizes it. Then the final path is shown on the map of magic kingdom.
 
 3-opt.py implements a 3-opt algorithm which improves upon the Christofides algorithm used in disney_tsp_final
