# def initial_cantilever2d_geometry():
## Initial design input file 
#
# *** THIS SCRIPT HAS TO BE CUSTOMIZED BY THE USER ***
#
# In this file, you must create two matrices that describe the initial
# design of bars.
#
# The first matrix contains the IDs (integer) and coordinates of the
# endpoints of the bars (point_matrix).
#
# The second matrix defines the IDs of the points that make up each bar.
# This matrix also sets the initial value of each bar's size variable, and
# the initial bar radius (half-width of the bar in 2-d).
#
# Note that this way of defining the bars allows for bars to be 'floating'
# (if the endpoints of a bar are not shared by any other bar) or
# 'connected' (if two or more bars share the same endpoint).
#

# *** Do not modify the line below ***
import numpy as np
# global FE, GEOM 

# Format of point_matrix is [ point_id, x, y] for 2-d problems, and 
# [ point_id, x, y, z] for 3-d problems)

point_matrix = np.array( (
    ( 0 , 0.25 , 4.75 ) , 
    ( 1 , 4.75 , 4.75 ) , 
    ( 2 , 0.25 , 2.75 ) , 
    ( 3 , 4.75 , 4.75 ) , 
    ( 4 , 0.25 , 4.75 ) , 
    ( 5 , 4.75 , 2.75 ) , 
    ( 6 , 0.25 , 2.75 ) , 
    ( 7 , 4.75 , 2.75 ) , 
    ( 8 , 0.25 , 2.25 ) , 
    ( 9 , 4.75 , 2.25 ) , 
    ( 10 , 0.25 , 0.25 ) , 
    ( 11 , 4.75 , 2.25 ) , 
    ( 12 , 0.25 , 2.25 ) , 
    ( 13 , 4.75 , 0.25 ) , 
    ( 14 , 0.25 , 0.25 ) , 
    ( 15 , 4.75 , 0.25 ) , 
    ( 16 , 5.25 , 4.75 ) , 
    ( 17 , 9.75 , 4.75 ) , 
    ( 18 , 5.25 , 2.75 ) , 
    ( 19 , 9.75 , 4.75 ) , 
    ( 20 , 5.25 , 4.75 ) , 
    ( 21 , 9.75 , 2.75 ) , 
    ( 22 , 5.25 , 2.75 ) , 
    ( 23 , 9.75 , 2.75 ) , 
    ( 24 , 5.25 , 2.25 ) , 
    ( 25 , 9.75 , 2.25 ) , 
    ( 26 , 5.25 , 0.25 ) , 
    ( 27 , 9.75 , 2.25 ) , 
    ( 28 , 5.25 , 2.25 ) , 
    ( 29 , 9.75 , 0.25 ) , 
    ( 30 , 5.25 , 0.25 ) , 
    ( 31 , 9.75 , 0.25 ) , 
    ( 32 , 10.25 , 4.75 ) , 
    ( 33 , 14.75 , 4.75 ) , 
    ( 34 , 10.25 , 2.75 ) , 
    ( 35 , 14.75 , 4.75 ) , 
    ( 36 , 10.25 , 4.75 ) , 
    ( 37 , 14.75 , 2.75 ) , 
    ( 38 , 10.25 , 2.75 ) , 
    ( 39 , 14.75 , 2.75 ) , 
    ( 40 , 10.25 , 2.25 ) , 
    ( 41 , 14.75 , 2.25 ) , 
    ( 42 , 10.25 , 0.25 ) , 
    ( 43 , 14.75 , 2.25 ) , 
    ( 44 , 10.25 , 2.25 ) , 
    ( 45 , 14.75 , 0.25 ) , 
    ( 46 , 10.25 , 0.25 ) , 
    ( 47 , 14.75 , 0.25 ) , 
    ( 48 , 15.25 , 4.75 ) , 
    ( 49 , 19.75 , 4.75 ) , 
    ( 50 , 15.25 , 2.75 ) , 
    ( 51 , 19.75 , 4.75 ) , 
    ( 52 , 15.25 , 4.75 ) , 
    ( 53 , 19.75 , 2.75 ) , 
    ( 54 , 15.25 , 2.75 ) , 
    ( 55 , 19.75 , 2.75 ) , 
    ( 56 , 15.25 , 2.25 ) , 
    ( 57 , 19.75 , 2.25 ) , 
    ( 58 , 15.25 , 0.25 ) , 
    ( 59 , 19.75 , 2.25 ) , 
    ( 60 , 15.25 , 2.25 ) , 
    ( 61 , 19.75 , 0.25 ) , 
    ( 62 , 15.25 , 0.25 ) , 
    ( 63 , 19.75 , 0.25 ) ) )

# Format of bar_matrix is [ bar_id, pt1, pt2, alpha, w/2 ], where alpha is
# the initial value of the bar's size variable, and w/2 the initial radius
# of the bar.
#
bar_matrix = np.array( (
    ( 0 , 0 , 1 , 0.5, 0.25 ) ,
    ( 1 , 2 , 3 , 0.5, 0.25 ) ,
    ( 2 , 4 , 5 , 0.5, 0.25 ) ,
    ( 3 , 6 , 7 , 0.5, 0.25 ) ,
    ( 4 , 8 , 9, 0.5, 0.25 ) ,
    ( 5 , 10, 11, 0.5, 0.25 ) ,
    ( 6 , 12, 13, 0.5, 0.25 ) ,
    ( 7 , 14, 15, 0.5, 0.25 ) ,
    ( 8 , 16 , 17 , 0.5, 0.25 ) ,
    ( 9 , 18 , 19 , 0.5, 0.25 ) ,
    ( 10 , 20 , 21 , 0.5, 0.25 ) ,
    ( 11 , 22 , 23 , 0.5, 0.25 ) ,
    ( 12 , 24 , 25, 0.5, 0.25 ) ,
    ( 13 , 26, 27, 0.5, 0.25 ) ,
    ( 14 , 28, 29, 0.5, 0.25 ) ,
    ( 15 , 30, 31, 0.5, 0.25 ) ,
    ( 16 , 32 , 33 , 0.5, 0.25 ) ,
    ( 17 , 34 , 35 , 0.5, 0.25 ) ,
    ( 18 , 36 , 37 , 0.5, 0.25 ) ,
    ( 19 , 38 , 39 , 0.5, 0.25 ) ,
    ( 20 , 40 , 41, 0.5, 0.25 ) ,
    ( 21 , 42, 43, 0.5, 0.25 ) ,
    ( 22 , 44, 45, 0.5, 0.25 ) ,
    ( 23 , 46, 47, 0.5, 0.25 ) ,
    ( 24 , 48 , 49 , 0.5, 0.25 ) ,
    ( 25 , 50 , 51 , 0.5, 0.25 ) ,
    ( 26 , 52 , 53 , 0.5, 0.25 ) ,
    ( 27 , 54 , 55 , 0.5, 0.25 ) ,
    ( 28 , 56 , 57, 0.5, 0.25 ) ,
    ( 29 , 58 , 59, 0.5, 0.25 ) ,
    ( 30 , 60 , 61, 0.5, 0.25 ) ,
    ( 31 , 62 , 63, 0.5, 0.25 ) ) )

# *** Do not modify the code below ***
GEOM['initial_design']['point_matrix'] = point_matrix
GEOM['initial_design']['bar_matrix'] = bar_matrix

print('initialized ' + str(FE['dim']) + 'd initial design with ' + 
    str( GEOM['initial_design']['point_matrix'].shape[0] ) + ' points and ' +
    str( GEOM['initial_design']['bar_matrix'].shape[0] ) + ' bars\n' )