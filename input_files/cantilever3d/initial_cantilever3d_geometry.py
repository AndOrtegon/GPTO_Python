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
    ( 0 , 2.4 , 2.5 , 2.5 ) ,
    ( 1 , 2.6 , 2.5 , 2.5 ) ,
    ( 2 , 7.4 , 2.5 , 2.5 ) ,
    ( 3 , 7.6 , 2.5 , 2.5 ) ,
    ( 4 , 12.4 , 2.5 , 2.5 ) ,
    ( 5 , 12.6 , 2.5 , 2.5 ) ,
    ( 6 , 17.4 , 2.5 , 2.5 ) ,
    ( 7 , 17.6 , 2.5 , 2.5 ) ,
    ( 8 ,  2.4 , 7.5 , 2.5 ) ,
    ( 9 , 2.6 , 7.5 , 2.5 ) ,
    ( 10 , 7.4 , 7.5 , 2.5 ) ,
    ( 11 , 7.6 , 7.5 , 2.5 ) ,
    ( 12 , 12.4 , 7.5 , 2.5 ) ,
    ( 13 , 12.6 , 7.5 , 2.5 ) ,
    ( 14 , 17.4 , 7.5 , 2.5 ) ,
    ( 15 , 17.6 , 7.5 , 2.5 ) ,
    ( 16 ,  2.4 , 2.5 , 7.5 ) ,
    ( 17 ,  2.6 , 2.5 , 7.5 ) ,
    ( 18 ,  7.4 , 2.5 , 7.5 ) ,
    ( 19 , 7.6 , 2.5 , 7.5 ) ,
    ( 20 , 12.4 , 2.5 , 7.5 ) ,
    ( 21 , 12.6 , 2.5 , 7.5 ) ,
    ( 22 , 17.4 , 2.5 , 7.5 ) ,
    ( 23 , 17.6 , 2.5 , 7.5 ) ,
    ( 24 , 2.4 , 7.5 , 7.5 ) ,
    ( 25 , 2.6 , 7.5 , 7.5 ) ,
    ( 26 , 7.4 , 7.5 , 7.5 ) ,
    ( 27 , 7.6 , 7.5 , 7.5 ) ,
    ( 28 , 12.4 , 7.5 , 7.5 ) ,
    ( 29 , 12.6 , 7.5 , 7.5 ) ,
    ( 30 , 17.4 , 7.5 , 7.5 ) ,
    ( 31 , 17.6 , 7.5 , 7.5 ) ) )


# Format of bar_matrix is [ bar_id, pt1, pt2, alpha, w/2 ], where alpha is
# the initial value of the bar's size variable, and w/2 the initial radius
# of the bar.
#
bar_matrix = np.array( (
    (  0 , 0 , 1 , 0.5, 0.75 ) ,
    (  1 , 2 , 3 , 0.5, 0.75 ) ,
    (  2 , 4 , 5 , 0.5, 0.75 ) ,
    (  3 , 6 , 7 , 0.5, 0.75 ) ,
    (  4 , 8 , 9 , 0.5, 0.75 ) ,
    (  5 , 10, 11, 0.5, 0.75 ) ,
    (  6 , 12, 13, 0.5, 0.75 ) ,
    (  7 , 14, 15, 0.5, 0.75 ) ,
    (  8 , 16, 17, 0.5, 0.75 ) ,
    (  9 , 18, 19, 0.5, 0.75 ) ,
    ( 10 , 20, 21, 0.5, 0.75 ) ,
    ( 11 , 22, 23, 0.5, 0.75 ) ,
    ( 12 , 24, 25, 0.5, 0.75 ) ,
    ( 13 , 26, 27, 0.5, 0.75 ) ,
    ( 14 , 28, 29, 0.5, 0.75 ) ,
    ( 15 , 30, 31, 0.5, 0.75 ) ) )

# *** Do not modify the code below ***
GEOM['initial_design']['point_matrix'] = point_matrix
GEOM['initial_design']['bar_matrix'] = bar_matrix

print('initialized ' + str(FE['dim']) + 'd initial design with ' + 
    str( GEOM['initial_design']['point_matrix'].shape[0] ) + ' points and ' +
    str( GEOM['initial_design']['bar_matrix'].shape[0] ) + ' bars\n' )