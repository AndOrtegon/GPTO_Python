import numpy as np
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
global FE , GEOM 

# Format of point_matrix is [ point_id, x, y] for 2-d problems, and 
# [ point_id, x, y, z] for 3-d problems)

point_matrix = np.array( (
    ( 0 , 12.5000 , 12.5000 ) ,
    ( 1 , 12.5000 , 37.5000 ) ,
    ( 2 , 12.5000 , 62.5000 ) ,
    ( 3 , 12.5000 , 87.5000 ) ,
    ( 4 , 37.5000 , 12.5000 ) ,
    ( 5 , 37.5000 , 37.5000 ) ,
    ( 6 , 37.5000 , 62.5000 ) ,
    ( 7 , 37.5000 , 87.5000 ) ,
    ( 8 , 62.5000 , 12.5000 ) ,
    ( 9 , 62.5000 , 37.5000 ) ,
    ( 10 , 87.5000 , 12.5000 ) ,
    ( 11 , 87.5000 , 37.5000 ) ) )

 # Format of bar_matrix is [ bar_id, pt1, pt2, alpha, w/2 ], where alpha is
 # the initial value of the bar's size variable, and w/2 the initial radius
 # of the bar.
 #
bar_matrix = np.array( (
    (  0 ,  0 , 1 , 0.5000 , 2.0000 ) ,
    (  1 ,  0 , 4 , 0.5000 , 2.0000 ) , 
    (  2 ,  1 , 2 , 0.5000 , 2.0000 ) ,
    (  3 ,  1 , 4 , 0.5000 , 2.0000 ) ,
    (  4 ,  1 , 5 , 0.5000 , 2.0000 ) ,
    (  5 ,  2 , 3 , 0.5000 , 2.0000 ) ,
    (  6 ,  2 , 5 , 0.5000 , 2.0000 ) ,
    (  7 ,  2 , 6 , 0.5000 , 2.0000 ) , 
    (  8 ,  3 , 6 , 0.5000 , 2.0000 ) ,
    (  9 ,  3 , 7 , 0.5000 , 2.0000 ) ,
    ( 10 ,  4 , 5 , 0.5000 , 2.0000 ) ,
    ( 11 ,  4 , 8 , 0.5000 , 2.0000 ) ,
    ( 12 ,  5 , 6 , 0.5000 , 2.0000 ) ,
    ( 13 ,  5 , 8 , 0.5000 , 2.0000 ) ,
    ( 14 ,  5 , 9 , 0.5000 , 2.0000 ) ,
    ( 15 ,  6 , 7 , 0.5000 , 2.0000 ) ,
    ( 16 ,  8 , 9 , 0.5000 , 2.0000 ) ,
    ( 17 ,  8 , 10 , 0.5000 , 2.0000 ) ,
    ( 18 ,  9 , 10 , 0.5000 , 2.0000 ) ,
    ( 19 ,  9 , 11 , 0.5000 , 2.0000 ) , 
    ( 20 , 10 , 11 , 0.5000 , 2.0000 ) ) ) 

# *** Do not modify the code below ***
GEOM['initial_design']['point_matrix'] = point_matrix
GEOM['initial_design']['bar_matrix'] = bar_matrix


print('initialized ' + str(FE['dim']) + 'd initial design with ' + 
    str( GEOM['initial_design']['point_matrix'].shape[0] ) + ' points and ' +
    str( GEOM['initial_design']['bar_matrix'].shape[0] ) + ' bars\n' )
