## Input file 
#
# *** THIS SCRIPT HAS TO BE CUSTOMIZED BY THE USER ***
#
# This script sets up the displacement boundary conditions and the forces
# for the analysis. 
#
# Important note: you must make sure you do not simultaneously impose 
# displacement boundary conditions and forces on the same degree of
# freedom.

# ** Do not modify this line **
import numpy as np
global FE

coord_x = FE['coords'][0,:]
coord_y = FE['coords'][1,:]
if FE['dim'] == 3:
    coord_z = FE['coords'][2,:]

## ============================
## Compute predefined node sets
# compute_predefined_node_sets({'T_edge','TR_pt'})
# for an overview of this function, use: help compute_predefined_node_sets

TR_pt  = np.where( np.logical_and( FE['coords'][0,:] == 100 , FE['coords'][1,:] == 40 ) )[0]
T_edge = np.where( FE['coords'][1,:] == 100 )[0]
## ============================        


## Applied forces
net_mag = -.1  # Force magnitude (net over all nodes where applied)
load_dir = 2   # Force direction 
    
load_region = TR_pt
load_mag = net_mag/len(load_region)

# Here, we build the array with all the loads.  If you have multiple
# applied loads, the load_mat array must contain all the loads as follows:
#  - There is one row per each load on a degree of freedom
#  - Column 1 has the node id where the load is applied
#  - Column 2 has the direction (1 -> x, 2 -> y, 3 -> z)
#  - Column 3 has the load magnitude.
#

load_mat = np.zeros( ( len(load_region),3) )
load_mat[:,0] = load_region
load_mat[:,1] = load_dir
load_mat[:,2] = load_mag

## Displacement boundary conditions
disp_region = T_edge[None,:].copy()
disp_mag    = np.zeros((1, disp_region.shape[1] ))
disp_dirs1  = np.zeros((1, disp_region.shape[1] ) , dtype=int )  
disp_dirs2  = np.ones ((1, disp_region.shape[1] ) , dtype=int)

# Combine displacement BC regions
disp_region = np.concatenate( ( disp_region , disp_region ) , axis = 1 )
disp_dirs   = np.concatenate( ( disp_dirs1 , disp_dirs2 ) , axis = 1 )
disp_mag    = np.concatenate( ( disp_mag , disp_mag ) , axis = 1 )

# In this example we are constraining both the x- and y-directions along
# the top edge.

# Here, we build the array with all the displacement BCs. 
# The disp_mat array must contain all the loads as follows:
#  - There is one row per each load on a degree of freedom
#  - Column 1 has the node id where the displacement BC is applied
#  - Column 2 has the direction (1 -> x, 2 -> y, 3 -> z)
#  - Column 3 has the displacement magnitude.

disp_mat = np.zeros( ( disp_region.shape[1] ,3) , dtype=int)
for idisp in range(0, disp_region.shape[1] ):
    disp_mat[idisp, 0] = disp_region[0,idisp]
    disp_mat[idisp, 1] = disp_dirs[0,idisp]
    disp_mat[idisp, 2] = disp_mag[0,idisp]

# *** Do not modify the code below ***
#
# Write displacement boundary conditions and forces to the global FE
# structure.
#
# Note: you must assign values for all of the variables below.
#
FE['BC'] = {}
FE['BC']['n_pre_force_dofs'] = load_mat.shape[0]  # # of prescribed force dofs
FE['BC']['n_pre_disp_dofs'] = disp_mat.shape[0] # # of prescribed displacement dofs
FE['BC']['force_node']      =  load_mat[:,0].T
FE['BC']['force_dof']       = load_mat[:,1].T
FE['BC']['force_value']     = load_mat[:,2].T
FE['BC']['disp_node']       = disp_mat[:,0].T.astype(int)
FE['BC']['disp_dof']        = disp_mat[:,1].T.astype(int)
FE['BC']['disp_value']      = disp_mat[:,2].T

