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

coord_x = FE['coords'][0,:]
coord_y = FE['coords'][1,:]
if FE['dim'] == 3:
    coord_z = FE['coords'][2,:]

## ============================
## Compute predefined node sets
compute_predefined_node_sets(FE,{ 0:'MR_pt' , 1:'TLK_pt' , 2:'TLF_pt' , 3:'BLK_pt' , 4:'BLF_pt' })
# for an overview of this function, use: help compute_predefined_node_sets

MR_pt  = FE['node_set']['MR_pt']
TLK_pt  = FE['node_set']['TLK_pt']
TLF_pt  = FE['node_set']['TLF_pt']
BLK_pt  = FE['node_set']['BLK_pt']
BLF_pt  = FE['node_set']['BLF_pt']
## ============================        


## Applied forces
net_mag = -0.1  # Force magnitude (net over all nodes where applied)
load_dir = 2   # Force direction 
    
load_region = MR_pt
load_mag = net_mag/len(load_region)

# Here, we build the array with all the loads.  If you have multiple
# applied loads, the load_mat array must contain all the loads as follows:
#  - There is one row per each load on a degree of freedom
#  - Column 1 has the node id where the load is applied
#  - Column 2 has the direction (1 -> x, 2 -> y, 3 -> z)
#  - Column 3 has the load magnitude.
#
load_mat = np.zeros( (len(load_region),3) )
load_mat[:,0] = load_region
load_mat[:,1] = load_dir
load_mat[:,2] = load_mag


## Displacement boundary conditions

# Symmetry boundary condition on left-hand side edge
disp_region = np.stack((TLK_pt,TLF_pt,BLK_pt,BLF_pt),axis=1)
disp_mag    = np.zeros( ( 1, disp_region.shape[1] ) )
disp_dirs0  = np.zeros( ( 1, disp_region.shape[1] ) )
disp_dirs1  = np.ones(  ( 1, disp_region.shape[1] ) )
disp_dirs2  = 2*np.ones(( 1, disp_region.shape[1] ) )


# Combine displacement BC regions
disp_region = np.concatenate((disp_region,disp_region,disp_region),axis=1)
disp_dirs   = np.concatenate((disp_dirs0,disp_dirs1,disp_dirs2),axis=1)
disp_mag    = np.concatenate((disp_mag,disp_mag,disp_mag),axis=1)

# In this example, we are constraining both the x- 
# and y- displacements. 
                      
# Here, we build the array with all the displacement BCs. 
# The disp_mat array must contain all the loads as follows:
#  - There is one row per each load on a degree of freedom
#  - Column 1 has the node id where the displacement BC is applied
#  - Column 2 has the direction (1 -> x, 2 -> y, 3 -> z)
#  - Column 3 has the displacement magnitude.
#print( disp_region )

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
#
FE['BC'] = {}
FE['BC']['n_pre_force_dofs']    = load_mat.shape[0] # # of prescribed force dofs
FE['BC']['n_pre_disp_dofs']     = disp_mat.shape[0] # # of prescribed displacement dofs
FE['BC']['force_node']  = load_mat[:,0].T
FE['BC']['force_dof']   = load_mat[:,1].T
FE['BC']['force_value'] = load_mat[:,2].T
FE['BC']['disp_node']   = disp_mat[:,0].T
FE['BC']['disp_dof']    = disp_mat[:,1].T
FE['BC']['disp_value']  = disp_mat[:,2].T
