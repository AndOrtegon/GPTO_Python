import numpy as np

#def inputs_cantilever2d(FE,OPT,GEOM):
## Input file 
#
# *** THIS SCRIPT HAS TO BE CUSTOMIZED BY THE USER ***
#
# This script sets up all the parameters for the optimization, and also
# indicates what files contain the mesh, boundary conditions, and initial
# design.
#
#
# It is recommended that for different problems you make copies of this
# file in the input_files subfolder, so that to switch from one problem to
# another you need only change the run statement in the inputs.m file in
# the root folder.
#


# ** Do not modify this line **
#global FE, OPT, GEOM

# FE = {}
# OPT = {}
# GEOM = {}


# Set this flag to True if you want plotting during the optimization
plot_cond = True 


## =======================================================================
## Mesh information
#
# First of all, an important clarification: in this code, we refer to mesh
# as exclusively the nodal coordinates and the element nodal connectivity
# array. You have to write/modify a separate matlab script (setup_bcs) to
# set up the loads and displacement boundary conditions).
#
# This code provides three options to populate a mesh, indicated in the 
# FE['mesh_input.type field: 'generate','read-home-made', and 'read-gmsh':
#
# 1- 'generate': Generate a rectangular/parallelepiped mesh on the fly by 
#                providing the dimensions and element size (using the 
#                generate_mesh function).
# 2- 'read-home-made': Load mesh from Matlab .mat file 
#                      (which you can create before running this code using 
#                      the makemesh function).
# 3- 'read-gmsh':  Read quadrilateral or hexahedral mesh generated by Gmsh 
#                  and exported to the Matlab format. For this code, we
#                  tested version 4.0.6 of Gmsh (but it is possible that
#                  earlier versions work if so, try at your own risk).
#
FE['mesh_input'] = {}
FE['mesh_input']['type'] = 'read-gmsh'

# If mesh input type is 'generate', you must specify the dimensions of
# the rectangle/cuboid and the number of elements along each direction:
FE['mesh_input']['box_dimensions'] = np.array( (20,10) )
FE['mesh_input']['elements_per_side'] = np.array( (128,64) )

# If mesh input type is 'read-home-made', you must provide a
# mesh file name including extension (*.mat).
# 
# NOTE: all folders are relative to the root folder where the GPTO_b.m
# script is located.
#
FE['mesh_input']['mesh_filename'] = 'input_files/cantilever2d/2drectangle.mat'

# If mesh input type is 'read-gmsh', you must provide a
# mesh file name including extension (*.m). To produce this file, you must
# first generate a transfinite mesh (only quad elements in 2-d, only hexa
# elements in 3-d) in Gmsh, and then export it with Matlab format
# (including the .m extension).
FE['mesh_input']['gmsh_filename'] = 'input_files/cantilever2d/cantilever2d.py'

    
## =======================================================================
## Boundary conditions

# Here, you must specify the path to a Matlab script file that sets up the
# boundary conditions (which you must modify according to the problem)
FE['mesh_input']['bcs_file'] = 'input_files/cantilever2d/setup_bcs_cantilever2d.py'


## =======================================================================
## Material information
#
# Specify the Young's modulus and Poisson ratio 
# Young's modulus of the design material
# Poisson ratio of the design material
FE['material']= { 'E' : 1 , 'nu' : 0.3 }  

FE['material']['rho_min'] = 1e-2 # Minimum density in void region
FE['material']['nu_void'] = 0.3  # Poisson ratio of the void material

## =======================================================================
## Initial design (geometry)
# This flag allows you to use an initial design saved in a previous ru
# If restart = False, this path should be the path to a Matlab script
# that initializes the geometry otherwise, it should be the path to a
# .mat file previously saved by the code
GEOM['initial_design'] = { 
    'path':'input_files/cantilever2d/initial_cantilever2d_geometry.py' ,
    'plot':plot_cond ,
    'restart':False}

# You must specify the lower and upper bounds on the bar radius.  If you 
# want a design with fixed bar radii, simply set both fields to the same
# value.
GEOM['min_bar_radius'] = 0.5
GEOM['max_bar_radius'] = 0.51
## =======================================================================    
## Finite element solver
FE['analysis'] = {'solver':{}}
FE['analysis']['solver']['type'] = 'iterative' # 'direct'  # Options: 'direct' or 'iterative'
FE['analysis']['solver']['tol'] = 1e-5          # only for iterative
FE['analysis']['solver']['maxit'] = 1e4         # only for iterative
FE['analysis']['solver']['use_gpu'] = False 
# NOTE: gpus can only be used if:
# (a) FE['analysis.solver.type = 'iterative'
# (b) the system has a compatible nvidia gpu (and drivers)
#     You can query your system's gpu with the matlab command
#     >> gpuDevice()
# The gpu solver may be slower than an iterative solver on the cpu for
# smaller problems because of the cost to transfer data to the gpu. 
    
## =======================================================================        
## Optimization problem definition
# functions:
OPT['functions'] = {}
# Name of objective function
OPT['functions']['objective'] = 'compliance'
# Names of inequality (<=) constraints
OPT['functions']['constraints'] = 'volume fraction'
# Inequality constraint (upper) limits vector: should have the
# constraint limit for each one of the constraints.
OPT['functions']['constraint_limit'] = [0.3]

## =======================================================================        
## Geometry projection parameters
# Sample window radius.  By default, the routine init_optimization
# computes this radius as sqrt(FE['dim)/2 * FE['elem_vol.^(1./FE['dim),
# which corresponds to the radius of the circle (in 2-d) or sphere (in
# 3-d) that circumscribes a square or cube element.  If you uncomment
# the following line, you can override this value (if you do, then the
# value corresponds to the actual radius dimension).
# 
OPT['parameters'] = {}

# OPT['parameters']['elem_r'] = .1   # Eq. (34)
#
# Type of smooth maximum function for aggregation (Boolean union)
# Options are: 'mod_p-norm', 'mod_p-mean', 'KS', and 'KS_under'
OPT['parameters']['smooth_max_scheme'] = 'mod_p-norm'
# Parameter to be used for the smooth_max function
OPT['parameters']['smooth_max_param'] = 8    
# Penalization scheme 
OPT['parameters']['penalization_scheme'] = 'SIMP' # Options: 'SIMP', 'RAMP' 
# Parameter to be used for the penalization
OPT['parameters']['penalization_param'] = 3
    
## =======================================================================        
## Optimization parameters
OPT['options'] = {}
# Optimizer (options: 'default' and 'mma')
# OPT['options']['optimizer'] = 'default'
OPT['options']['optimizer'] = 'mma'
# Whether plots should be produced or not 
OPT['options']['plot'] = plot_cond 
# Write to a vkt file options are 'none', 'last' (only write last 
# iteration)and 'all' (write all iterations).  
OPT['options']['write_to_vtk'] = 'last'
# Recall that paths are relative to the root folder 
OPT['options']['vtk_output_path'] = 'output_files'
# whether to scale the design variables to the range [0,1]
OPT['options']['dv_scaling'] = True 
# Move limits as a fraction of the range between bounds 
OPT['options']['move_limit'] = 0.05 
# Maximum number of iterations 
OPT['options']['max_iter'] = 10
# Minimum step size in design
OPT['options']['step_tol'] = 2e-3 
# Convergence tolerance on KKT norm
OPT['options']['kkt_tol'] = 1e-4 

## =======================================================================        
## Sensitivities finite difference check
#
# These options allow you to run a finite difference check on the cost
# function, and/or the constraint.
#
# Please note that if you do a finite difference check, the code will stop
# right after the check and not continue with the optimization (typically
# you want to do one or the other but not both).
#
# Whether or not to perform sensitivities finite difference check
OPT['make_fd_check'] = False
# Step size for finite difference
OPT['fd_step_size'] = 1e-7
# Whether or not to check cost function sensitivities
OPT['check_cost_sens'] = True
# Whether or not to check constraint sensitivities
OPT['check_cons_sens'] = True
    