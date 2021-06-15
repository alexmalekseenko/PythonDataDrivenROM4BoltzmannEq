#####################################
###  09/10/2020 A. Alekseenko
###  This code will investigate the asymptotic stability of the ROM model and will
###  Explore conservtive projection.
###
#####################################

### TRIMMING PARAMETERS
MM = 41
Mtrim = 0
###
k=12
###

import numpy as np
import my_readwrite


# first we load the three index array of the ROM kernel. The third index is the non-symmetric index -- the output index
#CKern_all = my_readwrite.readromkrnl("E:\\temp\\cleaned100_MkKrnl\\M41Trim0\\SpHomT2D1M41Trm0SVKrnl_k53.dat")
######################
##  Our first exercise  is to compute the linearization term matrix B^{j.}_{.k}in the error equation
##
##  \partial_{t} e_{k} = B^{j}_{k}f e_{j} + A_{..k}^{ij.} e_{i}e_{j}
##   Formula for the B_{jk} is A^{i..}_{.jk} \hat{u}_{i} , where \hat{u}_{i} is the projection of the steady state maxwellian
## Thus we will need to load projection matrix, the mech, evaluate maxwellian on the mesh, project the maxwellian and
## compute the array convolution.
######################

######################
# We need some prep work. We will need to get nodes and weights of the quadratures associated with the solution loaded.
# these will be used to evaluate the solution entropy.
# these points can be obtained from the DG-Boltzmann solution saves of mesh parameters:
filename = 'F:\\SimulationsLearningCollision\\take3_M41\\good\\080\\CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(filename)
#########################################################################
#########################################################################
Mat_Moments = np.zeros((nodes_u.shape[1], 5))
Mat_Moments[:, 0] = nodes_gwts[0,:]
Mat_Moments[:, 1] = nodes_gwts[0,:] * nodes_u[0, :]
Mat_Moments[:, 2] = nodes_gwts[0,:] * nodes_v[0,:]
Mat_Moments[:, 3] = nodes_gwts[0,:] * nodes_w[0,:]
scrp_array = (nodes_u ** 2 + nodes_v ** 2 + nodes_w ** 2)
Mat_Moments[:, 4] = nodes_gwts[0,:] * scrp_array[0,:]
########################################################################
##########################################################################
import my_distributions
moment_vector_0 = np.array([[1.0,0.0,0.0,0.0,0.3]]) # ATTENTION: Last entry is energy, not temperature
sol_maxwell = my_distributions.maxwellian(moment_vector_0,nodes_u,nodes_v,nodes_w)  # evaluate truncated maxwellian with the same macroparameters

#########################################################################
### Load the projection vectors
#########################################################################
path = "F:\\temp\\cleaned_SVs_100\\sol_SVD_100.dat"
### There is a file with all singular vectors and with just the first 100
### Please pay attention that the trimming is correct
import pickle
save_file = open(path, 'rb')
Vh = pickle.load(save_file)
s = pickle.load(save_file)
svect = pickle.load(save_file)
save_file.close()
for ii in range(100):
   print(s[ii])
#########################################################################
## we will need only a subset of the projection vectors
#########################################################################
# Read the initial data and the steady state compute the orthogonal compliment in the ROM basis
#########################################################################
#
#####################
#results will be written into a text file
out_file = "F:\\temp\\cleamed_100_ROMruns\\results\\rerun_06112021\\not_projectedtoROM\\413\\steadystmoms413.txt"
save_file = open(out_file, 'w')
#####################
#
import my_vtk_tools
import my_utils
#
klist=[16,27,35,42]
for k in klist:
 path1="F:\\temp\\sphomruns\\good\\413\\CollTrnDta413_2kc1su1sv1sw3NXU41MuUU41MvVU41MwWU_time0.0042000000_SltnColl.dat"
 sol0, coll0, solsize = my_readwrite.my_read_sol_coll_trim(path1, MM, Mtrim)
 path2="F:\\temp\\cleamed_100_ROMruns\\results\\rerun_06112021\\not_projectedtoROM\\413\\k{0}longrun\\CollTrnDta413_2kc1su1sv1sw3NXU41MuUU41MvVU41MwWU_time5.4012000000_SltnColl.dat".format(k)
 sol1, coll1, solsize = my_readwrite.my_read_sol_coll_trim(path2, MM, Mtrim)
 ######
 ######
 # now we compute the orthogonal projections
 ######
 U=svect[:,0:k]
 sol0_perp = sol0 - np.matmul(np.matmul(sol0,svect[:,0:k]),svect[:,0:k].T)
 sol0_proj = np.matmul(np.matmul(sol0,svect[:,0:k]),svect[:,0:k].T)
 sol1_perp = sol1 - np.matmul(np.matmul(sol1,svect[:,0:k]),svect[:,0:k].T)
 sol1subtrM = sol1 - sol_maxwell
 ###################################
 # compute moments of the residuals
 ###################################
 entry_moments0 = my_utils.get_moments(sol0_perp, nodes_u, nodes_v, nodes_w, nodes_gwts, 0.0)
 entry_moments3 = my_utils.get_moments(sol0_proj, nodes_u, nodes_v, nodes_w, nodes_gwts, 0.0)
 entry_moments1 = my_utils.get_moments(sol1_perp, nodes_u, nodes_v, nodes_w, nodes_gwts, 5.4012)
 entry_moments2 = my_utils.get_moments(sol1subtrM, nodes_u, nodes_v, nodes_w, nodes_gwts, 5.4012)
 # now we write the moments into the output file:
 #####
 save_file.write("\n")
 save_file.write("k={0},\n".format(k))
 save_file.write(path1+"\n")
 save_file.write("moments of the orthongoal compliment of the initial data: \n")
 saveline = ""
 for i in range(0,entry_moments0.shape[1]):
  saveline = saveline + "{0:14.12f}, \n ".format(entry_moments0[0,i])
 save_file.write(saveline+"\n ")
 save_file.write("\n")
 save_file.write("moments of the orthogonal projection of the initial data: \n")
 saveline = ""
 for i in range(0, entry_moments3.shape[1]):
  saveline = saveline + "{0:14.12f}, \n ".format(entry_moments3[0, i])
 save_file.write(saveline + "\n ")
 save_file.write("\n")
 save_file.write(path2 + "\n")
 save_file.write("moments of the orthongoal compliment of the steady state: \n")
 saveline = ""
 for i in range(0,entry_moments1.shape[1]):
  saveline = saveline + "{0:14.12f}, \n ".format(entry_moments1[0,i])
 save_file.write(saveline + "\n ")
 save_file.write("moments of the steady state minus correct maxwellian: \n")
 saveline = ""
 for i in range(0,entry_moments2.shape[1]):
  saveline = saveline + "{0:14.12f}, \n ".format(entry_moments2[0,i])
 save_file.write(saveline + "\n ")
 ####### now we visualize all three
 #######################################
 # open a file that contains information about the nodal points.
 filename = 'F:\\SimulationsLearningCollision\\take3_M41\\good\\080\\CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_grids.dat'
 grids_cap_u, grids_cap_v, grids_cap_w, grids_u_tmp, grids_v_tmp, grids_w_tmp = my_readwrite.read_grids(filename)
 #########################################################################
 # Next we need to extract the data for the first grid.
 #########################################################################
 mu=grids_cap_u[0,0]
 mv=grids_cap_v[0,0]
 mw=grids_cap_w[0,0]
 ## next we truncate the grids arrays to only include the first grid
 grids_u = grids_u_tmp[:,0:mu]
 grids_v = grids_v_tmp[:,0:mu]
 grids_w = grids_w_tmp[:,0:mu]
 ## now we can use these values in the subroutine that writes VTK files
 path="F:\\temp\\cleamed_100_ROMruns\\results\\rerun_06112021\\not_projectedtoROM\\413"
 my_vtk_tools.writeVTKsolcellvalsRLG(sol0_perp, grids_u, grids_v, grids_w, 0.0042, path, 'initdataortcompk{0}'.format(k))
 my_vtk_tools.writeVTKsolcellvalsRLG(sol1_perp, grids_u, grids_v, grids_w, 5.4, path, 'ROMsolortcompk{0}'.format(k))
 my_vtk_tools.writeVTKsolcellvalsRLG(sol1subtrM, grids_u, grids_v, grids_w, 5.4, path, 'ROMsolorsubtrMaxwl{0}'.format(k))
############
save_file.close()
range(5)

