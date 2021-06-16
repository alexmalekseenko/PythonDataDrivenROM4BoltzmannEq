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
BKern_all, mm = my_readwrite.readromBkrnl("F:\\temp\\cleaned_SVs_100_PlusMaxwell\\SVZeroCln100M41Tr0KrnlP5_Bkrnl.dat")
#########################################################################
## we will need only a subset of the projection vectors
#########################################################################
# We create an array that collects all the eigenvalues
all_eigs =np.zeros((mm+1,mm-3))
############################################################
#
make_csv_eigenvalues=True
if make_csv_eigenvalues:
 for k in range(3,mm):
  if k<mm:
      BKern=BKern_all[0:k,0:k]
  else:
      BKern=BKern_all
  ###
  ## Now we will study Bmat
  evals, evects = np.linalg.eig(BKern)
  all_eigs[0,k-3] = k
  all_eigs[1:k+1,k-3]=evals[:]
 ### All done, Now export Bmat to .csv file
 my_readwrite.my_writeMatrix_cvs(all_eigs)

range(5)

