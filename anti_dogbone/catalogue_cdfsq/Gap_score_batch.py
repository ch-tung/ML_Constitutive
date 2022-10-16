#!/usr/bin/env python
# coding: utf-8

# <h1><center>Sample design - Stress/Strain anisotropy tester</center></h1>

# In[1]:


# list available samples
from os import listdir
from os import path
meshpath = './output_files/'
mesh_list = [x for x in listdir(meshpath) if x.endswith('.xdmf')]
mesh_list = [
 '2hole.xdmf',
 '2notch.xdmf',
 'test_array_circle.xdmf',   
 'test_array_ellipse.xdmf',
 'test_clover.xdmf',
 'test_dumbell.xdmf',
 'test_dumbell_45.xdmf',
 'test_dumbell_BF3.xdmf',
#  'test_clover_rand.xdmf',
 'test_loop_array_2fold.xdmf',
#  'test_loop_array_3fold.xdmf',
#  'test_NPR.xdmf',
 'test_NPR_rounded.xdmf',
 'test_NPR_rounded_45.xdmf',
#  'test_NPR_rounded_90.xdmf',
#  'test_NPR_rounded_rand.xdmf',
#  'test_NPR_rounded_swivel.xdmf',
#  'test_NPR_rounded_h.xdmf',
#  'test_NPR_rounded_v.xdmf',
 'test_simplehole_discA1.xdmf',
 'test_simplehole_ellipseA12.xdmf',
 'test_simplehole_rectangleA12.xdmf',
 'test_simplehole_squareA1.xdmf',
 'test_simplehole_squareB1.xdmf',
 'test_withouthole.xdmf',
#  'example_2fold.xdmf',
#  'example_3fold.xdmf',
]
mesh_list


# In[2]:


# from fenics import *
# from dolfin_adjoint import *
# import pygmsh_mesh_functions
# from pygmsh_mesh_functions import *
import meshio
import numpy as np
from tqdm import tqdm

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# For quality evaluation
from anisotropyv3_cdf import *
import optisample as opti

from scipy.io import loadmat
from scipy.io import savemat


# In[3]:


Lx=1.0
Ly=1.0

for i in range(len(mesh_list)):

    #### Evaluate strain diversity ####
    #### load mesh
    meshfile = mesh_list[i]
    filename = './output_files/'+meshfile
    if not path.exists(filename):
        print(meshfile+' failed')
        continue
    
    #### Results_grid
    #### Calculate anisotropic score #### 
    thresholds = [0.00,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]
    results = opti.onmesh_FITsample_grid(filename,lx=Lx,ly=Ly,title='Grid_GapScore',gapscore=GapScore_cdf,onlystresspoints=True,saving=False)
    Score_stress_gap = [GapScore_cdfsq(results,quantileinf=0.50,strcoeffinf=x,quantilesup=1.00,strcoeffsup=1.00)
                        for x in thresholds]
    print('Score_stress_gap_grid = {}'.format(Score_stress_gap))

    #### Save results ####
    Score_gap = np.array(Score_stress_gap)
    
    from scipy.io import savemat
    mdic = {
            'Score_gap':Score_gap,
           }
    savemat('./results/'+meshfile+'_score_grid.mat', mdic)
    
    #### Results_rand
    #### Calculate anisotropic score #### 
    thresholds = [0.00,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]
    results = opti.onmesh_FITsample_randsamp(filename,lx=Lx,ly=Ly,title='Grid_GapScore',gapscore=GapScore_cdf,onlystresspoints=True,saving=False)
    Score_stress_gap = [GapScore_cdfsq(results,quantileinf=0.50,strcoeffinf=x,quantilesup=1.00,strcoeffsup=1.00)
                        for x in thresholds]
    print('Score_stress_gap_rand = {}'.format(Score_stress_gap))

    #### Save results ####
    Score_gap = np.array(Score_stress_gap)
    
    from scipy.io import savemat
    mdic = {
            'Score_gap':Score_gap,
           }
    savemat('./results/'+meshfile+'_score_rand.mat', mdic)


# In[ ]:




