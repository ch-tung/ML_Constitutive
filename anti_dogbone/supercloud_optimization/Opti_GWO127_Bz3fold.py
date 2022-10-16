#!/usr/bin/env python
# coding: utf-8

# ### Import packages

# In[ ]:


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from neorl import DE, XNES, GWO, ES
from anisotropyv3_cdf import *
# from harmonicholes import *
import optisample as opti

from scipy.io import savemat

from fenics import *
# from dolfin_adjoint import *
import pygmsh_mesh_functions
from pygmsh_mesh_functions import *
import meshio
from tqdm import tqdm

from datetime import datetime


# # First attempt with given number of holes

# ### Parameter Space

# In[ ]:


from genmesh import genmesh_loop_array_2fold, genmesh_loop_array_3fold, genmesh_network_2fold, genmesh_network_3fold, genmesh_Bezier_2fold, genmesh_Bezier_3fold


# In[ ]:


Lx=1e-2
Ly=1e-2


#Setup the parameter space
N_points=3

SimNb="GWO127"

BOUNDS={}

i=0
# edge_width
BOUNDS['width']=['float', 0.02, 0.02]

for k in range(N_points):
    #coeff
    BOUNDS['x'+str(i)]=['float', 0, 1]
    i+=1

nparam=i+1        
print(str(nparam)+" parameters")


Ngen=500
#Grey wolf parameters
Nwolves=80
#DE parameters
Npop=10*nparam
Ncores=8

#gapscore:
def gapscore(Stress_points):
    return GapScore_cdfsq(Stress_points,quantileinf=0.5,strcoeffinf=1.,quantilesup=1.,strcoeffsup=1.,tol=1e-14)



# In[ ]:


#MESH FUNCTION FOR THE MODEL TO OPTIMIZE

meshfunctionforindiv=genmesh_Bezier_3fold


# ### Fitness

# In[ ]:


N_iter=np.array([0,datetime.now()])

def FIT(individual,meshfunction=meshfunctionforindiv,lx=Lx,ly=Ly,N=N_iter):
    """Anisotropy score calculation.
            F(x) = formula based on FE calculation and point counting
            Range: [0.,1.]
            Minima: 0
    """
    
    return opti.FITsample_grid(individual,meshfunction,lx,ly,N=N_iter,title=SimNb,gapscore=gapscore)


# ### Grey wolf

# In[ ]:


N_iter=np.array([0,datetime.now()])
start=datetime.now()
gwo=GWO(mode='min', fit=FIT, bounds=BOUNDS, nwolves=Nwolves, ncores=Ncores, seed=1)
x_best, fitness_best, gwo_hist=gwo.evolute(ngen=Ngen, verbose=1)
end=datetime.now()

with open('RESULTS-'+SimNb+'.txt', 'w') as f:
    f.write("---GWO Results---\n")
    f.write("best sample : {}\n".format(x_best))
    f.write("best fitness : {}\n".format(fitness_best))
    f.write("\nEXECUTION TIME ---------> {}\n".format(end-start))
    
print('---GWO Results---', )
print('x:', x_best)
print('fitness:', fitness_best)

print("\nEXECUTION TIME ---------> {}\n".format(end-start))

opti.savesample(x_best,meshfunctionforindiv,Lx,Ly,title=SimNb+"mesh with fitness "+str(fitness_best)[0:7])


# ### Evolution Strategies $(\mu,\lambda)$

# In[ ]:


# N_iter=np.array([0,datetime.now()])
# es=ES(mode='min', bounds=BOUNDS, fit=FIT, lambda_=80, mu=40, mutpb=0.1,
#      cxmode='blend', cxpb=0.7, ncores=1, seed=1)
# x_best, fitness_best, es_hist=es.evolute(ngen=Ngen, verbose=1)
# print('---ES Results---', )
# print('x:', x_best)
# print('fitness:', fitness_best)
# opti.savesample(x_best,meshfunctionforindiv,Lx,Ly,title="ES result-sample with fitness "+str(fitness_best)[0:7])


# ### DE

# In[ ]:


# N_iter=np.array([0,datetime.now()])
# de=DE(mode='min', bounds=BOUNDS, fit=FITcosholes, npop=Npop, CR=0.2, F=0.7, ncores=1, seed=1)
# x_best, fitness_best, de_hist=de.evolute(ngen=Ngen, verbose=1)
# print('---DE Results---', )
# print('x:', x_best)
# print('fitness:', fitness_best)
# opti.savesample(x_best,meshfunctionforindiv,Lx,Ly,title="DE result-sample with fitness "+str(fitness_best)[0:7])


# ## Plot

# In[ ]:


#Plot fitness for both methods
plt.figure()

plt.plot(gwo_hist['alpha_wolf'], label='alpha_wolf')
plt.plot(gwo_hist['beta_wolf'], label='beta_wolf')
plt.plot(gwo_hist['delta_wolf'], label='delta_wolf')
plt.plot(gwo_hist['fitness'], label='best')
plt.xlabel('Generation')
plt.ylabel('Fitness')

plt.legend()
plt.savefig(SimNb+'fitness.png', dpi=96, bbox_inches='tight')


#save GWO results as .mat
from scipy.io import savemat
mdic = {'x_best':x_best,
       'fitness_best':fitness_best,
       'last_pop':gwo_hist['last_pop'],
       'alpha_wolf':gwo_hist['alpha_wolf'],
       'beta_wolf':gwo_hist['beta_wolf'],
       'delta_wolf':gwo_hist['delta_wolf'],
       'fitness':gwo_hist['fitness']
       }
savemat(SimNb+'.mat', mdic)