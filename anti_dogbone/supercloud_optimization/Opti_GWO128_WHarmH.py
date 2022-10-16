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


from genmesh import freeHH_withholewall


# In[ ]:


Lx=1e-2
Ly=1e-2


#Setup the parameter space
N_holes=2
N_harmonics=5
minmaxR=0.05*min(Lx,Ly)
maxmaxR=0.3*min(Lx,Ly)
minw=0.0005
maxw=0.0005

SimNb="GWO128"


BOUNDS={}
i=0
BOUNDS['x'+str(i)]=['float', minw, maxw]
i+=1
BOUNDS['x'+str(i)]=['int', N_holes, N_holes]
for n in range(N_holes):
    #x
    i+=1
    BOUNDS['x'+str(i)]=['float', 0+maxmaxR, Lx-maxmaxR]
    #y
    i+=1
    BOUNDS['x'+str(i)]=['float', 0+maxmaxR, Ly-maxmaxR]
    #maxR
    i+=1
    BOUNDS['x'+str(i)]=['float', minmaxR, maxmaxR]
    for k in range(N_harmonics):
        #coeff
        i+=1
        BOUNDS['x'+str(i)]=['float', 0., 1.]
        #rot
        i+=1
        BOUNDS['x'+str(i)]=['float', 0., 1.]

nparam=i+1        
print(str(nparam)+" parameters")

Ngen=500
#Grey wolf parameters
Nwolves=160
#DE parameters
# Npop=10*nparam
Ncores=48

#gapscore:
def gapscore(Stress_points):
    return GapScore_cdfsq(Stress_points,quantileinf=0.5,strcoeffinf=1.,quantilesup=1.,strcoeffsup=1.,tol=1e-14)


# In[ ]:


#MESH FUNCTION FOR THE MODEL TO OPTIMIZE

meshfunctionforindiv=freeHH_withholewall


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
    f.write("\n\n")
    f.write("LAST_POP:\n{}".format(gwo_hist['last_pop']))
    
print('---GWO Results---', )
print('x:', x_best)
print('fitness:', fitness_best)

print("\nEXECUTION TIME ---------> {}\n".format(end-start))

opti.savesample(x_best,meshfunctionforindiv,Lx,Ly,title=SimNb+"mesh with fitness "+str(fitness_best)[0:7],verbose=False)

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