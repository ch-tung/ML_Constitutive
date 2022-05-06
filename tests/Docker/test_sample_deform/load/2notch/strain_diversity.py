import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
from scipy.io import savemat

# import RSHE.py
# numba was used in RSHE.py to accelerate the calculation
from RSHE import RSHEscore

def strain_diversity(filename, save='False'):
    
    ## load data
    epsilon = loadmat(filename)['epsilon']
    n_element = epsilon.shape[0]
    
    ## Calculate anisotropic score
    n_frame = epsilon.shape[3]
    strain = np.arange(n_frame)/(n_frame-1)*0.05
    p = np.zeros((n_element,5,n_frame))
    p_ave = np.zeros((5,n_frame))
    p_std = np.zeros((5,n_frame))
    
    for it in tqdm(range(n_frame)):
    epsilon_t = epsilon[:,:,:,it].reshape(n_element,4)
    
    for ie in range(n_element):
        p[ie,:,it] = RSHEscore(epsilon_t[ie,[0,3,1]]) # xx, yy, xy
        
    p_ave[:,it] = np.mean(p[:,:,it],axis=0)
    p_std[:,it] = np.std(p[:,:,it],axis=0)
    
    if save == 'True':
        mdic = {'p':p, 'p_ave':p_ave, 'p_std':p_std}
        filename_export = 'strain_diversity.mat'
        savemat(filename_export, mdic)
        
    return p, p_ave, p_std