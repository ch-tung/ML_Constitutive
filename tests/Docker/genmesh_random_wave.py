from fenics import *
from dolfin_adjoint import *
import pygmsh_mesh_functions
from pygmsh_mesh_functions import *
import meshio
import numpy as np
import matplotlib.pyplot as plt

import skimage.io
from skimage.measure import find_contours, subdivide_polygon, approximate_polygon

# img2mesh.py: python script converting 2D binaary image to FEM mesh
from img2mesh import *

alpha = 2.5e-2

K_xx = 30**2
K_yy = 20**2
K_xy = 0
K0 = np.array([[K_xx,K_xy],[K_xy,K_yy]])

eps = 1e-4
K = K0 + np.eye(2)*eps

def genmesh_random_wave(alpha = 2.5e-2, K = K0 + np.eye(2)*eps,
                        n_wave = 500, n_x = 100, n_y = 100, 
                        filename = "test_random_wave.xdmf"):
    
    x = np.arange(n_x)/n_x
    y = np.arange(n_y)/n_y

    xx, yy = np.meshgrid(x,y)

    L = np.linalg.cholesky(K)

    phi = np.zeros_like(xx)

    for i in range(n_wave):
        u = np.random.normal(size=2)
        k_i = L.T@u

        phase = np.random.rand()*2*np.pi
        phi_i = np.exp(1j*(k_i[0]*xx + k_i[1]*yy + phase))

        phi = phi + phi_i

    phi = phi/n_wave
    phi_real = np.real(phi)
    
    markers = np.tanh((phi_real-alpha)*100)
    # markers[:, 0] = -1
    # markers[:,-1] = -1
    # markers[ 0,:] = -1
    # markers[-1,:] = -1

    img = markers
    imsize = img.shape
    contours_list = find_contours(img, 0)
    
    img2mesh(img, filename, meshsize_min = 0.025, meshsize_max = 0.05)