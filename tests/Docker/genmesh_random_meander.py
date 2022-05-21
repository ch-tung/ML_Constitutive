from fenics import *
from dolfin_adjoint import *
import pygmsh_mesh_functions
from pygmsh_mesh_functions import *
import meshio
import numpy as np
import matplotlib.pyplot as plt

import skimage.io
from skimage.measure import find_contours, subdivide_polygon, approximate_polygon
from skimage.segmentation import watershed
from skimage.morphology import dilation

from skimage.filters import sobel, gaussian

from scipy import ndimage as ndi

def hole_polygon(domain, polygon, scale, tol=1e-2):
    if np.array_equal(polygon[-1],polygon[0]):
        polygon = polygon[:-1]
    
    tool = add_polygon(polygon*scale)
    domain = subtract(domain, tool)
    
alpha = 0.25

Kx = 4
Ky = 4

K_xx = Kx**2
K_yy = Ky**2
K_xy = 0
K0 = np.array([[K_xx,K_xy],[K_xy,K_yy]])

eps = 1e-4
K = K0 + np.eye(2)*eps

def genmesh_random_meander(alpha = 0.25, Kx = 4, Ky = 4,
                           n_wave = 500, n_x = 100, n_y = 100, 
                           filename = "test_random_meander.xdmf"):

    ## spinodal
    x = np.arange(n_x)/n_x
    y = np.arange(n_y)/n_y

    xx, yy = np.meshgrid(x,y)

    phi = np.zeros_like(xx)

    for i in range(n_wave):
        theta = np.random.rand()*2*pi
        k_i = (np.array([Kx*np.cos(theta),Ky*np.sin(theta)]))

        phase = np.random.rand()
        phi_i = np.exp(2*pi*1j*(k_i[0]*xx + k_i[1]*yy + phase))

        phi = phi + phi_i

    ## Gaussian random wave
    # x = np.arange(n_x)/n_x
    # y = np.arange(n_y)/n_y

    # xx, yy = np.meshgrid(x,y)

    # L = np.linalg.cholesky(K)

    # phi = np.zeros_like(xx)

    # for i in range(n_wave):
    #     u = np.random.normal(size=2)
    #     k_i = L.T@u

    #     phase = np.random.rand()
    #     phi_i = np.exp(2*pi*1j*(k_i[0]*xx + k_i[1]*yy + phase))

    #     phi = phi + phi_i

    ## fluctuated grids
    if 0:
        phi_grids = np.exp(pi*1j*(Kx*xx+Ky*yy)/np.sqrt(2)) + np.exp(pi*1j*(Kx*xx-Ky*yy)/np.sqrt(2))
        phi = phi + phi_grids*n_wave/10

        n_wave = n_wave+n_wave/10

    phi = phi*np.sqrt(2/n_wave)

    img = -np.abs(np.real(phi))
    imsize = img.shape

    img_ridge = img.copy()
    img_ridge[img_ridge>= -alpha] = 1
    img_ridge[img_ridge< -alpha] = 0

    img_dilation = img_ridge
    for i in range(4):
        img_dilation = dilation(img_dilation)

    img_blur = gaussian(img_dilation,1)

    contours_list = find_contours(img_blur, 0.5)

    polygon_list = []
    for i in range(len(contours_list)):
        polygon = np.flip(contours_list[i],axis=1)

        if not np.array_equal(polygon[-1],polygon[0]):
            d_ends = polygon[-1]-polygon[0]
            if d_ends[0]*d_ends[1] != 0:
                continue
            else:
                polygon = polygon[:-1]

        polygon_subdivide = subdivide_polygon(polygon, degree=2, preserve_ends=True)
        polygon_approx = approximate_polygon(polygon_subdivide,1/2)

        polygon_list.append(polygon_approx)

    ## Generate Mesh
    Lx = 1
    Ly = 1

    meshsize_min = 0.07*min(Lx,Ly)#0.015*min(Lx,Ly)
    meshsize_max = 0.2*min(Lx,Ly)
    print(meshsize_min)

#     pygmsh_mesh_functions.hmeshmin = meshsize_min
#     pygmsh_mesh_functions.hmeshmax = meshsize_max
    
    pygmsh_mesh_functions.reset_geometry_parameters(meshsize_min,meshsize_max)

    domain = add_polygon([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    X0 = np.array([0, 0])

    #OPERATIONS ON DOMAIN
    for i in range(len(polygon_list)):
        hole_polygon(domain, polygon_list[i]-1, 1/(imsize[0]-3))


    #CREATE AND STORE MESH
    sample_name = filename
    out_dir = pathlib.Path("output_files")
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = create_fenics_mesh(mesh_path = out_dir / sample_name, verbose=True)
    cell_type = mesh.cell_name()
    with create_XDMFFile(out_dir / sample_name) as xf:
        xf.write(mesh)

    plot(mesh)
    len(mesh.coordinates())
    
    return mesh