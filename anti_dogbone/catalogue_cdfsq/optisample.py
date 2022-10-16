# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:18:29 2022

Module gathering function for mesh optimization

@author: aurel

Change logs
20220601 CHTUNG
removed the audio notification
updated anisotropyv2 to anisotropyv3

20220602 CHTUNG
Assign unique filename to .xdmf files to prevent confilct in parallel processing

20220618 CHTUNG
Added criterion in fitness function to return 1.5 if sum of stress less then some tol

20220620 CHTUNG
Changed out_dir = pathlib.Path("optimized_samples") in savesample()

20220627 CHTUNG
Added FITsample_grid to interpolate the stress from Fenics to a n_x by n_y mesh grid, then calculate the Gap score using these interpolated values

20220704 CHTUNG
Added FITsample_randsamp to interpolate the stress from Fenics uniform random sample points, then calculate the Gap score using these interpolated values

20220727 AULEGOUP
New function spaces (Sd,Vd,Td) for onmesh functions, to check Dr. Dokken idea.

20220728 CHTUNG
Changed the coordinates of the stress data points from mesh points to the centroid of each mesh cells during interpolation in onmesh_FITsample_*.

20220729 CHTUNG
Removed NaN value after interpolation.
"""
#%% IMPORTS____________________________________________________________________________________________________

import numpy as np
import matplotlib.pyplot as plt
from neorl import DE, XNES, GWO, ES
from anisotropyv3 import *
# from harmonicholes import *
from scipy.io import savemat
    
from fenics import *
# from dolfin_adjoint import *
import pygmsh_mesh_functions
from pygmsh_mesh_functions import *
import meshio
from tqdm import tqdm

from datetime import datetime
from scipy import interpolate

import os

#%% Constitutive function______________________________________________________________________________________
    
# Mechanical Properties for sample design evaluation
# (default is given for steel https://www.matweb.com/search/datasheet.aspx?bassnum=MS0001&ckck=1 )

E_float = 200*1e9                                     #Young modulus (Pa)        default 200*1e9
nu_float = 0.25                                       #Poisson ratio (no unit)   default 0.25
sigma_y0_float = 350*1e6                              #Yield stress (Pa)         default 350*1e6

E = Constant(E_float)
nu = Constant(nu_float)
sigma_y0 = Constant(sigma_y0_float)
mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)
lmbda = 2*mu*lmbda/(lmbda+2*mu)    

def epsilon(u):
    return sym(grad(u))

def sigma_el(epsilon):
    return lmbda*tr(epsilon)*Identity(2) + 2*mu*epsilon

def constitutive_function(sig_n, du, dt=1E-3):
    d_epsilon_el = epsilon(du) # elastic strain increment
    d_epsilon = d_epsilon_el # - d_epsilon_pl
    return sig_n + sigma_el(d_epsilon)#, dot_p

def mis_ep(u):
    eps_dev = epsilon(u) - Identity(2)*tr(epsilon(u))
    mis = sqrt(3/2*inner(eps_dev,eps_dev))
    return mis

def mis_sig(sig):
    sigma_dev = sig - Identity(2)*tr(sig)
    mis = sqrt(3/2*inner(sigma_dev,sigma_dev))
    return mis

#%% FITNESS____________________________________________________________________________________________________

N_iter=np.array([0,datetime.now()])

def FITsample(individual,meshfunction,lx,ly,N=N_iter,title="sample",details=False,
                constitutive_func=constitutive_function,num_stps=10,e_max=0.001,T_max=10,tol = 1e-14, N_seg=200,gapscore=GapScore):
    """Anisotropy score calculation.
            F(x) = formula based on FE calculation and point counting
            Range: [0.,1.]
            Minima: 0
    """
    N[0]+=1
    print("\nIteration {}\n".format(N[0]))
    print("Calculating fitness for sample : {}\n".format(individual))
    if N[0]==1:
        print(individual[0])
    
    now = datetime.now()
    filename="Score evaluation - "+now.strftime("%Y-%m-%d %Hh%Mmin%Ss") # dd/mm/YY H:M:S
    filepath="./resultstester/"+filename+" -4h - "+title
#     filepath+="/"
    filepath+=" - "
    outname = "output_"+filename+".xdmf"
    outdata = "ForcevsDisp_"+filename+".txt"
    
#     if (N[0]%20)==0:
#         clear_output(wait=True)
#         seconds=(datetime.now()-N[1]).total_seconds()
#         print("Total execution time at iteration {} since beginning \n   >>> {} seconds, or \n   >>> {} minutes, or \n   >>> {} hours".format(N[0],seconds,seconds/60,seconds/3600))

    
    sample_height=ly
    
    filename_unique = now.strftime("%Hh%Mmin%Ss.%f")
    mesh = meshfunction(individual,Lx=lx,Ly=ly,filename=filename_unique+".xdmf")  #<<<<<<<<<<<<<<<<<<<<<<<FUNCTION TO MODIFY FOR EACH MESH
#     with XDMFFile(str(meshpath)) as xdmf_infile:
#         xdmf_infile.read(mesh)
#     plot(mesh,wireframe = True,title="Mesh "+title)
    n_elements=len(mesh.coordinates())
    print("Nb of nodes : {}".format(n_elements))
    
    
    #Loading
    Tn = list(np.linspace(0,T_max,num_stps))
    Dn = list(np.linspace(0,e_max*sample_height,num_stps))
    
    #Loading
    Tn = list(np.linspace(0,T_max,num_stps))
    Dn = list(np.linspace(0,e_max*sample_height,num_stps))
    
    # Define function space for displacement
    S = FunctionSpace(mesh, 'P', 1)
    V = VectorFunctionSpace(mesh, 'P', 1)
    T = TensorFunctionSpace(mesh, 'P', 1)
    
    # Define test functions
    v  = TestFunction(V)
    du = TrialFunction(V)
    
    # Define functions for dsiplacement
    u     = Function(V)
    u_n   = Function(V)
    sig   = Function(T)
    sig_n = Function(T)
    ep    = Function(S)
    ep_n  = Function(S)

    # Define boundary 
    def BC_y0(x, on_boundary):
        return on_boundary and near(x[1], 0, tol)
    def BC_y1(x, on_boundary):
        return on_boundary and near(x[1], sample_height, tol)
    bcD = DirichletBC(V.sub(1), 0,    BC_y0)
#     def BC_corner(x, on_boundary):
#         return near(x[1], 0, tol) and near(x[0], 0, tol)
#     bc_corner = DirichletBC(V.sub(0), 0,    BC_corner)
    def point(x, on_boundary):
        return (abs(x[0] - 0) < tol) and (abs(x[1] -0) < tol)
    bc_corner = DirichletBC(V, Constant((0., 0.)), point,method="pointwise")
    
    
    # Time-stepping
    t = Tn[0]
    results = []
    
#     File_displacement = File(filepath+'displacement.pvd')
#     File_stress_Mis = File(filepath+'stress_Mis.pvd')
#     File_strain_Mis = File(filepath+'strain_Mis.pvd')
#     File_strain = File(filepath+'strain.pvd')
#     File_stess = File(filepath+'stess.pvd')
    
#     File_displacement = XDMFFile(filepath+'displacement.xdmf')
#     File_stress_Mis = XDMFFile(filepath+'stress_Mis.xdmf')
#     File_strain_Mis = XDMFFile(filepath+'strain_Mis.xdmf')
#     File_strain = XDMFFile(filepath+'strain.xdmf')
#     File_stess = XDMFFile(filepath+'stess.xdmf')
    
    array_u = np.zeros((n_elements,2,num_stps+1))
    array_sigma = np.zeros((n_elements,2,2,num_stps+1))
    array_epsilon = np.zeros((n_elements,2,2,num_stps+1))
    
    for n in tqdm(range(num_stps)):
        displacement = Dn[n]
        bcU = DirichletBC(V.sub(1), displacement, BC_y1)
        BC = [bcU, bcD, bc_corner]
        
        # Define variational problem
        if n==0:
            dt=0
        if n>0:
            dt=Tn[n]-Tn[n-1]
        sig = constitutive_func(sig_n, u-u_n, dt=dt)
        F = inner(sig, epsilon(v))*dx
        
        # Update current time
        t = Tn[n]
        
        # Solve variational problem for time step
        J = derivative(F, u, du)
        try:
            problem = NonlinearVariationalProblem(F, u, BC, J)
            solver = NonlinearVariationalSolver(problem)
            solver.solve()
        except RuntimeError:
            return 1.5
        
        # Update previous solution
        sig = project(sig, T)
        
        sig_n.assign(sig)
        u_n.assign(u)
        print('end')
        
        # Save solution to file in VTK format
        sig_Mis = project(mis_sig(sig),S)
        ep_Mis = project(mis_ep(u),S)
#         sig_Mis = mis_sig(sig)
#         ep_Mis = mis_ep(u)
        epsilon_u = epsilon(u)
        
        u.rename('$u$','displacement')
        sig.rename('$\sigma$','stress')
        ep.rename('$\epsilon$','strain')
        sig_Mis.rename('$\sigma_{VM}$','stress_Mis')
        ep_Mis.rename('$\epsilon_{VM}$','strain_Mis')
        
#         File_displacement << (u,t)
#         File_stress_Mis << (sig_Mis,t)
#         File_strain_Mis << (ep_Mis,t)
#         File_strain << (ep,t)
#         File_stess << (sig,t)
        
#         File_displacement.write(u,t)
#         File_stress_Mis.write(sig_Mis,t)
#         File_strain_Mis.write(ep_Mis,t)
#         File_strain.write(ep,t)
#         File_stess.write(sig,t)
        
        # arrange element vectors 
        array_u[:,:,n] = np.array(u.vector()).reshape(n_elements,2)
        
        epsilon_u = epsilon(u)
        epsilon_u = project(epsilon(u),T)
        array_epsilon[:,:,:,n] = np.array(epsilon_u.vector()).reshape(n_elements,2,2)
        
        array_sigma[:,:,:,n] = np.array(sig.vector()).reshape(n_elements,2,2)
        
    # visualize the final strain evolutions
#     plot(sig_Mis, mode='color')
#     plt.title("Equivalent Von Mises stress in the sample at step {}".format(n))
#     plt.savefig(filepath+'test_strain_MIS.png', dpi=96, bbox_inches='tight')
        
    # save element vectors in .mat format
#     from scipy.io import savemat
#     mdic = {'u':array_u, 'epsilon':array_epsilon, 'sigma':array_sigma}
#     savemat(filepath+'data.mat', mdic)
    
    
    RF = sig_Mis(0,0)
    if RF<1e-9:
        print("VM stress at corner = {}".format(RF))
        return 1.5
    
    # calculate anisotropy score
    sigma = array_sigma
    i_frame = 3
    n_element = sigma.shape[0]
    n_frame = sigma.shape[3]
    
    sigma_normal = np.zeros((n_element,n_frame))
    sigma_shear = np.zeros((n_element,n_frame))
    
    for it in tqdm(range(n_frame)):
        sigma_t = sigma[:,:,:,it].reshape(n_element,4)

        sigma_normal[:,it] = (sigma_t[:,0] - sigma_t[:,3])/2
        sigma_shear[:,it] = sigma_t[:,2]
        
    sigma_all = np.vstack((sigma_normal.reshape(n_frame*n_element),sigma_shear.reshape(n_frame*n_element))).T.tolist()
    # sigma_annular = [x for x in sigma_all if x[0]**2+x[1]**2>threshold]
    sigma_i = np.vstack((sigma_normal[:,i_frame],sigma_shear[:,i_frame])).T.tolist()
    
    Score_stress_gap = gapscore(sigma_i)
    Score = Score_stress_gap
    
#     os.system('find -type f -name "*{}*" -delete'.format(filename_unique))
    try:
        os.remove("output_files/"+filename_unique+'.xdmf')
        os.remove("output_files/"+filename_unique+'.h5')
        os.remove("output_files/"+'_pre_'+filename_unique+'.xdmf.vtu')
    except:
        pass
    
    
    if details:
        #maximum, 90th percentile, median, mean and standard deviation
        sigma_i=np.array(sigma_i)
        RAD=np.sqrt(sigma_i[:,0]**2+sigma_i[:,1]**2)
        det=[np.max(RAD) , np.percentile(RAD,90) , np.median(RAD) , np.mean(RAD) , np.std(RAD),n_elements]
        return Score,det
    
    return Score

#%% Interpolate the stress from Fenics to a 50*50 mesh grid, then calculate the Gap score using these interpolated values
def FITsample_grid(individual,meshfunction,lx,ly,N=N_iter,title="sample",details=False,
                constitutive_func=constitutive_function,num_stps=10,e_max=0.001,T_max=10,tol = 1e-14, N_seg=200,gapscore=GapScore):
    """Anisotropy score calculation.
            F(x) = formula based on FE calculation and point counting
            Range: [0.,1.]
            Minima: 0
    """
    N[0]+=1
    print("\nIteration {}\n".format(N[0]))
    print("Calculating fitness for sample : {}\n".format(individual))
    if N[0]==1:
        print(individual[0])
    
    now = datetime.now()
    filename="Score evaluation - "+now.strftime("%Y-%m-%d %Hh%Mmin%Ss") # dd/mm/YY H:M:S
    filepath="./resultstester/"+filename+" -4h - "+title
#     filepath+="/"
    filepath+=" - "
    outname = "output_"+filename+".xdmf"
    outdata = "ForcevsDisp_"+filename+".txt"
    
#     if (N[0]%20)==0:
#         clear_output(wait=True)
#         seconds=(datetime.now()-N[1]).total_seconds()
#         print("Total execution time at iteration {} since beginning \n   >>> {} seconds, or \n   >>> {} minutes, or \n   >>> {} hours".format(N[0],seconds,seconds/60,seconds/3600))

    
    sample_height=ly
    
    filename_unique = now.strftime("%Hh%Mmin%Ss.%f")
    mesh = meshfunction(individual,Lx=lx,Ly=ly,filename=filename_unique+".xdmf")  #<<<<<<<<<<<<<<<<<<<<<<<FUNCTION TO MODIFY FOR EACH MESH
#     with XDMFFile(str(meshpath)) as xdmf_infile:
#         xdmf_infile.read(mesh)
#     plot(mesh,wireframe = True,title="Mesh "+title)
    n_elements=len(mesh.coordinates())
    print("Nb of nodes : {}".format(n_elements))
    
    
    #Loading
    Tn = list(np.linspace(0,T_max,num_stps))
    Dn = list(np.linspace(0,e_max*sample_height,num_stps))
    
    #Loading
    Tn = list(np.linspace(0,T_max,num_stps))
    Dn = list(np.linspace(0,e_max*sample_height,num_stps))
    
    # Define function space for displacement
    S = FunctionSpace(mesh, 'P', 1)
    V = VectorFunctionSpace(mesh, 'P', 1)
    T = TensorFunctionSpace(mesh, 'P', 1)
    
    # Define test functions
    v  = TestFunction(V)
    du = TrialFunction(V)
    
    # Define functions for dsiplacement
    u     = Function(V)
    u_n   = Function(V)
    sig   = Function(T)
    sig_n = Function(T)
    ep    = Function(S)
    ep_n  = Function(S)

    # Define boundary 
    def BC_y0(x, on_boundary):
        return on_boundary and near(x[1], 0, tol)
    def BC_y1(x, on_boundary):
        return on_boundary and near(x[1], sample_height, tol)
    bcD = DirichletBC(V.sub(1), 0,    BC_y0)
#     def BC_corner(x, on_boundary):
#         return near(x[1], 0, tol) and near(x[0], 0, tol)
#     bc_corner = DirichletBC(V.sub(0), 0,    BC_corner)
    def point(x, on_boundary):
        return (abs(x[0] - 0) < tol) and (abs(x[1] -0) < tol)
    bc_corner = DirichletBC(V, Constant((0., 0.)), point,method="pointwise")
    
    
    # Time-stepping
    t = Tn[0]
    results = []
    
#     File_displacement = File(filepath+'displacement.pvd')
#     File_stress_Mis = File(filepath+'stress_Mis.pvd')
#     File_strain_Mis = File(filepath+'strain_Mis.pvd')
#     File_strain = File(filepath+'strain.pvd')
#     File_stess = File(filepath+'stess.pvd')
    
#     File_displacement = XDMFFile(filepath+'displacement.xdmf')
#     File_stress_Mis = XDMFFile(filepath+'stress_Mis.xdmf')
#     File_strain_Mis = XDMFFile(filepath+'strain_Mis.xdmf')
#     File_strain = XDMFFile(filepath+'strain.xdmf')
#     File_stess = XDMFFile(filepath+'stess.xdmf')
    
    array_u = np.zeros((n_elements,2,num_stps+1))
    array_sigma = np.zeros((n_elements,2,2,num_stps+1))
    array_epsilon = np.zeros((n_elements,2,2,num_stps+1))
    
    for n in tqdm(range(num_stps)):
        displacement = Dn[n]
        bcU = DirichletBC(V.sub(1), displacement, BC_y1)
        BC = [bcU, bcD, bc_corner]
        
        # Define variational problem
        if n==0:
            dt=0
        if n>0:
            dt=Tn[n]-Tn[n-1]
        sig = constitutive_func(sig_n, u-u_n, dt=dt)
        F = inner(sig, epsilon(v))*dx
        
        # Update current time
        t = Tn[n]
        
        # Solve variational problem for time step
        J = derivative(F, u, du)
        try:
            problem = NonlinearVariationalProblem(F, u, BC, J)
            solver = NonlinearVariationalSolver(problem)
            solver.solve()
        except RuntimeError:
            return 1.5
        
        # Update previous solution
        sig = project(sig, T)
        
        sig_n.assign(sig)
        u_n.assign(u)
        print('end')
        
        # Save solution to file in VTK format
        sig_Mis = project(mis_sig(sig),S)
        ep_Mis = project(mis_ep(u),S)
#         sig_Mis = mis_sig(sig)
#         ep_Mis = mis_ep(u)
        epsilon_u = epsilon(u)
        
        u.rename('$u$','displacement')
        sig.rename('$\sigma$','stress')
        ep.rename('$\epsilon$','strain')
        sig_Mis.rename('$\sigma_{VM}$','stress_Mis')
        ep_Mis.rename('$\epsilon_{VM}$','strain_Mis')
        
#         File_displacement << (u,t)
#         File_stress_Mis << (sig_Mis,t)
#         File_strain_Mis << (ep_Mis,t)
#         File_strain << (ep,t)
#         File_stess << (sig,t)
        
#         File_displacement.write(u,t)
#         File_stress_Mis.write(sig_Mis,t)
#         File_strain_Mis.write(ep_Mis,t)
#         File_strain.write(ep,t)
#         File_stess.write(sig,t)
        
        # arrange element vectors 
        array_u[:,:,n] = np.array(u.vector()).reshape(n_elements,2)
        
        epsilon_u = epsilon(u)
        epsilon_u = project(epsilon(u),T)
        array_epsilon[:,:,:,n] = np.array(epsilon_u.vector()).reshape(n_elements,2,2)
        
        array_sigma[:,:,:,n] = np.array(sig.vector()).reshape(n_elements,2,2)
        
    # visualize the final strain evolutions
#     plot(sig_Mis, mode='color')
#     plt.title("Equivalent Von Mises stress in the sample at step {}".format(n))
#     plt.savefig(filepath+'test_strain_MIS.png', dpi=96, bbox_inches='tight')
        
    # save element vectors in .mat format
#     from scipy.io import savemat
#     mdic = {'u':array_u, 'epsilon':array_epsilon, 'sigma':array_sigma}
#     savemat(filepath+'data.mat', mdic)
    
    
    RF = sig_Mis(0,0)
    if RF<1e-9:
        print("VM stress at corner = {}".format(RF))
        return 1.5
    
    # calculate anisotropy score
    mesh_coord = mesh.coordinates()
    
    sigma = array_sigma
    i_frame = 3
    n_element = sigma.shape[0]
    n_frame = sigma.shape[3]
    
    sigma_test = sigma[:,:,:,i_frame]

    # meshgrid
    n_x = 50
    n_y = 50
    xc = np.arange(n_x+1)/n_x*lx
    yc = np.arange(n_y+1)/n_y*ly
    xx, yy = np.meshgrid(xc,yc)
    N_xy = np.zeros_like(xx,dtype=bool)

    # remove data from hole
    x_grid = xx.reshape((n_x+1)*(n_y+1))
    x_grid = yy.reshape((n_x+1)*(n_y+1))
    for i in range(n_x+1):
        for j in range(n_y+1):
            contains = [cell.contains(Point(xc[i],yc[j])) for cell in cells(mesh)]
            N_xy[i,j] = any(contains)

    list_matrix_index = [[0,0],[0,1],[1,0],[1,1]]

    # interpolation
    sigma_interp = np.zeros((n_x+1,n_y+1,2,2))
    for i, matrix_index in enumerate(list_matrix_index):
    #     f = interpolate.interp2d(mesh_coord[:,0],mesh_coord[:,1],sigma_test[:,matrix_index[0],matrix_index[1]])
    #     sigma_interp[:,:,matrix_index[0],matrix_index[1]] = f(xc, yc)
        sigma_interp[:,:,matrix_index[0],matrix_index[1]] = interpolate.griddata(mesh_coord,sigma_test[:,matrix_index[0],matrix_index[1]],(xx, yy),method='linear')

    sigma_interp_inside = sigma_interp[N_xy.T,:,:]
    
    sigma_normal = (sigma_interp_inside[:,0,0] - sigma_interp_inside[:,1,1])/2
    sigma_shear = sigma_interp_inside[:,0,1]

    sigma_all = np.vstack((sigma_normal,sigma_shear)).T.tolist()
    sigma_i = np.vstack((sigma_normal,sigma_shear)).T.tolist()
    
    Score_stress_gap = gapscore(sigma_i)
    Score = Score_stress_gap
    
#     os.system('find -type f -name "*{}*" -delete'.format(filename_unique))
    try:
        os.remove("output_files/"+filename_unique+'.xdmf')
        os.remove("output_files/"+filename_unique+'.h5')
        os.remove("output_files/"+'_pre_'+filename_unique+'.xdmf.vtu')
    except:
        pass

    if details:
        #maximum, 90th percentile, median, mean and standard deviation
        sigma_i=np.array(sigma_i)
        RAD=np.sqrt(sigma_i[:,0]**2+sigma_i[:,1]**2)
        det=[np.max(RAD) , np.percentile(RAD,90) , np.median(RAD) , np.mean(RAD) , np.std(RAD),n_elements]
        return Score,det
    
    return Score

#%% Interpolate the stress from Fenics to uniform random sample points, then calculate the Gap score using these interpolated values
def FITsample_randsamp(individual,meshfunction,lx,ly,N=N_iter,title="sample",details=False,
                constitutive_func=constitutive_function,num_stps=10,e_max=0.001,T_max=10,tol = 1e-14, N_seg=200,gapscore=GapScore):
    """Anisotropy score calculation.
            F(x) = formula based on FE calculation and point counting
            Range: [0.,1.]
            Minima: 0
    """
    N[0]+=1
    print("\nIteration {}\n".format(N[0]))
    print("Calculating fitness for sample : {}\n".format(individual))
    if N[0]==1:
        print(individual[0])
    
    now = datetime.now()
    filename="Score evaluation - "+now.strftime("%Y-%m-%d %Hh%Mmin%Ss") # dd/mm/YY H:M:S
    filepath="./resultstester/"+filename+" -4h - "+title
#     filepath+="/"
    filepath+=" - "
    outname = "output_"+filename+".xdmf"
    outdata = "ForcevsDisp_"+filename+".txt"
    
#     if (N[0]%20)==0:
#         clear_output(wait=True)
#         seconds=(datetime.now()-N[1]).total_seconds()
#         print("Total execution time at iteration {} since beginning \n   >>> {} seconds, or \n   >>> {} minutes, or \n   >>> {} hours".format(N[0],seconds,seconds/60,seconds/3600))

    
    sample_height=ly
    
    filename_unique = now.strftime("%Hh%Mmin%Ss.%f")
    mesh = meshfunction(individual,Lx=lx,Ly=ly,filename=filename_unique+".xdmf")  #<<<<<<<<<<<<<<<<<<<<<<<FUNCTION TO MODIFY FOR EACH MESH
#     with XDMFFile(str(meshpath)) as xdmf_infile:
#         xdmf_infile.read(mesh)
#     plot(mesh,wireframe = True,title="Mesh "+title)
    n_elements=len(mesh.coordinates())
    print("Nb of nodes : {}".format(n_elements))
    
    
    #Loading
    Tn = list(np.linspace(0,T_max,num_stps))
    Dn = list(np.linspace(0,e_max*sample_height,num_stps))
    
    #Loading
    Tn = list(np.linspace(0,T_max,num_stps))
    Dn = list(np.linspace(0,e_max*sample_height,num_stps))
    
    # Define function space for displacement
    S = FunctionSpace(mesh, 'P', 1)
    V = VectorFunctionSpace(mesh, 'P', 1)
    T = TensorFunctionSpace(mesh, 'P', 1)
    
    # Define test functions
    v  = TestFunction(V)
    du = TrialFunction(V)
    
    # Define functions for dsiplacement
    u     = Function(V)
    u_n   = Function(V)
    sig   = Function(T)
    sig_n = Function(T)
    ep    = Function(S)
    ep_n  = Function(S)

    # Define boundary 
    def BC_y0(x, on_boundary):
        return on_boundary and near(x[1], 0, tol)
    def BC_y1(x, on_boundary):
        return on_boundary and near(x[1], sample_height, tol)
    bcD = DirichletBC(V.sub(1), 0,    BC_y0)
#     def BC_corner(x, on_boundary):
#         return near(x[1], 0, tol) and near(x[0], 0, tol)
#     bc_corner = DirichletBC(V.sub(0), 0,    BC_corner)
    def point(x, on_boundary):
        return (abs(x[0] - 0) < tol) and (abs(x[1] -0) < tol)
    bc_corner = DirichletBC(V, Constant((0., 0.)), point,method="pointwise")
    
    
    # Time-stepping
    t = Tn[0]
    results = []
    
#     File_displacement = File(filepath+'displacement.pvd')
#     File_stress_Mis = File(filepath+'stress_Mis.pvd')
#     File_strain_Mis = File(filepath+'strain_Mis.pvd')
#     File_strain = File(filepath+'strain.pvd')
#     File_stess = File(filepath+'stess.pvd')
    
#     File_displacement = XDMFFile(filepath+'displacement.xdmf')
#     File_stress_Mis = XDMFFile(filepath+'stress_Mis.xdmf')
#     File_strain_Mis = XDMFFile(filepath+'strain_Mis.xdmf')
#     File_strain = XDMFFile(filepath+'strain.xdmf')
#     File_stess = XDMFFile(filepath+'stess.xdmf')
    
    array_u = np.zeros((n_elements,2,num_stps+1))
    array_sigma = np.zeros((n_elements,2,2,num_stps+1))
    array_epsilon = np.zeros((n_elements,2,2,num_stps+1))
    
    for n in tqdm(range(num_stps)):
        displacement = Dn[n]
        bcU = DirichletBC(V.sub(1), displacement, BC_y1)
        BC = [bcU, bcD, bc_corner]
        
        # Define variational problem
        if n==0:
            dt=0
        if n>0:
            dt=Tn[n]-Tn[n-1]
        sig = constitutive_func(sig_n, u-u_n, dt=dt)
        F = inner(sig, epsilon(v))*dx
        
        # Update current time
        t = Tn[n]
        
        # Solve variational problem for time step
        J = derivative(F, u, du)
        try:
            problem = NonlinearVariationalProblem(F, u, BC, J)
            solver = NonlinearVariationalSolver(problem)
            solver.solve()
        except RuntimeError:
            return 1.5
        
        # Update previous solution
        sig = project(sig, T)
        
        sig_n.assign(sig)
        u_n.assign(u)
        print('end')
        
        # Save solution to file in VTK format
        sig_Mis = project(mis_sig(sig),S)
        ep_Mis = project(mis_ep(u),S)
#         sig_Mis = mis_sig(sig)
#         ep_Mis = mis_ep(u)
        epsilon_u = epsilon(u)
        
        u.rename('$u$','displacement')
        sig.rename('$\sigma$','stress')
        ep.rename('$\epsilon$','strain')
        sig_Mis.rename('$\sigma_{VM}$','stress_Mis')
        ep_Mis.rename('$\epsilon_{VM}$','strain_Mis')
        
#         File_displacement << (u,t)
#         File_stress_Mis << (sig_Mis,t)
#         File_strain_Mis << (ep_Mis,t)
#         File_strain << (ep,t)
#         File_stess << (sig,t)
        
#         File_displacement.write(u,t)
#         File_stress_Mis.write(sig_Mis,t)
#         File_strain_Mis.write(ep_Mis,t)
#         File_strain.write(ep,t)
#         File_stess.write(sig,t)
        
        # arrange element vectors 
        array_u[:,:,n] = np.array(u.vector()).reshape(n_elements,2)
        
        epsilon_u = epsilon(u)
        epsilon_u = project(epsilon(u),T)
        array_epsilon[:,:,:,n] = np.array(epsilon_u.vector()).reshape(n_elements,2,2)
        
        array_sigma[:,:,:,n] = np.array(sig.vector()).reshape(n_elements,2,2)
        
    # visualize the final strain evolutions
#     plot(sig_Mis, mode='color')
#     plt.title("Equivalent Von Mises stress in the sample at step {}".format(n))
#     plt.savefig(filepath+'test_strain_MIS.png', dpi=96, bbox_inches='tight')
        
    # save element vectors in .mat format
#     from scipy.io import savemat
#     mdic = {'u':array_u, 'epsilon':array_epsilon, 'sigma':array_sigma}
#     savemat(filepath+'data.mat', mdic)
    
    
    RF = sig_Mis(0,0)
    if RF<1e-9:
        print("VM stress at corner = {}".format(RF))
        return 1.5
    
    # calculate anisotropy score
    mesh_coord = mesh.coordinates()
    
    sigma = array_sigma
    i_frame = 3
    n_element = sigma.shape[0]
    n_frame = sigma.shape[3]
    
    sigma_test = sigma[:,:,:,i_frame]

    # random sampling
    n_samp = 2500
    np.random.seed(1)
    xc = np.random.rand(n_samp)*lx
    yc = np.random.rand(n_samp)*ly

    N_xy = np.zeros_like(xc,dtype=bool)

    # remove data from hole
    for i in range(n_samp):
        contains = [cell.contains(Point(xc[i],yc[i])) for cell in cells(mesh)]
        N_xy[i] = any(contains)

    list_matrix_index = [[0,0],[0,1],[1,0],[1,1]]

    # interpolation
    sigma_interp = np.zeros((n_samp,2,2))
    for i, matrix_index in enumerate(list_matrix_index):
        sigma_interp[:,matrix_index[0],matrix_index[1]] = interpolate.griddata(mesh_coord,sigma_test[:,matrix_index[0],matrix_index[1]],(xc, yc),
                                                                                 method='linear')

    sigma_interp_inside = sigma_interp[N_xy,:,:]
    n_element_inside = len(sigma_interp_inside)
    
    sigma_normal = (sigma_interp_inside[:,0,0] - sigma_interp_inside[:,1,1])/2
    sigma_shear = sigma_interp_inside[:,0,1]

    sigma_all = np.vstack((sigma_normal,sigma_shear)).T.tolist()
    sigma_i = np.vstack((sigma_normal,sigma_shear)).T.tolist()
    
    Score_stress_gap = gapscore(sigma_i)
    Score = Score_stress_gap
    
#     os.system('find -type f -name "*{}*" -delete'.format(filename_unique))
    try:
        os.remove("output_files/"+filename_unique+'.xdmf')
        os.remove("output_files/"+filename_unique+'.h5')
        os.remove("output_files/"+'_pre_'+filename_unique+'.xdmf.vtu')
    except:
        pass
    
    
    if details:
        #maximum, 90th percentile, median, mean and standard deviation #nb nodes
        sigma_i=np.array(sigma_i)
        RAD=np.sqrt(sigma_i[:,0]**2+sigma_i[:,1]**2)
        det=[np.max(RAD) , np.percentile(RAD,90) , np.median(RAD) , np.mean(RAD) , np.std(RAD),n_elements]
        return Score,det
    
    return Score

#%% FITNESS on a mesh____________________________________________________________________________________________________

N_iter=np.array([0,datetime.now()])

def onmesh_FITsample(MESHFILE,lx,ly,N=N_iter,title="sample",onlystresspoints=False,details=False,
                constitutive_func=constitutive_function,num_stps=10,e_max=0.001,T_max=10,tol = 1e-14, N_seg=200,gapscore=GapScore):
    """Anisotropy score calculation.
            F(x) = formula based on FE calculation and point counting
            Range: [0.,1.]
            Minima: 0
    """
    N[0]+=1
    print("\nIteration {}\n".format(N[0]))
#     print("Calculating fitness for sample : {}\n".format(individual))
#     if N[0]==1:
#         print(individual[0])
    
    now = datetime.now()
    filename="Score evaluation - "+now.strftime("%Y-%m-%d %Hh%Mmin%Ss") # dd/mm/YY H:M:S
    filepath="./resultstester/"+filename+" -4h - "+title
#     filepath+="/"
    filepath+=" - "
    outname = "output_"+filename+".xdmf"
    outdata = "ForcevsDisp_"+filename+".txt"
    
#     if (N[0]%20)==0:
#         clear_output(wait=True)
#         seconds=(datetime.now()-N[1]).total_seconds()
#         print("Total execution time at iteration {} since beginning \n   >>> {} seconds, or \n   >>> {} minutes, or \n   >>> {} hours".format(N[0],seconds,seconds/60,seconds/3600))

    
    sample_height = ly
    
    mesh = Mesh()
    with XDMFFile(str(MESHFILE)) as xdmf_infile:
        xdmf_infile.read(mesh)
#     plot(mesh,wireframe = True,title="Mesh "+title)
    n_elements=len(mesh.coordinates())
    n_cells = len(mesh.cells())
    print("Nb of nodes : {}".format(n_elements))
    print("Nb of cells : {}".format(n_cells))
    
    
    #Loading
    Tn = list(np.linspace(0,T_max,num_stps))
    Dn = list(np.linspace(0,e_max*sample_height,num_stps))
    
    # Define function space for displacement
    S = FunctionSpace(mesh, 'P', 1)
    V = VectorFunctionSpace(mesh, 'P', 1)
    T = TensorFunctionSpace(mesh, 'P', 1)
    #Define function space for gradients
    Sd = FunctionSpace(mesh, 'DG', 1)
    Vd = VectorFunctionSpace(mesh, 'DG', 1)
    Td = TensorFunctionSpace(mesh, 'DG', 1)
    
    # Define test functions
    v  = TestFunction(V)
    du = TrialFunction(V)
    
    # Define functions for dsiplacement
    u     = Function(V)
    u_n   = Function(V)
    sig   = Function(Td)
    sig_n = Function(Td)
    ep    = Function(Sd)
    ep_n  = Function(Sd)

    # Define boundary 
    def BC_y0(x, on_boundary):
        return on_boundary and near(x[1], 0, tol)
    def BC_y1(x, on_boundary):
        return on_boundary and near(x[1], sample_height, tol)
    bcD = DirichletBC(V.sub(1), 0,    BC_y0)
#     def BC_corner(x, on_boundary):
#         return near(x[1], 0, tol) and near(x[0], 0, tol)
#     bc_corner = DirichletBC(V.sub(0), 0,    BC_corner)
    def point(x, on_boundary):
        return (abs(x[0] - 0) < tol) and (abs(x[1] -0) < tol)
    bc_corner = DirichletBC(V, Constant((0., 0.)), point,method="pointwise")
    
    
    # Time-stepping
    t = Tn[0]
    results = []
    
#     File_displacement = File(filepath+'displacement.pvd')
#     File_stress_Mis = File(filepath+'stress_Mis.pvd')
#     File_strain_Mis = File(filepath+'strain_Mis.pvd')
#     File_strain = File(filepath+'strain.pvd')
#     File_stess = File(filepath+'stess.pvd')
    
#     File_displacement = XDMFFile(filepath+'displacement.xdmf')
#     File_stress_Mis = XDMFFile(filepath+'stress_Mis.xdmf')
#     File_strain_Mis = XDMFFile(filepath+'strain_Mis.xdmf')
#     File_strain = XDMFFile(filepath+'strain.xdmf')
#     File_stess = XDMFFile(filepath+'stess.xdmf')
    
    array_u = np.zeros((n_elements,2,num_stps+1))
    array_sigma = np.zeros((n_elements,2,2,num_stps+1))
    array_epsilon = np.zeros((n_elements,2,2,num_stps+1))
    
    for n in tqdm(range(num_stps)):
        displacement = Dn[n]
        bcU = DirichletBC(V.sub(1), displacement, BC_y1)
        BC = [bcU, bcD, bc_corner]
        
        # Define variational problem
        if n==0:
            dt=0
        if n>0:
            dt=Tn[n]-Tn[n-1]
        sig = constitutive_func(sig_n, u-u_n, dt=dt)
        F = inner(sig, epsilon(v))*dx
        
        # Update current time
        t = Tn[n]
        
        # Solve variational problem for time step
        J = derivative(F, u, du)
        try:
            problem = NonlinearVariationalProblem(F, u, BC, J)
            solver = NonlinearVariationalSolver(problem)
            solver.solve()
        except RuntimeError:
            return 1.5
        
        # Update previous solution
        sig = project(sig, Td)
        
        sig_n.assign(sig)
        u_n.assign(u)
        print('end')
        
        # Save solution to file in VTK format
        sig_Mis = project(mis_sig(sig),Sd)
        ep_Mis = project(mis_ep(u),Sd)
#         sig_Mis = mis_sig(sig)
#         ep_Mis = mis_ep(u)
        epsilon_u = epsilon(u)
        
        u.rename('$u$','displacement')
        sig.rename('$\sigma$','stress')
        ep.rename('$\epsilon$','strain')
        sig_Mis.rename('$\sigma_{VM}$','stress_Mis')
        ep_Mis.rename('$\epsilon_{VM}$','strain_Mis')
        
#         File_displacement << (u,t)
#         File_stress_Mis << (sig_Mis,t)
#         File_strain_Mis << (ep_Mis,t)
#         File_strain << (ep,t)
#         File_stess << (sig,t)
        
#         File_displacement.write(u,t)
#         File_stress_Mis.write(sig_Mis,t)
#         File_strain_Mis.write(ep_Mis,t)
#         File_strain.write(ep,t)
#         File_stess.write(sig,t)
        
        # arrange element vectors 
        array_u[:,:,n] = np.array(u.vector()).reshape(n_elements,2)
        
        epsilon_u = epsilon(u)
        epsilon_u = project(epsilon(u),Td)
        array_epsilon[:,:,:,n] = np.array(epsilon_u.vector()).reshape(n_elements,2,2)
        
        array_sigma[:,:,:,n] = np.array(sig.vector()).reshape(n_elements,2,2)
        
    # visualize the final strain evolutions
#     plot(sig_Mis, mode='color')
#     plt.title("Equivalent Von Mises stress in the sample at step {}".format(n))
#     plt.savefig(filepath+'test_strain_MIS.png', dpi=96, bbox_inches='tight')
        
    # save element vectors in .mat format
#     from scipy.io import savemat
#     mdic = {'u':array_u, 'epsilon':array_epsilon, 'sigma':array_sigma}
#     savemat(filepath+'data.mat', mdic)
    
    
    RF = sig_Mis(0,0)
    if RF<1e-9:
        print("VM stress at corner = {}".format(RF))
        return 1.5
    
    # calculate anisotropy score
    sigma = array_sigma
    i_frame = 3
    n_element = sigma.shape[0]
    n_frame = sigma.shape[3]
    
    sigma_normal = np.zeros((n_element,n_frame))
    sigma_shear = np.zeros((n_element,n_frame))
    
    for it in tqdm(range(n_frame)):
        sigma_t = sigma[:,:,:,it].reshape(n_element,4)

        sigma_normal[:,it] = (sigma_t[:,0] - sigma_t[:,3])/2
        sigma_shear[:,it] = sigma_t[:,2]
        
    sigma_all = np.vstack((sigma_normal.reshape(n_frame*n_element),sigma_shear.reshape(n_frame*n_element))).T.tolist()
    # sigma_annular = [x for x in sigma_all if x[0]**2+x[1]**2>threshold]
    sigma_i = np.vstack((sigma_normal[:,i_frame],sigma_shear[:,i_frame])).T.tolist()
    
    if onlystresspoints:
        return sigma_i
    
    Score_stress_gap = gapscore(sigma_i)
    Score = Score_stress_gap
    
    
    if details:
        #maximum, 90th percentile, median, mean and standard deviation
        sigma_i=np.array(sigma_i)
        RAD=np.sqrt(sigma_i[:,0]**2+sigma_i[:,1]**2)
        det=[np.max(RAD) , np.percentile(RAD,90) , np.median(RAD) , np.mean(RAD) , np.std(RAD),n_elements]
        return Score,det
    
    return Score

#%% Interpolate the stress from Fenics to a 50*50 mesh grid, then calculate the Gap score using these interpolated values
def onmesh_FITsample_grid(MESHFILE,lx,ly,N=N_iter,title="sample",onlystresspoints=False,details=False,
                constitutive_func=constitutive_function,num_stps=10,e_max=0.001,T_max=10,tol = 1e-14, N_seg=200,
                          gapscore=GapScore,saving=False, newtensorspace=True):
    """Anisotropy score calculation.
            F(x) = formula based on FE calculation and point counting
            Range: [0.,1.]
            Minima: 0
    """
    N[0]+=1
    print("\nIteration {}\n".format(N[0]))
#     print("Calculating fitness for sample : {}\n".format(individual))
#     if N[0]==1:
#         print(individual[0])
    
    now = datetime.now()
    filename="Score evaluation - "+now.strftime("%Y-%m-%d %Hh%Mmin%Ss") # dd/mm/YY H:M:S
    filepath="./resultstester/"+filename+" -4h - "+title
#     filepath+="/"
    filepath+=" - "
    outname = "output_"+filename+".xdmf"
    outdata = "ForcevsDisp_"+filename+".txt"
    
#     if (N[0]%20)==0:
#         clear_output(wait=True)
#         seconds=(datetime.now()-N[1]).total_seconds()
#         print("Total execution time at iteration {} since beginning \n   >>> {} seconds, or \n   >>> {} minutes, or \n   >>> {} hours".format(N[0],seconds,seconds/60,seconds/3600))
    
    sample_height = ly
    
    mesh = Mesh()
    with XDMFFile(str(MESHFILE)) as xdmf_infile:
        xdmf_infile.read(mesh)
#     plot(mesh,wireframe = True,title="Mesh "+title)
    n_elements=len(mesh.coordinates())
    n_cells = len(mesh.cells())
    print("Nb of nodes : {}".format(n_elements))
    print("Nb of cells : {}".format(n_cells))
    
    
    #Loading
    Tn = list(np.linspace(0,T_max,num_stps))
    Dn = list(np.linspace(0,e_max*sample_height,num_stps))
    
    # Define function space for displacement
    S = FunctionSpace(mesh, 'P', 1)
    V = VectorFunctionSpace(mesh, 'P', 1)
    T = TensorFunctionSpace(mesh, 'P', 1)
    #Define function space for gradients
    Sd = FunctionSpace(mesh, 'DG', 1)
    Vd = VectorFunctionSpace(mesh, 'DG', 1)
    Td = TensorFunctionSpace(mesh, 'DG', 1)
    
    # Define test functions
    v  = TestFunction(V)
    du = TrialFunction(V)
    
    # Define functions for dsiplacement
    u     = Function(V)
    u_n   = Function(V)
    sig   = Function(Td)
    sig_n = Function(Td)
    ep    = Function(Sd)
    ep_n  = Function(Sd)

    # Define boundary 
    def BC_y0(x, on_boundary):
        return on_boundary and near(x[1], 0, tol)
    def BC_y1(x, on_boundary):
        return on_boundary and near(x[1], sample_height, tol)
    bcD = DirichletBC(V.sub(1), 0,    BC_y0)
#     def BC_corner(x, on_boundary):
#         return near(x[1], 0, tol) and near(x[0], 0, tol)
#     bc_corner = DirichletBC(V.sub(0), 0,    BC_corner)
    def point(x, on_boundary):
        return (abs(x[0] - 0) < tol) and (abs(x[1] -0) < tol)
    bc_corner = DirichletBC(V, Constant((0., 0.)), point,method="pointwise")
    
    
    
    # Time-stepping
    t = Tn[0]
    results = []
    
    if saving:
#         File_displacement = File(filepath+'displacement.pvd')
#         File_stress_Mis = File(filepath+'stress_Mis.pvd')
#         File_strain_Mis = File(filepath+'strain_Mis.pvd')
#         File_strain = File(filepath+'strain.pvd')
#         File_stess = File(filepath+'stess.pvd')

        File_displacement = XDMFFile(filepath+'displacement.xdmf')
        File_stress_Mis = XDMFFile(filepath+'stress_Mis.xdmf')
        File_strain_Mis = XDMFFile(filepath+'strain_Mis.xdmf')
        File_strain = XDMFFile(filepath+'strain.xdmf')
        File_stess = XDMFFile(filepath+'stess.xdmf')
    
    array_u = np.zeros((n_elements,2,num_stps+1))
    array_sigma = np.zeros((n_cells*3,2,2,num_stps+1))
    array_epsilon = np.zeros((n_cells*3,2,2,num_stps+1))
    
    for n in tqdm(range(num_stps)):
        displacement = Dn[n]
        bcU = DirichletBC(V.sub(1), displacement, BC_y1)
        BC = [bcU, bcD, bc_corner]
        
        # Define variational problem
        if n==0:
            dt=0
        if n>0:
            dt=Tn[n]-Tn[n-1]
        sig = constitutive_func(sig_n, u-u_n, dt=dt)
        F = inner(sig, epsilon(v))*dx
        
        # Update current time
        t = Tn[n]
        
        # Solve variational problem for time step
        J = derivative(F, u, du)
        try:
            problem = NonlinearVariationalProblem(F, u, BC, J)
            solver = NonlinearVariationalSolver(problem)
            solver.solve()
        except RuntimeError:
            return 1.5
        
        # Update previous solution
        sig = project(sig, Td)
        
        sig_n.assign(sig)
        u_n.assign(u)
        print('end')
        
        # Save solution to file in VTK format
        sig_Mis = project(mis_sig(sig),Sd)
        ep_Mis = project(mis_ep(u),Sd)
#         sig_Mis = mis_sig(sig)
#         ep_Mis = mis_ep(u)
        epsilon_u = epsilon(u)
        
        u.rename('$u$','displacement')
        sig.rename('$\sigma$','stress')
        ep.rename('$\epsilon$','strain')
        sig_Mis.rename('$\sigma_{VM}$','stress_Mis')
        ep_Mis.rename('$\epsilon_{VM}$','strain_Mis')
        
        if saving:
#             File_displacement << (u,t)
#             File_stress_Mis << (sig_Mis,t)
#             File_strain_Mis << (ep_Mis,t)
#             File_strain << (ep,t)
#             File_stess << (sig,t)

            File_displacement.write(u,t)
            File_stress_Mis.write(sig_Mis,t)
            File_strain_Mis.write(ep_Mis,t)
            File_strain.write(ep,t)
            File_stess.write(sig,t)
        
        # arrange element vectors 
        array_u[:,:,n] = np.array(u.vector()).reshape(n_elements,2)
        
        epsilon_u = epsilon(u)
        epsilon_u = project(epsilon(u),Td)
        
        print("shape of epsilon_u: {}".format(np.array(epsilon_u.vector()).shape))
        
        array_epsilon[:,:,:,n] = np.array(epsilon_u.vector()).reshape(n_cells*3,2,2)
        
        array_sigma[:,:,:,n] = np.array(sig.vector()).reshape(n_cells*3,2,2)
        
    if saving:
        # visualize the final strain evolutions
        plot(sig_Mis, mode='color')
        plt.title("Equivalent Von Mises stress in the sample at step {}".format(n))
        plt.savefig(filepath+'test_strain_MIS.png', dpi=96, bbox_inches='tight')

        # save element vectors in .mat format
        from scipy.io import savemat
        mdic = {'u':array_u, 'epsilon':array_epsilon, 'sigma':array_sigma}
        savemat(filepath+'data.mat', mdic)
    
    RF = sig_Mis(0,0)
    if RF<1e-9:
        print("VM stress at corner = {}".format(RF))
        return 1.5
    
    # calculate anisotropy score
    mesh_coord = mesh.coordinates()
    mesh_coord_cell = mesh.coordinates()[mesh.cells()]
    mesh_cell_center = np.array([(x[0]+x[1]+x[2])/3 for x in mesh_coord_cell])
    
    sigma = array_sigma
    i_frame = 3
    n_element = sigma.shape[0]
    n_frame = sigma.shape[3]
    i_cell = np.arange(n_cells)*3
    
    sigma_test = sigma[i_cell,:,:,i_frame]
    i_finite = [np.array([np.isfinite(sigma_test[x,0,0]),
                         np.isfinite(sigma_test[x,1,1]),
                         np.isfinite(sigma_test[x,0,1])]).all() for x in range(len(i_cell))]

    # meshgrid
    n_x = 50
    n_y = 50
    xc = np.arange(n_x+1)/n_x*lx
    yc = np.arange(n_y+1)/n_y*ly
    xx, yy = np.meshgrid(xc,yc)
    N_xy = np.zeros_like(xx,dtype=bool)

    # remove data from hole
    x_grid = xx.reshape((n_x+1)*(n_y+1))
    x_grid = yy.reshape((n_x+1)*(n_y+1))
    for i in range(n_x+1):
        for j in range(n_y+1):
            contains = [cell.contains(Point(xc[i],yc[j])) for cell in cells(mesh)]
            N_xy[i,j] = any(contains)

    list_matrix_index = [[0,0],[0,1],[1,0],[1,1]]

    # interpolation
    sigma_interp = np.zeros((n_x+1,n_y+1,2,2))
    
    for i, matrix_index in enumerate(list_matrix_index):
    #     f = interpolate.interp2d(mesh_coord[:,0],mesh_coord[:,1],sigma_test[:,matrix_index[0],matrix_index[1]])
    #     sigma_interp[:,:,matrix_index[0],matrix_index[1]] = f(xc, yc)
        sigma_interp[:,:,matrix_index[0],matrix_index[1]] = interpolate.griddata(mesh_cell_center[i_finite],sigma_test[i_finite,matrix_index[0],matrix_index[1]],(xx, yy),method='linear')

    sigma_interp_inside = sigma_interp[N_xy.T,:,:]
    
    sigma_normal = (sigma_interp_inside[:,0,0] - sigma_interp_inside[:,1,1])/2
    sigma_shear = sigma_interp_inside[:,0,1]
    
    i_finite_interp = np.isfinite(sigma_shear)
    print('all data are finite after interpolation: {}'.format(i_finite_interp.all()))
    sigma_normal_finite = sigma_normal[i_finite_interp]
    sigma_shear_finite = sigma_shear[i_finite_interp]

    sigma_all = np.vstack((sigma_normal_finite,sigma_shear_finite)).T.tolist()
    sigma_i = np.vstack((sigma_normal_finite,sigma_shear_finite)).T.tolist()
    
    if onlystresspoints:
        return sigma_i
    
    Score_stress_gap = gapscore(sigma_i)
    Score = Score_stress_gap

    if details:
        #maximum, 90th percentile, median, mean and standard deviation
        sigma_i=np.array(sigma_i)
        RAD=np.sqrt(sigma_i[:,0]**2+sigma_i[:,1]**2)
        det=[np.max(RAD) , np.percentile(RAD,90) , np.median(RAD) , np.mean(RAD) , np.std(RAD),n_elements]
        return Score,det
    
    return Score

#%% Interpolate the stress from Fenics to uniform random sample points, then calculate the Gap score using these interpolated values
def onmesh_FITsample_randsamp(MESHFILE,lx,ly,N=N_iter,title="sample",onlystresspoints=False,details=False,
                constitutive_func=constitutive_function,num_stps=10,e_max=0.001,T_max=10,tol = 1e-14, N_seg=200,gapscore=GapScore,saving=False, newtensorspace=True):
    """Anisotropy score calculation.
            F(x) = formula based on FE calculation and point counting
            Range: [0.,1.]
            Minima: 0
    """
    N[0]+=1
    print("\nIteration {}\n".format(N[0]))
#     print("Calculating fitness for sample : {}\n".format(individual))
#     if N[0]==1:
#         print(individual[0])
    
    now = datetime.now()
    filename="Score evaluation - "+now.strftime("%Y-%m-%d %Hh%Mmin%Ss") # dd/mm/YY H:M:S
    filepath="./resultstester/"+filename+" -4h - "+title
#     filepath+="/"
    filepath+=" - "
    outname = "output_"+filename+".xdmf"
    outdata = "ForcevsDisp_"+filename+".txt"
    
#     if (N[0]%20)==0:
#         clear_output(wait=True)
#         seconds=(datetime.now()-N[1]).total_seconds()
#         print("Total execution time at iteration {} since beginning \n   >>> {} seconds, or \n   >>> {} minutes, or \n   >>> {} hours".format(N[0],seconds,seconds/60,seconds/3600))
    
    sample_height = ly
    
    mesh = Mesh()
    with XDMFFile(str(MESHFILE)) as xdmf_infile:
        xdmf_infile.read(mesh)
#     plot(mesh,wireframe = True,title="Mesh "+title)
    n_elements=len(mesh.coordinates())
    n_cells = len(mesh.cells())
    print("Nb of nodes : {}".format(n_elements))
    print("Nb of cells : {}".format(n_cells))
    
    #Loading
    Tn = list(np.linspace(0,T_max,num_stps))
    Dn = list(np.linspace(0,e_max*sample_height,num_stps))
    
    # Define function space for displacement
    S = FunctionSpace(mesh, 'P', 1)
    V = VectorFunctionSpace(mesh, 'P', 1)
    T = TensorFunctionSpace(mesh, 'P', 1)
    #Define function space for gradients
    Sd = FunctionSpace(mesh, 'DG', 1)
    Vd = VectorFunctionSpace(mesh, 'DG', 1)
    Td = TensorFunctionSpace(mesh, 'DG', 1)
    
    # Define test functions
    v  = TestFunction(V)
    du = TrialFunction(V)
    
    # Define functions for dsiplacement
    u     = Function(V)
    u_n   = Function(V)
    sig   = Function(Td)
    sig_n = Function(Td)
    ep    = Function(Sd)
    ep_n  = Function(Sd)

    # Define boundary 
    def BC_y0(x, on_boundary):
        return on_boundary and near(x[1], 0, tol)
    def BC_y1(x, on_boundary):
        return on_boundary and near(x[1], sample_height, tol)
    bcD = DirichletBC(V.sub(1), 0,    BC_y0)
#     def BC_corner(x, on_boundary):
#         return near(x[1], 0, tol) and near(x[0], 0, tol)
#     bc_corner = DirichletBC(V.sub(0), 0,    BC_corner)
    def point(x, on_boundary):
        return (abs(x[0] - 0) < tol) and (abs(x[1] -0) < tol)
    bc_corner = DirichletBC(V, Constant((0., 0.)), point,method="pointwise")
    
    
    # Time-stepping
    t = Tn[0]
    results = []
    
    if saving:
#         File_displacement = File(filepath+'displacement.pvd')
#         File_stress_Mis = File(filepath+'stress_Mis.pvd')
#         File_strain_Mis = File(filepath+'strain_Mis.pvd')
#         File_strain = File(filepath+'strain.pvd')
#         File_stess = File(filepath+'stess.pvd')

        File_displacement = XDMFFile(filepath+'displacement.xdmf')
        File_stress_Mis = XDMFFile(filepath+'stress_Mis.xdmf')
        File_strain_Mis = XDMFFile(filepath+'strain_Mis.xdmf')
        File_strain = XDMFFile(filepath+'strain.xdmf')
        File_stess = XDMFFile(filepath+'stess.xdmf')
    
    array_u = np.zeros((n_elements,2,num_stps+1))
    array_sigma = np.zeros((n_cells*3,2,2,num_stps+1))
    array_epsilon = np.zeros((n_cells*3,2,2,num_stps+1))
    
    for n in tqdm(range(num_stps)):
        displacement = Dn[n]
        bcU = DirichletBC(V.sub(1), displacement, BC_y1)
        BC = [bcU, bcD, bc_corner]
        
        # Define variational problem
        if n==0:
            dt=0
        if n>0:
            dt=Tn[n]-Tn[n-1]
        sig = constitutive_func(sig_n, u-u_n, dt=dt)
        F = inner(sig, epsilon(v))*dx
        
        # Update current time
        t = Tn[n]
        
        # Solve variational problem for time step
        J = derivative(F, u, du)
        try:
            problem = NonlinearVariationalProblem(F, u, BC, J)
            solver = NonlinearVariationalSolver(problem)
            solver.solve()
        except RuntimeError:
            return 1.5
        
        # Update previous solution
        sig = project(sig, Td)
        
        sig_n.assign(sig)
        u_n.assign(u)
        print('end')
        
        # Save solution to file in VTK format
        sig_Mis = project(mis_sig(sig),Sd)
        ep_Mis = project(mis_ep(u),Sd)
#         sig_Mis = mis_sig(sig)
#         ep_Mis = mis_ep(u)
        epsilon_u = epsilon(u)
        
        u.rename('$u$','displacement')
        sig.rename('$\sigma$','stress')
        ep.rename('$\epsilon$','strain')
        sig_Mis.rename('$\sigma_{VM}$','stress_Mis')
        ep_Mis.rename('$\epsilon_{VM}$','strain_Mis')
        
        if saving:
#             File_displacement << (u,t)
#             File_stress_Mis << (sig_Mis,t)
#             File_strain_Mis << (ep_Mis,t)
#             File_strain << (ep,t)
#             File_stess << (sig,t)

            File_displacement.write(u,t)
            File_stress_Mis.write(sig_Mis,t)
            File_strain_Mis.write(ep_Mis,t)
            File_strain.write(ep,t)
            File_stess.write(sig,t)
        
        # arrange element vectors 
        array_u[:,:,n] = np.array(u.vector()).reshape(n_elements,2)
        
        epsilon_u = epsilon(u)
        epsilon_u = project(epsilon(u),Td)
        
        print("shape of epsilon_u: {}".format(np.array(epsilon_u.vector()).shape))
        
        array_epsilon[:,:,:,n] = np.array(epsilon_u.vector()).reshape(n_cells*3,2,2)
        
        array_sigma[:,:,:,n] = np.array(sig.vector()).reshape(n_cells*3,2,2)
        
    # visualize the final strain evolutions
#     plot(sig_Mis, mode='color')
#     plt.title("Equivalent Von Mises stress in the sample at step {}".format(n))
#     plt.savefig(filepath+'test_strain_MIS.png', dpi=96, bbox_inches='tight')
        
    # save element vectors in .mat format
#     from scipy.io import savemat
#     mdic = {'u':array_u, 'epsilon':array_epsilon, 'sigma':array_sigma}
#     savemat(filepath+'data.mat', mdic)
    
    
    RF = sig_Mis(0,0)
    if RF<1e-9:
        print("VM stress at corner = {}".format(RF))
        return 1.5
    
    # calculate anisotropy score
    mesh_coord = mesh.coordinates()
    mesh_coord_cell = mesh.coordinates()[mesh.cells()]
    mesh_cell_center = np.array([(x[0]+x[1]+x[2])/3 for x in mesh_coord_cell])
    
    sigma = array_sigma
    i_frame = 3
    n_element = sigma.shape[0]
    n_frame = sigma.shape[3]
    i_cell = np.arange(n_cells)*3
    
    sigma_test = sigma[i_cell,:,:,i_frame]
    i_finite = [np.array([np.isfinite(sigma_test[x,0,0]),
                         np.isfinite(sigma_test[x,1,1]),
                         np.isfinite(sigma_test[x,0,1])]).all() for x in range(len(i_cell))]

    # random sampling
    n_samp = 2500
    np.random.seed(1)
    xc = np.random.rand(n_samp)*lx
    yc = np.random.rand(n_samp)*ly

    N_xy = np.zeros_like(xc,dtype=bool)

    # remove data from hole
    for i in range(n_samp):
        contains = [cell.contains(Point(xc[i],yc[i])) for cell in cells(mesh)]
        N_xy[i] = any(contains)

    list_matrix_index = [[0,0],[0,1],[1,0],[1,1]]

    # interpolation
    sigma_interp = np.zeros((n_samp,2,2))
    for i, matrix_index in enumerate(list_matrix_index):
        sigma_interp[:,matrix_index[0],matrix_index[1]] = interpolate.griddata(mesh_cell_center[i_finite],sigma_test[i_finite,matrix_index[0],matrix_index[1]],(xc, yc),
                                                                                 method='linear')

    sigma_interp_inside = sigma_interp[N_xy,:,:]
    n_element_inside = len(sigma_interp_inside)
    
    sigma_normal = (sigma_interp_inside[:,0,0] - sigma_interp_inside[:,1,1])/2
    sigma_shear = sigma_interp_inside[:,0,1]
    
    i_finite_interp = np.isfinite(sigma_shear)
    print('all data are finite after interpolation: {}'.format(i_finite_interp.all()))
    sigma_normal_finite = sigma_normal[i_finite_interp]
    sigma_shear_finite = sigma_shear[i_finite_interp]

    sigma_all = np.vstack((sigma_normal_finite,sigma_shear_finite)).T.tolist()
    sigma_i = np.vstack((sigma_normal_finite,sigma_shear_finite)).T.tolist()
    
    if onlystresspoints:
        return sigma_i
    
    Score_stress_gap = gapscore(sigma_i)
    Score = Score_stress_gap
    
    
    if details:
        #maximum, 90th percentile, median, mean and standard deviation #nb nodes
        sigma_i=np.array(sigma_i)
        RAD=np.sqrt(sigma_i[:,0]**2+sigma_i[:,1]**2)
        det=[np.max(RAD) , np.percentile(RAD,90) , np.median(RAD) , np.mean(RAD) , np.std(RAD),n_elements]
        return Score,det
    
    return Score

#%% Function to plot and save result sample____________________________________________________________________________

def savesample(individual,meshfunction,lx,ly,title="result-sample",SHOW=False,verbose=False,nodate=False):
    now = datetime.now()
    print("\nSaving sample : {}\n".format(individual))
    sample_name = now.strftime("%Y-%m-%d %Hh%Mmin%Ss") + title+".xdmf"  # dd/mm/YY H:M:S
    if nodate:
        sample_name = title+".xdmf"
#     out_dir = pathlib.Path("output_files")
    out_dir = pathlib.Path("optimized_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = meshfunction(individual,Lx=lx,Ly=ly,Verbose=verbose)
    cell_type = mesh.cell_name()
    with create_XDMFFile(out_dir / sample_name) as xf:
        xf.write(mesh)
    plot(mesh,wireframe = True,title=title)
    if SHOW:
        plt.show()
    plt.savefig('savedsamples/mesh_'+title+'.png', dpi=96, bbox_inches='tight')
    
def showsample(individual,meshfunction,lx,ly,title="SampleShown",verbose=False):
    if verbose:
        print("\nShowing sample : {}\n".format(individual))
    mesh = meshfunction(individual,Lx=lx,Ly=ly,Verbose=verbose)
    plot(mesh,wireframe = True,title=title)