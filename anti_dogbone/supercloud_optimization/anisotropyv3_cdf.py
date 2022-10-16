import matplotlib.pyplot as plt
import numpy as np

def SortCut1(Stress_points,threshold=0.5):
    """
    Threshold on the number of points
    """
    A=np.array(Stress_points)
    n=len(Stress_points)
    A=np.flipud(A[(A[:,0]**2+A[:,1]**2).argsort()])
    return A[:int(threshold*n)]

def SortCut2(Stress_points,threshold=0.5):
    """
    Threshold on radius
    """
    A=np.array(Stress_points)
    RAD=np.sqrt(A[:,0]**2+A[:,1]**2)
    return A[RAD>=threshold*RAD.max()]

def SortCut3(Stress_points,threshold=0.5,nptsmax=10):
    """
    Threshold on radius
    """
    A=np.array(Stress_points)
    RAD=np.sqrt(A[:,0]**2+A[:,1]**2)
    RAD=np.sort(RAD)
    if nptsmax>len(RAD):
        m=np.mean(RAD)
    m=np.mean(RAD[-nptsmax:])
    return A[RAD>=threshold*m]

def SortCut4(Stress_points,threshold=0.5,percentile=0.9):
    """
    Threshold on radius of a certain percentile
    """
    A=np.array(Stress_points)
    RAD=np.sqrt(A[:,0]**2+A[:,1]**2)
    RAD2=np.sort(RAD)
    stressforTH=RAD2[int(percentile*len(RAD2))]
    return A[RAD>=threshold*stressforTH]

def SortCut5(Stress_points,quantileinf=0.5,strcoeffinf=1.,quantilesup=0.99,strcoeffsup=1.1):
    """
    Threshold on radius of a certain percentile
    """
    A=np.array(Stress_points)
    RAD=np.sqrt(A[:,0]**2+A[:,1]**2)
    qinf=np.quantile(RAD,quantileinf)
    qsup=np.quantile(RAD,quantilesup)
    coeffbool=np.array([qinf*strcoeffinf <= RAD[i] <= qsup*strcoeffsup for i in range(len(RAD))])
    return A[coeffbool]

def GapScore(Stress_points,threshold=0.5,tol=1e-14):
    A=SortCut2(Stress_points,threshold=threshold)
    THETAS=np.sort(np.arctan2(A[:,1],A[:,0]))
#     print(THETAS/np.pi)
    GAPS=(np.roll(THETAS,-1,axis=0)-THETAS)%(2*np.pi)
#     print(GAPS/np.pi)
    GAPS=GAPS[GAPS>tol]
#     print(GAPS/np.pi)
    Ngaps=len(GAPS)
    if Ngaps<=1:
        return 1
    return 1/(2*np.pi)*np.sqrt((np.sum((GAPS-np.mean(GAPS))**2))/(1-1/Ngaps))

def GapScore3(Stress_points,threshold=0.5,nptsmax=10,tol=1e-14):
    A=SortCut3(Stress_points,threshold=threshold,nptsmax=nptsmax)
    THETAS=np.sort(np.arctan2(A[:,1],A[:,0]))
#     print(THETAS/np.pi)
    GAPS=(np.roll(THETAS,-1,axis=0)-THETAS)%(2*np.pi)
#     print(GAPS/np.pi)
    GAPS=GAPS[GAPS>tol]
#     print(GAPS/np.pi)
    Ngaps=len(GAPS)
    if Ngaps<=1:
        return 1
    return 1/(2*np.pi)*np.sqrt((np.sum((GAPS-np.mean(GAPS))**2))/(1-1/Ngaps))

def GapScore4(Stress_points,threshold=0.5,percentile=0.9,tol=1e-14):
    A=SortCut4(Stress_points,threshold=threshold,percentile=percentile)
    THETAS=np.sort(np.arctan2(A[:,1],A[:,0]))
#     print(THETAS/np.pi)
    GAPS=(np.roll(THETAS,-1,axis=0)-THETAS)%(2*np.pi)
#     print(GAPS/np.pi)
    GAPS=GAPS[GAPS>tol]
#     print(GAPS/np.pi)
    Ngaps=len(GAPS)
    if Ngaps<=1:
        return 1
    return 1/(2*np.pi)*np.sqrt((np.sum((GAPS-np.mean(GAPS))**2))/(1-1/Ngaps))

def GapScore5(Stress_points,quantileinf=0.5,strcoeffinf=1.,quantilesup=0.99,strcoeffsup=1.1,tol=1e-14):
    A=SortCut5(Stress_points,quantileinf=quantileinf,strcoeffinf=strcoeffinf,quantilesup=quantilesup,strcoeffsup=strcoeffsup)
    THETAS=np.sort(np.arctan2(A[:,1],A[:,0]))
#     print(THETAS/np.pi)
    GAPS=(np.roll(THETAS,-1,axis=0)-THETAS)%(2*np.pi)
#     print(GAPS/np.pi)
    GAPS=GAPS[GAPS>tol]
#     print(GAPS/np.pi)
    Ngaps=len(GAPS)
    if Ngaps<=1:
        return 1
    return 1/(2*np.pi)*np.sqrt((np.sum((GAPS-np.mean(GAPS))**2))/(1-1/Ngaps))

def GapScore_cdf(Stress_points,quantileinf=0.5,strcoeffinf=1.,quantilesup=0.99,strcoeffsup=1.1,tol=1e-14):
    A=SortCut5(Stress_points,quantileinf=quantileinf,strcoeffinf=strcoeffinf,quantilesup=quantilesup,strcoeffsup=strcoeffsup)
    THETAS_0=np.sort(np.arctan2(A[:,1],A[:,0]))
    theta_bar = np.arctan2(np.mean(np.sin(THETAS_0)),np.mean(np.cos(THETAS_0)))
    THETAS = np.sort((THETAS_0-theta_bar+np.pi)%(2*np.pi))
    N = len(THETAS)
    N_bins = 100
    c_prime = (np.arange(N_bins)/N_bins)/N #(0<c_prime<1/N)
    
    def AREA(i,theta):
        r = np.abs(2*np.pi*(c_prime+i/N)-theta) #integrate the residue
        return np.mean(r)/N       
        
    AREAS = [AREA(i,THETAS[i]) for i in np.arange(N)]
    
    if N<1:
        return 1
    return np.sum(AREAS)/np.pi*2

#######################################################################

# from IPython.display import Audio
# sound_positive = './NOTIFICATION SOUNDS/mixkit-positive-notification-951.wav'
# musical_reveal = './NOTIFICATION SOUNDS/mixkit-musical-reveal-961.wav'
# from IPython.core.display import display
# def beep_pos():
#     display(Audio(sound_positive, autoplay=True))
# def beep_reveal():
#     display(Audio(musical_reveal, autoplay=True))

# from fenics import *
# from dolfin_adjoint import *
# import pygmsh_mesh_functions
# from pygmsh_mesh_functions import *
# import meshio
# # import numpy as np
# # import matplotlib.pyplot as plt
# from tqdm import tqdm

# # # For quality evaluation
# # from anisotropy import anisotropy_score

# #date&time
# from datetime import datetime

# # #(((I don't know if it is useful
# # %reload_ext autoreload
# # #)))

# def plotdensity(rho,Lx=1e-2,Ly=1e-2,nx=100,ny=100,wireframe=False):
#     #mesh
#     sample_height=Ly
#     mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx-1, ny-1,"right")
#     if wireframe:
#         plot(mesh,wireframe = True,title="RectangleMesh")
#         print("Nb of nodes : {}".format(len(mesh.coordinates())))
#         print("Mesh element caracteristic length. Horizontal : {} mm. Vertical : {} mm".format(Lx/nx*1e3,Ly/ny*1e3))
#     C = mesh.coordinates()
    
    
#     # Define function space for projection of density
#     S = FunctionSpace(mesh, 'P', 1)
#     density = Function(S)
#     #Projection
#     for i in range(S.dim()):
#         density.vector()[i] = rho[i]
    
#     plt.figure()
#     P=plot(density, mode='color',title="Density in sample (yellow=max,blue=min)", vmin=0, vmax=1)
#     plt.colorbar(P)
#     plt.show()
    

# # plotdensity(np.random.rand(110),nx=10,ny=11,wireframe=False)

# def mesh(Lx=1e-2,Ly=1e-2,nx=100,ny=100):
#     return RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx-1, ny-1,"right")

# def solve_plastic_system_and_score(rho=np.random.rand(10000),title='notitle',sound=False,saverunfiles=False,num_stps=15,e_max=0.01,T_max=40, loss_func=lambda n,x: 0,
#                          tol = 1.0E-14, N_seg=200,Lx=1e-2,Ly=1e-2,nx=100,ny=100,p_SIMP=3,mesh=None):
    
#     now = datetime.now()
#     filename="Score evaluation - "+now.strftime("%Y-%m-%d %Hh%Mmin%Ss") # dd/mm/YY H:M:S
#     filepath="./resultstester/"+filename+" -4h - "+title
# #     filepath+="/"
#     filepath+=" - "
#     outname = "output_"+filename+".xdmf"
#     outdata = "ForcevsDisp_"+filename+".txt"
    
#     #Create square mesh
#     sample_height=Ly
#     if mesh==None:
#         mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx-1, ny-1,"right")
#     plot(mesh,wireframe = True,title="RectangleMesh")
#     print("Nb of nodes : {}".format(len(mesh.coordinates())))
#     print("Mesh element caracteristic length. Horizontal : {} mm. Vertical : {} mm".format(Lx/nx*1e3,Ly/ny*1e3))
#     n_elements = mesh.coordinates().shape[0]
# #     plot(mesh)
#     C=mesh.coordinates()
#     plt.scatter(C[:,0],C[:,1])
#     plt.show()
    
#     #Loading
#     Tn = list(np.linspace(0,T_max,num_stps))
#     Dn = list(np.linspace(0,e_max*sample_height,num_stps))
    
#     # Define function space for displacement
#     S = FunctionSpace(mesh, 'P', 1)
#     V = VectorFunctionSpace(mesh, 'P', 1)
#     T = TensorFunctionSpace(mesh, 'P', 1)
    
#     # Define test functions
#     v  = TestFunction(V)
#     du = TrialFunction(V)
    
#     # Define functions for dsiplacement
#     u     = Function(V)
#     u_n   = Function(V)
#     sig   = Function(T)
#     sig_n = Function(T)
#     ep    = Function(S)
#     ep_n  = Function(S)
#     density = Function(S)
    
# #     Coords = S.tabulate_dof_coordinates()
# #     print(Coords)
    
#     if len(rho)!=nx*ny:
#         raise NameError('WrongDensityLength')
        
#     plotdensity(rho,Lx=Lx,Ly=Ly,nx=nx,ny=ny,wireframe=False)
    
#     for i in range(S.dim()):
#         density.vector()[i] = rho[i]#(Coords[i][0],Coords[i][1])
    
#     # Define boundary 
#     def BC_y0(x, on_boundary):
#         return on_boundary and near(x[1], 0, tol)
#     def BC_y1(x, on_boundary):
#         return on_boundary and near(x[1], sample_height, tol)
#     bcD = DirichletBC(V.sub(1), 0,    BC_y0)
#     def BC_corner(x, on_boundary):
#         return near(x[1], 0, tol) and near(x[0], 0, 2e-2)
#     bc_corner = DirichletBC(V.sub(0), 0,    BC_corner)
    
#     #Constitutive function______________________________________________________________________________________
    
#     # Mechanical Properties for sample design evaluation
#     # (default is given for steel https://www.matweb.com/search/datasheet.aspx?bassnum=MS0001&ckck=1 )

#     E_float = 200*1e9                                     #Young modulus (Pa)        default 200*1e9
#     nu_float = 0.25                                       #Poisson ratio (no unit)   default 0.25
#     sigma_y0_float = 350*1e6                              #Yield stress (Pa)         default 350*1e6
    
#     E_normal = Constant(E_float)
#     E_low = E_normal/1e6
#     E = E_low+density**p_SIMP*(E_normal-E_low)
#     nu = Constant(nu_float)
#     sigma_y0 = Constant(sigma_y0_float)
#     mu = E/2/(1+nu)
#     lmbda = E*nu/(1+nu)/(1-2*nu)
#     lmbda = 2*mu*lmbda/(lmbda+2*mu)    

#     def epsilon(u):
#         return sym(grad(u))

#     def sigma_el(epsilon):
#         return lmbda*tr(epsilon)*Identity(2) + 2*mu*epsilon

#     def constitutive_func(sig_n, du, dt=1E-3):
#         d_epsilon_el = epsilon(du) # elastic strain increment
#         d_epsilon = d_epsilon_el # - d_epsilon_pl
#         return sig_n + sigma_el(d_epsilon)#, dot_p
    
#     def mis_ep(u):
#         eps_dev = epsilon(u) - Identity(2)*tr(epsilon(u))
#         mis = sqrt(3/2*inner(eps_dev,eps_dev))
#         return mis

#     def mis_sig(sig):
#         sigma_dev = sig - Identity(2)*tr(sig)
#         mis = sqrt(3/2*inner(sigma_dev,sigma_dev))
#         return mis
#     #___________________________________________________________________________________________________________
    
#     # Time-stepping
#     t = Tn[0]
#     results = []
    
#     if saverunfiles:
#         File_displacement = File(filepath+'displacement_AG.pvd')
#         File_stress_Mis = File(filepath+'stress_Mis_AG.pvd')
#         File_strain_Mis = File(filepath+'strain_Mis_AG.pvd')
#         File_strain_AG = File(filepath+'strain_AG.pvd')
#         File_stess_AG = File(filepath+'stess_AG.pvd')

#     array_u = np.zeros((n_elements,2,num_stps+1))
#     array_sigma = np.zeros((n_elements,2,2,num_stps+1))
#     array_epsilon = np.zeros((n_elements,2,2,num_stps+1))
    
#     for n in tqdm(range(num_stps)):
# #         print(n)
#         displacement = Dn[n]
#         bcU = DirichletBC(V.sub(1), displacement, BC_y1)
#         BC = [bcU, bcD, bc_corner]
        
#         # Define variational problem
#         if n==0:
#             dt=0
#         if n>0:
#             dt=Tn[n]-Tn[n-1]
#         sig = constitutive_func(sig_n, u-u_n, dt=dt)
#         F = inner(sig, epsilon(v))*dx
        
#         # Update current time
#         t = Tn[n]
        
#         # Solve variational problem for time step
#         J = derivative(F, u, du)
#         problem = NonlinearVariationalProblem(F, u, BC, J)
#         solver = NonlinearVariationalSolver(problem)
#         solver.solve()
        
#         # Update previous solution
#         sig = project(sig, T)
        
#         sig_n.assign(sig)
#         u_n.assign(u)
# #         print('end')
        
#         # Save solution to file in VTK format
#         sig_Mis = project(mis_sig(sig),S)
#         ep_Mis = project(mis_ep(u),S)
# #         sig_Mis = mis_sig(sig)
# #         ep_Mis = mis_ep(u)
#         epsilon_u = epsilon(u)
        
#         u.rename('$u$','displacement')
#         sig.rename('$\sigma$','stress')
#         ep.rename('$\epsilon$','strain')
#         sig_Mis.rename('$\sigma_{VM}$','stress_Mis')
#         ep_Mis.rename('$\epsilon_{VM}$','strain_Mis')
        
#         if saverunfiles:
#             File_displacement << (u,t)
#             File_stress_Mis << (sig_Mis,t)
#             File_strain_Mis << (ep_Mis)
#             File_strain_AG << (ep,t)
#             File_stess_AG << (sig,t)
        
#         # Save element vectors in .mat format
#         from scipy.io import savemat
#         array_u[:,:,n] = np.array(u.vector()).reshape(n_elements,2)
        
#         epsilon_u = epsilon(u)
#         epsilon_u = project(epsilon(u),T)
#         array_epsilon[:,:,:,n] = np.array(epsilon_u.vector()).reshape(n_elements,2,2)
        
#         array_sigma[:,:,:,n] = np.array(sig.vector()).reshape(n_elements,2,2)
        
# #         plot(sig_Mis, mode='color')
# #         plt.title("Equivalent Von Mises stress in the sample at step {}".format(n))
# #         plt.show()
        
#     PTSSTRESS=[]
#     for n in range(n_elements):
#         for st in range(num_stps+1):
#             PTSSTRESS+=[[(array_sigma[n,0,0,st]-array_sigma[n,1,1,st])/2,array_sigma[n,0,1,st]]]
    
#     if saverunfiles:
#         mdic = {'u':array_u, 'epsilon':array_epsilon, 'sigma':array_sigma}
#         savemat(filepath+'data.mat', mdic)
    
#     if sound:
#         beep_pos()
    
# #     S = FunctionSpace(mesh, 'P', 2)
# #     sigma_Mis = project(mis_sig(sig),S)
# #     plot(sig_Mis, mode='color')
# #     plt.title("Equivalent Von Mises stress in the sample")
# #     plt.show()
    
#     PTSSTRESS = np.array(PTSSTRESS)
#     Score=anisotropy_score(PTSSTRESS,n_segments=N_seg)
    
#     with open(filepath+'Score.txt', 'w') as f:
#         f.write("Anisotropy score : {}\n\n".format(Score))
#         f.write("title   = {}\nnum_stps= {}\ne_max   = {}\nT_max   = {}\ntol     = {}\nN_seg   = {}\nLx      = {}\nLy      = {}\nnx      = {}\nny      = {}\np_SIMP  = {}\n".format(
#         title,num_stps,e_max,T_max,tol,N_seg,Lx,Ly,nx,ny,p_SIMP))
    
#     plt.plot(PTSSTRESS[:,0],PTSSTRESS[:,1],marker='o',markersize=0.3,linewidth=0)
#     plt.title("Stress paths : anisotropy score = "+str(Score)[0:7])
#     plt.savefig(filepath+"Stress paths and anisotropy score.png")
#     plt.show()
    
#     #_______________________________________________________________
#     del(mesh)
    
#     del(S)
#     del(V)
#     del(T)
    
#     del(v)
#     del(du)
    
#     del(u)
#     del(u_n)
#     del(sig)
#     del(sig_n)
#     del(ep)
#     del(ep_n)
#     del(density)
    
#     del(E_normal)
#     del(E_low)
#     del(E)
#     del(nu)
#     del(sigma_y0)
#     del(mu)
#     del(lmbda)
     
#     #_______________________________________________________________
    
#     return Score

# # Title='random sqrt distrib density'
# # Score = solve_plastic_system_and_score(title=Title)
# # print("The anisotropy score for '{}' is : {}".format(Title,Score))

