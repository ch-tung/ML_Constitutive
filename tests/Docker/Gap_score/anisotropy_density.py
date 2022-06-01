import numpy as np

def anisotropy_score_density(Stress_points,n_segments=60):
    PTSCOUNT=np.zeros(n_segments)
    div=2*np.pi/n_segments
    for [x,y] in Stress_points:
        if x==0 and y==0:
            continue
        else :
            theta=np.arctan2(y,x)
        PTSCOUNT[int(theta//div)]+=1
    # X=[div*k for k in range(n_segments)]
    n_mean=np.mean(PTSCOUNT)
#     plt.plot(X,PTSCOUNT,marker='+')
#     plt.title("$n(\Theta)$")
#     plt.plot(X,[n_mean]*n_segments,color='r')
#     plt.show()
#     print(n_segments)
    Sigma=1/len(Stress_points)*np.sqrt((np.sum((PTSCOUNT-n_mean)**2))/(1-1/n_segments))
    return Sigma

def anisotropy_score_radius(Stress_points,n_segments=60):
    PTSCOUNT=np.zeros(n_segments)
    RADICOUNT=np.zeros(n_segments)
    div=2*np.pi/n_segments
    for [x,y] in Stress_points:
        if x==0 and y==0:
            continue
        else :
            theta=np.arctan2(y,x)
        PTSCOUNT[int(theta//div)]+=1
        RADICOUNT[int(theta//div)]+=np.sqrt(x**2+y**2)
    n_mean=np.mean(PTSCOUNT)
#     print(n_segments)
    
    MEANRADS=RADICOUNT/PTSCOUNT
    MEANRADS/=np.max(MEANRADS)
    r_mean=np.mean(MEANRADS)
    RadiusScore=np.sqrt((np.sum((MEANRADS-r_mean)**2))/(1-1/n_segments))
    
    return RadiusScore

def anisotropy_score_harmonics(Stress_points,n_segments=60):
    PTSCOUNT=np.zeros(n_segments)
    div=2*np.pi/n_segments
    for [x,y] in Stress_points:
        if x==0 and y==0:
            continue
        else :
            theta=np.arctan2(y,x)
        PTSCOUNT[int(theta//div)]+=1
    n_mean=np.mean(PTSCOUNT)
    
    rho = (PTSCOUNT-n_mean)/len(Stress_points)
    theta = np.arange(n_segments)*div
    f_cos = np.sum(np.cos(theta)*rho)
    f_sin = np.sum(np.sin(theta)*rho)
    
    score = f_cos**2 + f_sin**2
    return score