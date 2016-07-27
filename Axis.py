import numpy as np
import numpy as np
import astropy.constants as const


class Axis(object):
    '''A generic parent class - set up the disk on a chosen 2d grid (can later make 3d is needed .. )
    Specify either the points (2 - 1D r and phi arrays) or bounds on r and phi + num_points or deltas
    - The output of this class is 2d arrays of r & phi using meshgrid ...
    '''
    def __init__(self, points_r=None,points_phi=None, bounds_r=None,bounds_phi=None, \
                 num_points=10,delta_r=None,delta_phi=None,axis_type='TwoD_cyl'):
        self.axis_type = axis_type
        defaultUnits = {'r': 'AU',
                        'phi': 'degree'}
        if points_r is not None:
            try:
                points_r = np.atleast_1d(np.array(points_r, dtype=float))
                points_phi = np.atleast_1d(np.array(points_phi, dtype=float))
            except:
                raise ValueError('points must be array_like.')
        if bounds_r is not None:
            try:
                bounds_r = np.sort(np.atleast_1d(np.array(bounds_r, dtype=float)))
                bounds_phi = np.sort(np.atleast_1d(np.array(bounds_phi, dtype=float)))
            except:
                raise ValueError('bounds must be array_like.')

        if bounds_r is None:
            if points_r is not None:
                # only points are given - so use the default bounds in addition to the points.
                bounds_r = np.array([points_r.min(),points_r.max()])
                bounds_phi = np.array([points_phi.min(),points_phi.max()])
            else:
                raise ValueError('Specify either bounds or points')
        else:  # bounds are given
            if points_r is None:
                # create an evenly spaced axis
                if delta_r is not None :
                    end0 = np.min(bounds_r)
                    end1 = np.max(bounds_r)
                    points_r = np.arange(end0,end1+delta_r/2.,delta_r)
                    end0 = np.min(bounds_phi)
                    end1 = np.max(bounds_phi)
                    points_phi = np.arange(end0,end1+delta_phi/2.,delta_phi)
                else :
                    end0 = np.min(bounds_r)
                    end1 = np.max(bounds_r)
                    delta = (end1 - end0) / num_points
                    points_r = np.linspace(end0, end1, num_points)
                    end0 = np.min(bounds_phi)
                    end1 = np.max(bounds_phi)
                    delta = (end1 - end0) / num_points
                    points_phi = np.linspace(end0, end1, num_points)
            else:
                # points and bounds both given, over-write the bounds ...
                print('Overwriting the bounds to match the points')
                bounds_r = np.array([points_r.min(),points_r.max()])
                bounds_phi = np.array([points_phi.min(),points_phi.max()])

        X, Y = np.meshgrid(points_r, points_phi)
        self.num_points = X.size
        self.shape = np.array([points_r.size,points_phi.size])
        self.units = defaultUnits
        self.points_r = X
        self.bounds_r = bounds_r
        self.points_phi = Y
        self.bounds_phi = bounds_phi
        area1 = np.pi*(bounds_r[1]**2. - bounds_r[0]**2.)
        area2 = (bounds_phi[1] - bounds_phi[0])/2./np.pi
        tmp1  = X**2.
        tmp2 = np.diff(tmp1,axis=1)
        tmp3 = np.diff(Y,axis=0)/2./np.pi
        self.area_all = area1*area2
        self.area_grid = np.pi*tmp2[:-1,:]*tmp3[:,:-1]
        self.num_grid_pt = self.area_grid.flatten().shape[0]

## Some code to get general disk properties ..

# mmsn - sigma1 (g/cm^2) and T1 (Kelvin) are density and temp at 1 AU. r_disk in AU
def mmsn_model(R_disk,alphav,Sigma1 =1700.,T1 = 270.):
    sigma_r = Sigma1*R_disk**(-1.5)
    t_gas_r = T1*R_disk**(-.5)
    tmp1 = R_disk*const.au.cgs.value
    tmp2 = const.G.cgs.value*const.M_sun.cgs.value
    mean_mol_weigth = 2.34*const.m_p.cgs.value
    omega = np.sqrt(tmp2/(tmp1)**3.) # orbital angular velocity
    c_s = np.sqrt(const.k_B.cgs.value*t_gas_r/mean_mol_weigth) # gas sound speed, cm
    H_disk = c_s/omega
    rhog = (1./np.sqrt(2.*np.pi))*sigma_r/H_disk           # Gas Volume Density
    t_L,L_Scale,vel_L,t_kol,l_kol,v_kol,Re = turb_param(c_s,H_disk,alphav,rhog,omega)
    lmfp = mean_mol_weigth/np.sqrt(2.)/2e-15/rhog    # cm
    vk = R_disk*omega*const.au.cgs.value
    return sigma_r,t_gas_r,c_s,t_L,L_Scale,vel_L, t_kol,l_kol,v_kol,Re,lmfp,vk,H_disk

def turb_param(c_s,H_disk,alphav,rhog,omega):
    t_L = 1./omega
    L_Scale = np.sqrt(alphav)*H_disk # equate L^2/t_overturn ~ L^2*(omega) ~ alphav*c_s*H_disk
    vel_L  = np.sqrt(alphav)*c_s
    v_tub = alphav*c_s*H_disk
    v_mol = .5*np.sqrt(8./np.pi)*c_s*2.4*const.m_p.cgs.value/np.sqrt(2.)/2e-15/rhog
    Re = v_tub/v_mol
    l_kol = L_Scale/Re**(3./4.)
    t_kol = t_L/np.sqrt(Re)
    v_kol = l_kol/t_kol
    return t_L,L_Scale,vel_L,t_kol,l_kol,v_kol,Re


## This is for giving an array of particle1, particle2 and output a grid of relative velocities for all combinations ...
## Need an equal shapes of particle_1 and particle_2 arrays
def rel_velocity_grd(particle_1,particle_2,M_star,R_dust,Temp_g,St1=None,St2=None,alphav = 1e-3,sigmaD = 1000.,psi = .01,H_disk = .1,rhod1=3.,rhod2=3.):
    # Input Paramters
    #Temp = 300 # Kelvin
    #R = 1. #AU
    #alphav = 1e-3          # Alpha-turbulence
    #sigmaD = 1000.           # gm/cm^2 - Gas Surface Density at Chosen Location ...
    #psi = .01	# Dust/Gas Mass ratio
    #H = .1 		# Scale Heigh (AU)
    #particle_1 = logspace(-4,6,100)
    #particle_2 = logspace(-4,6,100)
    #if(particle_1.shape != particle_2.shape):
    #    print('Need equal shapes of particle_1 and particle_2 arrays')
    #    pass
    H_disk = H_disk/const.au.cgs.value
    tmp1 = R_dust*const.au.cgs.value
    tmp2 = M_star*const.G.cgs.value*const.M_sun.cgs.value
    vk = np.sqrt(tmp2/tmp1)
    omega = np.sqrt(tmp2/(tmp1)**3.) # orbital angular velocity
    mean_mol_weigth = 2.4*const.m_p.cgs.value
    cs = np.sqrt(const.k_B.cgs.value*Temp_g/mean_mol_weigth) # gas sound speed
    rhog = (1./np.sqrt(2.*np.pi))*sigmaD/H_disk/const.au.cgs.value           # Gas Volume Density
    mfp = (2.4*const.m_p.cgs.value)/rhog/2e-15/np.sqrt(2)	# Gas mean Free Path
    cnst1_1 = (4./9.)*1e-15*(rhod1*omega)/((2.4*const.m_p.cgs.value)*cs)
    cnst1_2 = (4./9.)*1e-15*(rhod2*omega)/((2.4*const.m_p.cgs.value)*cs)
    u_n = cs**2./2./vk
    vg2 = (np.sqrt(alphav)*cs)**2.
    beta = (H_disk/R_dust)**2.
    Re_u = 1./np.sqrt(alphav*cs*cs/omega/mfp/cs)
    ############################################################
    ############################################################
    m1 = (4.*np.pi/3.)*(particle_1**3.)*rhod1
    m2 = (4.*np.pi/3.)*(particle_2**3.)*rhod2
    ############################################################
    if St1 is None :
        St1 = (particle_1)*(rhod1*omega)/(rhog*cs)
        tmp1 = np.where(particle_1 > mfp*9./4.)
        St1[tmp1] = cnst1_1*(particle_1[tmp1])**2.  # Stokes Drag Regime

        St2 = (particle_2)*(rhod2*omega)/(rhog*cs)
        tmp1 = np.where(particle_2 > mfp*9./4.)
        St2[tmp1] = cnst1_2*(particle_2[tmp1])**2.  # Stokes Drag Regime
    ############################################################
    num = particle_1.shape[0]
    num2 = particle_2.shape[0]
    vel_browm = np.zeros([num,num2])
    vel_drift = np.zeros([num,num2])
    vel_drift_th = np.zeros([num,num2])
    vel_turb= np.zeros([num,num2])
    cross_sec= np.zeros([num,num2])
    St_st1 = 1.6*St1.copy()
    St_st1[St_st1<Re_u]=Re_u.copy()
    St_st1[St_st1 >= 1.]=1.
    St_st2 = 1.6*St2.copy()
    St_st2[St_st2<Re_u]=Re_u.copy()
    St_st2[St_st2 >= 1.]=1.
    for i in range(0,num):
        vel_browm[i,:] = np.sqrt((8.*Temp_g*const.k_B.cgs.value/np.pi)*(m1[i]+m2)/(m1[i]*m2))
        vel_drift[i,:] = np.abs(u_n*2./(St1[i]+1./St1[i]) - u_n*2./(St2+1./St2))
        vel_drift_th[i,:] = np.abs(u_n*(1./(1.+St1[i]**2.) - 1./(1.+St2**2.)))
        st12 = np.fmax(St_st1[i],St_st2)
        tmp1= vg2*((-St2 + St1[i])/(St2 + St1[i]))
        tmp2 = (St1[i]**2.)/(st12 + St1[i]) - (St1[i]**2.)/(1 + St1[i]) - (St2**2.)/(st12 + St2) + (St2**2.)/(1 + St2)
        deltav1 = tmp1*tmp2

        tmp1= (St1[i]**2.)/(st12 + St1[i]) + (St2**2.)/(st12 + St2) - (St2**2.)/(Re_u + St2) - (St1[i]**2.)/(Re_u + St1[i])
        tmp2 = 2.*(st12-Re_u)
        deltav2 = vg2*(tmp1+tmp2)
        vel_turb[i,:] = np.sqrt(np.abs(deltav2+deltav1))
        cross_sec[i,:] = np.pi*(particle_1[i]+particle_2)**2.

    #X, Y = np.meshgrid(particle_1, particle_2)
    net_vel = np.sqrt(vel_browm**2. + vel_drift**2. +vel_drift_th**2. +vel_turb**2.)
    return net_vel,cross_sec
