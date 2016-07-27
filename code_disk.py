######### This version has an extra Destruction section  -
        #tmp2 = np.where((calc_D_val > 1.) & (calc_D_val <= 100.))
        #indx_rem = tmp1[0][tmp2[0]]
        #dn_grid[indx_rem] = dn_grid[indx_rem]*(1. - np.exp(-calc_D_val[tmp2]/20.))        # Not conserving mass here (Bad ... )
        #By default is turned off

from Axis import *
import numpy as np
from scipy import integrate
import astropy.constants as const
import timeit

class Disk_process(object):
    '''A generic parent class - set up the disk on a chosen 2d grid (can later make 3d is needed .. )
    Every process object has a set of state variables on a spatial grid.
    Generally pass the state as a dictionary of ndarray  - one state is size_dist_grid.. -> original
    '''
    def __init__(self, axis=None,state=None,initial_size_dist=None):
        if axis is None :
                raise ValueError('Must pass an axis class object ..')
        self.process_name = 'TwoD_disk'
        self.numdims = 2
        setattr(self, axis.axis_type, axis)
        self.shape = axis.shape
        self.state = state
        # dictionary of model parameters
        print('Setting constants to default')
        self.param = self.get_constants()
        self.param['max_size_cascade'] = self.state['size_dist_grid'].max()
        self.param['min_size_cascade'] = self.state['size_dist_grid'].min()
        self.param['max_part_size'] = self.state['size_dist_grid'].max()
        self.calculate_norm(initial_size_dist)
        self.param['init_point_val'] = self.state['size_dist'][0]
        self.param['volume'] = axis.area_all*self.param['h_disk_dust']*const.au.cgs.value**2.
        self.state['rhogN'] = (1./np.sqrt(2.*np.pi))*self.param['mean_density_G']/self.param['H_disk'] # Gas Volume Density
        self.state['st_grid'] = self.func_St_particle(self.state['size_dist_grid'])
        self.param['Max_St_Cascade'] = self.state['st_grid'].max()
        self.param['step_count'] = 0
        self.param['Area_all'] = axis.area_all*const.au.cgs.value**2.
        self.param['regrid_size'] = 200
        self.param['dm_0'] = (4.*self.param['mean_density_grain']*np.pi/3.)*(self.param['min_size_cascade']**3.) - \
                             (4.*self.param['mean_density_grain']*np.pi/3.)*((self.param['min_size_cascade']*0.95)**3.)
        self.param['dist_size_orig'] = self.state['size_dist_grid'].shape[0]
        dn_grid = self.grid_dN(self.state['size_dist'],self.state['size_dist_grid'])
        self.state['size_dist_dm_int'] = dn_grid

    def get_constants(self):
        alpha_slope = 2.1 # Slope of the bkg cascade - this remains fixed in the calc ..
        dust_to_gas = 0.01 # Initial dust to gas ratio
        # ( Set's the surface density of the dust - integrated over all the sizes .. )
        alphav = 0.01
        R_dust = np.mean(self.TwoD_cyl.points_r)
        sigma_r,t_gas_r,c_s,t_L,L_Scale,vel_L, t_kol,l_kol,v_kol,Re,lmfp,v_k,H_disk = \
        mmsn_model(R_dust,alphav,Sigma1 =1700.,T1 = 270.)
        Mach_n = v_k/c_s
        a_val = 0.5 # for compressive, 1./3. for solenoidal
        density_disp = np.sqrt(np.log(1. + a_val*Mach_n))
        t_violent_relax = 1. # in Kepler times i.e. 2.*pi/omega
        max_size_cascade = 1. # 1 cm
        min_size_cascade = 1e-4 # 1 micron
        mean_density_grain = 3. # gm/cm^3
        omega = c_s/H_disk # orbital angular velocity
        St_cnst1 = mean_density_grain*omega/c_s
        mean_mol_weigth = 2.4*const.m_p.cgs.value
        mfp_const = mean_mol_weigth/2e-15/np.sqrt(2.)                 # Gas mean Free Path
        St_cnst2 = (4./9.)*1e-15*(mean_density_grain*omega)/(mean_mol_weigth*c_s)
        max_mass_ratio = 0.001 # rule 1 (destroy wihting a threshold of 1/1000th of the mass)
        max_vel_grow = 10.*1e2 # rule 1 (destroy if velocity > 50 m/s)
        return {'dust_to_gas':dust_to_gas,'mean_density_G':sigma_r,'density_disp':density_disp,'t_gas_r':t_gas_r,
        'c_s':c_s,'t_L':t_L,'L_Scale':L_Scale,'vel_L':vel_L,'t_kol':t_kol,'l_kol':l_kol,'v_kol':v_kol,'Re':Re,
        'lmfp':lmfp,'mean_density_grain':mean_density_grain,'omega':omega,'alphav':alphav,
        't_violent_relax':t_violent_relax,'h_disk_dust':H_disk*0.25,'alpha_slope':alpha_slope,
               'mfp_const':mfp_const,'St_cnst1':St_cnst1,'St_cnst2':St_cnst2, 'H_disk':H_disk,'max_mass_ratio':max_mass_ratio,
               'max_vel_grow':max_vel_grow,'R_dust':R_dust,'extra_dest':0}

    # Note that the normalization is defined such that the surface mass density matches
    #  what is given (matches it at the mean value)
    def calculate_norm(self,initial_size_dist=None):
        if self.param['alpha_slope'] == 2 :
            raise ValueError('slope cannot be exactly 2')
        dust_density = self.param['dust_to_gas']*self.param['mean_density_G']
        t1 = 2. - self.param['alpha_slope']
        m_max = (4.*np.pi/3.)*self.param['mean_density_grain']*(self.param['max_size_cascade']**3.)
        m_min = (4.*np.pi/3.)*self.param['mean_density_grain']*(self.param['min_size_cascade']**3.)
        t2 = (1./t1)*(m_max**t1 - m_min**t1)
        t3 = dust_density/(self.param['h_disk_dust'])
        self.param['norm_param'] = t3/t2
        if initial_size_dist is None :
            self.state['size_dist'] = self.nm_pl((4.*np.pi*self.param['mean_density_grain']*self.state['size_dist_grid']**3.)/3.)
        else :
            self.state['size_dist'] = initial_size_dist.copy()

    def nm_pl(self,m) :  # Specify a power law as the n(m) = m^{-alpha}, properly normalized
        p1 = self.param['norm_param']*(m**(-self.param['alpha_slope']))
        return p1  # return num/volume/mass

    def func_St_particle(self,particle_grid,gas_density=None,single=None):
        if gas_density is None :
            gas_density = self.state['rhogN']
        St1_grid = (particle_grid)*(self.param['St_cnst1']/gas_density)
        mfp = self.param['mfp_const']/gas_density              # Gas mean Free Path
        if (single ==1) :
            if (particle_grid > mfp*9./4.):
                St1_grid = self.param['St_cnst2']*(particle_grid)**2.
            return St1_grid
        tmp1 = np.where(particle_grid > mfp*9./4.)
        St1_grid[tmp1] = self.param['St_cnst2']*(particle_grid[tmp1])**2.  # Stokes Drag Regime
        return St1_grid

    def func_St_particle_invert(self,St_num,gas_density=None):
        if gas_density is None :
            gas_density = self.state['rhogN']
        particle_grid = St_num/(self.param['St_cnst1']/gas_density)
        mfp = self.param['mfp_const']/gas_density              # Gas mean Free Path
        if(particle_grid > mfp*9./4.):
            particle_grid = np.sqrt(St_num/self.param['St_cnst2'])  # Stokes Drag Regime
        return particle_grid

    def reset_density(self):
        self.state['rhogN'] = (1./np.sqrt(2.*np.pi))*self.param['mean_density_G']/self.param['H_disk'] # Gas Volume Density
        return 1

    def G_tm(self,dn_grid,size_dist_grid,st_num_grid,tmp1,sigmaG_sc,R_dust) :  #only dm/dt
        ##  dm/dt - G
        m_tm = (4.*self.param['mean_density_grain']*np.pi/3.)*(size_dist_grid**3.) # mass of the particles in the grid
        #dm_grid = np.diff(m_tm)
        #integrnd_D = dn_grid.copy()
        net_vel,cross_sec = rel_velocity_grd(size_dist_grid[tmp1],size_dist_grid,1.,R_dust,self.param['t_gas_r'],\
                             alphav = self.param['alphav'],sigmaD = self.param['mean_density_G']*sigmaG_sc,\
                             psi=self.param['dust_to_gas'],H_disk = self.param['H_disk'],\
                         rhod1=self.param['mean_density_grain'],rhod2=self.param['mean_density_grain'],\
                         St1=st_num_grid[tmp1],St2=st_num_grid)
        m_tm_act = np.tile(m_tm,(net_vel.shape[0],1))
        m_tm_act1 = np.transpose(np.tile(m_tm[tmp1],(net_vel.shape[1],1)))*self.param['max_mass_ratio'] # impose the mass-cutoff
        m_tm_act2 = m_tm_act1/self.param['max_mass_ratio']*1.1
        m_tm_act_filt = m_tm_act.copy()*0.0 + 1.
        m_tm_act_filt[m_tm_act<m_tm_act1] = 0.
        m_tm_act_filt = 1. - m_tm_act_filt
        m_tm_act_filt[net_vel<self.param['max_vel_grow']] = 1.  # This is the rule that destroy only if velocity > 50 m/s)
        m_tm_act_filt[m_tm_act>m_tm_act2] = 0.
        net_vel_cross2 = ((net_vel*cross_sec)*m_tm_act)*m_tm_act_filt
        tmp1 = net_vel_cross2[:,1:]*dn_grid[1:]
        dm_G = np.sum(tmp1,axis=1)
        return dm_G,m_tm[-1]

    def G_D_tm(self,dn_grid,size_dist_grid,st_num_grid,tmp1,sigmaG_sc,R_dust) :  # D destruction time-scale (in seconds)
        ## D - timescale, dm/dt - G
        net_vel,cross_sec = rel_velocity_grd(size_dist_grid[tmp1],size_dist_grid,1.,R_dust,self.param['t_gas_r'],\
                             alphav = self.param['alphav'],sigmaD = self.param['mean_density_G']*sigmaG_sc,\
                             psi=self.param['dust_to_gas'],H_disk = self.param['H_disk'],\
                         rhod1=self.param['mean_density_grain'],rhod2=self.param['mean_density_grain'],\
                         St1=st_num_grid[tmp1],St2=st_num_grid)
        m_tm = (4.*self.param['mean_density_grain']*np.pi/3.)*(size_dist_grid**3.) # mass of the particles in the grid
        # integrnd_D = dn_grid.copy()
        net_vel_D_filt = net_vel.copy()*0.0 + 1.
        m_tm_act = np.tile(m_tm,(net_vel.shape[0],1))
        m_tm_act1 = np.transpose(np.tile(m_tm[tmp1],(net_vel.shape[1],1)))*self.param['max_mass_ratio'] # impose the mass-cutoff
        m_tm_act2 = (m_tm_act1/self.param['max_mass_ratio'])*1.1
        m_tm_act_filt = m_tm_act.copy()*0.0 + 1.
        net_vel_D_filt[net_vel<self.param['max_vel_grow']] =0.  # This is the rule that destroy only if velocity > 50 m/s)
        m_tm_act_filt[m_tm_act<m_tm_act1] = 0.
        #m_tm_act_filt[m_tm_act>m_tm_act2] = 0.
        net_vel_cross = ((net_vel*cross_sec)*net_vel_D_filt)*m_tm_act_filt
        tmp1 = net_vel_cross[:,1:]*dn_grid[1:]
        sm = np.sum(tmp1,axis=1)
        sm[sm==0] = -1
        t_D = 1./sm

        m_tm_act_filt = 1. - m_tm_act_filt
        m_tm_act_filt[net_vel<self.param['max_vel_grow']] = 1.  # This is the rule that destroy only if velocity > 50 m/s)
        m_tm_act_filt[m_tm_act>m_tm_act2] = 0. # Get destroyed by anybody larger than you by at least 10% ...
        net_vel_cross2 = ((net_vel*cross_sec)*m_tm_act)*m_tm_act_filt
        tmp1 = net_vel_cross2[:,1:]*dn_grid[1:]
        dm_G = np.sum(tmp1,axis=1)
        return t_D,dm_G

    def rule1_growth_dest(self,dn_grid,size_dist_grid,st_num_grid,tmp1,new_dens_sc,time_step,r_val,only_D = 0.):
        '''
        implement rule 1 here ..
        '''
        calc_D_val,calc_G_val = self.G_D_tm(dn_grid,size_dist_grid,st_num_grid,tmp1,\
                                      new_dens_sc,r_val)  # output is D (in sec) and dm/dt
        calc_D_val = calc_D_val/time_step   # Non-dimensional ..
        tmp2 = np.where((calc_D_val <= 1.) & (calc_D_val > 0.))
        indx_rem = tmp1[0][tmp2[0]]
        dn_grid[indx_rem] = 0.       # Not conserving mass here (Bad ... )
        if (self.param['extra_dest'] ==1):
            tmp2 = np.where((calc_D_val > 1.) & (calc_D_val <= 100.))
            indx_rem = tmp1[0][tmp2[0]]
            dn_grid[indx_rem] = dn_grid[indx_rem]*(1. - np.exp(-calc_D_val[tmp2]/20.))        # Not conserving mass here (Bad ... )
        ######### tmp3 = tmp1[0][tmp2[0]]
        ######### tmp4 = np.where(size_dist_grid < 1e3) # only dest for <1e3
        ######### indx_rem = np.intersect1d(tmp3,tmp4[0])
        ## For the rest (where D/time-step > 1,
        # i.e would not be destroyed in mean at least (can have Poisson stats here))
        if only_D != 1 :
            tmp2 = np.where((calc_D_val > 1.) | (calc_D_val <= 0.))
            indx_rem = tmp1[0][tmp2[0]]
            delta_m = calc_G_val[tmp2]*time_step # Not conserving mass here (Bad ... )
            delta_m2 = delta_m/(4.*np.pi*self.param['mean_density_grain']/3.)
            delta_r = (size_dist_grid[indx_rem]**3. +  delta_m2)**(1./3.) - size_dist_grid[indx_rem] #    delta_r
            for i in range(indx_rem.shape[0]-1):
                if size_dist_grid[indx_rem[i]]+delta_r[i] == size_dist_grid[indx_rem[i]+1] :
                    print('Duplicate ... - fixed ')  ## Check for duplicates in the new size_grid ..
                    delta_r[i] = delta_r[i]*.99
            if ( delta_r < 0.).any():
                print('negative delta_r - BAD ...')
            size_dist_grid[indx_rem]  += delta_r
            tmp1 = np.argsort(size_dist_grid)
            size_dist_grid = size_dist_grid[tmp1]
            dn_grid = dn_grid[tmp1]
        if (dn_grid<0).any():
            raise ValueError('Error - negative number dN - BAD ... ')  ## Check for any negatives - bad ...
        return dn_grid,size_dist_grid

    def grid_dN(self,size_dist,size_dist_grid): # get dN/volume grid instead of dN/dm/volume which is the original power-law ..
        m_tm = (4.*self.param['mean_density_grain']*np.pi/3.)*(size_dist_grid**3.) # mass of the particles in the grid
        dm_grid = np.diff(m_tm)
        dn_grid = size_dist[1:]*dm_grid # this is the dN/volume grid (size is 1 less than the size_dist_grid .. )
        dn_grid = np.insert(dn_grid,0,self.param['dm_0']*size_dist[0]) # prepend a slope*.05 times slope of first part for the dn_grid to make it the same size as size_dist_grid
        return dn_grid

    def rule2_growth(self,size_max_cascade,dn_grid,size_dist_grid,st_num_grid,new_dens_sc,time_step,r_val,gas_density_inp):
        '''
        implement rule 2 here ..
        '''
        size_dist = new_dens_sc*self.state['size_dist'].copy()
        new_size_growth = size_dist_grid.max()
        time_counter = 0.
        first_pass_max_cas = 0.
        dn_grid_New = dn_grid.copy()
        # Note that the following stops if mass doubling time is greater than time-step
        # Make a note of when the size exceeds size_max_cascade, we stop growing (can introduce a destruction term later .. )
        while (time_counter <= 1.) : # really not used ...
            new_size_growth = (2.**(1./3.))*size_dist_grid[-1]
            tmp1 = np.array([-1])
            calc_G_val,m_lrg = self.G_tm(dn_grid_New,size_dist_grid,st_num_grid,tmp1,\
                                          new_dens_sc,r_val)
            if calc_G_val ==0:
                time_dbl = 2.
            else :
                time_dbl = (m_lrg/calc_G_val)/time_step   # Non-dimensional ..
            time_counter += time_dbl
            if (time_counter > 1.):  # as long as the cumulative  mass doubling time is greater than 1
                break
            tmp1 = self.nm_pl((4.*np.pi*self.param['mean_density_grain']*size_dist_grid[-1]**3.)/3.)
            scale_fac = size_dist[-1]/tmp1  # this takes care of the diff btw the scales and normalizations ..
            size_dist_grid = np.append(size_dist_grid,new_size_growth)
            ############################################
            new_size_dist = scale_fac*\
            self.nm_pl((4.*np.pi*self.param['mean_density_grain']*new_size_growth**3.)/3.)
            size_dist = np.append(size_dist,new_size_dist)
            dn_grid_New = self.grid_dN(size_dist,size_dist_grid)
            ############################################
            st_new_grwth = self.func_St_particle(new_size_growth,gas_density = gas_density_inp,single=1)
            st_num_grid = np.append(st_num_grid,st_new_grwth)
            ############################################
            ## This is for later if we want to also introduce a destruction for some fraction of the time-step (too detailed .. for our purposes)
            if (new_size_growth >= size_max_cascade) & (first_pass_max_cas ==0.) :
                first_pass_max_cas = 1. - time_counter
                break
        # No need to do regiding here (just save this and regrid at end of the time-step)
        #mp1 = np.where(st_num_grid >= self.param['Max_St_Cascade'])
        #ize_dist_grid_N = np.logspace(np.log10(size_dist_grid.min()),np.log10(size_dist_grid.max()),self.param['regrid_size'])
        #ize_dist_N = self.regrid(size_dist,size_dist_grid,size_dist_grid_N)
        return dn_grid_New,size_dist_grid #tmp1,first_pass_max_cas

    def regrid(self,dn_grid,size_dist_grid,new_size_dist_grid):
        new_dn_grid = new_size_dist_grid.copy()*0.0
        new_size_da = new_size_dist_grid.copy()*0.0
        new_size_dist = new_size_dist_grid.copy()*0.0
        new_dn_grid[0] = dn_grid[0]
        for i in range(1,new_size_dist_grid.shape[0]):
            tmp1 = np.where((size_dist_grid > new_size_dist_grid[i-1]) & (size_dist_grid <= new_size_dist_grid[i]))
            new_dn_grid[i] = np.sum(dn_grid[tmp1])
        m_tm = (4.*self.param['mean_density_grain']*np.pi/3.)*(new_size_dist_grid**3.) # mass of the particles in the grid
        dm_grid = np.diff(m_tm)  # Note the size is one smaller than new_dn_grid
        da_grid = np.diff(new_size_dist_grid) # Note the size is one smaller than new_dn_grid
        #####################
        new_size_da[0] = new_dn_grid[0]/(0.05*new_size_dist_grid[0])
        new_size_da[1:] = new_dn_grid[1:]/da_grid
        #####################
        new_size_dist[0] = new_dn_grid[0]/self.param['dm_0']
        new_size_dist[1:] = new_dn_grid[1:]/dm_grid
        #####################
        tmp_del = np.where((new_dn_grid*self.param['volume'])<= 1e-2)
        new_size_dist  = np.delete(new_size_dist, tmp_del)
        new_size_da  = np.delete(new_size_da, tmp_del)
        new_dn_grid  = np.delete(new_dn_grid, tmp_del)
        new_size_dist_grid  = np.delete(new_size_dist_grid, tmp_del)
        return new_size_dist,new_size_da,new_dn_grid,new_size_dist_grid

    def step_forward(self):
        '''
        Call this after every time-step ... - time-step is t_violent_relax
        '''
        mean_dd = self.param['dust_to_gas']*self.param['mean_density_G']
        mean_gd = self.param['mean_density_G']
        mu = np.log(self.param['mean_density_G']) - .5*self.param['density_disp']**2.
        new_dens_sc =  (np.random.lognormal(mu,self.param['density_disp'],self.TwoD_cyl.area_grid.shape))/mean_gd
        # note that this should be scaled by grid area ...
        self.state['scale'] = new_dens_sc.copy()
        self.state['rhogNew']= self.state['rhogN']*new_dens_sc       # This is gas density in each grid bin ..
        size_dist_dm_int_New = self.state['size_dist_dm_int'].copy()*0.0 # this is the dN grid ..
        size_dist_grid_New =  self.state['size_dist_grid'].copy()
        #########################################
        # Set-up the size distribution - how to distribute from overall distrib after a time-step ..
        #########################################
        # Evolve in each grid bin separately -
        # Rule : If the size distrib in the grid has St greater than self.param['Max_St_Cascade'],
        # then calculate the D (Destruction time in units of time_step) - do this for all St > ma_st_cascade
        # As long as D is smaller than 1 (i.e. smaller than timestep), remove all the particles of that mass/size
        # else allow growth of the particle as rho*sigma*v ..
        # Second Rule - For cases where St max is < Max_St_cascade -> calculate the G timescale -
        # time to double the mass in units of time-step. As long as G<1 keep doubling mass of the system until
        # hitting max_St_Cascade*2. Then the first rule works ..
        #
        # Other choices :
        # a. how to set cascade in Rule 2 : Fix the normalization so that same number of particles at end of cascade at originally
        #    at largest size (i.e. assume all the largest particles grow to bigger sizes + fill from bkg influx)
        # b. How to distribute the mass of the destroyed particles ? -
        #     ideally integrate mass and put it all in the cascade, currently just removed and put in the 'bkg'.
        # c. Anything else ?? - Choose that Destruction (D) only by particles within 1/100th of the mass and with velocities of
        #    at least 50 m/s (can change both of these later). Similarly for G (grow due to everything with either mass < 1/100th )
        #    or velocity less than 50 m/s.
        #
        #  Note - not conserving mass currently
        #########################################
        r_val = self.TwoD_cyl.points_r.mean()
        time_step = (self.param['t_violent_relax']*2.*np.pi/self.param['omega'])
        for i in range(self.state['scale'].shape[0]) :
            for k in range(self.state['scale'].shape[1]):
                size_dist_dm_int = new_dens_sc[i,k]*self.state['size_dist_dm_int'].copy()
                size_dist_grid = self.state['size_dist_grid'].copy()
                st_num_grid = self.func_St_particle(size_dist_grid,gas_density = self.state['rhogNew'][i,k])
                tmp1 = np.where(st_num_grid > self.param['Max_St_Cascade'])
                if np.size(tmp1) != 0:
                    #a1 = timeit.timeit()
                    size_dist_dm_int,size_dist_grid = \
                    self.rule1_growth_dest(size_dist_dm_int,size_dist_grid,st_num_grid,tmp1,new_dens_sc[i,k],\
                                      time_step,r_val,only_D = 0.)
                    #print('R1: ',timeit.timeit() - a1)
                else:
                    #a1 = timeit.timeit()
                    ## Now in the regime of Rule 2 since nothing is larger than the cascade end ..
                    size_max_cascade = self.func_St_particle_invert(self.param['Max_St_Cascade'],\
                                    gas_density=self.state['rhogNew'][i,k])
                    size_dist_dm_int,size_dist_grid = \
                    self.rule2_growth(size_max_cascade,size_dist_dm_int,size_dist_grid,st_num_grid,new_dens_sc[i,k],\
                                     time_step,r_val,self.state['rhogNew'][i,k])
                    #
                    #print('R2: ',timeit.timeit() - a1)
                    # For the rest of the time-step (after the max_cascade is reached .. -> can have destruction also ..)
                    #if (first_pass_max_cas < 1) & (first_pass_max_cas>0.05):
                    #    st_num_grid = self.func_St_particle(size_dist_grid,gas_density = self.state['rhogNew'][i,k])
                    #    size_dist,size_dist_grid = \
                    #    self.rule1_growth_dest(self,size_dist,size_dist_grid,st_num_grid,tmp1,new_dens_sc[i,k],\
                    #                  time_step*first_pass_max_cas,r_val,only_D = 1)
                    ############################################
                #### end of else here ...
                #a1 = timeit.timeit()
                tmp1 = np.in1d(size_dist_grid_New, size_dist_grid, assume_unique=True) # index of size_dist_grid_New
                tmp1b = np.in1d(size_dist_grid,size_dist_grid_New, assume_unique=True)  # index of size_dist_grid
                size_dist_dm_int_New[tmp1] += size_dist_dm_int[tmp1b]
                # then merge other size_dist_grid
                inv_tmp1b = np.invert(tmp1b)
                del tmp1,tmp1b
                size_dist_grid_New = np.append(size_dist_grid_New,size_dist_grid[inv_tmp1b])
                size_dist_dm_int_New = np.append(size_dist_dm_int_New,size_dist_dm_int[inv_tmp1b])
                # sort it to make sure ascending sequence
                tmp1 = np.argsort(size_dist_grid_New)
                size_dist_grid_New = size_dist_grid_New[tmp1]
                size_dist_dm_int_New = size_dist_dm_int_New[tmp1]
                del size_dist_grid,size_dist_dm_int
                #print('R3: ',timeit.timeit() - a1)
        # Assume that the first point of the new grid is same as the original grid
        #a1 = timeit.timeit()
        print('Max_part : ',size_dist_grid_New.max()/1e6)
        if (size_dist_grid_New.max() > self.param['max_size_cascade']*1.05):
            size_dist_grid_N = np.append(\
            np.logspace(np.log10(self.param['min_size_cascade']),np.log10(self.param['max_size_cascade']),self.param['dist_size_orig']),\
            np.logspace(np.log10(self.param['max_size_cascade'])*1.02,np.log10(2e5),self.param['regrid_size']))
            size_dist_grid_N = np.append(size_dist_grid_N,np.array([3e5,4e5,5e5,6e5,7e5,8e5,9e5,1e6,2e6,5e6,7e6,1e7,3e7,5e7,1e8]))
        else :
            size_dist_grid_N = np.logspace(np.log10(self.param['min_size_cascade']),np.log10(size_dist_grid_New.max()),self.param['dist_size_orig'])
        size_dist_N,new_size_da,size_dist_dm_int,size_dist_grid_N = self.regrid(size_dist_dm_int_New,size_dist_grid_New,size_dist_grid_N)
        ## At the end of the time-step -->
        self.state['size_dist'] = size_dist_N/self.state['scale'].sum()  # Note that size_dist_New has to be corrected for the scaling to restore back ..
        self.state['size_dist_grid'] = size_dist_grid_N # The new corresponding grid ...
        self.param['step_count'] +=1
        self.param['max_part_size'] = size_dist_grid_N.max()
        self.state['size_dist_da'] = new_size_da/self.state['scale'].sum()
        self.state['size_dist_dm_int'] = size_dist_dm_int/self.state['scale'].sum()
        #print('R4: ',timeit.timeit() - a1)
        tmp1A = np.argmin(1e3 >= size_dist_grid_N )
        print('Num_part : ',np.sum(size_dist_dm_int[tmp1A:])*self.param['volume']/new_dens_sc.sum(),'  ',tmp1A)
        #print('Done, step_number : ',self.param['step_count'],'  ',size_dist_grid_New.max())
        #scale_fact = self.param['init_point_val']/self.state['size_dist'][0]
        #print(scale_fact),
        #self.state['size_dist'] = scale_fact*self.state['size_dist']
        #a1 = self.reset_density()
        return 1
        #############################################################################################################
