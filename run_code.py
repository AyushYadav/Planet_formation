import numpy as np
import warnings
from code_disk import *
from Axis import *
import pylab as plt
import matplotlib.pyplot as plt

ax1 = Axis(bounds_r=np.array([.9,1.1]),bounds_phi=np.array([0.,2.*np.pi]),delta_r=.05,delta_phi=np.pi/18.)

#grid = np.append(np.array([1e-4,2e-4,1e-3,1e-2]),np.logspace(-1,.1,100))
grid = np.logspace(-4,.1,100)
disk1 = Disk_process(axis =ax1,state={'size_dist_grid':grid})
disk1.param['alpha_slope'] = 2.1
disk1.param['density_disp'] = .5
disk1.calculate_norm()
disk1.param['init_point_val'] = disk1.state['size_dist'][0]
dn_grid = disk1.grid_dN(disk1.state['size_dist'],disk1.state['size_dist_grid'])
disk1.state['size_dist_dm_int'] = dn_grid

disk1.param['regrid_size'] = 200
## destruction parameters
disk1.param['max_mass_ratio'] = 0.001 # ie. within 1/100 th of the mass
disk1.param['max_vel_grow'] = 10.*1e2 # 10 m/s

plt.figure(1)
plt.clf()
orig = disk1.state['size_dist_dm_int'],disk1.state['size_dist_grid']
#orig1 = disk1.state['size_dist_da'],disk1.state['size_dist_grid']
#orig2 = disk1.state['size_dist'],disk1.state['size_dist_grid']
print(disk1.param['alpha_slope'])
num_steps = 1350
for i in range(num_steps):
    disk1.step_forward()
    print(i)
    plt.loglog(orig[1][1:],orig[0][1:]*disk1.param['volume'],'r+')
    new2 = disk1.state['size_dist_dm_int'],disk1.state['size_dist_grid']
    plt.loglog(new2[1][1:],new2[0][1:]*disk1.param['volume'],marker='o',linestyle='None')
    plt.title('time-step : '+str(disk1.param['step_count']))
    plt.xlabel(' Size  - cm')
    plt.ylabel(' dN (over the volume of annulus of .1 AU at 1 AU)')
    plt.savefig('Run2p1_N/Run2p1_stp'+str(i)+'_intm.png')
    plt.clf()
    if (disk1.param['max_part_size'] > 5e8):
        print('Reached Maximum size of 1e6, Finish calculation')
        break
    #############################################################################################################
    #plt.loglog(orig1[1][1:],orig1[0][1:]*disk1.param['volume'],'r+')
    #new2 = disk1.state['size_dist_da'],disk1.state['size_dist_grid']
    #plt.loglog(new2[1][1:],new2[0][1:]*disk1.param['volume'],marker='o',linestyle='None')
    #plt.title('time-step : '+str(disk1.param['step_count']))
    #plt.xlabel(' Size  - cm')
    #plt.ylabel(' dN/da (over the volume of annulus of .1 AU at 1 AU)')
    #plt.savefig('Run2e/Run2e_stp'+str(i)+'_da.png')
    #plt.clf()
    #############################################################################################################
    #plt.loglog(orig2[1][1:],orig2[0][1:]*disk1.param['volume'],'r+')
    #new2 = disk1.state['size_dist'],disk1.state['size_dist_grid']
    #plt.loglog(new2[1][1:],new2[0][1:]*disk1.param['volume'],marker='o',linestyle='None')
    #plt.title('time-step : '+str(disk1.param['step_count']))
    #plt.xlabel(' Size  - cm')
    #plt.ylabel(' dN/dm (over the volume of annulus of .1 AU at 1 AU)')
    #plt.savefig('Run2e/Run2e_stp'+str(i)+'_dm.png')
    #plt.clf()

### to save the class - disk1 (useful since it records all the parameter choices etc..)
import pickle
# Saving the objects:
with open('Run1p9.pickle', 'wb') as f:
    pickle.dump(disk1, f)
