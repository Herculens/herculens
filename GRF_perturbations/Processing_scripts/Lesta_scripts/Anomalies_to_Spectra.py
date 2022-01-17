import sys
import argparse
import os
sys.path.append('../../')

import numpy as np
#from tqdm import tqdm
import time
import math

import jax
import jax.numpy as jnp
from jax.config import config
config.update('jax_platform_name','cpu')


from GRF_perturbations.Modules.Data_generation import Observation_conditions_class
from GRF_perturbations.Modules.Image_processing import compute_radial_spectrum
from GRF_perturbations.Modules.GRF_generation import get_k_grid

#CPUs parallelization
import multiprocessing
max_thread_numbers=multiprocessing.cpu_count()
#os.system('echo CPU count %d'%)

Thread_num=32
#Parallelize on several CPUs
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=%d"%(Thread_num)

grid_size=128

#In practice I can do fft over the entire grid. Then aggregate spectra in parallel.

def anomalies_to_radial_spectrum(anomaly_image, Observation_conditions, k_grid):
    radial_spectrum = compute_radial_spectrum(anomaly_image, Observation_conditions.annulus_mask, k_grid,
                                              Observation_conditions.frequencies)
    return radial_spectrum

def get_radial_spectra_grid(anomalies_grid):


    anomalies_to_radial_spectrum_pure = lambda anomaly: \
        anomalies_to_radial_spectrum(anomaly, Observation_conditions, k_grid)

    anomalies_to_radial_spectrum_seeds = lambda anomaly_AB_row: \
        jax.vmap(anomalies_to_radial_spectrum_pure, in_axes=0)(anomaly_AB_row)
    anomalies_to_radial_spectrum_Betas = lambda anomaly_A_row: \
        jax.vmap(anomalies_to_radial_spectrum_seeds, in_axes=0)(anomaly_A_row)


    #Function computing loop over GRF amplitudes logA in parallel
    anomalies_to_radial_spectrum_grid_pmapped = jax.pmap(anomalies_to_radial_spectrum_Betas)

    spectra_grid=anomalies_to_radial_spectrum_grid_pmapped(anomalies_grid)

    return spectra_grid

if __name__ == '__main__':
    try:
        assert grid_size%Thread_num==0
    except:
        os.system('echo grid_size %d is not multiple of Thread_num %d'%(grid_size,Thread_num))


    try:
        assert Thread_num<=max_thread_numbers
    except:
        os.system('echo Max_thread_number=%d, but computation is trying to be paralelized into %d threads'%(max_thread_numbers,Thread_num))
        os.system('echo advised number of CPUs is %d'%(Thread_num))


    try:
        assert grid_size%Thread_num==0
    except:
        os.system('echo grid_size is not a multiple of Thread_num. grid_size=%d , Thread_num=%d'%(grid_size,Thread_num))

    nodes=grid_size//Thread_num

    pixel_number=100
    pixel_scale=0.08

    PSF_FWHM=0.1 #arcsec, PSF of HST
    exposure_time=2028 #COSMOS exposure
    SNR=200 #75th percentile of COSMOS gaussian noise distribution


    os.system('echo Precompiling Observation and Inference classes')
    start_time=time.time()
    Observation_conditions=Observation_conditions_class(pixel_number,pixel_scale,PSF_FWHM,SNR,exposure_time)
    time_string="elapsed time {:.5f}".format(time.time()-start_time)
    os.system('echo Compilation took:  %s seconds' %time_string)

    k_grid, _ = get_k_grid(Observation_conditions.pixel_number, Observation_conditions.pixel_scale)

    nodes_array = np.arange(4)
    Batches_array = np.arange(0, 128, 16)
    #for loop over nodes 0-3 and batches ... with appending

    spectra_ABF_grid = np.zeros((0, 128, 100, 46))

    os.system('echo Processing grid')
    start_time = time.time()
    for node_index in nodes_array:
        spectra_BF_row = np.zeros((32, 0, 100, 46))
        for batch_index in Batches_array:
            anomalies_grid=np.load('./results/Anomalies/anomalies_grid_node_{:.0f}_batch_{:.0f}.npz'.format(node_index,batch_index))['arr_0']
            spectra_F_grid=get_radial_spectra_grid(anomalies_grid)
            spectra_BF_row=np.append(spectra_BF_row, spectra_F_grid, axis=1)
            time_string = "node_index {:.0f}, batch_index {:.0f} elapsed time {:.5f}".format(node_index,batch_index,time.time() - start_time)
            os.system('echo %s' % time_string)

        spectra_ABF_grid = np.append(spectra_ABF_grid, spectra_BF_row, axis=0)

    os.makedirs('./results/Radial_spectra', exist_ok=True)
    np.savez('./results/Radial_spectra/Spectra_grid',spectra_ABF_grid)


