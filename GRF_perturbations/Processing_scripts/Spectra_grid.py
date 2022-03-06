# For this computation we used 4 nodes of cluster, which had 16 Intel Xeon (E5-2640 v3, 2.6GHz) cores per node and 128GB (2133 MHz) of memory per node.
# The whole computation was parallelized between 4 nodes and for every node it was parallelized on 32 threads with the hyper-threading.
# The whole computation took  approximately 2 days.

import sys
import argparse
import os
sys.path.append('../../')

import numpy as np
#from tqdm import tqdm
import time

import jax.numpy as jnp
from jax.config import config
config.update('jax_platform_name','cpu')

from GRF_perturbations.Modules.GRF_inhomogeneities_class import GRF_inhomogeneities_class # Class that handles generation of GRF
from GRF_perturbations.Modules.Surface_Brightness_class import Surface_brightness_class # Class that handles all the inference related procedures
from GRF_perturbations.Modules.Inference_class import Inference_class # Class that handles all the inference related procedures
from GRF_perturbations.Modules.Utils import jax_pmap_over_grid # jaxified function that maps given function over given grid

#CPUs parallelization
import multiprocessing
max_thread_numbers=multiprocessing.cpu_count() # Maximal number of threads (usually cores*2)
# you can use the max number of threads, but it is better to keep it under control
Thread_num=32
#os.system('echo CPU count %d'%max_thread_numbers)
#Thread_num=max_thread_numbers
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=%d"%(Thread_num) # Parallelize on several CPUs

# Define the full grid of parameters logA,Beta,Phi
grid_size=128 # resolution of logA,Beta axes
logA_array=np.linspace(-10.075,-6.9,grid_size)
Beta_array=np.linspace(0,4.7,grid_size)
Phase_seeds_number=100  # number of random seeds that would ensure statistical significance of estimated spectrum mean and uncertainties

def check_computation_configuration():
    """ Check that multiprocessing is properly set up for selected parameters grid resolution"""
    try:
        assert grid_size % Thread_num == 0
    except:
        os.system('echo grid_size %d is not multiple of Thread_num %d' % (grid_size, Thread_num))

    try:
        assert Thread_num <= max_thread_numbers
    except:
        os.system('echo Max_thread_number=%d, but computation is trying to be paralelized into %d threads' % (
        max_thread_numbers, Thread_num))
        os.system('echo advised number of CPUs is %d' % (Thread_num))

    try:
        assert grid_size % Thread_num == 0
    except:
        os.system(
            'echo grid_size is not a multiple of Thread_num. grid_size=%d , Thread_num=%d' % (grid_size, Thread_num))


if __name__ == '__main__':

    check_computation_configuration() # check multiprocessing configuration
    nodes = grid_size // Thread_num # on how many nodes we will split computation of the grid

    # read index of the node that is used in the current script call
    parser = argparse.ArgumentParser(description='Node split')
    parser.add_argument('--node_num', type=int, help='Which node to compute')
    args = parser.parse_args()
    node_num = args.node_num
    os.system('echo nodes=%d, node_num=%d' % (nodes, node_num))

    if node_num is None:
        node_num = 0
    try:
        assert node_num < nodes
    except:
        os.system('echo node_num is greater of equal to nodes. node_num=%d , nodes=%d' % (node_num, nodes))

    pixel_number = 100  # The image is 100x100 pixels
    pixel_scale = 0.08  # The resolution is 0.08 arcsec/pixel

    # Observation conditions
    PSF_FWHM = 0.1  # arcsec, PSF of HST
    exposure_time = 2028  # COSMOS exposure time for Poisson noise
    SNR = 200  # 75th percentile of COSMOS dataset's distribution of Peak-SNR for Background noise

    os.system('echo Precompiling Observation and Inference classes')
    start_time = time.time()
    # GRF generating class
    GRF_class = GRF_inhomogeneities_class(pixel_number, pixel_scale, Phase_seeds_number)
    # All the unperturbed source-lens parameters are encoded in the class as defaults
    Surface_brightness = Surface_brightness_class(pixel_number, pixel_scale, PSF_FWHM, SNR, exposure_time)
    # Handling all the inference-related procedures
    Inference=Inference_class(GRF_class,Surface_brightness,Grad_descent_max_iter=1000) # Set 1000 steps of source-lens fitting gradient descent
    time_string = "elapsed time {:.5f}".format(time.time() - start_time)
    os.system('echo Compilation took:  %s seconds' % time_string)

    # We split computation of full grid over $node_num nodes, so every nodes computes part of the grid
    logA_to_process = logA_array[node_num * Thread_num:(node_num + 1) * Thread_num]

    os.system('echo Process logA:')
    for i in range(len(logA_to_process)):
        os.system('echo i={:.0f} logA={:.2f}'.format(i, logA_to_process[i]))

    # Set of functions that map the anomalies power spectrum getter over a given dimension of the log(A),Beta,phi grid
    compute_anomalies_spectrum_pure = lambda logA, Beta, Fourier_phase: Inference.Anomalies_Radial_Power_Spectrum(
        jnp.array([logA, Beta]), Fourier_phase)
    # Parallel mapping of given function over given logA_array_batch,Beta_array,random_seed. Computation is parallelized over logA_array
    compute_spectra_grid_batch = lambda logA_array_batch: jax_pmap_over_grid(compute_anomalies_spectrum_pure,
                                                                             logA_array_batch, Beta_array,
                                                                             np.arange(Phase_seeds_number))

    # Parallel mapping of given function over given logA_array,Beta_array,random_seed. Computation is parallelized over logA_array
    os.system('echo Computing grid')
    start_time=time.time()
    Spectra_grid_batch=compute_spectra_grid_batch(logA_to_process) # Computation parallelized over each value of log(A) in the logA_to_process array
    end_time=time.time()
    time_string="elapsed time {:.5f}".format(end_time-start_time)
    os.system('echo Grid processing time: %s seconds' %(time_string))

    os.makedirs('./results/', exist_ok=True)
    np.savez('./results/Spectra_grid_node_%d'%(node_num),Spectra_grid_batch)
    # Append all the nodes to get Spectra_grid after the computation
