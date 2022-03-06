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

import jax
from GRF_perturbations.Modules.Utils import jax_map
from GRF_perturbations.Modules.Spectra_grid_processing import get_confidence_intervals

#CPUs parallelization
import multiprocessing
max_thread_numbers=multiprocessing.cpu_count() # Maximal number of threads (usually cores*2)
# you can use the max number of threads, but it is better to keep it under control
Thread_num=32
#os.system('echo CPU count %d'%max_thread_numbers)
#Thread_num=max_thread_numbers
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=%d"%(Thread_num) # Parallelize on several CPUs

#The grid computed for the unperturbed Source-Lens setup used in this notebook
Spectra_grid=np.load('../Processing_scripts/results/Spectra_grid.npz')['arr_0']
grid_size=Spectra_grid.shape[0]
Phase_seeds_number=Spectra_grid.shape[2]
logA_array=np.linspace(-10.075,-6.9,grid_size)

def get_predictions(Data_spectrum,MU_tensor,Sigma_tensor):

    Data_logspectrum=jnp.log(Data_spectrum) # Anomalies spectrum that we want to infer logA,Beta for
    Chi_squared=jnp.mean(jnp.power((Data_logspectrum-MU_tensor)/Sigma_tensor,2),axis=-1) # negative log-likelihood
    Likelihood_grid=jnp.exp(-Chi_squared/2)
    Likelihood_grid/=Likelihood_grid.sum() # Normalised grid of likelihoods for every logA,Beta

    prediction, logA_conf_intervals, Beta_conf_intervals=get_confidence_intervals(Likelihood_grid)

    res_matrix=jnp.array([prediction])
    res_matrix=jnp.append(res_matrix,logA_conf_intervals,axis=0)
    res_matrix=jnp.append(res_matrix,Beta_conf_intervals,axis=0)

    return res_matrix

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

    # We split computation of full grid over $node_num nodes, so every nodes computes part of the grid
    indices_to_process=np.arange(node_num * Thread_num,(node_num + 1) * Thread_num)
    Spectra_grid_batch=Spectra_grid[indices_to_process]
    os.system('echo Process logA:')
    for i in range(len(indices_to_process)):
        os.system('echo i={:.0f} logA={:.2f}'.format(indices_to_process[i], logA_array[indices_to_process[i]]))

    # Statistics of the spectra
    # log(Spectrum) Mean over random seed phi
    MU_tensor = np.log(Spectra_grid).mean(axis=2)
    MU_tensor_ext = np.tile(MU_tensor, Spectra_grid.shape[2])
    MU_tensor_ext = MU_tensor_ext.reshape(Spectra_grid.shape)
    # log(Spectrum) standard deviation over random seed phi
    Sigma_tensor = np.sqrt(np.power(np.log(Spectra_grid) - MU_tensor_ext, 2).sum(axis=2) / (Spectra_grid.shape[2] - 1))

    # parallel computation
    compute_predictions = lambda Data_spectrum: get_predictions(Data_spectrum, MU_tensor, Sigma_tensor)
    compute_predictions_seeds = lambda Spectra_Phi_row: \
        jax_map(compute_predictions, Spectra_Phi_row)
    compute_predictions_Beta = jax.jit(lambda Spectra_Beta_Phi_matrix: \
                                           jax_map(compute_predictions_seeds, Spectra_Beta_Phi_matrix))

    compute_grid_predictions_batch=jax.pmap(compute_predictions_Beta)

    # Parallel mapping of given function over given logA_array,Beta_array,random_seed. Computation is parallelized over logA_array
    os.system('echo Computing grid')
    start_time=time.time()
    Predictions_grid_batch=compute_grid_predictions_batch(Spectra_grid_batch) # Computation parallelized over each value of log(A) in the logA_to_process array
    end_time=time.time()
    time_string="elapsed time {:.5f}".format(end_time-start_time)
    os.system('echo Grid processing time: %s seconds' %(time_string))

    os.makedirs('./results/', exist_ok=True)
    np.savez('./results/Predictions_grid_node_%d'%(node_num),Predictions_grid_batch)
    # Append all the nodes to get Predictions_grid after the computation
