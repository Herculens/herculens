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


from GRF_perturbations.Modules.Jax_Utils import jax_map
from GRF_perturbations.Modules.Inference import get_likelihood,get_conf_intervals

#CPUs parallelization
import multiprocessing
max_thread_numbers=multiprocessing.cpu_count()
#os.system('echo CPU count %d'%)

Thread_num=1
#Parallelize on several CPUs
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=%d"%(Thread_num)

grid_size=128

logA_array=np.linspace(-10.075,-6.9,grid_size)
Beta_array=np.linspace(0,4.7,grid_size)
GRF_seeds_number=100

Spectra_grid=np.load('./results/Radial_spectra/Spectra_grid.npz')['arr_0']

MU_tensor=np.log(Spectra_grid).mean(axis=2)
MU_tensor_ext=np.tile(MU_tensor,GRF_seeds_number)
MU_tensor_ext=MU_tensor_ext.reshape((grid_size,grid_size,GRF_seeds_number,Spectra_grid.shape[-1]))
Sigma_tensor=np.sqrt(np.power(np.log(Spectra_grid)-MU_tensor_ext,2).sum(axis=2)/(Spectra_grid.shape[2]-1))

def compute_predictions_seeds(spectra_F_row,MU_tensor,Sigma_tensor):


    likelihoods=jax_map(lambda data_spectrum: get_likelihood(data_spectrum,MU_tensor,Sigma_tensor),spectra_F_row)

    res=jax_map(get_conf_intervals,likelihoods)

    return res

def get_Predictions_grid(spectra_ABF_subgrid):

    compute_Predictions_Betas= jax.jit(lambda spectra_BF_row: jax_map(lambda spectra_F_row: compute_predictions_seeds(spectra_F_row,MU_tensor,Sigma_tensor),spectra_BF_row))

    #Function computing loop over GRF amplitudes logA in parallel
    compute_Predictions_grid_pmapped=jax.pmap(compute_Predictions_Betas)

    os.system('echo Computing grid')

    start_time=time.time()
    Predictions_grid=compute_Predictions_grid_pmapped(spectra_ABF_subgrid)
    end_time=time.time()


    time_string="elapsed time {:.5f}".format(end_time-start_time)
    os.system('echo Grid processing time: %s seconds' %(time_string))

    np.save('./results/prediction/shape.npy',Predictions_grid.shape)
    #shape_string="{}".format(Predictions_grid.shape)
    #os.system('echo Grid shape: seconds' %(shape_string))

    return Predictions_grid


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

    #This part will allow to split the computation on several nodes
    #Keeping it parallelized on 32 CPUs
    parser = argparse.ArgumentParser(description='Node split')

    parser.add_argument('--node_num', type=int, help='Which node to compute')

    args = parser.parse_args()


    #Index of which part of the splitted array to compute
    node_num=args.node_num

    os.system('echo nodes=%d, node_num=%d'%(nodes,node_num))

    if node_num is None:
        node_num=0

    try:
        assert node_num<nodes
    except:
        os.system('echo node_num is greater of equal to nodes. node_num=%d , nodes=%d'%(node_num,nodes))


    logA_indices_to_process=np.arange(node_num*Thread_num,(node_num+1)*Thread_num)

    os.system('echo Process logA:')
    for i in range(len(logA_indices_to_process)):
        os.system('echo i={:.0f} logA_index={:.0f} logA={:.2f}'.format(i,logA_indices_to_process[i],logA_array[logA_indices_to_process[i]]))

    spectra_ABF_subgrid=Spectra_grid[node_num*Thread_num:(node_num+1)*Thread_num]
    Predictions_grid=get_Predictions_grid(spectra_ABF_subgrid)

    os.makedirs('./results/', exist_ok=True)
    np.savez('./results/prediction/Predictions_grid_test__node_%d'%(node_num),Predictions_grid)
