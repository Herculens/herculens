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
from GRF_perturbations.Modules.Jax_Utils import jax_map
from GRF_perturbations.Modules.Inference import Inference_class

#CPUs parallelization
import multiprocessing
max_thread_numbers=multiprocessing.cpu_count()
#os.system('echo CPU count %d'%)

Thread_num=32
#Parallelize on several CPUs
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=%d"%(Thread_num)

grid_size=128

logA_array=np.linspace(-10.075,-6.9,grid_size)
Beta_array=np.linspace(0,4.7,0)
GRF_seeds_number=1

def Box_Muller_transform():
    s = 1.1
    while s > 1. :
        u = np.random.uniform (-1.,1.)
        v = np.random.uniform (-1.,1.)
        s = u**2. + v**2.
    fac = np.sqrt(-2.*np.log(s)/s)
    z1 = u*fac
    z2 = v*fac
    return z1,z2

def Compute_anomalies(Inference,GRF_params,Fourier_phase,Noise=True):
        get_GRF=Inference.GRF_getters(False)

        GRF_potential=get_GRF(GRF_params,Fourier_phase)

        noise_seed=jnp.round(jnp.abs(GRF_params[0]*(GRF_params[1]+1)*(Fourier_phase[0,1].real*(1e+3)+1)*1e+5)).astype(int)

        simulate_perturbed_image=Inference.Observation_conditions.perturbed_image_getter
        simulated_image=simulate_perturbed_image(GRF_potential,Inference.Observation_conditions.kwargs_data,Noise,noise_seed)

        args_fit=Inference.differentiable_fit_image(simulated_image)
        simulate_unperturbed_image=Inference.Observation_conditions.unperturbed_image_getter
        fit_image=simulate_unperturbed_image(Inference.Observation_conditions.parameters.args2kwargs(args_fit),Noise_flag=False)
        residuals=simulated_image-fit_image
        return residuals

def get_anomalies_grid(logA_to_process):

    pixel_number=100
    pixel_scale=0.08

    PSF_FWHM=0.1 #arcsec, PSF of HST
    exposure_time=2028 #COSMOS exposure
    SNR=200 #75th percentile of COSMOS gaussian noise distribution


    os.system('echo Precompiling Observation and Inference classes')
    start_time=time.time()
    Observation_conditions=Observation_conditions_class(pixel_number,pixel_scale,PSF_FWHM,SNR,exposure_time)
    Inference=Inference_class(Observation_conditions,GRF_seeds_number=GRF_seeds_number)
    time_string="elapsed time {:.5f}".format(time.time()-start_time)
    os.system('echo Compilation took:  %s seconds' %time_string)

    compute_anomalies_pure=lambda logA,Beta,Fourier_phase: Compute_anomalies(jnp.array([logA,Beta]),Fourier_phase)
    compute_anomalies_seeds= lambda logA,Beta: jax_map(lambda Fourier_phase: compute_anomalies_pure(logA,Beta,Fourier_phase),Inference.Fourier_phase_tensor)
    compute_anomalies_Betas= jax.jit(lambda logA: jax_map(lambda Beta: compute_anomalies_seeds(logA,Beta),Beta_array))

    #Function computing loop over GRF amplitudes logA in parallel
    compute_anomalies_grid_pmapped=jax.pmap(compute_anomalies_Betas)

    os.system('echo Computing grid')

    start_time=time.time()
    anomalies_grid=compute_anomalies_grid_pmapped(logA_to_process)
    end_time=time.time()

    time_string="elapsed time {:.5f}".format(end_time-start_time)
    os.system('echo Grid processing time: %s seconds' %(time_string))

    return anomalies_grid

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

    pixel_number=100
    pixel_scale=0.08

    PSF_FWHM=0.1 #arcsec, PSF of HST
    exposure_time=2028 #COSMOS exposure
    SNR=200 #75th percentile of COSMOS gaussian noise distribution


    os.system('echo Precompiling Observation and Inference classes')
    start_time=time.time()
    Observation_conditions=Observation_conditions_class(pixel_number,pixel_scale,PSF_FWHM,SNR,exposure_time)
    Inference=Inference_class(Observation_conditions,GRF_seeds_number=GRF_seeds_number)
    time_string="elapsed time {:.5f}".format(time.time()-start_time)
    os.system('echo Compilation took:  %s seconds' %time_string)


    logA_to_process=logA_array[node_num*Thread_num:(node_num+1)*Thread_num]

    os.system('echo Process logA:')
    for i in range(len(logA_to_process)):
        os.system('echo i={:.0f} logA={:.2f}'.format(i,logA_to_process[i]))

    anomalies_grid=get_anomalies_grid(logA_to_process)

    os.makedirs('./results/', exist_ok=True)
    np.savez('./results/anomalies_grid_node_%d'%(node_num),anomalies_grid)
