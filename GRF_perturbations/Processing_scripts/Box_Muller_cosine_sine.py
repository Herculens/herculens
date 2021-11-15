import numpy as np
from tqdm import tqdm
import argparse
import time


nx=100
ny=100
number_of_phases=100



if __name__ == '__main__':

    #Read shape of required tensor
    parser = argparse.ArgumentParser(\
        description='Number of phases and size of Fourier grid')

    parser.add_argument('--seeds_num', type=int, help='For how many random seeds you want to generate your Fourier phase images?')
    parser.add_argument('--grid_y_size', type=int, help="Number of pixels in Fourier image's y axis")
    parser.add_argument('--grid_x_size', type=int, help="Number of pixels in Fourier image's x axis")
    args = parser.parse_args()

    number_of_phases=args.seeds_num
    nx=args.grid_x_size
    ny=args.grid_y_size

    #Default case
    if number_of_phases is None:
        number_of_phases=100
    if nx is None:
        nx=100
    if ny is None:
        ny=100

    #These are U1 and U2 from Box-Muller polar transform (wiki)
    Fourier_cos_sin_tensor=np.zeros((2,number_of_phases,ny,nx))

    start_time=time.time()

    for seed in tqdm(range(number_of_phases)):
        np.random.seed(seed+1)
        for y in range(ny):
            for x in range(nx):
                s = 1.1
                while s>1. :
                    u = np.random.uniform (-1.,1.)
                    v = np.random.uniform (-1.,1.)
                    s=u**2.+v**2.
                Fourier_cos_sin_tensor[0,seed,y,x]=u
                Fourier_cos_sin_tensor[1,seed,y,x]=v

    print('BM_cos_sine generation took {:.2f} seconds'.format(time.time()-start_time))
    np.savez('../Notebooks/data/Box_Muller_cosine_sine',cosine=Fourier_cos_sin_tensor[0],sine=Fourier_cos_sin_tensor[1])
