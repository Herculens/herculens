#!/bin/bash
#SBATCH --mem=64G
#SBATCH --exclusive

source /astro/scripts/setup-modules.sh
module add anaconda
#echo $PYTHONPATH
export PYTHONPATH=''

source /astro/soft/common/anaconda/3-2020.07/bin/activate
conda init bash
conda activate env_jax

#ls
#conda list

srun python Spectra_grid.py --node_num 0
