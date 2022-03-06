#!/bin/bash
#SBATCH --mem=64G
#SBATCH --exclusive

# Call conda using lesta cluster's scripts
source /astro/scripts/setup-modules.sh
module add anaconda
# on lesta cluster PYTHONPATH points at py2.7 packages first. Get rid of that, cause we use py 3.*
export PYTHONPATH=''

# Activate preassembled conda env
source /astro/soft/common/anaconda/3-2020.07/bin/activate
conda init bash
conda activate env_jax

# If in the script you set up 'nodes' variable to 'N',
# when you call this script on the node number 'n' you need to specify 'n' here
# 'n': 'n' is in [0,1,2,...,'N'-1]
srun python Spectra_grid.py --node_num 0
