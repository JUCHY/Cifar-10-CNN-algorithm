#!/bin/bash
#
#SBATCH --job-name=assign5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=6GB
#SBATCH --mail-type=END
##SBATCH --mail-user=jnt297@nyu.edu
#SBATCH --output=assign5_output.out

cd /home/$USER/assign5
source /share/apps/anaconda3/2019.10/etc/profile.d/conda.sh
conda activate Ptorch
python assign5.py