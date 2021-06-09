#!/bin/sh

#SBATCH --qos=long
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4096
#SBATCH --mail-type=END
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

# srun python main.py --dataset-path '/tudelft.net/staff-bulk/ewi/insy/VisionLab/attilalengyel/datasets' --model 'resnet44_gcp4m'

srun python main.py --dataset-path '/tudelft.net/staff-bulk/ewi/insy/VisionLab/attilalengyel/datasets' --model 'resnet44_gcp4m' --hflip --rc 4
