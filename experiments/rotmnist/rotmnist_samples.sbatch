#!/bin/sh

#SBATCH --qos=short
#SBATCH --partition=general
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4096
#SBATCH --mail-type=END
#SBATCH --gres=gpu:turing:1

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

# srun python main.py --planes 20 --seed 1 --equivariance 'Z2' --wd 1e-5 --samples 2000
# srun python main.py --planes 20 --seed 2 --equivariance 'Z2' --wd 1e-5 --samples 2000
# srun python main.py --planes 20 --seed 3 --equivariance 'Z2' --wd 1e-5 --samples 2000
# srun python main.py --planes 20 --seed 4 --equivariance 'Z2' --wd 1e-5 --samples 2000
# srun python main.py --planes 20 --seed 5 --equivariance 'Z2' --wd 1e-5 --samples 2000

# srun python main.py --planes 20 --seed 1 --equivariance 'Z2' --wd 1e-5 --samples 4000
# srun python main.py --planes 20 --seed 2 --equivariance 'Z2' --wd 1e-5 --samples 4000
# srun python main.py --planes 20 --seed 3 --equivariance 'Z2' --wd 1e-5 --samples 4000
# srun python main.py --planes 20 --seed 4 --equivariance 'Z2' --wd 1e-5 --samples 4000
# srun python main.py --planes 20 --seed 5 --equivariance 'Z2' --wd 1e-5 --samples 4000

# srun python main.py --planes 20 --seed 1 --equivariance 'Z2' --wd 1e-5 --samples 6000
# srun python main.py --planes 20 --seed 2 --equivariance 'Z2' --wd 1e-5 --samples 6000
# srun python main.py --planes 20 --seed 3 --equivariance 'Z2' --wd 1e-5 --samples 6000
# srun python main.py --planes 20 --seed 4 --equivariance 'Z2' --wd 1e-5 --samples 6000
# srun python main.py --planes 20 --seed 5 --equivariance 'Z2' --wd 1e-5 --samples 6000

# srun python main.py --planes 20 --seed 1 --equivariance 'Z2' --wd 1e-5 --samples 8000
# srun python main.py --planes 20 --seed 2 --equivariance 'Z2' --wd 1e-5 --samples 8000
# srun python main.py --planes 20 --seed 3 --equivariance 'Z2' --wd 1e-5 --samples 8000
# srun python main.py --planes 20 --seed 4 --equivariance 'Z2' --wd 1e-5 --samples 8000
# srun python main.py --planes 20 --seed 5 --equivariance 'Z2' --wd 1e-5 --samples 8000

# srun python main.py --planes 20 --seed 1 --equivariance 'Z2' --wd 1e-5 --samples 10000
# srun python main.py --planes 20 --seed 2 --equivariance 'Z2' --wd 1e-5 --samples 10000
# srun python main.py --planes 20 --seed 3 --equivariance 'Z2' --wd 1e-5 --samples 10000
# srun python main.py --planes 20 --seed 4 --equivariance 'Z2' --wd 1e-5 --samples 10000
# srun python main.py --planes 20 --seed 5 --equivariance 'Z2' --wd 1e-5 --samples 10000


# srun python main.py --planes 57 --seed 1 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 2000
# srun python main.py --planes 57 --seed 2 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 2000
# srun python main.py --planes 57 --seed 3 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 2000
# srun python main.py --planes 57 --seed 4 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 2000
# srun python main.py --planes 57 --seed 5 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 2000

# srun python main.py --planes 57 --seed 1 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 4000
# srun python main.py --planes 57 --seed 2 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 4000
# srun python main.py --planes 57 --seed 3 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 4000
# srun python main.py --planes 57 --seed 4 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 4000
# srun python main.py --planes 57 --seed 5 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 4000

# srun python main.py --planes 57 --seed 1 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 6000
# srun python main.py --planes 57 --seed 2 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 6000
# srun python main.py --planes 57 --seed 3 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 6000
# srun python main.py --planes 57 --seed 4 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 6000
# srun python main.py --planes 57 --seed 5 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 6000

# srun python main.py --planes 57 --seed 1 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 8000
# srun python main.py --planes 57 --seed 2 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 8000
# srun python main.py --planes 57 --seed 3 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 8000
# srun python main.py --planes 57 --seed 4 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 8000
# srun python main.py --planes 57 --seed 5 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 8000

# srun python main.py --planes 57 --seed 1 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 10000
# srun python main.py --planes 57 --seed 2 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 10000
# srun python main.py --planes 57 --seed 3 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 10000
# srun python main.py --planes 57 --seed 4 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 10000
# srun python main.py --planes 57 --seed 5 --equivariance 'Z2' --separable 'g' --wd 1e-5 --samples 10000


# srun python main.py --planes 10 --seed 1 --equivariance 'P4' --wd 1e-5 --samples 2000
# srun python main.py --planes 10 --seed 2 --equivariance 'P4' --wd 1e-5 --samples 2000
# srun python main.py --planes 10 --seed 3 --equivariance 'P4' --wd 1e-5 --samples 2000
# srun python main.py --planes 10 --seed 4 --equivariance 'P4' --wd 1e-5 --samples 2000
# srun python main.py --planes 10 --seed 5 --equivariance 'P4' --wd 1e-5 --samples 2000

# srun python main.py --planes 10 --seed 1 --equivariance 'P4' --wd 1e-5 --samples 4000
# srun python main.py --planes 10 --seed 2 --equivariance 'P4' --wd 1e-5 --samples 4000
# srun python main.py --planes 10 --seed 3 --equivariance 'P4' --wd 1e-5 --samples 4000
# srun python main.py --planes 10 --seed 4 --equivariance 'P4' --wd 1e-5 --samples 4000
# srun python main.py --planes 10 --seed 5 --equivariance 'P4' --wd 1e-5 --samples 4000

# srun python main.py --planes 10 --seed 1 --equivariance 'P4' --wd 1e-5 --samples 6000
# srun python main.py --planes 10 --seed 2 --equivariance 'P4' --wd 1e-5 --samples 6000
# srun python main.py --planes 10 --seed 3 --equivariance 'P4' --wd 1e-5 --samples 6000
# srun python main.py --planes 10 --seed 4 --equivariance 'P4' --wd 1e-5 --samples 6000
# srun python main.py --planes 10 --seed 5 --equivariance 'P4' --wd 1e-5 --samples 6000

# srun python main.py --planes 10 --seed 1 --equivariance 'P4' --wd 1e-5 --samples 8000
# srun python main.py --planes 10 --seed 2 --equivariance 'P4' --wd 1e-5 --samples 8000
# srun python main.py --planes 10 --seed 3 --equivariance 'P4' --wd 1e-5 --samples 8000
# srun python main.py --planes 10 --seed 4 --equivariance 'P4' --wd 1e-5 --samples 8000
# srun python main.py --planes 10 --seed 5 --equivariance 'P4' --wd 1e-5 --samples 8000

# srun python main.py --planes 10 --seed 1 --equivariance 'P4' --wd 1e-5 --samples 10000
# srun python main.py --planes 10 --seed 2 --equivariance 'P4' --wd 1e-5 --samples 10000
# srun python main.py --planes 10 --seed 3 --equivariance 'P4' --wd 1e-5 --samples 10000
# srun python main.py --planes 10 --seed 4 --equivariance 'P4' --wd 1e-5 --samples 10000
# srun python main.py --planes 10 --seed 5 --equivariance 'P4' --wd 1e-5 --samples 10000


srun python main.py --planes 17 --seed 1 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 2000
srun python main.py --planes 17 --seed 2 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 2000
srun python main.py --planes 17 --seed 3 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 2000
srun python main.py --planes 17 --seed 4 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 2000
srun python main.py --planes 17 --seed 5 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 2000

srun python main.py --planes 17 --seed 1 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 4000
srun python main.py --planes 17 --seed 2 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 4000
srun python main.py --planes 17 --seed 3 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 4000
srun python main.py --planes 17 --seed 4 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 4000
srun python main.py --planes 17 --seed 5 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 4000

srun python main.py --planes 17 --seed 1 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 6000
srun python main.py --planes 17 --seed 2 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 6000
srun python main.py --planes 17 --seed 3 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 6000
srun python main.py --planes 17 --seed 4 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 6000
srun python main.py --planes 17 --seed 5 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 6000

srun python main.py --planes 17 --seed 1 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 8000
srun python main.py --planes 17 --seed 2 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 8000
srun python main.py --planes 17 --seed 3 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 8000
srun python main.py --planes 17 --seed 4 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 8000
srun python main.py --planes 17 --seed 5 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 8000

srun python main.py --planes 17 --seed 1 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 10000
srun python main.py --planes 17 --seed 2 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 10000
srun python main.py --planes 17 --seed 3 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 10000
srun python main.py --planes 17 --seed 4 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 10000
srun python main.py --planes 17 --seed 5 --equivariance 'P4' --separable 'g' --wd 1e-5 --samples 10000



# srun python main.py --planes 30 --seed 1 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 2000
# srun python main.py --planes 30 --seed 2 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 2000
# srun python main.py --planes 30 --seed 3 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 2000
# srun python main.py --planes 30 --seed 4 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 2000
# srun python main.py --planes 30 --seed 5 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 2000

# srun python main.py --planes 30 --seed 1 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 4000
# srun python main.py --planes 30 --seed 2 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 4000
# srun python main.py --planes 30 --seed 3 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 4000
# srun python main.py --planes 30 --seed 4 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 4000
# srun python main.py --planes 30 --seed 5 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 4000

# srun python main.py --planes 30 --seed 1 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 6000
# srun python main.py --planes 30 --seed 2 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 6000
# srun python main.py --planes 30 --seed 3 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 6000
# srun python main.py --planes 30 --seed 4 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 6000
# srun python main.py --planes 30 --seed 5 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 6000

# srun python main.py --planes 30 --seed 1 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 8000
# srun python main.py --planes 30 --seed 2 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 8000
# srun python main.py --planes 30 --seed 3 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 8000
# srun python main.py --planes 30 --seed 4 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 8000
# srun python main.py --planes 30 --seed 5 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 8000

# srun python main.py --planes 30 --seed 1 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 10000
# srun python main.py --planes 30 --seed 2 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 10000
# srun python main.py --planes 30 --seed 3 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 10000
# srun python main.py --planes 30 --seed 4 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 10000
# srun python main.py --planes 30 --seed 5 --equivariance 'P4' --separable 'gi' --wd 1e-5 --samples 10000
