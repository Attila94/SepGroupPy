#!/bin/sh

#SBATCH --qos=long
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4096
#SBATCH --mail-type=END
#SBATCH --gres=gpu:turing:1

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

# srun python main.py --planes 10 --seed 1 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 10 --seed 2 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 10 --seed 3 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 10 --seed 4 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 10 --seed 5 --equivariance 'Z2' --wd 1e-5

# srun python main.py --planes 20 --seed 1 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 20 --seed 2 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 20 --seed 3 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 20 --seed 4 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 20 --seed 5 --equivariance 'Z2' --wd 1e-5

# srun python main.py --planes 40 --seed 1 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 40 --seed 2 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 40 --seed 3 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 40 --seed 4 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 40 --seed 5 --equivariance 'Z2' --wd 1e-5

# srun python main.py --planes 60 --seed 1 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 60 --seed 2 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 60 --seed 3 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 60 --seed 4 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 60 --seed 5 --equivariance 'Z2' --wd 1e-5

# srun python main.py --planes 80 --seed 1 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 80 --seed 2 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 80 --seed 3 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 80 --seed 4 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 80 --seed 5 --equivariance 'Z2' --wd 1e-5

# srun python main.py --planes 100 --seed 1 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 100 --seed 2 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 100 --seed 3 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 100 --seed 4 --equivariance 'Z2' --wd 1e-5
# srun python main.py --planes 100 --seed 5 --equivariance 'Z2' --wd 1e-5



# srun python main.py --planes 25 --seed 1 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 25 --seed 2 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 25 --seed 3 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 25 --seed 4 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 25 --seed 5 --equivariance 'Z2' --separable 'g' --wd 1e-5

# srun python main.py --planes 57 --seed 1 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 57 --seed 2 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 57 --seed 3 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 57 --seed 4 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 57 --seed 5 --equivariance 'Z2' --separable 'g' --wd 1e-5

# srun python main.py --planes 120 --seed 1 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 120 --seed 2 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 120 --seed 3 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 120 --seed 4 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 120 --seed 5 --equivariance 'Z2' --separable 'g' --wd 1e-5

# srun python main.py --planes 184 --seed 1 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 184 --seed 2 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 184 --seed 3 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 184 --seed 4 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 184 --seed 5 --equivariance 'Z2' --separable 'g' --wd 1e-5

# srun python main.py --planes 247 --seed 1 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 247 --seed 2 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 247 --seed 3 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 247 --seed 4 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 247 --seed 5 --equivariance 'Z2' --separable 'g' --wd 1e-5

# srun python main.py --planes 311 --seed 1 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 311 --seed 2 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 311 --seed 3 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 311 --seed 4 --equivariance 'Z2' --separable 'g' --wd 1e-5
# srun python main.py --planes 311 --seed 5 --equivariance 'Z2' --separable 'g' --wd 1e-5



# srun python main.py --planes 5 --seed 1 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 5 --seed 2 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 5 --seed 3 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 5 --seed 4 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 5 --seed 5 --equivariance 'P4' --wd 1e-5

# srun python main.py --planes 10 --seed 1 --equivariance 'P4' --wd 1e-5 --savestr 'p4_seed1'
# srun python main.py --planes 10 --seed 2 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 10 --seed 3 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 10 --seed 4 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 10 --seed 5 --equivariance 'P4' --wd 1e-5

# srun python main.py --planes 20 --seed 1 --equivariance 'P4' --wd 1e-5 --savestr 'p4_seed1'
# srun python main.py --planes 20 --seed 2 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 20 --seed 3 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 20 --seed 4 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 20 --seed 5 --equivariance 'P4' --wd 1e-5

# srun python main.py --planes 30 --seed 1 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 30 --seed 2 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 30 --seed 3 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 30 --seed 4 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 30 --seed 5 --equivariance 'P4' --wd 1e-5

# srun python main.py --planes 40 --seed 1 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 40 --seed 2 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 40 --seed 3 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 40 --seed 4 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 40 --seed 5 --equivariance 'P4' --wd 1e-5

# srun python main.py --planes 50 --seed 1 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 50 --seed 2 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 50 --seed 3 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 50 --seed 4 --equivariance 'P4' --wd 1e-5
# srun python main.py --planes 50 --seed 5 --equivariance 'P4' --wd 1e-5

srun python main.py --planes 10 --seed 1 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 10 --seed 2 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 10 --seed 3 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 10 --seed 4 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 10 --seed 5 --equivariance 'P4' --separable 'g' --wd 1e-5

srun python main.py --planes 8 --seed 1 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 8 --seed 2 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 8 --seed 3 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 8 --seed 4 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 8 --seed 5 --equivariance 'P4' --separable 'g' --wd 1e-5

srun python main.py --planes 17 --seed 1 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 17 --seed 2 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 17 --seed 3 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 17 --seed 4 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 17 --seed 5 --equivariance 'P4' --separable 'g' --wd 1e-5

srun python main.py --planes 34 --seed 1 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 34 --seed 2 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 34 --seed 3 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 34 --seed 4 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 34 --seed 5 --equivariance 'P4' --separable 'g' --wd 1e-5

srun python main.py --planes 51 --seed 1 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 51 --seed 2 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 51 --seed 3 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 51 --seed 4 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 51 --seed 5 --equivariance 'P4' --separable 'g' --wd 1e-5

srun python main.py --planes 68 --seed 1 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 68 --seed 2 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 68 --seed 3 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 68 --seed 4 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 68 --seed 5 --equivariance 'P4' --separable 'g' --wd 1e-5

srun python main.py --planes 85 --seed 1 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 85 --seed 2 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 85 --seed 3 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 85 --seed 4 --equivariance 'P4' --separable 'g' --wd 1e-5
srun python main.py --planes 58 --seed 5 --equivariance 'P4' --separable 'g' --wd 1e-5



# srun python main.py --planes 10 --seed 1 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 10 --seed 2 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 10 --seed 3 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 10 --seed 4 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 10 --seed 5 --equivariance 'P4' --separable 'gi' --wd 1e-5

# srun python main.py --planes 30 --seed 1 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 30 --seed 2 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 30 --seed 3 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 30 --seed 4 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 30 --seed 5 --equivariance 'P4' --separable 'gi' --wd 1e-5

# srun python main.py --planes 62 --seed 1 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 62 --seed 2 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 62 --seed 3 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 62 --seed 4 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 62 --seed 5 --equivariance 'P4' --separable 'gi' --wd 1e-5

# srun python main.py --planes 94 --seed 1 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 94 --seed 2 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 94 --seed 3 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 94 --seed 4 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 94 --seed 5 --equivariance 'P4' --separable 'gi' --wd 1e-5

# srun python main.py --planes 126 --seed 1 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 126 --seed 2 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 126 --seed 3 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 126 --seed 4 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 126 --seed 5 --equivariance 'P4' --separable 'gi' --wd 1e-5

# srun python main.py --planes 158 --seed 1 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 158 --seed 2 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 158 --seed 3 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 158 --seed 4 --equivariance 'P4' --separable 'gi' --wd 1e-5
# srun python main.py --planes 158 --seed 5 --equivariance 'P4' --separable 'gi' --wd 1e-5
