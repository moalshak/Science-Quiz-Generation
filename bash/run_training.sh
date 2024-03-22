#!/bin/bash

#SBATCH --time=2-03:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --job-name=unet
#SBATCH --output=unet.out


source /home3/s3799174/machinelearning/venv/bin/activate

module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load scikit-learn/1.1.2-foss-2022a
module load matplotlib/3.7.0-gfbf-2022b 
module load Pillow/9.4.0-GCCcore-12.2.0
module load transformers
module load datasets


python questions-generator.py
