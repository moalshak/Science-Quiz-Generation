#!/bin/bash

#SBATCH --time=0-06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --job-name=questions
#SBATCH --output=questions.out


source /home3/s3799174/machinelearning/venv/bin/activate

module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/2.1.2-foss-2022b
module load matplotlib/3.7.0-gfbf-2022b
module load Pillow/9.4.0-GCCcore-12.2.0
pip install 'transformers'
pip install 'datasets'


python questions-generator.py
