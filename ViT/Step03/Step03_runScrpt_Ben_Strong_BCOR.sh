#!/bin/bash

# run scripts to compute BCOR only for strong MJO events

#SBATCH -J Step03_BC
#SBATCH -t 10:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=devel
#SBATCH --mem-per-cpu=20G


module purge
module load miniconda
conda activate ml_for_mjo_env_2

cd /home/bec32/project/ML_for_MJO/ViT/Ensemble_Scripts/Main_Scripts/Step03

#process lead lead 1 first to setup output file
python ViT_MJO_Ben_Compute_Strong_BCOR_leadTm1.py > ViT_MJO_Ben_Compute_Strong_BCOR_leadTm1.txt

# loop over leadTms
for i in {2..40}
do
  python ViT_MJO_Ben_Compute_Strong_BCOR_leadTm${i}.py > ViT_MJO_Ben_Compute_Strong_BCOR_leadTm${i}.txt &
done

wait

