#!/bin/bash

# submit and run scripts to compute predictions only for strong MJO events

#SBATCH -J Step03_Strong_Preds
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=5G


module purge
module load miniconda
conda activate ml_for_mjo_env_2

cd /home/bec32/project/ML_for_MJO/ViT/Ensemble_Scripts/Main_Scripts/Step03

#loop over lead Tms
for i in {1..40}
 do
 # loop over ensembles
 for j in {1..5}
  do
   python ViT_MJO_Ben_Compute_Strong_Preds_leadTm${i}_ensm${j}.py > ViT_MJO_Ben_Compute_Strong_Preds_leadTm${i}_ensm${j}.txt &
   python ViT_MJO_Ben_Compute_Strong_Preds_leadTm${i}_ensm$((j+5)).py > ViT_MJO_Ben_Compute_Strong_Preds_leadTm${i}_ensm$((j+5)).txt &
   python ViT_MJO_Ben_Compute_Strong_Preds_leadTm${i}_ensm$((j+10)).py > ViT_MJO_Ben_Compute_Strong_Preds_leadTm${i}_ensm$((j+10)).txt &
   python ViT_MJO_Ben_Compute_Strong_Preds_leadTm${i}_ensm$((j+15)).py > ViT_MJO_Ben_Compute_Strong_Preds_leadTm${i}_ensm$((j+15)).txt & 
   wait
  done
done

