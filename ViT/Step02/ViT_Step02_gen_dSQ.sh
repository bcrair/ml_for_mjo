#!/bin/bash

# generate joblist for dSQ submission to slurm

> Step02_joblist.txt 

for i in {1..40}
do
 echo -n  "module purge ; module load miniconda ; conda activate ml_for_mjo_env_2 ; cd  /home/bec32/project/ML_for_MJO/ViT/Ensemble_Scripts/Main_Scripts/Step02 ;" >> Step02_joblist.txt
 for j in {1..10}
 do
  echo -n  " python ViT_MJO_Ben_OBS_Train_leadTm${i}_ensm${j}.py > ViT_MJO_Ben_OBS_Train_leadTm${i}_ensm${j}.txt & " >> Step02_joblist.txt
  echo -n  " python ViT_MJO_Ben_OBS_Train_leadTm${i}_ensm$((j+10)).py > ViT_MJO_Ben_OBS_Train_leadTm${i}_ensm$((j+10)).txt & " >> Step02_joblist.txt
  echo -n "wait ;" >> Step02_joblist.txt
  done
 echo "" >> Step02_joblist.txt
done




