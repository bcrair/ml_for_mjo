#!/bin/bash

# create joblist for dSQ submission

> Step01_joblist.txt

for i in {1..40}
do
 echo -n  "module purge ; module load miniconda ; conda activate ml_for_mjo_env_2 ; cd  /home/bec32/project/ML_for_MJO/ViT/Ensemble_Scripts/Main_Scripts/Step01 ;" >> Step01_joblist.txt
 for j in {1..10}
 do
  echo -n  " python ViT_MJO_Ben_MDL_Train_leadTm${i}_ensm${j}.py > ViT_MJO_Ben_MDL_Train_leadTm${i}_ensm${j}.txt & " >> Step01_joblist.txt
  echo -n  "wait ; " >> Step01_joblist.txt
 done
 echo >> Step01_joblist.txt
 echo -n  "module purge ; module load miniconda ; conda activate ml_for_mjo_env_2 ; cd  /home/bec32/project/ML_for_MJO/ViT/Ensemble_Scripts/Main_Scripts/Step01 ;" >> Step01_joblist.txt
 for j in {11..20}
 do
  echo -n  " python ViT_MJO_Ben_MDL_Train_leadTm${i}_ensm${j}.py > ViT_MJO_Ben_MDL_Train_leadTm${i}_ensm${j}.txt & " >> Step01_joblist.txt
  echo -n  "wait ; " >> Step01_joblist.txt
 done
 echo >> Step01_joblist.txt
done
