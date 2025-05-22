#!/bin/bash

# Script to generate a dSQ job submission list

> Step02_joblist.txt 

echo -n  "module purge ; module load miniconda ; conda activate ml_for_mjo_env_2 ; cd  /home/bec32/project/ML_for_MJO/CNN/Ensemble_Scripts/Main_Scripts/Step02 ;" >> Step02_joblist.txt
for i in {1..10}
do
 for j in {1..10}
 do
  echo -n  " python CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm${j}.py > CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm${j}.txt & " >> Step02_joblist.txt
  echo -n  " python CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm$((j+10)).py > CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm$((j+10)).txt & " >> Step02_joblist.txt
  echo -n  "wait ; " >> Step02_joblist.txt
 done
done
echo >> Step02_joblist.txt

echo -n  "module purge ; module load miniconda ; conda activate ml_for_mjo_env_2 ; cd  /home/bec32/project/ML_for_MJO/CNN/Ensemble_Scripts/Main_Scripts/Step02 ;" >> Step02_joblist.txt
for i in {11..20}
do
 for j in {1..10}
 do
  echo -n  " python CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm${j}.py > CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm${j}.txt & " >> Step02_joblist.txt
  echo -n  " python CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm$((j+10)).py > CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm$((j+10)).txt & " >> Step02_joblist.txt
  echo -n  "wait ; " >> Step02_joblist.txt
 done
done
echo >> Step02_joblist.txt

echo -n  "module purge ; module load miniconda ; conda activate ml_for_mjo_env_2 ; cd  /home/bec32/project/ML_for_MJO/CNN/Ensemble_Scripts/Main_Scripts/Step02 ;" >> Step02_joblist.txt
for i in {21..30}
do
 for j in {1..10}
 do
  echo -n  " python CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm${j}.py > CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm${j}.txt & " >> Step02_joblist.txt
  echo -n  " python CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm$((j+10)).py > CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm$((j+10)).txt & " >> Step02_joblist.txt
  echo -n  "wait ; " >> Step02_joblist.txt
 done
done
echo >> Step02_joblist.txt

echo -n  "module purge ; module load miniconda ; conda activate ml_for_mjo_env_2 ; cd  /home/bec32/project/ML_for_MJO/CNN/Ensemble_Scripts/Main_Scripts/Step02 ;" >> Step02_joblist.txt
for i in {31..40}
do
 for j in {1..10}
 do
  echo -n  " python CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm${j}.py > CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm${j}.txt & " >> Step02_joblist.txt
  echo -n  " python CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm$((j+10)).py > CNN_MJO_Ben_OBS_Train_leadTm${i}_ensm$((j+10)).txt & " >> Step02_joblist.txt
  echo -n  "wait ; " >> Step02_joblist.txt
 done
done
echo >> Step02_joblist.txt



