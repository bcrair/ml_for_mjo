#!/bin/bash

# take main training script and generate versions for each lead time and ensemble number

#+++++++++++++++++++++++++++
# Sectn A: modify .py files
#+++++++++++++++++++++++++++
echo "Generating python.py ..."
# loop over leadTms
for i in {1..40}
do
  # loop over ensembles
  for j in {1..20}
  do
    echo "Working on leadTm i=$i and ensm j=$j "
    cp ViT_MJO_Ben_OBS_Train.py temp.py
    `sed -i "s/seed_num = 1/seed_num = $j/" temp.py`
    `sed -i "s/leadTm = 15/leadTm = $i/" temp.py`
    `mv temp.py ViT_MJO_Ben_OBS_Train_leadTm${i}_ensm${j}.py`
  done
done

