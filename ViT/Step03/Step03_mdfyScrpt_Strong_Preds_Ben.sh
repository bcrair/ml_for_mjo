#!/bin/bash

# create for each ensemble and lead time a script to make predictions only for strong MJO events

echo "Generating python.py ..."
# loop over leadTms
for i in {1..40}
do	
   for j in {1..20}
   do 
    echo "Working on leadTm i=$i and ensm j=$j "
    cp ViT_MJO_Ben_Compute_Strong_Preds.py temp.py
    `sed -i "s/seed_num = 1/seed_num = $j/" temp.py`
    `sed -i "s/leadTm = 15/leadTm = $i/" temp.py` 
    `mv temp.py ViT_MJO_Ben_Compute_Strong_Preds_leadTm${i}_ensm${j}.py`
   done
done
