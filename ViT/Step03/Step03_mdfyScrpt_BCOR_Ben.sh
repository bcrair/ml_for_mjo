#!/bin/bash

# take base code to compute BCOR and generate a version for each lead time

echo "Generating python.py ..."
# loop over leadTms
for i in {1..40}
do	
  #loop over rounds
   echo "Working on leadTm i=$i "
   cp ViT_MJO_Ben_Compute_BCOR_New.py temp.py
   `sed -i "s/leadTm = 15/leadTm = $i/" temp.py` 
   `mv temp.py ViT_MJO_Ben_Compute_BCOR_New_leadTm${i}.py`
done
