#!/bin/bash

# Script to generate files to compute BCOR

echo "Generating python.py ..."
# loop over leadTms
for i in {1..40}
do	
  #loop over rounds
   echo "Working on leadTm i=$i "
   cp CNN_MJO_Ben_Compute_BCOR.py temp.py
   `sed -i "s/leadTm = 15/leadTm = $i/" temp.py` 
   `mv temp.py CNN_MJO_Ben_Compute_BCOR_New_leadTm${i}.py`
done
