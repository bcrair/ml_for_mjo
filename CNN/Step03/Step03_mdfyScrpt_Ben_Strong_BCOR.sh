#!/bin/bash

# Generate scripts for all lead time to compute BCOR for only strong MJO events

echo "Generating python.py ..."
# loop over leadTms
for i in {1..40}
do	
  #loop over rounds
   echo "Working on leadTm i=$i "
   cp CNN_MJO_Ben_Compute_Strong_BCOR.py temp.py
   `sed -i "s/leadTm = 15/leadTm = $i/" temp.py` 
   `mv temp.py CNN_MJO_Ben_Compute_Strong_BCOR_leadTm${i}.py`
done
