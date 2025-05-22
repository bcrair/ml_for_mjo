#!/bin/bash

# Script to submit dSQ job array to slurm

module load dSQ

dsq --job-file Step02_joblist.txt -t 1:00:00 --cpus-per-task=2 --partition=gpu --gpus=a5000:1 --mem-per-gpu=5G -J CNN_Step02_dSQ --max-jobs 4 --submit








