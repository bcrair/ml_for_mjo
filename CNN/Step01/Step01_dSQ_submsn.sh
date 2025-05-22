#!/bin/bash

# Script to submit dSQ batch job to slurm

module load dSQ

dsq --job-file Step01_joblist.txt -t 1:00:00 --cpus-per-task=2 --partition=gpu --gpus=a5000:1 --mem-per-cpu=10G -J Step01_dSQ --max-jobs 20 --submit


