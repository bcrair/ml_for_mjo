#!/bin/bash

# submit dSQ joblist to slurm

module load dSQ

dsq --job-file Step01_joblist.txt -t 5:00:00 --cpus-per-task=1 --partition=gpu --gpus=a5000:1 --mem-per-cpu=10G -J ViT_Step01_dSQ --max-jobs 20 --submit


