#!/bin/bash

# submit joblist to slurm with dSQ

module load dSQ

dsq --job-file Step02_joblist.txt -t 30:00 --cpus-per-task=2 --partition=gpu --gpus=a5000:1 --mem-per-gpu=5G -J ViT_Step02_dSQ --max-jobs 20 --submit








