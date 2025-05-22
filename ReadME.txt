Files are splitup by model: Convolutional Neural Network (CNN), Vision Transformer (ViT), and Vision Transformer with Current Index (ViTCI).

Since I never ran the full ViTCI model there is only training code for ViTCI, but ViT and CNN include full slurm scripts.

Step01 corresponds to pretraining on simulation data, Step02 corresponds to fine-tuning on observational data.
The base Step02 script also makes forecasts for all MJO events.

Step03 is post processing. There is a set of scripts to compute the BCOR value for all MJO events (the predictions were made automatically in Step02).
There are also scripts in Step03 to compute predictions for just Strong MJO events, and then subsequently compute the BCOR values for predictions
made just for Strong MJO events.
