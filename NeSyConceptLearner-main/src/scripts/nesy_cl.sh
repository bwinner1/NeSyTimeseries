#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
MODEL="ts-concept-learner-$NUM"

#-------------------------------------------------------------------------------#

# CUDA_LAUNCH_BLOCKING=1 
CUDA_VISIBLE_DEVICES=$DEVICE python nesy_cl.py --dataset p2s \
    --mode train --num-tries 5 \
    --concept sax --n-segments 8 --alphabet-size 4 --n-heads 4 --set-transf-hidden 128 \
    --epochs 50 --name $MODEL --lr 0.0001 --batch-size 64 --seed 42 --num-workers 0 \
    --explain

# In the following, the different args with default values for each summarizer are shown.
# More detailed explanations for each argument can be found in args.py

# ### Exaplanations:
# Enable xai explanations (needed for further xai functions)
# --explain
# Save xai outputs as pdfs, in folder xai
# --save-pdf
# Enable global xai explanations (only for tsfresh)
# --explain-all

# ### Modes
# gridsearch | train 

# In gridsearch, different settings can be experimented with. These are set in the gridsearch method
# in nesy_cl.py. The same parameter configuration is run 5 times per default, as specified by num-tries
# At the end of each configuration, the result are saved under the gridsearch folder. 
# In train mode, the following example parameters for each summarizer can be used: 


# ### SAX
# --concept sax --n-segments 32 --alphabet-size 32 --n-heads 4 --set-transf-hidden 128 \
# --epochs 50 --name $MODEL --lr 0.0001 --batch-size 64 --seed 42 --num-workers 0 \

# If p2s should be used in decoy mode
# --p2s-decoy \

# If xil should be applied. If yes, the corresponding xil weight
# --xil --xil-weight 0.001


# ### tsfresh
# --concept tsfresh --ts-setting mid --n-heads 4 --set-transf-hidden 128 \
# --load-tsf \

# ts-setting options
# --ts-setting: fast | mid | slow
# --load-tsf --filter-tsf --normalize-tsf


# ### VQShape
# --concept vqshape --n-heads 4 --set-transf-hidden 32 \
# --epochs 50 --name $MODEL --lr 0.0025 --batch-size 64 --seed 42 --num-workers 0 \

# --no-cuda  # for cpu usage
