#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
# MODE=$3
MODEL="ts-concept-learner-$NUM"

#-------------------------------------------------------------------------------#
# CLEVR-Hans3

# Set epochs back to 50
CUDA_VISIBLE_DEVICES=$DEVICE python nesy_cl_p2s.py --dataset p2s --mode train \
--concept tsfresh --load-ts --ts-setting slow \
--epochs 50 --name $MODEL --lr 0.0001 --batch-size 128 --seed 0 --num-workers 4


### SAX
# --concept sax --n-segments 7 --alphabet-size 3 \


# ### tsfresh
# --load-ts
# fast | mid| slow

# --no-cuda  # for cpu usage

# ## Modes
# gridsearch | train | test | plot


# CUDA_VISIBLE_DEVICES=$DEVICE python nesy_cl_p2s.py --dataset p2s --concept tsfresh --n-segments 8 --alphabet-size 4 \
# --epochs 50 --name $MODEL --lr 0.0001 --batch-size 128 --seed 0 --num-workers 4 --mode train \




# For gpu (old version)
#CUDA_VISIBLE_DEVICES=$DEVICE python train_nesy_concept_learner_clevr_hans.py --data-dir $DATA --dataset $DATASET \
#--epochs 50 --name $MODEL --lr 0.0001 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 0 \
#--mode train

# For cpu
#CUDA_VISIBLE_DEVICES=$DEVICE python train_nesy_concept_learner_clevr_hans.py --data-dir $DATA --dataset $DATASET \
#--epochs 50 --name $MODEL --lr 0.0001 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 0 \
#--mode train --no-cuda
