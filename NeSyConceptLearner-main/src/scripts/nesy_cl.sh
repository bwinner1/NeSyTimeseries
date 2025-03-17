#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
# MODE=$3
MODEL="ts-concept-learner-$NUM"

#-------------------------------------------------------------------------------#
# CLEVR-Hans3

# CUDA_LAUNCH_BLOCKING=1 
CUDA_VISIBLE_DEVICES=$DEVICE python nesy_cl.py --dataset p2s \
--mode train --num-tries 5 \
--concept sax --n-segments 32 --alphabet-size 10 --n-heads 4 --set-transf-hidden 128 \
--epochs 50 --name $MODEL --lr 0.0001 --batch-size 6 --seed 2 --num-workers 0 \
--xil



# --explain 

# todo batch-size to 64

# Added one zero 

# --load-tsf \

# ### SAX
# --concept sax --n-segments 32 --alphabet-size 10 --n-heads 4 --set-transf-hidden 128 \
# --concept sax --n-segments 7 --alphabet-size 3 --n-heads 4 --set-transf-hidden 128 \
# --epochs 50 --name $MODEL --lr 0.0001 --batch-size 64 --seed 42 --num-workers 0 \

# ### tsfresh
# --concept tsfresh --ts-setting slow --n-heads 4 --set-transf-hidden 128 \
# --load-tsf \

# ### VQShape
# --concept vqshape --n-heads 4 --set-transf-hidden 32 \
# --epochs 50 --name $MODEL --lr 0.0025 --batch-size 128 --seed 42 --num-workers 0 \


# --explain --explain-all --xil

# --load-tsf --filter-tsf --normalize-tsf
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
