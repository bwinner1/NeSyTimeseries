#!/bin/bash

# old version
#DEVICE=$1
#NUM=$2
#DATA=$3
#MODEL="concept-learner-$NUM"
#DATASET=clevr-hans-state
#OUTPATH="out/clevr-state/$MODEL-$ITER"


# CUDA DEVICE ID
DEVICE=$1
NUM=$2
#MODE=$3
MODEL="ts-concept-learner-$NUM"

#-------------------------------------------------------------------------------#
# CLEVR-Hans3

# For gpu
CUDA_VISIBLE_DEVICES=$DEVICE python train_cl_p2s.py --concept sax --n_segments 8 --alphabet_size 4\
--epochs 50 --name $MODEL --lr 0.0001 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 0 \
--mode train

# For gpu (old version)
#CUDA_VISIBLE_DEVICES=$DEVICE python train_nesy_concept_learner_clevr_hans.py --data-dir $DATA --dataset $DATASET \
#--epochs 50 --name $MODEL --lr 0.0001 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 0 \
#--mode train

# For cpu
#CUDA_VISIBLE_DEVICES=$DEVICE python train_nesy_concept_learner_clevr_hans.py --data-dir $DATA --dataset $DATASET \
#--epochs 50 --name $MODEL --lr 0.0001 --batch-size 128 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --seed 0 \
#--mode train --no-cuda
