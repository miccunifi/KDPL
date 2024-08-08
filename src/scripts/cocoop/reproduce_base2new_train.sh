#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/dataset/folder"
TRAINER=CoCoOp

# rn50_c4_ep10_batch1_ctxv1, vit_b32_c4_ep10_batch1_ctxv1
CFG=vit_b32_c4_ep10_batch1_ctxv1
SHOTS=16

DATASETS=(oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet)

for DATASET in "${DATASETS[@]}"; do
    for SEED in 1 2 3; do

        DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}.  Skip this job"
        else
            echo "Run this job and save the output to ${DIR}"
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base
        fi

    done
done
