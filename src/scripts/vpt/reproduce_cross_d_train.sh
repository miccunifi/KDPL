#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/dataset/folder"
TRAINER=VPT

CFG=vit_b32_c2_ep5_batch4_4
SHOTS=16

DATASET=imagenet

for SEED in 1 2 3; do

    DIR=output/cross_domain_and_datasets/train/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi

done
