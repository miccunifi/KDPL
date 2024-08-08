#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/dataset/folder"
TRAINER=CoCoOp_KDPL

# rn50_c4_ep10_batch1_ctxv1, vit_b32_c4_ep10_batch1_ctxv1
CFG=vit_b32_c4_ep10_batch1_ctxv1
SHOTS=16
LOADEP=10
SUB=new

CLASS_AGNOSTIC=False # whether to use class-agnostic KDPL (KDPL-CA) or not

DATASETS=(oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet)

for DATASET in "${DATASETS[@]}"; do
    for SEED in 1 2 3; do

        if [ "$CLASS_AGNOSTIC" = True ]; then # KDPL-CA
          COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}-CA/${CFG}/seed${SEED}
        else # KDPL
          COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        fi

        MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
        DIR=output/base2new/test_${SUB}/${COMMON_DIR}

        echo "Evaluating model"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}

    done
done
