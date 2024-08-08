#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/dataset/folder"
TRAINER=CoCoOp_KDPL

# rn50_c4_ep10_batch1_ctxv1, vit_b32_c4_ep10_batch1_ctxv1
CFG=vit_b32_c4_ep10_batch1_ctxv1
SHOTS=16
LOADEP=10

CLASS_AGNOSTIC=False # whether to use class-agnostic KDPL (KDPL-CA) or not

# unccoment to run CROSS-DOMAIN
# DATASETS=(imagenet imagenev2 imagenet_r imagenet_a imagenet_sketch)

# unccoment to run CROSS-DATASET
DATASETS=(oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet)


for DATASET in "${DATASETS[@]}"; do
    for SEED in 1 2 3; do

        if [ "$CLASS_AGNOSTIC" = True ]; then # KDPL-CA
          DIR=output/cross_domain_and_datasets/test/${DATASET}/shots_${SHOTS}/${TRAINER}-CA/${CFG}/seed${SEED}
         else # KDPL
          DIR=output/cross_domain_and_datasets/test/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        fi

        echo "Evaluating model"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir output/cross_domain_and_datasets/train_imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
        --load-epoch ${LOADEP} \
        --eval-only

    done
done