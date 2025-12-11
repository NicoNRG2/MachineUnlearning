#!/bin/bash

# CONFIGURATION OF UNLEARNING PHASE: ADJUST THESE PARAMETERS BASED ON POISON RATE:
# 1: Experiment name and poison rate
EXPERIMENT_NAME="unlearn_50_poison" # for 50% poison
POISON_RATE=0.50                    # for 50% poison

# EXPERIMENT_NAME="unlearn_20_poison" # for 20% poison
# POISON_RATE=0.20                    # for 20% poison

# 2: Paths
POISONED_MODEL="runs/50_poison/gan2:pre&gan3:pre&sdXL:pre&real:pre/checkpoints/best.pt"  # for 50% poison
#POISONED_MODEL="runs/20_poison/gan2:pre&gan3:pre&sdXL:pre&real:pre/checkpoints/best.pt"  # for 20% poison
DATA_ROOT="/media/NAS/TrueFake"
SPLIT_PATH="../splits"

# 3: Unlearning hyperparameters
NUM_EPOCHS=3  # For 50% poison
START_LR=0.05 # For 50% poison
END_LR=0.01   # For 50% poison

# NUM_EPOCHS=10   # For 20% poison
# START_LR=0.001  # For 20% poison
# END_LR=0.0001   # For 20% poison


# END OF CONFIGURATION

RETAINED_VAR=0.95  # Keep 95% of variance in SVD
OFFSET=0.1
LOSS1_W=1.0
LOSS2_W=0.2

# Run unlearning
python unlearn_trueface.py \
    --name ${EXPERIMENT_NAME} \
    --poisoned_model ${POISONED_MODEL} \
    --poison_rate ${POISON_RATE} \
    --data_root ${DATA_ROOT} \
    --split_path ${SPLIT_PATH} \
    --data "gan2:pre&gan3:pre&sdXL:pre&real:pre" \
    --model nodown \
    --freeze \
    --num_epochs ${NUM_EPOCHS} \
    --start_lr ${START_LR} \
    --end_lr ${END_LR} \
    --retained_var ${RETAINED_VAR} \
    --offset ${OFFSET} \
    --loss1_w ${LOSS1_W} \
    --loss2_w ${LOSS2_W} \
    --device cuda:0 \
    --batch_size 16 \
    --num_threads 8 \
    --resize_prob 0.2 \
    --resize_size 512 \
    --resize_scale 0.2 1.0 \
    --resize_ratio 0.75 1.33 \
    --jpeg_prob 0.2 \
    --jpeg_qual 30 100 \
    --blur_prob 0.2 \
    --blur_sigma 1e-6 3 \
    --patch_size 96 \
    --save_model \
    --output_dir runs/unlearning \
    --early_stop_thres 5.0

echo "Unlearning completed! Results in: runs/unlearning/${EXPERIMENT_NAME}/"