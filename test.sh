DATA_DIR="/home/bkabas/raid/data/IXI"
SOURCE_MODALITY="PD"
TARGET_MODALITY="T1"
MODEL_PATH="/home/bkabas/raid/improved-diffusion/log/BRATS-FLAIR-T2-2024-07-11-15-23-10/model000020.pt"
BATCH_SIZE=1
export MANUAL_RANK=0

MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --dropout 0.3 --learn_sigma False"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --predict_xstart True"

DATASET_NAME=$(basename "$DATA_DIR")
DATE=$(date '+%Y-%m-%d-%H-%M-%S')
export OPENAI_LOGDIR="/home/bkabas/raid/improved-diffusion/log/$DATASET_NAME-$SOURCE_MODALITY-$TARGET_MODALITY-$DATE"

python scripts/image_sample.py --data_dir $DATA_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --source_modality $SOURCE_MODALITY --target_modality $TARGET_MODALITY --model_path $MODEL_PATH --batch_size $BATCH_SIZE