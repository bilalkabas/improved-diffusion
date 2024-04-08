DATA_DIR="/home/bkabas/raid/data/IXI"
SOURCE_MODALITY="PD"
TARGET_MODALITY="T1"
export MANUAL_RANK=3

MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --dropout 0.3 --learn_sigma False"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --predict_xstart True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 2"

DATASET_NAME=$(basename "$DATA_DIR")
DATE=$(date '+%Y-%m-%d-%H-%M-%S')
export OPENAI_LOGDIR="/home/bkabas/raid/improved-diffusion/log/$DATASET_NAME-$SOURCE_MODALITY-$TARGET_MODALITY-$DATE"

python scripts/image_train.py --data_dir $DATA_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --source_modality $SOURCE_MODALITY --target_modality $TARGET_MODALITY