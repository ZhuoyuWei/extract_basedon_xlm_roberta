#!/bin/sh

EXP_ID=$1
MODEL_TYPE=$2
MODEL_NAME=$3

#output dir setting
WDATA_DIR=/data/zhuoyu/extraction/wdata
OUTPUT_DIR=$WDATA_DIR/$EXP_ID
mkdir $OUTPUT_DIR


#envirement
sudo apt-get install zip -y
cd /
EXP_ROOT_DIR=/zhuoyu_exp
sudo mkdir $EXP_ROOT_DIR
sudo chmod 777 $EXP_ROOT_DIR
cd $EXP_ROOT_DIR
pwd
ls

#data
DATA_DIR=${EXP_ROOT_DIR}/data
mkdir $DATA_DIR
cd $DATA_DIR
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
cd $EXP_ROOT_DIR

#code
CODE_DIR=${EXP_ROOT_DIR}/code
mkdir $CODE_DIR
cd $CODE_DIR
git clone https://github.com/ZhuoyuWei/extract_basedon_xlm_roberta.git
cd extract_basedon_xlm_roberta
sudo pip install --upgrade torch torchvision
sudo pip install .

#running
python -m torch.distributed.launch --nproc_per_node=8 ./examples/run_squad.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --version_2_with_negative \
    --train_file $DATA_DIR/train-v2.0.json \
    --predict_file $DATA_DIR/dev-v2.0.json \
    --learning_rate 1.2e-4 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir  $OUTPUT_DIR\
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \

#evaluating
#cd scripts/extract
#python evaluate-2.0.py ${DATA_DIR}/dev-v2.0.json ${OUTPUT_DIR}/predictions_.json --out-file ${OUTPUT_DIR}/res.json
