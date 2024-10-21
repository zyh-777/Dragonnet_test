#!/bin/bash

#SBATCH -A sml
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=victorveitch@gmail.com
#SBATCH --mail-type=ALL

#source activate ce-dev

source D:/ZhangYihang/Dragonnet/dragonnet/dragonnet-env/Scripts/activate
export PYTHONPATH=D:/ZhangYihang/Dragonnet/dragonnet/src

DATA_DIR=D:/ZhangYihang/Dragonnet/dragonnet/dat/ihdp/csv

OUTPUT_DIR=D:/ZhangYihang/Dragonnet/dragonnet/result/ihdp

echo "  python -m experiment.ihdp_main --data_base_dir $DATA_DIR\
                                 --knob $SET\
                                 --output_base_dir $FOD"


python -m experiment.ihdp_main --data_base_dir $DATA_DIR\
                                 --knob $SET\
                                 --output_base_dir $OUTPUT_DIR
