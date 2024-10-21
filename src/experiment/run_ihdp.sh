#!/usr/bin/env bash


options=(
    dragonnet
    tarnet

)



for i in ${options[@]}; do
    echo $i
    python -m experiment.ihdp_main --data_base_dir D:/ZhangYihang/Dragonnet/dragonnet/dat/ihdp/csv\
                                 --knob $i\
                                 --output_base_dir D:/ZhangYihang/Dragonnet/dragonnet/result/ihdp\


done
