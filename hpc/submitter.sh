#!/bin/bash

declare seeds=(1)
declare splits=(1)


declare learning_rates=(0.0003)
declare learning_rate_decay_rates=(0.8)
declare dropouts=(0.1)
declare rnn_sizes=(1000)
declare input_encoding_sizes=(1000)
declare att_hid_sizes=(1000)
declare num_layers=(1)


for seed in "${seeds[@]}"; do
    for split in "${splits[@]}"; do
        for learning_rate in "${learning_rates[@]}"; do
            for learning_rate_decay_rate in "${learning_rate_decay_rates[@]}"; do
                for dropout in "${dropouts[@]}"; do
                    for rnn_size in "${rnn_sizes[@]}"; do
                        for input_encoding_size in "${input_encoding_sizes[@]}"; do
                            for att_hid_size in "${att_hid_sizes[@]}"; do
                                for num_layer in "${num_layers[@]}"; do
                                    echo "Submitting job with seed: $seed, split: $split, learning_rate: $learning_rate, learning_rate_decay_rate: $learning_rate_decay_rate, dropout: $dropout, rnn_size: $rnn_size, input_encoding_size: $input_encoding_size, att_hid_size: $att_hid_size, num_layer: $num_layer"
                                    # qsub -N "weighted_geometry_seed-$seed-split-$split-learning_rate-$learning_rate-learning_rate_decay_rate-$learning_rate_decay_rate-dropout-$dropout-rnn_size-$rnn_size-input_encoding_size-$input_encoding_size-att_hid_size-$att_hid_size-num_layer-$num_layer" -v seed=$seed,split=$split,learning_rate=$learning_rate,learning_rate_decay_rate=$learning_rate_decay_rate,dropout=$dropout,rnn_size=$rnn_size,input_encoding_size=$input_encoding_size,att_hid_size=$att_hid_size,num_layer=$num_layer weighted_geometry.qsub
                                    # qsub -N "weighted_semantic_seed-$seed-split-$split-learning_rate-$learning_rate-learning_rate_decay_rate-$learning_rate_decay_rate-dropout-$dropout-rnn_size-$rnn_size-input_encoding_size-$input_encoding_size-att_hid_size-$att_hid_size-num_layer-$num_layer" -v seed=$seed,split=$split,learning_rate=$learning_rate,learning_rate_decay_rate=$learning_rate_decay_rate,dropout=$dropout,rnn_size=$rnn_size,input_encoding_size=$input_encoding_size,att_hid_size=$att_hid_size,num_layer=$num_layer weighted_semantic.qsub
                                    # qsub -N "unweighted_geometry_seed-$seed-split-$split-learning_rate-$learning_rate-learning_rate_decay_rate-$learning_rate_decay_rate-dropout-$dropout-rnn_size-$rnn_size-input_encoding_size-$input_encoding_size-att_hid_size-$att_hid_size-num_layer-$num_layer" -v seed=$seed,split=$split,learning_rate=$learning_rate,learning_rate_decay_rate=$learning_rate_decay_rate,dropout=$dropout,rnn_size=$rnn_size,input_encoding_size=$input_encoding_size,att_hid_size=$att_hid_size,num_layer=$num_layer unweighted_geometry.qsub
                                    # qsub -N "unweighted_semantic_seed-$seed-split-$split-learning_rate-$learning_rate-learning_rate_decay_rate-$learning_rate_decay_rate-dropout-$dropout-rnn_size-$rnn_size-input_encoding_size-$input_encoding_size-att_hid_size-$att_hid_size-num_layer-$num_layer" -v seed=$seed,split=$split,learning_rate=$learning_rate,learning_rate_decay_rate=$learning_rate_decay_rate,dropout=$dropout,rnn_size=$rnn_size,input_encoding_size=$input_encoding_size,att_hid_size=$att_hid_size,num_layer=$num_layer unweighted_semantic.qsub


                                    sbatch --job-name=weighted_geometry_seed-$seed-split-$split-learning_rate-$learning_rate-learning_rate_decay_rate-$learning_rate_decay_rate-dropout-$dropout-rnn_size-$rnn_size-input_encoding_size-$input_encoding_size-att_hid_size-$att_hid_size-num_layer-$num_layer --export=seed=$seed,split=$split,learning_rate=$learning_rate,learning_rate_decay_rate=$learning_rate_decay_rate,dropout=$dropout,rnn_size=$rnn_size,input_encoding_size=$input_encoding_size,att_hid_size=$att_hid_size,num_layer=$num_layer --output=/jmain02/home/J2AD007/txk47/hxs67-txk47/VSUA-Captioning/log/weighted_geometric_seed-$seed-split-$split-learning_rate-$learning_rate-learning_rate_decay_rate-$learning_rate_decay_rate-dropout-$dropout-rnn_size-$rnn_size-input_encoding_size-$input_encoding_size-att_hid_size-$att_hid_size-num_layer-$num_layer.out weighted_geometry.sh
                                    sbatch --job-name=weighted_semantic_seed-$seed-split-$split-learning_rate-$learning_rate-learning_rate_decay_rate-$learning_rate_decay_rate-dropout-$dropout-rnn_size-$rnn_size-input_encoding_size-$input_encoding_size-att_hid_size-$att_hid_size-num_layer-$num_layer --export=seed=$seed,split=$split,learning_rate=$learning_rate,learning_rate_decay_rate=$learning_rate_decay_rate,dropout=$dropout,rnn_size=$rnn_size,input_encoding_size=$input_encoding_size,att_hid_size=$att_hid_size,num_layer=$num_layer --output=/jmain02/home/J2AD007/txk47/hxs67-txk47/VSUA-Captioning/log/weighted_semantic_seed-$seed-split-$split-learning_rate-$learning_rate-learning_rate_decay_rate-$learning_rate_decay_rate-dropout-$dropout-rnn_size-$rnn_size-input_encoding_size-$input_encoding_size-att_hid_size-$att_hid_size-num_layer-$num_layer.out weighted_semantic.sh
                                    # sbatch --job-name=unweighted_geometry_seed-$seed-split-$split-learning_rate-$learning_rate-learning_rate_decay_rate-$learning_rate_decay_rate-dropout-$dropout-rnn_size-$rnn_size-input_encoding_size-$input_encoding_size-att_hid_size-$att_hid_size-num_layer-$num_layer --export=seed=$seed,split=$split,learning_rate=$learning_rate,learning_rate_decay_rate=$learning_rate_decay_rate,dropout=$dropout,rnn_size=$rnn_size,input_encoding_size=$input_encoding_size,att_hid_size=$att_hid_size,num_layer=$num_layer --output=/jmain02/home/J2AD007/txk47/hxs67-txk47/VSUA-Captioning/log/unweighted_geometric_seed-$seed-split-$split-learning_rate-$learning_rate-learning_rate_decay_rate-$learning_rate_decay_rate-dropout-$dropout-rnn_size-$rnn_size-input_encoding_size-$input_encoding_size-att_hid_size-$att_hid_size-num_layer-$num_layer.out unweighted_geometry.sh
                                    # sbatch --job-name=unweighted_semantic_seed-$seed-split-$split-learning_rate-$learning_rate-learning_rate_decay_rate-$learning_rate_decay_rate-dropout-$dropout-rnn_size-$rnn_size-input_encoding_size-$input_encoding_size-att_hid_size-$att_hid_size-num_layer-$num_layer --export=seed=$seed,split=$split,learning_rate=$learning_rate,learning_rate_decay_rate=$learning_rate_decay_rate,dropout=$dropout,rnn_size=$rnn_size,input_encoding_size=$input_encoding_size,att_hid_size=$att_hid_size,num_layer=$num_layer --output=/jmain02/home/J2AD007/txk47/hxs67-txk47/VSUA-Captioning/log/unweighted_semantic_seed-$seed-split-$split-learning_rate-$learning_rate-learning_rate_decay_rate-$learning_rate_decay_rate-dropout-$dropout-rnn_size-$rnn_size-input_encoding_size-$input_encoding_size-att_hid_size-$att_hid_size-num_layer-$num_layer.out unweighted_semantic.sh

                                done
                            done
                        done
                    done
                done
            done
        done
    done
done