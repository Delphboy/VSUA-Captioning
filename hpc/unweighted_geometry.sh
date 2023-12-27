#!/bin/bash

#SBATCH --chdir=/jmain02/home/J2AD007/txk47/hxs67-txk47/VSUA-Captioning
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=small


# Load modules
module load cuda/10.2
module load python/anaconda3

# Activate virtual environment
conda activate vsua

# Run numbers through bc
seed=$(echo "$seed" | bc)
split=$(echo "$split" | bc)
learning_rate=$(echo "$learning_rate" | bc)
learning_rate_decay_rate=$(echo "$learning_rate_decay_rate" | bc)
dropout=$(echo "$dropout" | bc)
rnn_size=$(echo "$rnn_size" | bc)
input_encoding_size=$(echo "$input_encoding_size" | bc)
att_hid_size=$(echo "$att_hid_size" | bc)
num_layer=$(echo "$num_layer" | bc)

# Generate the data for the split and seed
python3 dataset_split_generator.py -p ${split} -s ${seed}

split=$(echo "$split * 100" | bc | sed 's/\.0*$//')

# Run the training script
python3 train.py --batch_size 32 \
                --geometry_relation True \
                --max_epochs 30 \
                --self_critical_after 20 \
                --beam_size 3 \
                --checkpoint_root /jmain02/home/J2AD007/txk47/hxs67-txk47/VSUA-Captioning/vsua-checkpoints/ \
                --id unweighted-geometry-seed=${seed}-split=${split}-learning_rate=${learning_rate}-learning_rate_decay_rate=${learning_rate_decay_rate}-dropout=${dropout}-rnn_size=${rnn_size}-input_encoding_size=${input_encoding_size}-att_hid_size=${att_hid_size}-num_layer=${num_layer} \
                --input_json "data/cocotalk_${split}_${seed}.json" \
                --input_label_h5 "data/cocotalk_${split}_${seed}_label.h5" \
                --fixed_seed ${seed} \
                --learning_rate ${learning_rate} \
                --learning_rate_decay_rate ${learning_rate_decay_rate} \
                --drop_prob_lm ${dropout} \
                --rnn_size ${rnn_size} \
                --input_encoding_size ${input_encoding_size} \
                --att_hid_size ${att_hid_size} \
                --num_layers ${num_layer} \
                --objectid_to_cocotalkid "" \