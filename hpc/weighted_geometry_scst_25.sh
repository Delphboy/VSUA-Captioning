#!/bin/bash
#$ -l h_rt=24:0:0
#$ -l h_vmem=11G
#$ -pe smp 8
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -wd /data/home/eey362/VSUA-Captioning
#$ -j y
#$ -m ea
#$ -o log/
### $ -l cluster=andrena


# Load modules
module load python/3.8.5
module load cuda/11.6.2
module load cudnn/8.4.1-cuda11.6
module load java/1.8.0_241-oracle

# Activate virtual environment
source .venv/bin/activate

# Run the script
python3 train.py --batch_size 64 \
                --geometry_relation True \
                --input_json "data/cocotalk_25.json" \
                --save_checkpoint_every 2215 \
                --max_epochs 30 \
                --self_critical_after 20 \
                --beam_size 3 \
                --fixed_seed True \
                --id weighted-geometry-scst-25 \
                --relationship_weights "/data/home/eey362/VSUA-Captioning/data/relationship_weights.json" \