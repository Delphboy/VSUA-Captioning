#!/bin/bash
#$ -l h_rt=24:0:0
#$ -l h_vmem=11G
#$ -pe smp 8
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -wd /data/home/eey362/image-captioning
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

# pip install
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
python3 -m pip install -r requirements.txt


# Run the script
python3 train.py --batch_size 32 \
                --geometry_relation True
                --save_checkpoint_every 3540 \
                --max_epochs 30 \
                --self_critical_after 20 \
                --beam_size 3 \
                --fixed_seed True \
                --id weighted-geometry \
                --relationship_weights "/dir/to/weights" \