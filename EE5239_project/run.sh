#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.out
#SBATCH --job-name=name_PLACEHOLDER
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100-8

module load miniforge
source activate /projects/standard/ee5239/shared/ee5239_proj2
cd /users/0/avela019/EE5239_project/sam2/EE5239_project || exit

python train.py \
--name name_PLACEHOLDER \
--lr lr_PLACEHOLDER \
--batch_size batch_size_PLACEHOLDER \
--LR_sch LR_sch_PLACEHOLDER \
--n_train n_train_PLACEHOLDER \
--n_val n_val_PLACEHOLDER \
--peft peft_PLACEHOLDER \
--lora_rank lora_rank_PLACEHOLDER \
--model model_PLACEHOLDER \
--wandb_group wandb_group_PLACEHOLDER \
--target_layers target_layers_PLACEHOLDER \
--tqdm 0 \
--wandb 1

exit