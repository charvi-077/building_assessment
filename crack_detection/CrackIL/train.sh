#!/bin/bash
#SBATCH -A mobility_arfs
#SBATCH --partition ihub
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode101
# nohup python train.py --save_dir '/scratch/kushagra0301/Crack_IL_step_1_40' --num_epochs 150 --source_dataset_path '/scratch/kushagra0301/CrackDataset40' > "crack_IL_step_1_40.log" 2>&1
# nohup python train.py --save_dir '/scratch/kushagra0301/target_as_source' --num_epochs 150 --source_dataset_path '/scratch/kushagra0301/IIITDataset' > "target_as_source.log" 2>&1
python homework4_part2.py 