#!/bin/bash
#SBATCH -A mobility_arfs
#SBATCH --partition ihub
#SBATCH -n 20
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode105

# python eval.py --result_dir '/scratch/kushagra0301/Crack_IL_step_1/target_model_0_best' --weight '/scratch/kushagra0301/Crack_IL_step_1/best_model.pth' --source_dataset_path '/scratch/kushagra0301/CrackDataset' 
python domain_shift.py --weight1 '/scratch/kushagra0301/Crack_IL_step_1/model_10.pth' --weight2 '/scratch/kushagra0301/Crack_IL_step_2/10_checkpoint.pth.tar' --source_dataset_path '/scratch/kushagra0301/CrackDataset' 
python domain_shift.py --weight1 '/scratch/kushagra0301/Crack_IL_step_1/model_20.pth' --weight2 '/scratch/kushagra0301/Crack_IL_step_2/20_checkpoint.pth.tar' --source_dataset_path '/scratch/kushagra0301/CrackDataset' 
python domain_shift.py --weight1 '/scratch/kushagra0301/Crack_IL_step_1/model_30.pth' --weight2 '/scratch/kushagra0301/Crack_IL_step_2/30_checkpoint.pth.tar' --source_dataset_path '/scratch/kushagra0301/CrackDataset' 
