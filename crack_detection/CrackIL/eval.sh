#!/bin/bash
#SBATCH -A mobility_arfs
#SBATCH --partition ihub
#SBATCH -n 20
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode105

# python eval.py --result_dir '/scratch/kushagra0301/Crack_IL_step_2_without_mason/results/source_10_checkpoint' --weight '/scratch/kushagra0301/Crack_IL_step_2_without_mason/10_checkpoint.pth.tar' --source_dataset_path '/scratch/kushagra0301/CrackDataset1' --target_dataset_path '/scratch/kushagra0301/CustomCrackDetectionModified' --dataset_avoided 'Mason'
python eval.py --result_dir '/scratch/kushagra0301/Crack_IL_step_2_without_ceramic/results/target_as_source' --weight '/scratch/kushagra0301/target_as_source/best_model.pth' --source_dataset_path '/scratch/kushagra0301/CrackDataset1' --target_dataset_path '/scratch/kushagra0301/IIITDataset' --dataset_avoided 'NA'
# python eval.py --result_dir '/scratch/kushagra0301/ForLabelling/target' --weight '/scratch/kushagra0301/Crack_IL_step_1/best_model.pth' --source_dataset_path '/scratch/kushagra0301/CrackDataset1' --target_dataset_path '/scratch/kushagra0301/CustomCrackDetectionModified' --dataset_avoided 'CFD'
