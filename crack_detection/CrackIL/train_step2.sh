#!/bin/bash
#SBATCH -A mobility_arfs
#SBATCH --partition ihub
#SBATCH -n 15
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode108


# python train_step2.py --save_dir '/scratch/kushagra0301/kld_1' --num_epochs 30   --dataset_avoided 'NA' --saved_model /scratch/kushagra0301/Crack_IL_step_1/best_model.pth --kld_weight 1 > 'kld1_step2.log' 2>&1
# python train_step2.py --save_dir '/scratch/kushagra0301/kld_0.8' --num_epochs 30   --dataset_avoided 'NA' --saved_model /scratch/kushagra0301/Crack_IL_step_1/best_model.pth --kld_weight 0.8 > 'kld0.8_step2.log' 2>&1
# python train_step2.py --save_dir '/scratch/kushagra0301/kld_0.6' --num_epochs 30   --dataset_avoided 'NA' --saved_model /scratch/kushagra0301/Crack_IL_step_1/best_model.pth --kld_weight 0.6 > 'kld0.6_step2.log' 2>&1
# python train_step2.py --save_dir '/scratch/kushagra0301/kld_0.4' --num_epochs 30   --dataset_avoided 'NA' --saved_model /scratch/kushagra0301/Crack_IL_step_1/best_model.pth --kld_weight 0.4 > 'kld0.4_step2.log' 2>&1
# python train_step2.py --save_dir '/scratch/kushagra0301/kld_0.2' --num_epochs 30   --dataset_avoided 'NA' --saved_model /scratch/kushagra0301/Crack_IL_step_1/best_model.pth --kld_weight 0.2 > 'kld0.2_step2.log' 2>&1
# python train_step2.py --save_dir '/scratch/kushagra0301/ce_0.1' --num_epochs 5   --dataset_avoided 'NA' --saved_model /scratch/kushagra0301/Crack_IL_step_1/best_model.pth --ce_weight 0.1 > 'ce0.1_step2.log' 2>&1
# python train_step2.py --save_dir '/scratch/kushagra0301/ce_0.2' --num_epochs 5   --dataset_avoided 'NA' --saved_model /scratch/kushagra0301/Crack_IL_step_1/best_model.pth --ce_weight 0.2 > 'ce0.2_step2.log' 2>&1
# python train_step2.py --save_dir '/scratch/kushagra0301/ce_0.3' --num_epochs 5   --dataset_avoided 'NA' --saved_model /scratch/kushagra0301/Crack_IL_step_1/best_model.pth --ce_weight 0.3 > 'ce0.3_step2.log' 2>&1
# python train_step2.py --save_dir '/scratch/kushagra0301/ce_0.4' --num_epochs 5   --dataset_avoided 'NA' --saved_model /scratch/kushagra0301/Crack_IL_step_1/best_model.pth --ce_weight 0.4 > 'ce0.4_step2.log' 2>&1
# python train_step2.py --save_dir '/scratch/kushagra0301/ce_0.5' --num_epochs 5   --dataset_avoided 'NA' --saved_model /scratch/kushagra0301/Crack_IL_step_1/best_model.pth --ce_weight 0.5 > 'ce0.5_step2.log' 2>&1

# python train_step2.py --save_dir '/scratch/kushagra0301/Crack_IL_step_2_30' --num_epochs 150 --source_dataset_path '/scratch/kushagra0301/CrackDataset30' --dataset_avoided 'NA' --saved_model /scratch/kushagra0301/Crack_IL_step_1_30/best_model.pth  > 'train_step2_30.log' 2>&1
python train_step2_source_as_target.py --save_dir '/scratch/kushagra0301/target_as_source' --num_epochs 150  --dataset_avoided 'NA' --saved_model /scratch/kushagra0301/target_as_source/best_model.pth  > 'target_as_source_step2.log' 2>&1

