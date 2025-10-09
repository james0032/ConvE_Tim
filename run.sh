#!/bin/bash
#SBATCH --job-name=ConvE_cggd
##SBATCH --job-name=ConvE_preprocess

#SBATCH -p volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --mem 100G
#SBATCH -t 7-
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=end
#SBATCH --mail-user=damiosh@unc.edu



#bash preprocess.sh
# Run the ConvE training script
#export PYTHONPATH=$PYTHONPATH:/work/users/d/o/damiosh/ConvE/spodernet:/work/users/d/o/damiosh/ConvE/bashmagic
export PYTHONPATH=$PYTHONPATH:/proj/jchunglab/projects/ec_moa/ConvE_2DCNN/ConvE_benchmark/spodernet:/proj/jchunglab/projects/ec_moa/ConvE_2DCNN/ConvE_benchmark/bashmagic

#python main.py --model conve --data FB15k-237 --preprocess
#python main.py --model conve --data matrix_subgraph --preprocess
#python main.py --model conve --data ROBOKOP_30fd_baseline2_CCDD_noSubclassOf --preprocess
#python main.py --data ROBOKOP_30fd_baseline2_CCDD_noSubclassOf --preprocess
#python main.py --model conve --data ROBOKOP_30fd_baseline2_CCGGDD_noSubclassOf --preprocess
python main.py --model conve --data FB15k-237 --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2  --lr 0.003 --preprocess

#python model.py
# Run the evaluation script
#python evaluation.py
