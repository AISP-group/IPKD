#!/bin/sh
env="football"
scenario="academy_3_vs_1_with_keeper"
# academy_pass_and_shoot_with_keeper
# academy_3_vs_1_with_keeper
# academy_counterattack_easy
#academy_corner
n_agent=3
algo="ipkd"
exp="single"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=1 python train/train_football.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --n_agent ${n_agent} --lr 7e-4 --entropy_coef 0.01 --max_grad_norm 0.5 --eval_episodes 32 --n_training_threads 16 --n_rollout_threads 20 --num_mini_batch 1 --episode_length 400 --eval_interval 10 --num_env_steps 10000000 --ppo_epoch 15 --clip_param 0.2 --use_eval --use_value_active_masks --use_policy_active_masks --use_wandb --ifdistill --ifevalstudent --ifofa
