#!/bin/bash
env="Overcooked"

layout=$1
version="new"

num_env_steps="1e7"
num_agents=2
algo="hmarl"
exp="sp"
seed_begin=6
seed_max=10
ulimit -n 65536

entropy_coefs="0.2 0.05 0.01"
entropy_coef_horizons="0 5e6 1e7"
reward_shaping_horizon="1e8"
agent_policy_names="ppo ppo"

vae_checkpoint_path="/home/juliecandoit98/neurocontroller/checkpoints/vae_epoch_010_1tasks_4d.pt"
skill_dim=4
intrinsic_scale=1.0
intrinsic_alpha=0.3
t_seg=5

w0="0,0,0,0,0,0.1,0.1,0,0,0.1,0,3,0,10,-2,3,2,2,-2,-2,5,5,0,20,-5,0,7,20,-5,-0.01,-0.01,-0.01,-0.01,30"
w1="0,0,0,0,0,0.1,0.1,0,0,0.1,0,3,0,10,-2,3,2,2,-2,-2,5,5,0,20,-5,0,7,20,-5,-0.01,-0.01,-0.01,-0.01,30"

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, seed from ${seed_begin} to ${seed_max}"
for seed in $(seq ${seed_begin} ${seed_max});
do
    echo "seed is ${seed}:"
    python train/train_hmarl.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents}  \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 50 --dummy_batch_size 2 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    --overcooked_version ${version} --agent_policy_names ${agent_policy_names}\
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --use_hsp --w0 ${w0} --w1 ${w1} --share_policy \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
    --use_proper_time_limits \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads 10 \
    --skill_dim ${skill_dim} --t_seg ${t_seg} --intrinsic_alpha ${intrinsic_alpha} --intrinsic_scale ${intrinsic_scale} --vae_checkpoint_path ${vae_checkpoint_path} \
    --use_render --save_gifs --n_render_rollout_threads 1 --render_episodes 1 \
    --wandb_name "kyungyoon"
done

# --agent_policy_names ${agent_policy_names}