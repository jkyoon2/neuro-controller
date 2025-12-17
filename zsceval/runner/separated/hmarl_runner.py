from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
import numpy as np
import torch
import wandb
from loguru import logger

from zsceval.algorithms.hmarl.hmarl_policy import HMARLPolicy
from zsceval.algorithms.hmarl.hmarl_trainer import HMARLTrainer
from zsceval.algorithms.hmarl.vae import SkillVAE
from zsceval.runner.separated.base_runner import _t2n
from zsceval.utils.hmarl_buffer import HighLevelReplayBuffer, LowLevelRolloutBuffer
from zsceval.utils.log_util import eta


class HMARLRunner:
    """
    Two-level runner:
      - High level selects skills every t_seg steps from shared/global obs.
      - Low level executes actions conditioned on the current skill every env step.
    """

    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]
        self.run_dir = Path(config["run_dir"])

        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        # Enable render if either use_render or save_gifs is set
        self.use_wandb = self.all_args.use_wandb
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        self.save_gifs = getattr(self.all_args, "save_gifs", True)
        self.use_render = getattr(self.all_args, "use_render", True)
        self.n_render_rollout_threads = getattr(self.all_args, "n_render_rollout_threads", 1)
        self.t_seg = getattr(self.all_args, "t_seg", 5)
        self.skill_dim = getattr(self.all_args, "skill_dim", getattr(self.all_args, "latent_dim", 4))
        self.high_batch_size = getattr(self.all_args, "high_batch_size", 64)
        self.high_buffer_size = getattr(self.all_args, "high_buffer_size", 50000)
        if self.env_name == "Overcooked":
            if self.all_args.overcooked_version == "old":
                from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
            else:
                from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
            self.shaped_info_keys = list(SHAPED_INFOS)
        else:
            self.shaped_info_keys = []

        # Load pretrained SkillVAE for intrinsic rewards if provided
        self._maybe_load_skill_vae()

        # Composite policy & trainer
        share_observation_space = (
            self.envs.share_observation_space if self.use_centralized_V else self.envs.observation_space
        )
        self.policy = HMARLPolicy(
            self.all_args,
            self.envs.observation_space,
            share_observation_space,
            self.envs.action_space,
            device=self.device,
        )
        self.trainer = HMARLTrainer(self.all_args, self.policy, device=self.device)

        # Convenience handles
        self.low_policies = self.policy.low_levels
        self.high_policy = self.policy.high_level
        self.low_trainers = self.trainer.low_trainers
        self.high_trainer = self.trainer.high_trainer

        # Buffers
        self.low_buffers = []
        for agent_id in range(self.num_agents):
            share_obs_space = (
                self.envs.share_observation_space[agent_id]
                if self.use_centralized_V
                else self.envs.observation_space[agent_id]
            )
            bu = LowLevelRolloutBuffer(
                self.all_args, self.envs.observation_space[agent_id], share_obs_space, self.envs.action_space[agent_id], self.skill_dim
            )
            self.low_buffers.append(bu)
        self.high_buffer = HighLevelReplayBuffer(
            self.high_buffer_size,
            get_shape_from_space(self.envs.observation_space[0]),
            get_shape_from_space(share_observation_space),
            self.skill_dim,
            device=self.device,
        )
        
        # States
        self.current_skills = np.zeros((self.n_rollout_threads, self.num_agents, self.skill_dim), dtype=np.float32)
        recurrent_N = self.all_args.recurrent_N
        self.rnn_states_actor_high = np.zeros(
            (self.n_rollout_threads, self.num_agents, recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        self.rnn_states_critic_high = np.zeros_like(self.rnn_states_actor_high)
        self.rnn_states_low = np.zeros(
            (self.num_agents, self.n_rollout_threads, recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        self.rnn_states_low_critic = np.zeros_like(self.rnn_states_low)
        # Track high-level rnn states used at segment boundaries
        self.prev_rnn_states_actor_high = np.zeros_like(self.rnn_states_actor_high)
        self.prev_rnn_states_critic_high = np.zeros_like(self.rnn_states_critic_high)
        self.prev_masks_high = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.high_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        self.segment_reward = np.zeros((self.n_rollout_threads, 1), dtype=np.float32)
        self.segment_step = np.zeros(self.n_rollout_threads, dtype=np.int32)
        self.segment_start_obs = None
        self.segment_start_share_obs = None
        self.segment_traj_obs = [[[] for _ in range(self.num_agents)] for _ in range(self.n_rollout_threads)]
        self.segment_traj_actions = [[[] for _ in range(self.num_agents)] for _ in range(self.n_rollout_threads)]

        # output directories (models and gifs only)
        self.save_dir = self.run_dir / "models"
        self.gif_dir = self.run_dir / "gifs"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        class _DummyWriter:
            def export_scalars_to_json(self, *args, **kwargs): ...
            def close(self): ...
        self.writter = _DummyWriter()

    def _maybe_load_skill_vae(self):
            """
            Load SkillVAE checkpoint and attach to args for intrinsic reward computation.
            Also resolves the correct Task ID (One-Hot) based on the current layout.
            """
            vae_ckpt = getattr(self.all_args, "vae_checkpoint_path", None)
            if not vae_ckpt:
                if not hasattr(self.all_args, "skill_decoder"):
                    self.all_args.skill_decoder = None
                return

            ckpt_path = Path(vae_ckpt)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"SkillVAE checkpoint not found at {ckpt_path}")

            # 1. 체크포인트 파일 전체 로드 (Dictionary)
            try:
                checkpoint = torch.load(str(ckpt_path), map_location=self.device)
            except Exception as exc:
                raise RuntimeError(f"Failed to load VAE checkpoint from {ckpt_path}") from exc

            # 2. State Dict 추출 (model 가중치)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # 3. 모델 초기화를 위한 차원 정보 설정
            obs_shape = get_shape_from_space(self.envs.observation_space[0])
            obs_channels = obs_shape[2] if len(obs_shape) > 0 else 1 
            
            action_space = self.envs.action_space[0]
            action_dim = action_space.n if hasattr(action_space, "n") else int(np.prod(action_space.shape))
            
            # [핵심] 체크포인트에서 layout_to_id 정보 가져오기
            layout_to_id = checkpoint.get("layout_to_id", None)
            
            # Task Dim 결정 (매핑 정보가 있으면 그 길이, 없으면 args 설정값)
            if layout_to_id:
                task_dim = len(layout_to_id)
            else:
                task_dim = getattr(self.all_args, "task_dim", 1)

            # 4. 모델 초기화 및 가중치 로드
            skill_vae = SkillVAE(
                obs_channels=obs_channels,
                action_dim=action_dim,
                task_dim=task_dim,
                hidden_dim=128,  # args.hidden_size가 있다면 그것을 사용
                latent_dim=self.skill_dim,
                t_seg=self.t_seg,
            )
            skill_vae.load_state_dict(state_dict)
            skill_vae.to(self.device)
            skill_vae.eval()
            
            self.all_args.skill_decoder = skill_vae
            if not hasattr(self.all_args, "skill_decoder_type"):
                self.all_args.skill_decoder_type = "vae"
                
            logger.info(f"Loaded SkillVAE from checkpoint: {ckpt_path}")

            # =========================================================================
            # [NEW] Task ID 자동 설정 (One-Hot Encoding)
            # =========================================================================
            if layout_to_id:
                # 현재 실행 중인 환경의 레이아웃 이름 가져오기
                current_layout = getattr(self.all_args, "layout_name", None)
                
                if current_layout and current_layout in layout_to_id:
                    task_idx = layout_to_id[current_layout]
                    
                    # One-Hot Vector 생성: (1, task_dim)
                    task_onehot = torch.zeros((1, task_dim), device=self.device)
                    task_onehot[0, task_idx] = 1.0
                    
                    # Trainer가 가져갈 수 있도록 args에 저장
                    self.all_args.current_task_id = task_onehot
                    self.all_args.task_dim = task_dim # 차원 정보도 업데이트
                    
                    logger.success(f"Task ID Resolved: '{current_layout}' -> ID {task_idx} (One-Hot)")
                else:
                    logger.warning(f"Current layout '{current_layout}' not found in VAE checkpoint map: {list(layout_to_id.keys())}")
                    self.all_args.current_task_id = None # Trainer에서 Dummy 생성하도록 유도
            else:
                logger.warning("No 'layout_to_id' found in VAE checkpoint. Cannot assign automatic Task ID.")
                self.all_args.current_task_id = None

    def run(self):
        self.warmup()
        start = 0
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.policy.lr_decay(episode, episodes)

            # Episode-level accumulators (per env, per agent)
            ep_sparse_r = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
            ep_extrinsic_r = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
            ep_intrinsic_r = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
            ep_low_total_r = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
            ep_high_r = np.zeros((self.n_rollout_threads, 1), dtype=np.float32)
            ep_events = defaultdict(lambda: np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32))

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                # values : (50, 2, 1)
                # actions: (50, 2, 1)
                # action_log_probs: (50, 2, 1)
                # rnn_states: (50, 2, 1, 64)
                # rnn_states_critic: (50, 2, 1, 64)
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                # obs: (50, 2, 13, 5, 20) -- actual agent-specific obs is in infos["all_agent_obs"]
                # share_obs: (50, 2, 13, 5, 20) -- actual shared obs is in infos["share_obs"]
                # rewards: (50, 2, 1)
                # dones: (50, 2)
                # infos: list of 50 dicts (dict_keys(['agent_infos', 'sparse_r_by_agent', 'shaped_r_by_agent', 'shaped_info_by_agent', 'phi_s', 'phi_s_prime', 'shaped_info_timestep', 'stuck', 'bad_transition', 'all_agent_obs', 'share_obs', 'available_actions']))
                # available_actions: (50, 2, 6)

                # Use authoritative obs/share_obs/available_actions from info to ensure correct channels
                obs = np.array([info["all_agent_obs"] for info in infos]) 
                # obs: (50, 2, 13, 5, 20)
                share_obs = np.array([info["share_obs"] for info in infos])
                available_actions = np.array([info["available_actions"] for info in infos])

                total_num_steps += self.n_rollout_threads

                # reward bookkeeping (extrinsic/high-level)
                rewards = self.process_rewards(rewards, ep_extrinsic_r, ep_high_r)
                # track shaped stats for debugging/events
                self.record_event_logs(infos, ep_events)
                # store trajectories for intrinsic reward computation
                self.store_segment_trajectories(obs, actions)
                # sparse reward logging from env-provided episode info
                self.update_sparse_episode_rewards(infos, ep_sparse_r)

                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                self.insert(data)

                if step % self.t_seg == self.t_seg - 1 or np.any(dones):
                    self.finish_segment(obs, share_obs, dones, ep_intrinsic_r, ep_low_total_r)

            self.compute()
            train_infos = self.train(total_num_steps)

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(total_num_steps)

            if episode % self.log_interval == 0 or episode == episodes - 1:
                logger.info(
                    f"HMARL updates {episode}/{episodes}, total timesteps {total_num_steps}/{self.num_env_steps}"
                )
                # Episode-total returns averaged over envs
                env_mean_sparse = np.mean(ep_sparse_r, axis=0)
                env_mean_extrinsic = np.mean(ep_extrinsic_r, axis=0)
                env_mean_intrinsic = np.mean(ep_intrinsic_r, axis=0)
                env_mean_low_train = np.mean(ep_low_total_r, axis=0)
                env_mean_high_train = float(np.mean(ep_high_r))

                avg_rewards = {}
                for a in range(self.num_agents):
                    train_infos["low"][a]["average_episode_rewards"] = env_mean_low_train[a]
                    avg_rewards[a] = env_mean_low_train[a]
                    logger.info(f"agent {a} average episode rewards {env_mean_low_train[a]}")

                if self.use_wandb:
                    for a in range(self.num_agents):
                        wandb.log(
                            {
                                f"reward/sparse/agent{a}": env_mean_sparse[a],
                                f"reward/extrinsic/agent{a}": env_mean_extrinsic[a],
                                f"reward/intrinsic/agent{a}": env_mean_intrinsic[a],
                                f"reward/low_level_total/agent{a}": env_mean_low_train[a],
                                f"train/low_level_return/agent{a}": env_mean_low_train[a],
                            },
                            step=total_num_steps,
                        )
                    wandb.log({f"reward/high_level_team": env_mean_high_train}, step=total_num_steps)
                    wandb.log({f"train/high_level_return": env_mean_high_train}, step=total_num_steps)

                logger.info(f"Episode Return (Low): {env_mean_low_train}")
                logger.info(f"Episode Return (High): {env_mean_high_train}")

                self.log_train(train_infos, avg_rewards, total_num_steps)
                self.log_events(ep_events, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval or episode == episodes - 1:
                pass  # eval placeholder

    def warmup(self):
        obs_batch, info_list = self.envs.reset()
        all_agent_obs = np.array([info["all_agent_obs"] for info in info_list])
        share_obs = np.array([info["share_obs"] for info in info_list])
        available_actions = np.array([info["available_actions"] for info in info_list])

        self.segment_start_obs = all_agent_obs
        self.segment_start_share_obs = share_obs
        self.current_skills, self.rnn_states_actor_high = self.sample_skills(share_obs, all_agent_obs, 0, None, masks=self.high_masks)

        for agent_id in range(self.num_agents):
            self.low_buffers[agent_id].share_obs[0] = share_obs[:, agent_id] if self.use_centralized_V else all_agent_obs[:, agent_id]
            self.low_buffers[agent_id].obs[0] = all_agent_obs[:, agent_id]
            self.low_buffers[agent_id].skills[0] = self.current_skills[:, agent_id]
            self.low_buffers[agent_id].available_actions[0] = available_actions[:, agent_id]

    @torch.no_grad()
    def sample_skills(self, share_obs, obs, step_in_seg, prev_skills, masks=None):
        if masks is None:
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # remember current rnn states and masks for buffer logging
        self.prev_rnn_states_actor_high = self.rnn_states_actor_high.copy()
        self.prev_rnn_states_critic_high = self.rnn_states_critic_high.copy()
        self.prev_masks_high = masks.copy()

        (q1, q2), skills, _, rnn_actor, rnn_critic = self.high_policy.get_actions(
            share_obs, obs, self.rnn_states_actor_high, self.rnn_states_critic_high, masks, step_in_seg=step_in_seg, prev_skills=prev_skills
        )
        self.rnn_states_actor_high = _t2n(rnn_actor)
        self.rnn_states_critic_high = _t2n(rnn_critic)
        return _t2n(skills), _t2n(rnn_actor)

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.low_policies[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.low_policies[agent_id].get_actions(
                self.low_buffers[agent_id].share_obs[step], # (401, 50, 13, 5, 20)
                self.low_buffers[agent_id].obs[step], # (401, 50, 13, 5, 20)
                self.rnn_states_low[agent_id], # (50, 1, 64)
                self.rnn_states_low_critic[agent_id],  # (50, 1, 64)
                self.low_buffers[agent_id].masks[step], # (401, 50, 1)
                self.low_buffers[agent_id].available_actions[step], # (401, 50, 6)
                deterministic=False,
                skill=self.current_skills[:, agent_id], # (50, 4)
            )
            values.append(_t2n(value))
            # list of (n_rollout_threads, 1)
            actions.append(_t2n(action))
            action_log_probs.append(_t2n(action_log_prob))  # list of (n_rollout_threads, 1), tensor 
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        values_env = np.stack(values, axis=1)  # (B, N, 1)
        actions_env = np.stack(actions, axis=1)  # (B, N, 1)
        action_log_probs_env = np.stack(action_log_probs, axis=1)  # (B, N, 1)
        rnn_states_env = np.stack(rnn_states, axis=1)  # (B, N, 1, H)
        rnn_states_critic_env = np.stack(rnn_states_critic, axis=1)  # (B, N, 1, H)
        return values_env, actions_env, action_log_probs_env, rnn_states_env, rnn_states_critic_env

    def process_rewards(self, rewards, ep_extrinsic_r, ep_high_r):
        """
        Handle extrinsic reward accumulation and high-level segment reward.
        rewards: (n_threads, n_agents, 1)
        """
        rewards_sq = rewards.squeeze(-1)
        ep_extrinsic_r += rewards_sq

        team_reward = rewards_sq.mean(axis=1, keepdims=True)
        self.segment_reward += team_reward
        self.segment_step += 1
        ep_high_r += team_reward
        return rewards

    def record_event_logs(self, infos, ep_events):
        """
        Accumulate shaped event counts for the current episode.

        infos: list of env info dicts
        ep_events: defaultdict mapping event name -> (n_threads, n_agents) array of counts
        """
        if not self.shaped_info_keys:
            return

        for env_idx, info in enumerate(infos):
            shaped_info = info.get("shaped_info_by_agent", None)
            if shaped_info is None:
                continue

            for a in range(self.num_agents):
                agent_info = shaped_info[a]
                if isinstance(agent_info, dict):
                    values = [agent_info.get(k, 0.0) for k in self.shaped_info_keys]
                else:
                    values = agent_info

                for idx, val in enumerate(values):
                    key = self.shaped_info_keys[idx] if idx < len(self.shaped_info_keys) else f"event_{idx}"
                    ep_events[key][env_idx, a] += float(val)

    def store_segment_trajectories(self, obs, actions):
        """Accumulate obs/actions for intrinsic reward calculation at segment end."""
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                self.segment_traj_obs[e][a].append(obs[e, a])
                act = actions[e, a]
                act_arr = np.asarray(act)
                if act_arr.ndim == 0 or act_arr.shape == ():
                    onehot = np.zeros(self.envs.action_space[a].n, dtype=np.float32)
                    onehot[int(act_arr)] = 1.0
                elif act_arr.shape == (1,):
                    # discrete action wrapped in array
                    onehot = np.zeros(self.envs.action_space[a].n, dtype=np.float32)
                    onehot[int(act_arr[0])] = 1.0
                else:
                    onehot = act_arr
                self.segment_traj_actions[e][a].append(onehot)

    def update_sparse_episode_rewards(self, infos, ep_sparse_r):
        """Grab sparse episode totals from env info when available."""
        for env_idx, info in enumerate(infos):
            episode_info = info.get("episode", None)
            if episode_info is None:
                continue
            if "ep_sparse_r_by_agent" in episode_info:
                ep_sparse_r[env_idx] = np.array(episode_info["ep_sparse_r_by_agent"], dtype=np.float32)

    def insert(self, data):
        (
                obs,
                share_obs,
                rewards,
                dones,
                infos,
                available_actions,
                values,
                actions,
                action_log_probs,
                rnn_states,
                rnn_states_critic,
            ) = data

        for agent_id in range(self.num_agents):
            masks = np.ones((self.n_rollout_threads, 1), dtype=np.float32)
            agent_done = np.array(dones)[:, agent_id]
            masks[agent_done] = 0.0
            self.low_buffers[agent_id].insert(
                share_obs[:, agent_id] if self.use_centralized_V else obs[:, agent_id],
                obs[:, agent_id],
                self.current_skills[:, agent_id],
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id].reshape(-1, 1),
                masks,
                available_actions=available_actions[:, agent_id] if available_actions is not None else None,
            )
            self.rnn_states_low[agent_id] = rnn_states[:, agent_id]
            self.rnn_states_low_critic[agent_id] = rnn_states_critic[:, agent_id]

    def finish_segment(self, obs, share_obs, dones, ep_intrinsic_r, ep_low_total_r):
        next_masks = 1.0 - dones.astype(np.float32).reshape(self.n_rollout_threads, self.num_agents, 1)

        # add to high-level buffer with masks and rnn states (per env entry)
        for env_idx in range(self.n_rollout_threads):
            self.high_buffer.add(
                self.segment_start_obs[env_idx],
                self.segment_start_share_obs[env_idx],
                self.current_skills[env_idx],
                self.segment_reward[env_idx].copy()/ 100.0,  # scale down high-level reward
                obs[env_idx],
                share_obs[env_idx],
                dones[env_idx].astype(np.float32).reshape(-1, 1),
                mask=self.prev_masks_high[env_idx],
                next_mask=next_masks[env_idx],
                rnn_states_actor=self.prev_rnn_states_actor_high[env_idx],
                rnn_states_critic=self.prev_rnn_states_critic_high[env_idx],
                next_rnn_states_actor=self.rnn_states_actor_high[env_idx],
                next_rnn_states_critic=self.rnn_states_critic_high[env_idx],
            )
        # compute intrinsic and reset trajectories
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                seg_len = len(self.segment_traj_obs[e][a])
                if seg_len == 0:
                    continue

                obs_seq = torch.as_tensor(np.stack(self.segment_traj_obs[e][a])[None], device=self.device).float()
                obs_seq = obs_seq.permute(0, 1, 4, 2, 3)  # to NCHW
                act_seq = torch.as_tensor(np.stack(self.segment_traj_actions[e][a])[None], device=self.device)
                skill = torch.as_tensor(self.current_skills[e, a][None], device=self.device)

                intrinsic_val = self.low_trainers[a].compute_intrinsic_reward_value(obs_seq, act_seq, skill)
                ep_intrinsic_r[e, a] += intrinsic_val

                per_step_intrinsic = intrinsic_val / max(1, seg_len)

                start_idx = (self.low_buffers[a].step - seg_len) % self.episode_length
                idxs = [(start_idx + k) % self.episode_length for k in range(seg_len)]
                current_extrinsic = self.low_buffers[a].rewards[idxs, e, 0]
                alpha = self.low_trainers[a].intrinsic_alpha
                new_reward = alpha * current_extrinsic + (1 - alpha) * per_step_intrinsic
                self.low_buffers[a].rewards[idxs, e, 0] = new_reward

                ep_low_total_r[e, a] += float(np.sum(new_reward))
        # reset
        self.segment_reward[:] = 0
        self.segment_step[:] = 0
        self.segment_start_obs = obs
        self.segment_start_share_obs = share_obs
        self.high_masks = next_masks
        self.segment_traj_obs = [[[] for _ in range(self.num_agents)] for _ in range(self.n_rollout_threads)]
        self.segment_traj_actions = [[[] for _ in range(self.num_agents)] for _ in range(self.n_rollout_threads)]
        # resample skills
        self.current_skills, self.rnn_states_actor_high = self.sample_skills(share_obs, obs, 0, None, masks=self.high_masks)

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            next_value = self.low_policies[agent_id].get_values(
                self.low_buffers[agent_id].share_obs[-1],
                self.low_buffers[agent_id].rnn_states_critic[-1],
                self.low_buffers[agent_id].masks[-1],
                skill=self.low_buffers[agent_id].skills[-1],
            )
            next_value = _t2n(next_value)
            self.low_buffers[agent_id].compute_returns(next_value, self.low_trainers[agent_id].value_normalizer)

    def train(self, num_steps):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.low_trainers[agent_id].prep_training()
            self.low_trainers[agent_id].adapt_entropy_coef(num_steps)
            train_info = self.low_trainers[agent_id].train(self.low_buffers[agent_id])
            train_infos.append(train_info)
            self.low_buffers[agent_id].after_update()

        high_info = None
        # high-level off-policy update
        if len(self.high_buffer) >= self.high_batch_size:
            self.high_trainer.prep_training()
            high_info = self.high_trainer.train(self.high_buffer, batch_size=self.high_batch_size)
        return {"low": train_infos, "high": high_info}

    def log_train(self, train_infos, avg_rewards, total_num_steps):
        if train_infos is None:
            return
        low_infos = train_infos.get("low", [])
        high_info = train_infos.get("high", None)
        for agent_id, info in enumerate(low_infos):
            for k, v in info.items():
                if isinstance(v, Iterable):
                    if len(v) == 0:
                        continue
                    v = np.mean(v)
                metric = {f"train/low/agent{agent_id}/{k}": float(v)}
                if self.use_wandb:
                    wandb.log(metric, step=total_num_steps)
        if high_info is not None:
            for agent_id in range(self.num_agents):
                for k, v in high_info.items():
                    metric = {f"train/high/agent{agent_id}/{k}": float(v)}
                    if self.use_wandb:
                        wandb.log(metric, step=total_num_steps)
        if avg_rewards:
            for agent_id, v in avg_rewards.items():
                metric = {f"train/agent{agent_id}/average_episode_rewards": float(v)}
                if self.use_wandb:
                    wandb.log(metric, step=total_num_steps)

    def log_events(self, ep_events, total_num_steps):
        """Log per-episode event averages across environments."""
        if not self.use_wandb or not ep_events:
            return

        for event_name, data in ep_events.items():
            env_mean_events = np.mean(data, axis=0)
            for agent_id in range(self.num_agents):
                wandb.log({f"events/agent{agent_id}/{event_name}": float(env_mean_events[agent_id])}, step=total_num_steps)

    def save(self, steps=None):
        postfix = f"_{steps}.pt" if steps else ".pt"
        for agent_id in range(self.num_agents):
            torch.save(self.low_policies[agent_id].actor.state_dict(), str(self.save_dir / f"ll_actor_agent{agent_id}{postfix}"))
            torch.save(self.low_policies[agent_id].critic.state_dict(), str(self.save_dir / f"ll_critic_agent{agent_id}{postfix}"))
        torch.save(self.high_policy.critic.state_dict(), str(self.save_dir / f"hl_critic{postfix}"))
        for i, actor in enumerate(self.high_policy.actors):
            torch.save(actor.state_dict(), str(self.save_dir / f"hl_actor_agent{i}{postfix}"))


def get_shape_from_space(space):
    if hasattr(space, "shape"):
        return space.shape
    if isinstance(space, (list, tuple)):
        return space[0].shape
    raise ValueError("Unknown space")
