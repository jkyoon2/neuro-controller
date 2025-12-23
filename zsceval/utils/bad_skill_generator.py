import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from zsceval.algorithms.hmarl.hmarl_policy import HMARLPolicy
from zsceval.algorithms.hmarl.vae import SkillVAE
from zsceval.envs.env_wrappers import ShareDummyVecEnv
from zsceval.envs.overcooked.Overcooked_Env import Overcooked as OvercookedOld
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as OvercookedNew
from zsceval.runner.separated.base_runner import _t2n
from zsceval.utils.train_util import get_base_run_dir, setup_seed


def _get_shape_from_space(space):
    if hasattr(space, "shape"):
        return space.shape
    if isinstance(space, (list, tuple)):
        return space[0].shape
    raise ValueError("Unknown space")


class BadSkillGenerator:
    def __init__(self, config):
        self.all_args = config["all_args"]
        self.device = config.get("device", torch.device("cpu"))
        self.run_dir = Path(config.get("run_dir", "."))
        self.num_agents = config.get("num_agents", getattr(self.all_args, "num_agents", 2))
        self.n_rollout_threads = getattr(self.all_args, "n_rollout_threads", 1)
        self.episode_length = getattr(self.all_args, "episode_length", 200)
        self.t_seg = getattr(self.all_args, "t_seg", 5)
        self.skill_dim = getattr(self.all_args, "skill_dim", getattr(self.all_args, "latent_dim", 4))
        self.segment_pre = getattr(self.all_args, "bad_segment_pre", 3)
        self.segment_post = getattr(self.all_args, "bad_segment_post", 1)
        self.segment_len = self.segment_pre + self.segment_post + 1
        self.target_error_types = getattr(self.all_args, "target_error_types", None)
        self.model_log_info = []

        setup_seed(getattr(self.all_args, "seed", 1))
        self._init_env()
        self._load_policy()
        self._load_skill_vae()

    def _init_env(self):
        if self.all_args.env_name != "Overcooked":
            raise NotImplementedError(f"BadSkillGenerator only supports Overcooked, got {self.all_args.env_name}")

        def get_env_fn(rank):
            def init_env():
                if self.all_args.overcooked_version == "old":
                    env = OvercookedOld(self.all_args, self.run_dir, rank=rank)
                else:
                    env = OvercookedNew(self.all_args, self.run_dir, rank=rank)
                env.seed(getattr(self.all_args, "seed", 1) + rank * 1000)
                return env

            return init_env

        self.envs = ShareDummyVecEnv([get_env_fn(i) for i in range(self.n_rollout_threads)])

    def _load_policy(self):
        share_observation_space = (
            self.envs.share_observation_space if self.all_args.use_centralized_V else self.envs.observation_space
        )
        self.policy = HMARLPolicy(
            self.all_args,
            self.envs.observation_space,
            share_observation_space,
            self.envs.action_space,
            device=self.device,
        )
        self.policy.prep_rollout()

        self.low_policies = self.policy.low_levels
        self.high_policy = self.policy.high_level

        self._load_policy_weights()

    def _load_policy_weights(self):
        """
        Simplified checkpoint loading:
        - Uses load_seeds/load_steps from args (e.g., injected via YAML)
        - Broadcasts single entries to all agents
        - Falls back to the latest checkpoint step when step is None
        """

        load_seeds = getattr(self.all_args, "load_seeds", None)
        if load_seeds is None:
            load_seeds = [getattr(self.all_args, "seed", 1)]

        if len(load_seeds) == 1:
            seeds = [load_seeds[0]] * self.num_agents
        elif len(load_seeds) == self.num_agents:
            seeds = list(load_seeds)
        else:
            raise ValueError(f"load_seeds length ({len(load_seeds)}) must be 1 or {self.num_agents}")

        load_steps = getattr(self.all_args, "load_steps", None)
        if load_steps is None:
            steps = [None] * self.num_agents
        elif len(load_steps) == 1:
            steps = [load_steps[0]] * self.num_agents
        elif len(load_steps) == self.num_agents:
            steps = list(load_steps)
        else:
            raise ValueError(f"load_steps length ({len(load_steps)}) must be 1 or {self.num_agents}")

        self.model_log_info = []

        for agent_id in range(self.num_agents):
            seed = seeds[agent_id]
            step = steps[agent_id]

            model_dir = self._build_model_dir(seed)
            if step is None:
                step = self._find_latest_step(model_dir, agent_id)
                if step is None:
                    raise FileNotFoundError(f"No checkpoint found for agent {agent_id} in {model_dir}")

            self.model_log_info.append(f"Ag{agent_id}(Seed{seed}/Step{step})")

            logger.info(f"[Agent {agent_id}] Loading model from Seed {seed}, Step {step}")
            self._load_weights_from_file(agent_id, model_dir, int(step))
        if self.model_log_info:
            logger.info(f"Model Loaded: {', '.join(self.model_log_info)}")

    def _build_model_dir(self, seed):
        base_dir = Path(get_base_run_dir())
        return (
            base_dir
            / self.all_args.env_name
            / self.all_args.layout_name
            / self.all_args.algorithm_name
            / self.all_args.experiment_name
            / f"seed{seed}"
            / "models"
        )

    def _find_latest_step(self, model_dir, agent_id):
        patterns = [f"*_agent{agent_id}_*.pt", "hl_critic_*.pt"]
        steps = []
        for pattern in patterns:
            for path in Path(model_dir).glob(pattern):
                match = re.search(r"_([0-9]+)\\.pt$", path.name)
                if match:
                    steps.append(int(match.group(1)))
        return max(steps) if steps else None

    def _load_weights_from_file(self, agent_id, model_dir, step):
        model_dir = Path(model_dir)

        ll_actor_path = model_dir / f"ll_actor_agent{agent_id}_{step}.pt"
        if self.policy.share_policy and not ll_actor_path.exists():
            fallback = model_dir / f"ll_actor_agent0_{step}.pt"
            if fallback.exists():
                ll_actor_path = fallback
        if not ll_actor_path.exists():
            raise FileNotFoundError(f"Low-level actor checkpoint not found: {ll_actor_path}")

        ll_critic_path = model_dir / f"ll_critic_agent{agent_id}_{step}.pt"
        if self.policy.share_policy and not ll_critic_path.exists():
            fallback = model_dir / f"ll_critic_agent0_{step}.pt"
            if fallback.exists():
                ll_critic_path = fallback
        if not ll_critic_path.exists():
            raise FileNotFoundError(f"Low-level critic checkpoint not found: {ll_critic_path}")

        logger.info(f"Loading low-level actor for agent {agent_id} from {ll_actor_path}")
        self.low_policies[agent_id].actor.load_state_dict(torch.load(ll_actor_path, map_location=self.device))

        logger.info(f"Loading low-level critic for agent {agent_id} from {ll_critic_path}")
        self.low_policies[agent_id].critic.load_state_dict(torch.load(ll_critic_path, map_location=self.device))

        hl_actor_path = model_dir / f"hl_actor_agent{agent_id}_{step}.pt"
        if self.policy.share_policy and not hl_actor_path.exists():
            fallback = model_dir / f"hl_actor_agent0_{step}.pt"
            if fallback.exists():
                hl_actor_path = fallback
        if not hl_actor_path.exists():
            raise FileNotFoundError(f"High-level actor checkpoint not found: {hl_actor_path}")

        logger.info(f"Loading high-level actor for agent {agent_id} from {hl_actor_path}")
        self.high_policy.actors[agent_id].load_state_dict(torch.load(hl_actor_path, map_location=self.device))

        if agent_id == 0:
            hl_critic_path = model_dir / f"hl_critic_{step}.pt"
            if hl_critic_path.exists():
                logger.info(f"Loading high-level critic from {hl_critic_path}")
                self.high_policy.critic.load_state_dict(torch.load(hl_critic_path, map_location=self.device))
            else:
                logger.warning(f"High-level critic checkpoint not found: {hl_critic_path}")

    def _load_skill_vae(self):
        vae_ckpt = getattr(self.all_args, "vae_checkpoint_path", None)
        if not vae_ckpt:
            raise FileNotFoundError("vae_checkpoint_path is required to generate bad skills.")

        ckpt_path = Path(vae_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SkillVAE checkpoint not found at {ckpt_path}")

        checkpoint = torch.load(str(ckpt_path), map_location=self.device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        obs_shape = _get_shape_from_space(self.envs.observation_space[0])
        obs_channels = obs_shape[2] if len(obs_shape) > 2 else obs_shape[-1]
        action_space = self.envs.action_space[0]
        action_dim = action_space.n if hasattr(action_space, "n") else int(np.prod(action_space.shape))

        layout_to_id = checkpoint.get("layout_to_id", None) if isinstance(checkpoint, dict) else None
        task_dim = len(layout_to_id) if layout_to_id else getattr(self.all_args, "task_dim", 1)

        skill_vae = SkillVAE(
            obs_channels=obs_channels,
            action_dim=action_dim,
            task_dim=task_dim,
            hidden_dim=128,
            latent_dim=self.skill_dim,
            t_seg=self.t_seg,
        )
        skill_vae.load_state_dict(state_dict)
        skill_vae.to(self.device)
        skill_vae.eval()

        self.skill_vae = skill_vae
        self.skill_decoder_type = getattr(self.all_args, "skill_decoder_type", "vae")
        self.task_id = None

        if layout_to_id:
            current_layout = getattr(self.all_args, "layout_name", None)
            if current_layout and current_layout in layout_to_id:
                task_idx = layout_to_id[current_layout]
                task_onehot = torch.zeros((1, task_dim), device=self.device)
                task_onehot[0, task_idx] = 1.0
                self.task_id = task_onehot
            else:
                logger.warning("Layout not found in VAE checkpoint. Using zero task id.")
        if self.task_id is None:
            self.task_id = torch.zeros((1, task_dim), device=self.device)

        logger.info(f"Loaded SkillVAE from checkpoint: {ckpt_path}")

    def generate_bad_skills(self, num_episodes=10, output_path=None):
        bad_skills = []
        obs, share_obs, available_actions = self._reset_envs()

        self._init_rnn_states()
        self.current_skills = np.zeros((self.n_rollout_threads, self.num_agents, self.skill_dim), dtype=np.float32)
        self.high_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.low_masks = np.ones((self.num_agents, self.n_rollout_threads, 1), dtype=np.float32)

        traj_buffers = [[] for _ in range(self.n_rollout_threads)]
        pending_errors = [[] for _ in range(self.n_rollout_threads)]
        total_events_detected = 0
        event_type_counts = defaultdict(int)

        logger.info(f"Start collecting bad skills... (Target: {self.target_error_types if self.target_error_types else 'ALL'})")

        for _episode in range(num_episodes):
            for step in range(self.episode_length):
                self.current_skills = self._sample_skills(share_obs, obs, step, self.current_skills)

                actions = self._collect_actions(obs, share_obs, available_actions)
                prev_obs = obs
                next_obs, next_share_obs, rewards, dones, infos, next_available_actions = self.envs.step(actions)

                obs = self._extract_obs_from_infos(next_obs, infos)
                share_obs = self._extract_share_obs_from_infos(next_share_obs, infos)
                available_actions = self._extract_available_actions(next_available_actions, infos)

                dones_arr = np.array(dones)
                if dones_arr.ndim == 1:
                    dones_env = dones_arr
                else:
                    dones_env = np.all(dones_arr, axis=1)

                for env_idx in range(self.n_rollout_threads):
                    traj_buffers[env_idx].append(
                        {
                            "obs": prev_obs[env_idx].copy(),
                            "actions": actions[env_idx, :, 0].copy(),
                        }
                    )

                    current_detect_error = infos[env_idx].get("detect_error", {})
                    triggered_events = []
                    for err_type, counts in current_detect_error.items():
                        if self.target_error_types and err_type not in self.target_error_types:
                            continue
                        if sum(counts) > 0:
                            triggered_events.append(f"{err_type}{list(counts)}")
                            event_type_counts[err_type] += 1
                            total_events_detected += 1

                    if triggered_events:
                        event_str = ", ".join(triggered_events)
                        model_str = ", ".join(self.model_log_info)
                        logger.info(
                            f"[Ep {_episode}/{num_episodes} Step {step}] "
                            f"Event Detected: {event_str} | Model: {model_str}"
                        )

                    error_agents = self._check_error(infos[env_idx])
                    if error_agents:
                        pending_errors[env_idx].append(
                            {"step": len(traj_buffers[env_idx]) - 1, "agents": error_agents}
                        )

                    self._drain_pending_errors(
                        traj_buffers[env_idx],
                        pending_errors[env_idx],
                        len(traj_buffers[env_idx]) - 1,
                        bad_skills,
                    )

                if np.any(dones_env):
                    for env_idx, done_flag in enumerate(dones_env):
                        if not done_flag:
                            continue
                        self._flush_pending_errors(
                            traj_buffers[env_idx],
                            pending_errors[env_idx],
                            bad_skills,
                        )
                        traj_buffers[env_idx] = []
                        pending_errors[env_idx] = []
                        self._reset_rnn_states(env_idx)

                self._update_masks(dones_arr)

        self.envs.close()

        logger.success("=" * 50)
        logger.success("Bad Skill Generation Summary")
        logger.success(f"Models Used       : {', '.join(self.model_log_info)}")
        logger.success(f"Total Episodes    : {num_episodes}")
        logger.success(f"Total Detection   : {total_events_detected}")
        logger.success("Breakdown by Type :")
        for k, v in event_type_counts.items():
            logger.success(f"  - {k:<25}: {v} times")
        logger.success("=" * 50)
        return self._save_bad_skills(bad_skills, output_path)

    def _reset_envs(self):
        reset_out = self.envs.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs_list, infos = reset_out
            obs = self._extract_obs_from_infos(obs_list, infos)
            share_obs = self._extract_share_obs_from_infos(None, infos)
            available_actions = self._extract_available_actions(None, infos)
            return obs, share_obs, available_actions
        obs, share_obs, available_actions = reset_out
        obs = np.array(obs)
        return obs, np.array(share_obs), np.array(available_actions)

    def _extract_obs_from_infos(self, obs, infos):
        if infos and isinstance(infos[0], dict) and "all_agent_obs" in infos[0]:
            return np.array([info["all_agent_obs"] for info in infos])
        return np.array(obs)

    def _extract_share_obs_from_infos(self, share_obs, infos):
        if infos and isinstance(infos[0], dict) and "share_obs" in infos[0]:
            return np.array([info["share_obs"] for info in infos])
        return np.array(share_obs) if share_obs is not None else None

    def _extract_available_actions(self, available_actions, infos):
        if infos and isinstance(infos[0], dict) and "available_actions" in infos[0]:
            return np.array([info["available_actions"] for info in infos])
        return np.array(available_actions) if available_actions is not None else None

    def _init_rnn_states(self):
        recurrent_N = getattr(self.all_args, "recurrent_N", 1)
        hidden_size = getattr(self.all_args, "hidden_size", 64)

        self.rnn_states_actor_high = np.zeros(
            (self.n_rollout_threads, self.num_agents, recurrent_N, hidden_size),
            dtype=np.float32,
        )
        self.rnn_states_critic_high = np.zeros_like(self.rnn_states_actor_high)
        self.rnn_states_low = np.zeros(
            (self.num_agents, self.n_rollout_threads, recurrent_N, hidden_size),
            dtype=np.float32,
        )
        self.rnn_states_low_critic = np.zeros_like(self.rnn_states_low)

    def _reset_rnn_states(self, env_idx):
        self.rnn_states_actor_high[env_idx] = 0
        self.rnn_states_critic_high[env_idx] = 0
        self.rnn_states_low[:, env_idx] = 0
        self.rnn_states_low_critic[:, env_idx] = 0

    def _update_masks(self, dones):
        if dones.ndim == 1:
            dones = np.repeat(dones[:, None], self.num_agents, axis=1)
        elif dones.ndim == 2 and dones.shape[1] != self.num_agents:
            if dones.shape[1] == 1:
                dones = np.repeat(dones, self.num_agents, axis=1)
        next_masks = 1.0 - dones.astype(np.float32)
        self.high_masks = next_masks.reshape(self.n_rollout_threads, self.num_agents, 1)
        for agent_id in range(self.num_agents):
            self.low_masks[agent_id] = next_masks[:, agent_id].reshape(self.n_rollout_threads, 1)

    @torch.no_grad()
    def _sample_skills(self, share_obs, obs, step, prev_skills):
        if np.any(self.high_masks == 0):
            step_in_seg = 0
            prev = None
        else:
            step_in_seg = step % self.t_seg
            reuse = step_in_seg != 0 and prev_skills is not None
            prev = prev_skills if reuse else None
        (q1, q2), skills, _, rnn_actor, rnn_critic = self.high_policy.get_actions(
            share_obs,
            obs,
            self.rnn_states_actor_high,
            self.rnn_states_critic_high,
            self.high_masks,
            step_in_seg=step_in_seg,
            prev_skills=prev,
        )
        self.rnn_states_actor_high = _t2n(rnn_actor)
        self.rnn_states_critic_high = _t2n(rnn_critic)
        return _t2n(skills)

    @torch.no_grad()
    def _collect_actions(self, obs, share_obs, available_actions):
        actions = []
        for agent_id in range(self.num_agents):
            self.low_policies[agent_id].prep_rollout()
            share_in = share_obs[:, agent_id] if self.all_args.use_centralized_V else obs[:, agent_id]
            value, action, _, rnn_state, rnn_state_critic = self.low_policies[agent_id].get_actions(
                share_in,
                obs[:, agent_id],
                self.rnn_states_low[agent_id],
                self.rnn_states_low_critic[agent_id],
                self.low_masks[agent_id],
                available_actions[:, agent_id] if available_actions is not None else None,
                deterministic=False,
                skill=self.current_skills[:, agent_id],
            )
            actions.append(_t2n(action))
            self.rnn_states_low[agent_id] = _t2n(rnn_state)
            self.rnn_states_low_critic[agent_id] = _t2n(rnn_state_critic)

        return np.stack(actions, axis=1).astype(np.int64)

    def _check_error(self, info):
        detect_error = info.get("detect_error")
        if not detect_error:
            return []
        if self.target_error_types:
            keys_to_check = [key for key in self.target_error_types if key in detect_error]
        else:
            keys_to_check = detect_error.keys()

        error_agents = []
        for agent_id in range(self.num_agents):
            for key in keys_to_check:
                values = detect_error[key]
                if values[agent_id] > 0:
                    error_agents.append(agent_id)
                    break
        return error_agents

    def _drain_pending_errors(self, buffer, pending, current_step, bad_skills):
        ready = []
        remaining = []
        for item in pending:
            if item["step"] + self.segment_post <= current_step:
                ready.append(item)
            else:
                remaining.append(item)
        pending[:] = remaining
        for item in ready:
            self._process_error_item(buffer, item, bad_skills)

    def _flush_pending_errors(self, buffer, pending, bad_skills):
        for item in pending:
            self._process_error_item(buffer, item, bad_skills)
        pending.clear()

    def _process_error_item(self, buffer, item, bad_skills):
        for agent_id in item["agents"]:
            segment = self._extract_segment(buffer, item["step"], agent_id)
            if segment is None:
                continue
            z_bad = self._encode_segment(*segment)
            if z_bad is not None:
                bad_skills.append(z_bad)

    def _extract_segment(self, buffer, center_step, agent_id):
        if not buffer:
            return None
        last_idx = len(buffer) - 1
        if center_step > last_idx:
            return None
        indices = []
        for offset in range(-self.segment_pre, self.segment_post + 1):
            idx = center_step + offset
            if idx < 0:
                idx = 0
            elif idx > last_idx:
                idx = last_idx
            indices.append(idx)

        obs_seq = [buffer[idx]["obs"][agent_id] for idx in indices]
        act_seq = [buffer[idx]["actions"][agent_id] for idx in indices]
        return np.asarray(obs_seq), np.asarray(act_seq)

    def _encode_segment(self, obs_seq, act_seq):
        obs_formatted = self._format_obs(obs_seq)
        act_onehot = self._to_onehot(act_seq)
        if obs_formatted is None or act_onehot is None:
            return None

        obs_tensor = torch.as_tensor(obs_formatted[None], device=self.device, dtype=torch.float32)
        act_tensor = torch.as_tensor(act_onehot[None], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            if self.skill_decoder_type == "vae":
                z_bad = self.skill_vae.encode(obs_tensor, act_tensor, self.task_id, deterministic=True)
            else:
                z_bad = self.skill_vae.encode(obs_tensor, act_tensor, self.task_id)
        return z_bad.detach().cpu().numpy()

    def _format_obs(self, obs_seq):
        obs_arr = np.asarray(obs_seq)
        if obs_arr.ndim != 4:
            return None
        obs_shape = _get_shape_from_space(self.envs.observation_space[0])
        obs_channels = obs_shape[-1] if len(obs_shape) >= 3 else obs_shape[0]
        if obs_arr.shape[-1] == obs_channels:
            obs_arr = obs_arr.transpose(0, 3, 1, 2)
        elif obs_arr.shape[1] != obs_channels:
            return None
        if obs_arr.shape[0] != self.segment_len:
            return None
        return obs_arr.astype(np.float32)

    def _to_onehot(self, actions):
        act_arr = np.asarray(actions)
        if act_arr.shape[0] != self.segment_len:
            return None
        action_space = self.envs.action_space[0]
        action_dim = action_space.n if hasattr(action_space, "n") else int(np.prod(action_space.shape))
        if act_arr.ndim == 1:
            return np.eye(action_dim, dtype=np.float32)[act_arr.astype(int)]
        if act_arr.ndim == 2 and act_arr.shape[1] == 1:
            return np.eye(action_dim, dtype=np.float32)[act_arr[:, 0].astype(int)]
        if act_arr.ndim == 2 and act_arr.shape[1] == action_dim:
            return act_arr.astype(np.float32)
        return None

    def _save_bad_skills(self, bad_skills, output_path=None):
        if not bad_skills:
            logger.warning("No bad skills collected.")
            return None

        out_path = output_path or getattr(self.all_args, "bad_skill_output", None)
        if out_path is None:
            out_path = self.run_dir / "bad_skills.npy"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        skills_arr = np.concatenate(bad_skills, axis=0)
        np.save(out_path, skills_arr)
        logger.info(f"Saved {skills_arr.shape[0]} bad skills to {out_path}")
        return out_path
