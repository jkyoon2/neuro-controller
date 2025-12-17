import numpy as np
import torch

from zsceval.utils.util import get_shape_from_act_space, get_shape_from_obs_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


class HighLevelReplayBuffer:
    """
    Off-policy buffer for high-level TD3. Stores one entry per skill segment.
    """

    def __init__(self, capacity, obs_shape, share_obs_shape, skill_shape, device=torch.device("cpu")):
        self.capacity = capacity
        self.device = device
        self.obs_shape = obs_shape
        self.share_obs_shape = share_obs_shape
        self.skill_shape = skill_shape

        self.reset()

    def reset(self):
        self.obs = []
        self.share_obs = []
        self.skills = []
        self.rewards = []
        self.next_obs = []
        self.next_share_obs = []
        self.dones = []
        self.masks = []
        self.next_masks = []
        self.rnn_states_actor = []
        self.rnn_states_critic = []
        self.next_rnn_states_actor = []
        self.next_rnn_states_critic = []
        self._idx = 0

    def add(
        self,
        obs,
        share_obs,
        skills,
        reward,
        next_obs,
        next_share_obs,
        done,
        mask=None,
        next_mask=None,
        rnn_states_actor=None,
        rnn_states_critic=None,
        next_rnn_states_actor=None,
        next_rnn_states_critic=None,
    ):
        if len(self.obs) < self.capacity:
            self.obs.append(obs)
            self.share_obs.append(share_obs)
            self.skills.append(skills)
            self.rewards.append(reward)
            self.next_obs.append(next_obs)
            self.next_share_obs.append(next_share_obs)
            self.dones.append(done)
            self.masks.append(mask)
            self.next_masks.append(next_mask)
            self.rnn_states_actor.append(rnn_states_actor)
            self.rnn_states_critic.append(rnn_states_critic)
            self.next_rnn_states_actor.append(next_rnn_states_actor)
            self.next_rnn_states_critic.append(next_rnn_states_critic)
        else:
            idx = self._idx % self.capacity
            self.obs[idx] = obs
            self.share_obs[idx] = share_obs
            self.skills[idx] = skills
            self.rewards[idx] = reward
            self.next_obs[idx] = next_obs
            self.next_share_obs[idx] = next_share_obs
            self.dones[idx] = done
            self.masks[idx] = mask
            self.next_masks[idx] = next_mask
            self.rnn_states_actor[idx] = rnn_states_actor
            self.rnn_states_critic[idx] = rnn_states_critic
            self.next_rnn_states_actor[idx] = next_rnn_states_actor
            self.next_rnn_states_critic[idx] = next_rnn_states_critic
        self._idx += 1

    def __len__(self):
        return len(self.obs)

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.obs), size=batch_size, replace=len(self.obs) < batch_size)

        def _stack(lst):
            if lst[0] is None:
                return None
            return torch.as_tensor(np.stack([lst[i] for i in idxs]), device=self.device, dtype=torch.float32)

        batch = {
            "obs": torch.as_tensor(np.stack([self.obs[i] for i in idxs]), device=self.device, dtype=torch.float32),
            "share_obs": torch.as_tensor(
                np.stack([self.share_obs[i] for i in idxs]), device=self.device, dtype=torch.float32
            ),
            "skills": torch.as_tensor(np.stack([self.skills[i] for i in idxs]), device=self.device, dtype=torch.float32),
            "rewards": torch.as_tensor(
                np.stack([self.rewards[i] for i in idxs]), device=self.device, dtype=torch.float32
            ),
            "next_obs": torch.as_tensor(
                np.stack([self.next_obs[i] for i in idxs]), device=self.device, dtype=torch.float32
            ),
            "next_share_obs": torch.as_tensor(
                np.stack([self.next_share_obs[i] for i in idxs]), device=self.device, dtype=torch.float32
            ),
            "dones": torch.as_tensor(np.stack([self.dones[i] for i in idxs]), device=self.device, dtype=torch.float32),
        }

        batch["masks"] = _stack(self.masks)
        batch["next_masks"] = _stack(self.next_masks)
        batch["rnn_states_actor"] = _stack(self.rnn_states_actor)
        batch["rnn_states_critic"] = _stack(self.rnn_states_critic)
        batch["next_rnn_states_actor"] = _stack(self.next_rnn_states_actor)
        batch["next_rnn_states_critic"] = _stack(self.next_rnn_states_critic)
        return batch


class LowLevelRolloutBuffer:
    """
    On-policy buffer for low-level PPO, adapted from SeparatedReplayBuffer with skill storage.
    """

    def __init__(self, args, obs_space, share_obs_space, act_space, skill_dim):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.rnn_hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self._use_peb = args.use_peb

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)
        if isinstance(obs_shape[-1], list):
            obs_shape = obs_shape[:1]
        if isinstance(share_obs_shape[-1], list):
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *share_obs_shape),
            dtype=np.float32,
        )
        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *obs_shape),
            dtype=np.float32,
        )
        self.skills = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, skill_dim),
            dtype=np.float32,
        )

        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.recurrent_N, self.rnn_hidden_size),
            dtype=np.float32,
        )
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        if act_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones(
                (self.episode_length + 1, self.n_rollout_threads, act_space.n),
                dtype=np.float32,
            )
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)
        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)
        self.step = 0

    def insert(
        self,
        share_obs,
        obs,
        skills,
        rnn_states,
        rnn_states_critic,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        bad_masks=None,
        active_masks=None,
        available_actions=None,
    ):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.skills[self.step + 1] = skills.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None and self.available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.skills[0] = self.skills[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:
            terminated_masks = 1 - np.logical_and(self.masks == 0, self.bad_masks == 1).astype(float)
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = (
                            self.rewards[step]
                            + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * terminated_masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = delta + self.gamma * self.gae_lambda * terminated_masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = (
                            self.rewards[step]
                            + self.gamma * self.value_preds[step + 1] * terminated_masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = delta + self.gamma * self.gae_lambda * terminated_masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma * terminated_masks[step + 1] + self.rewards[step]
                    )
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = (
                            self.rewards[step]
                            + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = (
                            self.rewards[step]
                            + self.gamma * self.value_preds[step + 1] * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        if mini_batch_size is None:
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        skills = self.skills[:-1].reshape(-1, self.skills.shape[-1])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        else:
            available_actions = None

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            skills_batch = skills[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            adv_targ = advantages[indices]
            if available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None

            yield (
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                skills_batch,
            )

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        n_rollout_threads = self.rewards.shape[1]
        num_envs_per_batch = n_rollout_threads // num_mini_batch
        perm = torch.randperm(n_rollout_threads).numpy()
        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            skills_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(self.share_obs[:-1, ind])
                obs_batch.append(self.obs[:-1, ind])
                skills_batch.append(self.skills[:-1, ind])
                rnn_states_batch.append(self.rnn_states[0:1, ind])
                rnn_states_critic_batch.append(self.rnn_states_critic[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(self.available_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                active_masks_batch.append(self.active_masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.episode_length, num_envs_per_batch
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            skills_batch = np.stack(skills_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            rnn_states_batch = np.stack(rnn_states_batch, 1).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch, 1).reshape(N, *self.rnn_states_critic.shape[2:])

            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            skills_batch = _flatten(T, N, skills_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield (
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                skills_batch,
            )