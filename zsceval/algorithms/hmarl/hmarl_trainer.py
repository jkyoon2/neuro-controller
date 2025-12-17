from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from zsceval.algorithms.hmarl.hmarl_policy import HMARLHighLevelPolicy, HMARLLowLevelPolicy, HMARLPolicy
from zsceval.algorithms.utils.util import check
from zsceval.utils.util import get_gard_norm, huber_loss, mse_loss
from zsceval.utils.valuenorm import ValueNorm


class LowLevelTrainer:
    """rMAPPO-style trainer for the worker, with skill conditioning."""

    def __init__(self, args, policy: HMARLLowLevelPolicy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.t_seg = getattr(args, "t_seg", 1)
        self.intrinsic_alpha = getattr(args, "intrinsic_alpha", 0.0)
        self.intrinsic_scale = getattr(args, "intrinsic_scale", 1.0)
        self.skill_decoder_type = getattr(args, "skill_decoder_type", "vae")
        self.vae = getattr(args, "skill_decoder", None)
        self.current_task_id = getattr(args, "current_task_id", None)

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.policy_value_loss_coef = args.policy_value_loss_coef
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coefs = args.entropy_coefs
        self.entropy_coef_horizons = args.entropy_coef_horizons

        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.share_policy = args.share_policy

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_policy_vhead = args.use_policy_vhead
        self._use_task_v_out = getattr(args, "use_task_v_out", False)

        assert (
            self._use_popart and self._use_valuenorm
        ) == False, "self._use_popart and self._use_valuenorm can not be set True simultaneously"

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
            if self._use_policy_vhead:
                self.policy_value_normalizer = self.policy.actor.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
            if self._use_policy_vhead:
                self.policy_value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
            if self._use_policy_vhead:
                self.policy_value_normalizer = None

    def adapt_entropy_coef(self, num_steps: int):
        n = len(self.entropy_coef_horizons)
        for i in range(n - 1):
            if self.entropy_coef_horizons[i] <= num_steps < self.entropy_coef_horizons[i + 1]:
                start_steps = self.entropy_coef_horizons[i]
                end_steps = self.entropy_coef_horizons[i + 1]
                start_coef = self.entropy_coefs[i]
                end_coef = self.entropy_coefs[i + 1]
                fraction = (num_steps - start_steps) / (end_steps - start_steps)
                self.entropy_coef = (1 - fraction) * start_coef + fraction * end_coef
                break
        else:
            self.entropy_coef = self.entropy_coefs[-1]

    def cal_value_loss(
        self,
        value_normalizer,
        values,
        value_preds_batch,
        return_batch,
        active_masks_batch,
    ):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)

        if self._use_popart or self._use_valuenorm:
            value_normalizer.update(return_batch)
            error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(
        self,
        sample,
        actor_zero_grad: bool = True,
        critic_zero_grad: bool = True,
    ):
        # detect sample layout and optional skill
        if self.share_policy:
            (
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
                *rest,
            ) = sample
        else:
            (
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
                *rest,
            ) = sample

        skill_batch = None
        other_policy_id_batch = None
        if rest:
            # handle (other_policy_id_batch,) or (other_policy_id_batch, skill_batch) or just (skill_batch,)
            if self._use_task_v_out:
                other_policy_id_batch = rest[0]
                if len(rest) > 1:
                    skill_batch = rest[1]
            else:
                skill_batch = rest[0]

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        if self._use_task_v_out and other_policy_id_batch is not None:
            other_policy_id_batch = check(other_policy_id_batch).to(**self.tpdv)

        (
            values,
            action_log_probs,
            dist_entropy,
            policy_values,
        ) = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            task_id=other_policy_id_batch if self._use_task_v_out else None,
            skill=skill_batch,
        )

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        if self._use_policy_vhead:
            policy_value_loss = self.cal_value_loss(
                self.policy_value_normalizer,
                policy_values,
                value_preds_batch,
                return_batch,
                active_masks_batch,
            )
        else:
            policy_value_loss = 0

        entropy_coef = getattr(self, "entropy_coef", self.entropy_coefs[-1])
        dist_entropy = dist_entropy.mean()
        actor_loss = policy_action_loss - entropy_coef * dist_entropy + policy_value_loss

        self.policy.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.tensor(0.0)
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(
            self.value_normalizer if self.value_normalizer is not None else None,
            values,
            value_preds_batch,
            return_batch,
            active_masks_batch,
        )

        self.policy.critic_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        critic_grad_norm = torch.tensor(0.0)
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()

        clipped_ratio = torch.min(ratio, torch.ones_like(ratio) * (1.0 + self.clip_param))
        lower_rate = (ratio < (1.0 - self.clip_param)).float().mean()
        upper_rate = (ratio > (1.0 + self.clip_param)).float().mean()

        return (
            value_loss,
            critic_grad_norm,
            policy_action_loss,
            dist_entropy,
            actor_grad_norm,
            clipped_ratio,
            upper_rate,
            lower_rate,
            entropy_coef,
        )

    def train(self, buffer, **kwargs):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = defaultdict(float)
        keys = [
            "value_loss",
            "policy_loss",
            "dist_entropy",
            "actor_grad_norm",
            "critic_grad_norm",
            "ratio",
            "upper_clip_rate",
            "lower_clip_rate",
            "entropy_coef",
        ]
        for k in keys:
            train_info[k] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    ratio,
                    upper_rate,
                    lower_rate,
                    entropy_coef,
                ) = self.ppo_update(sample, **kwargs)

                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm.item()
                train_info["critic_grad_norm"] += critic_grad_norm.item()
                train_info["ratio"] += ratio.mean().item()
                train_info["upper_clip_rate"] += upper_rate.item()
                train_info["lower_clip_rate"] += lower_rate.item()
                train_info["entropy_coef"] += entropy_coef

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

    def to(self, device):
        self.policy.to(device)

    def compute_intrinsic_reward_value(self, obs_seq: torch.Tensor, action_seq: torch.Tensor, target_skill: torch.Tensor):
        """
        Compute intrinsic reward value (scaled) without blending with extrinsic.
        """
        if self.vae is None:
            return 0.0
        with torch.no_grad():
            if self.skill_decoder_type == "vae":
                z_actual = self.vae.encode(obs_seq, action_seq, self.current_task_id, deterministic=True)
            else:
                z_actual = self.vae.encode(obs_seq, action_seq, self.current_task_id)
            mse = torch.sum((target_skill - z_actual) ** 2, dim=-1)
            intrinsic = self.intrinsic_scale * (1.0 / (1.0 + mse))
        return intrinsic.item()

    def compute_intrinsic_reward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor, target_skill: torch.Tensor):
        """
        Tensor-returning wrapper preserved for compatibility.
        """
        value = self.compute_intrinsic_reward_value(obs_seq, action_seq, target_skill)
        return torch.as_tensor(value, device=self.device).repeat(target_skill.shape[0])

    def mix_extrinsic_intrinsic(self, extrinsic, intrinsic_val):
        """
        Formula: alpha * extrinsic + (1 - alpha) * intrinsic.
        """
        if self.intrinsic_alpha == 0.0:
            return extrinsic
        return self.intrinsic_alpha * extrinsic + (1 - self.intrinsic_alpha) * intrinsic_val


class HighLevelTrainer:
    """Minimal rTD3-style trainer for the manager."""

    def __init__(self, args, policy: HMARLHighLevelPolicy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.gamma = args.gamma
        self.tau = getattr(args, "tau", 0.005)
        self.policy_delay = getattr(args, "policy_delay", 2)
        self.train_step = 0
        self.t_seg = getattr(args, "t_seg", 1)

    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self, buffer, batch_size=None):
        """Assumes HighLevelReplayBuffer.sample returns dict with obs/next_obs/skills/rewards/dones/masks/rnn states."""
        batch = buffer.sample(batch_size)
        share_obs = check(batch["share_obs"]).to(**self.tpdv)
        obs = check(batch["obs"]).to(**self.tpdv)
        skills = check(batch["skills"]).to(**self.tpdv)
        rewards = check(batch["rewards"]).to(**self.tpdv)
        next_share_obs = check(batch["next_share_obs"]).to(**self.tpdv)
        next_obs = check(batch["next_obs"]).to(**self.tpdv)
        dones = check(batch["dones"]).to(**self.tpdv)
        masks = check(batch.get("masks", None)) if "masks" in batch else None
        next_masks = check(batch.get("next_masks", None)) if "next_masks" in batch else None

        rnn_states_actor = check(batch.get("rnn_states_actor", None)) if "rnn_states_actor" in batch else None
        rnn_states_critic = check(batch.get("rnn_states_critic", None)) if "rnn_states_critic" in batch else None
        next_rnn_states_actor = check(batch.get("next_rnn_states_actor", None)) if "next_rnn_states_actor" in batch else None
        next_rnn_states_critic = check(batch.get("next_rnn_states_critic", None)) if "next_rnn_states_critic" in batch else None

        # If rewards provided per-agent, average for central critic target
        if rewards.dim() > 1:
            rewards = rewards.mean(dim=1, keepdim=True)
        elif rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() > 2:
            dones = dones[:, 0]
        elif dones.dim() == 1:
            dones = dones.unsqueeze(-1)

        # Keep full masks for per-agent actors, but use a single mask channel for the shared critic.
        actor_masks = masks
        actor_next_masks = next_masks
        critic_masks = masks[:, 0] if masks is not None and masks.dim() > 2 else masks
        critic_next_masks = next_masks[:, 0] if next_masks is not None and next_masks.dim() > 2 else next_masks
        # Critic update
        with torch.no_grad():
            next_skills, _ = self.policy.act(
                obs=next_obs,
                rnn_states_actor=next_rnn_states_actor,
                masks=actor_next_masks,
            )
            q1_next, q2_next, _ = self.policy.critic(
                next_share_obs,
                next_skills,
                next_rnn_states_critic,
                critic_next_masks,
            )
            target_q = rewards + self.gamma * (1.0 - dones) * torch.min(q1_next, q2_next)

        q1, q2, _ = self.policy.critic(
            share_obs,   # torch.Size([64, 2, 13, 5, 20])
            skills,  # torch.Size([64, 2, 4])
            rnn_states_critic,  # torch.Size([64, 2, 1, 64])
            critic_masks,  # torch.Size([64, 1])
        )

        # breakpoint()
        # Compute critic loss with explicit reduction to keep scalar objective
        critic_loss = mse_loss(q1 - target_q).mean() + mse_loss(q2 - target_q).mean()
        self.policy.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 1.0)
        self.policy.critic_optimizer.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.train_step % self.policy_delay == 0:
            # Deterministic policy gradient: maximize Q
            curr_skills, _ = self.policy.act(obs=obs, rnn_states_actor=rnn_states_actor, masks=actor_masks)
            q1_pi, _, _ = self.policy.critic(
                share_obs,
                curr_skills,
                rnn_states_critic,
                critic_masks,
            )
            actor_loss = -q1_pi.mean()
            self.policy.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.actors.parameters(), 1.0)
            self.policy.actor_optimizer.step()

            # soft update targets if they exist
            if hasattr(self.policy, "target_critic"):
                self.soft_update(self.policy.critic, self.policy.target_critic)
            if hasattr(self.policy, "target_actors"):
                for src, tgt in zip(self.policy.actors, self.policy.target_actors):
                    self.soft_update(src, tgt)

        self.train_step += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss,
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "target_q_mean": target_q.mean().item(),
        }

    def prep_training(self):
        for actor in self.policy.actors:
            actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        for actor in self.policy.actors:
            actor.eval()
        self.policy.critic.eval()

    def to(self, device):
        self.policy.to(device)


class HMARLTrainer:
    """Wrapper that coordinates high-level (manager) and low-level (worker) trainers."""

    def __init__(self, args, policy: HMARLPolicy, device):
        self.policy = policy
        self.low_trainers = [LowLevelTrainer(args, lp, device=device) for lp in policy.low_levels]
        self.high_trainer = HighLevelTrainer(args, policy.high_level, device=device)
        self.value_normalizers = [lt.value_normalizer for lt in self.low_trainers]

    def train(self, buffers, high_buffer=None, batch_size=None, **kwargs):
        low_infos = [tr.train(buf, **kwargs) for tr, buf in zip(self.low_trainers, buffers)]
        high_info = None
        if high_buffer is not None:
            high_info = self.high_trainer.train(high_buffer, batch_size=batch_size)
        return {"low": low_infos, "high": high_info}

    def train_low(self, buffer, **kwargs):
        return [tr.train(buf, **kwargs) for tr, buf in zip(self.low_trainers, buffer)]

    def train_high(self, buffer, batch_size=None):
        return self.high_trainer.train(buffer, batch_size=batch_size)

    def prep_training(self):
        for tr in self.low_trainers:
            tr.prep_training()
        self.high_trainer.prep_training()

    def prep_rollout(self):
        for tr in self.low_trainers:
            tr.prep_rollout()
        self.high_trainer.prep_rollout()

    def adapt_entropy_coef(self, num_steps: int):
        for tr in self.low_trainers:
            tr.adapt_entropy_coef(num_steps)

    def to(self, device):
        self.policy.to(device)
        for tr in self.low_trainers:
            tr.to(device)
        self.high_trainer.to(device)
