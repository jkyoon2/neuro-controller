import torch
from loguru import logger

from zsceval.algorithms.hmarl.manager_worker import (
    HighLevelActor,
    HighLevelCritic,
    LowLevelActor,
    LowLevelCritic,
)
from zsceval.utils.util import update_linear_schedule


def _get_space_for_agent(space, agent_idx):
    """Handle both per-agent list/tuple of spaces and shared single space."""
    if isinstance(space, (list, tuple)):
        return space[agent_idx]
    return space


class HMARLLowLevelPolicy:
    """rMAPPO-style worker policy that conditions on latent skills."""

    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.data_parallel = getattr(args, "data_parallel", False)

        self.actor = LowLevelActor(args, self.obs_space, self.act_space, self.device)
        self.critic = LowLevelCritic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False,
        task_id=None,
        skill=None,
        **kwargs,
    ):
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic, skill=skill
        )
        values, rnn_states_critic = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id, skill=skill)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, share_obs, rnn_states_critic, masks, task_id=None, skill=None):
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id, skill=skill)
        return values

    def evaluate_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        task_id=None,
        skill=None,
    ):
        (
            action_log_probs,
            dist_entropy,
            policy_values,
        ) = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks, skill=skill
        )
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id, skill=skill)
        return values, action_log_probs, dist_entropy, policy_values

    def evaluate_transitions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        task_id=None,
        skill=None,
    ):
        (
            action_log_probs,
            dist_entropy,
            policy_values,
            rnn_states_actor,
        ) = self.actor.evaluate_transitions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks, skill=skill
        )
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id, skill=skill)
        return values, action_log_probs, dist_entropy, policy_values, rnn_states_actor

    def act(
        self,
        obs,
        rnn_states_actor,
        masks,
        available_actions=None,
        deterministic=False,
        skill=None,
        **kwargs,
    ):
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic, skill=skill
        )
        return actions, rnn_states_actor

    def get_probs(self, obs, rnn_states_actor, masks, available_actions=None, skill=None):
        action_probs, rnn_states_actor = self.actor.get_probs(
            obs, rnn_states_actor, masks, available_actions=available_actions, skill=skill
        )
        return action_probs, rnn_states_actor

    def get_action_log_probs(
        self,
        obs,
        rnn_states_actor,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        skill=None,
    ):
        action_log_probs, _, _, rnn_states_actor = self.actor.get_action_log_probs(
            obs, rnn_states_actor, action, masks, available_actions, active_masks, skill=skill
        )
        return action_log_probs, rnn_states_actor

    def load_checkpoint(self, ckpt_path):
        if "actor" in ckpt_path:
            self.actor.load_state_dict(torch.load(ckpt_path["actor"], map_location=self.device))
        if "critic" in ckpt_path:
            self.critic.load_state_dict(torch.load(ckpt_path["critic"], map_location=self.device))

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    def prep_training(self):
        self.actor.train()
        self.critic.train()


class HMARLHighLevelPolicy:
    """rTD3-style manager policy with per-agent actors and shared twin critic."""

    def __init__(self, args, obs_space, share_obs_space, act_space=None, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = getattr(args, "critic_lr", args.lr)
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.num_agents = args.num_agents
        self.t_seg = getattr(args, "t_seg", 1)

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space

        self.actors = torch.nn.ModuleList(
            [HighLevelActor(args, _get_space_for_agent(obs_space, i), device=self.device) for i in range(self.num_agents)]
        )
        self.critic = HighLevelCritic(args, self.share_obs_space, device=self.device)

        actor_params = []
        for actor in self.actors:
            actor_params += list(actor.parameters())

        self.actor_optimizer = torch.optim.Adam(
            actor_params,
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        step_in_seg: int = 0,
        prev_skills=None,
        deterministic=False,
        **kwargs,
    ):
        # obs, rnn_states_actor, masks expected shape: (B, num_agents, ...)
        reuse_skill = step_in_seg % self.t_seg != 0 and prev_skills is not None

        if reuse_skill:
            skills = prev_skills
            next_rnn_actor = rnn_states_actor
        else:
            skills = []
            next_rnn_actor = []
            for i, actor in enumerate(self.actors):
                agent_obs = obs[:, i]
                agent_rnn = rnn_states_actor[:, i]
                agent_mask = masks[:, i] if masks is not None else masks
                skill, next_rnn = actor(agent_obs, agent_rnn, agent_mask)
                skills.append(skill)
                next_rnn_actor.append(next_rnn)
            skills = torch.stack(skills, dim=1)  # (B, N, skill_dim)
            next_rnn_actor = torch.stack(next_rnn_actor, dim=1)

        critic_mask = masks[:, 0] if masks is not None else masks
        q1, q2, next_rnn_critic = self.critic(share_obs, skills, rnn_states_critic, critic_mask)

        # Keep return signature similar to rMAPPOPolicy: values, actions, action_log_probs, rnn_actor, rnn_critic
        return (q1, q2), skills, None, next_rnn_actor, next_rnn_critic

    def get_values(self, share_obs, skills, rnn_states_critic, masks):
        critic_mask = masks[:, 0] if masks is not None else masks
        q1, q2, _ = self.critic(share_obs, skills, rnn_states_critic, critic_mask)
        return q1, q2

    def evaluate_actions(
        self,
        share_obs,
        skills,
        rnn_states_critic,
        masks,
    ):
        critic_mask = masks[:, 0] if masks is not None else masks
        q1, q2, _ = self.critic(share_obs, skills, rnn_states_critic, critic_mask)
        # TD3 does not use log_probs; return placeholder None for interface symmetry
        return (q1, q2), None, None, None

    def act(self, obs, rnn_states_actor, masks, step_in_seg: int = 0, prev_skills=None, deterministic=True, **kwargs):
        reuse_skill = step_in_seg % self.t_seg != 0 and prev_skills is not None
        if reuse_skill:
            return prev_skills, rnn_states_actor

        skills = []
        next_rnn_actor = []
        for i, actor in enumerate(self.actors):
            agent_obs = obs[:, i] # obs: torch.Size([64, 2, 13, 5, 25]) / agent_obs: torch.Size([64, 13, 5, 25])
            # breakpoint()
            agent_rnn = rnn_states_actor[:, i] # torch.Size([64, 2, 1, 64])
            agent_mask = masks[:, i] if masks is not None else masks
            skill, next_rnn = actor(agent_obs, agent_rnn, agent_mask) # skill: torch.Size([64, 4])
            skills.append(skill)
            next_rnn_actor.append(next_rnn) # next_rnn: torch.Size([64, 1, 64])
        skills = torch.stack(skills, dim=1)
        next_rnn_actor = torch.stack(next_rnn_actor, dim=1)
        return skills, next_rnn_actor

    def prep_rollout(self):
        for actor in self.actors:
            actor.eval()
        self.critic.eval()

    def prep_training(self):
        for actor in self.actors:
            actor.train()
        self.critic.train()

    def to(self, device):
        for actor in self.actors:
            actor.to(device)
        self.critic.to(device)


class HMARLPolicy:
    """Composite policy that bundles high- and low-level policies."""

    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.num_agents = getattr(args, "num_agents", 2)
        self.share_policy = getattr(args, "share_policy", False)

        def _as_agent_list(space, count):
            if isinstance(space, (list, tuple)):
                return list(space)
            return [space for _ in range(count)]

        obs_spaces = _as_agent_list(obs_space, self.num_agents)
        share_obs_spaces = _as_agent_list(share_obs_space, self.num_agents)
        act_spaces = _as_agent_list(act_space, self.num_agents) if act_space is not None else [None] * self.num_agents

        # Low-level policy drives environment actions.
        self.low_levels = []
        if self.share_policy:
            shared_low = HMARLLowLevelPolicy(args, obs_spaces[0], share_obs_spaces[0], act_spaces[0], device=device)
            self.low_levels = [shared_low for _ in range(self.num_agents)]
        else:
            for i in range(self.num_agents):
                self.low_levels.append(
                    HMARLLowLevelPolicy(args, obs_spaces[i], share_obs_spaces[i], act_spaces[i], device=device)
                )
        self.low_level = self.low_levels[0]  # backward compatibility
        # High-level policy outputs latent skills.
        high_share_obs_space = share_obs_space[0] if isinstance(share_obs_space, (list, tuple)) else share_obs_space
        self.high_level = HMARLHighLevelPolicy(args, obs_space, high_share_obs_space, act_space=None, device=device)

        # Expose actor/critic for compatibility with existing trainer/save flows.
        self.actor = self.low_level.actor
        self.critic = self.low_level.critic
        self.actor_optimizer = self.low_level.actor_optimizer
        self.critic_optimizer = self.low_level.critic_optimizer

    # Low-level conveniences -------------------------------------------------
    def get_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False,
        task_id=None,
        skill=None,
        **kwargs,
    ):
        return self.low_level.get_actions(
            share_obs,
            obs,
            rnn_states_actor,
            rnn_states_critic,
            masks,
            available_actions=available_actions,
            deterministic=deterministic,
            task_id=task_id,
            skill=skill,
            **kwargs,
        )

    def get_values(self, share_obs, rnn_states_critic, masks, task_id=None, skill=None):
        return self.low_level.get_values(share_obs, rnn_states_critic, masks, task_id=task_id, skill=skill)

    def evaluate_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        task_id=None,
        skill=None,
    ):
        return self.low_level.evaluate_actions(
            share_obs,
            obs,
            rnn_states_actor,
            rnn_states_critic,
            action,
            masks,
            available_actions=available_actions,
            active_masks=active_masks,
            task_id=task_id,
            skill=skill,
        )

    def evaluate_transitions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        task_id=None,
        skill=None,
    ):
        return self.low_level.evaluate_transitions(
            share_obs,
            obs,
            rnn_states_actor,
            rnn_states_critic,
            action,
            masks,
            available_actions=available_actions,
            active_masks=active_masks,
            task_id=task_id,
            skill=skill,
        )

    def act(
        self,
        obs,
        rnn_states_actor,
        masks,
        available_actions=None,
        deterministic=False,
        skill=None,
        **kwargs,
    ):
        return self.low_level.act(
            obs,
            rnn_states_actor,
            masks,
            available_actions=available_actions,
            deterministic=deterministic,
            skill=skill,
            **kwargs,
        )

    def get_probs(self, obs, rnn_states_actor, masks, available_actions=None, skill=None):
        return self.low_level.get_probs(obs, rnn_states_actor, masks, available_actions=available_actions, skill=skill)

    def get_action_log_probs(
        self,
        obs,
        rnn_states_actor,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        skill=None,
    ):
        return self.low_level.get_action_log_probs(
            obs,
            rnn_states_actor,
            action,
            masks,
            available_actions=available_actions,
            active_masks=active_masks,
            skill=skill,
        )

    # High-level conveniences -----------------------------------------------
    def get_high_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        step_in_seg: int = 0,
        prev_skills=None,
        deterministic=False,
        **kwargs,
    ):
        return self.high_level.get_actions(
            share_obs,
            obs,
            rnn_states_actor,
            rnn_states_critic,
            masks,
            step_in_seg=step_in_seg,
            prev_skills=prev_skills,
            deterministic=deterministic,
            **kwargs,
        )

    def get_high_values(self, share_obs, skills, rnn_states_critic, masks):
        return self.high_level.get_values(share_obs, skills, rnn_states_critic, masks)

    def evaluate_high_actions(self, share_obs, skills, rnn_states_critic, masks):
        return self.high_level.evaluate_actions(share_obs, skills, rnn_states_critic, masks)

    def act_high(self, obs, rnn_states_actor, masks, step_in_seg: int = 0, prev_skills=None, deterministic=True, **kwargs):
        return self.high_level.act(
            obs,
            rnn_states_actor,
            masks,
            step_in_seg=step_in_seg,
            prev_skills=prev_skills,
            deterministic=deterministic,
            **kwargs,
        )

    # Shared helpers ---------------------------------------------------------
    def lr_decay(self, episode, episodes):
        for low in self.low_levels:
            low.lr_decay(episode, episodes)
        self.high_level.lr_decay(episode, episodes)

    def prep_rollout(self):
        for low in self.low_levels:
            low.prep_rollout()
        self.high_level.prep_rollout()

    def prep_training(self):
        for low in self.low_levels:
            low.prep_training()
        self.high_level.prep_training()

    def to(self, device):
        for low in self.low_levels:
            low.to(device)
        self.high_level.to(device)

    def load_checkpoint(self, ckpt_path):
        if isinstance(ckpt_path, dict):
            if "low_level" in ckpt_path:
                if isinstance(ckpt_path["low_level"], (list, tuple)):
                    for low, path in zip(self.low_levels, ckpt_path["low_level"]):
                        low.load_checkpoint(path)
                else:
                    for low in self.low_levels:
                        low.load_checkpoint(ckpt_path["low_level"])
            else:
                for low in self.low_levels:
                    low.load_checkpoint(ckpt_path)
            if "high_level" in ckpt_path:
                high_ckpt = ckpt_path["high_level"]
                if "critic" in high_ckpt:
                    self.high_level.critic.load_state_dict(torch.load(high_ckpt["critic"], map_location=self.device))
                if "actors" in high_ckpt:
                    for actor, path in zip(self.high_level.actors, high_ckpt["actors"]):
                        actor.load_state_dict(torch.load(path, map_location=self.device))
        else:
            for low in self.low_levels:
                low.load_checkpoint(ckpt_path)
