import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from zsceval.algorithms.utils.act import ACTLayer
from zsceval.algorithms.utils.cnn import CNNBase
from zsceval.algorithms.utils.mix import MIXBase
from zsceval.algorithms.utils.mlp import MLPBase, MLPLayer
from zsceval.algorithms.utils.popart import PopArt
from zsceval.algorithms.utils.rnn import RNNLayer
from zsceval.algorithms.utils.util import check, init
from zsceval.utils.util import get_shape_from_obs_space


class LowLevelActor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.skill_dim = getattr(args, "skill_dim", getattr(args, "latent_dim", 0))

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._influence_layer_N = args.influence_layer_N
        self._use_policy_vhead = args.use_policy_vhead
        self._use_popart = args.use_popart
        self._recurrent_N = args.recurrent_N
        self._layer_after_N = getattr(args, "layer_after_N", 0)
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)

        logger.trace(f"actor obs shape: {obs_shape}")
        if "Dict" in obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(args, obs_shape, cnn_layers_params=args.cnn_layers_params)
        else:
            self._mixed_obs = False
            self.base = (
                CNNBase(args, obs_shape, cnn_layers_params=args.cnn_layers_params)
                if len(obs_shape) == 3
                else MLPBase(
                    args,
                    obs_shape,
                    use_attn_internal=args.use_attn_internal,
                    use_cat_self=True,
                )
            )

        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                input_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(
                obs_shape[0],
                self.hidden_size,
                self._influence_layer_N,
                self._use_orthogonal,
                self._activation_id,
            )
            input_size += self.hidden_size

        if self._layer_after_N > 0:
            self.mlp_after = MLPLayer(
                input_size,
                input_size,
                self._layer_after_N,
                self._use_orthogonal,
                self._activation_id,
            )

        action_input_size = input_size + (self.skill_dim if self.skill_dim else 0)

        self.act = ACTLayer(action_space, action_input_size, self._use_orthogonal, self._gain)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_policy_vhead:
            if self._use_popart:
                self.v_out = init_(PopArt(action_input_size, 1, device=device))
            else:
                self.v_out = init_(nn.Linear(action_input_size, 1))

        self.to(device)

    def _append_skill(self, features, skill):
        if not self.skill_dim:
            return features
        if skill is None:
            skill = torch.zeros(features.shape[0], self.skill_dim, **self.tpdv)
        else:
            skill = check(skill).to(**self.tpdv)
            if skill.dim() > 2:
                skill = skill.view(skill.shape[0], -1)
        return torch.cat([features, skill], dim=1)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False, skill=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        actor_features = self._append_skill(actor_features, skill)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_transitions(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None, skill=None
    ):
        # ! only work for rnn model
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        actor_features = self._append_skill(actor_features, skill)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        values = self.v_out(actor_features) if self._use_policy_vhead else None

        return action_log_probs, dist_entropy, values, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None, skill=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        actor_features = self._append_skill(actor_features, skill)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        values = self.v_out(actor_features) if self._use_policy_vhead else None

        return action_log_probs, dist_entropy, values

    def get_policy_values(self, obs, rnn_states, masks, skill=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        actor_features = self._append_skill(actor_features, skill)

        values = self.v_out(actor_features)

        return values

    def get_probs(self, obs, rnn_states, masks, available_actions=None, skill=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        actor_features = self._append_skill(actor_features, skill)

        action_probs = self.act.get_probs(actor_features, available_actions)

        return action_probs, rnn_states

    def get_action_log_probs(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None, skill=None
    ):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        actor_features = self._append_skill(actor_features, skill)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        values = self.v_out(actor_features) if self._use_policy_vhead else None

        return action_log_probs, dist_entropy, values, rnn_states


class LowLevelCritic(nn.Module):
    def __init__(self, args, share_obs_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.skill_dim = getattr(args, "skill_dim", getattr(args, "latent_dim", 0))
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._use_popart = args.use_popart
        self._influence_layer_N = args.influence_layer_N
        self._recurrent_N = args.recurrent_N
        self._layer_after_N = getattr(args, "layer_after_N", 0)
        self._num_v_out = getattr(args, "num_v_out", 1)
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        logger.trace(f"critic share obs shape: {share_obs_shape}")

        if "Dict" in share_obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(args, share_obs_shape, cnn_layers_params=args.cnn_layers_params)
        else:
            self._mixed_obs = False
            self.base = (
                CNNBase(args, share_obs_shape, cnn_layers_params=args.cnn_layers_params)
                if len(share_obs_shape) == 3
                else MLPBase(
                    args,
                    share_obs_shape,
                    use_attn_internal=True,
                    use_cat_self=args.use_cat_self,
                )
            )

        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(
                share_obs_shape[0],
                self.hidden_size,
                self._influence_layer_N,
                self._use_orthogonal,
                self._activation_id,
            )
            input_size += self.hidden_size

        if self._layer_after_N > 0:
            self.mlp_after = MLPLayer(
                input_size,
                input_size,
                self._layer_after_N,
                self._use_orthogonal,
                self._activation_id,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        value_input_size = input_size + (self.skill_dim if self.skill_dim else 0)

        if self._use_popart:
            self.v_out = init_(PopArt(value_input_size, self._num_v_out, device=device))
        else:
            self.v_out = init_(nn.Linear(value_input_size, self._num_v_out))

        self.to(device)

    def _append_skill(self, features, skill):
        if not self.skill_dim:
            return features
        if skill is None:
            skill = torch.zeros(features.shape[0], self.skill_dim, **self.tpdv)
        else:
            skill = check(skill).to(**self.tpdv)
            if skill.dim() > 2:
                skill = skill.view(skill.shape[0], -1)
        return torch.cat([features, skill], dim=1)

    def forward(self, share_obs, rnn_states, masks, task_id=None, skill=None):
        if self._mixed_obs:
            for key in share_obs.keys():
                share_obs[key] = check(share_obs[key]).to(**self.tpdv)
        else:
            share_obs = check(share_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(share_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_share_obs = self.mlp(share_obs)
            critic_features = torch.cat([critic_features, mlp_share_obs], dim=1)

        if self._layer_after_N > 0:
            critic_features = self.mlp_after(critic_features)

        critic_features = self._append_skill(critic_features, skill)

        values = self.v_out(critic_features)

        if self._num_v_out > 1 and task_id is not None:
            assert len(task_id.shape) == len(values.shape) and np.prod(task_id.shape) * self._num_v_out == np.prod(
                values.shape
            ), (task_id.shape, values.shape)
            values = torch.gather(values, -1, task_id.long())

        return values, rnn_states


class HighLevelActor(nn.Module):
    """RNN-augmented TD3-style actor that outputs continuous latent skills."""

    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.skill_dim = getattr(args, "skill_dim", getattr(args, "latent_dim", 4))
        self.skill_range = getattr(args, "skill_range", 2.0)
        self.t_seg = getattr(args, "t_seg", 5)

        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._layer_after_N = getattr(args, "layer_after_N", 0)
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        logger.trace(f"high-level actor obs shape: {obs_shape}")

        if "Dict" in obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(args, obs_shape, cnn_layers_params=args.cnn_layers_params)
        else:
            self._mixed_obs = False
            self.base = (
                CNNBase(args, obs_shape, cnn_layers_params=args.cnn_layers_params)
                if len(obs_shape) == 3
                else MLPBase(
                    args,
                    obs_shape,
                    use_attn_internal=args.use_attn_internal,
                    use_cat_self=True,
                )
            )

        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                input_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )
            input_size = self.hidden_size

        if self._layer_after_N > 0:
            self.mlp_after = MLPLayer(
                input_size,
                input_size,
                self._layer_after_N,
                self._use_orthogonal,
                self._activation_id,
            )

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.skill_head = init_(nn.Linear(input_size, self.skill_dim))

        self.to(device)

    def forward(self, obs, rnn_states, masks):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        skill = torch.tanh(self.skill_head(actor_features)) * self.skill_range
        return skill, rnn_states


class HighLevelCritic(nn.Module):
    """RNN-augmented TD3-style twin critic. Inputs global state and all agents' skills."""

    def __init__(self, args, share_obs_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.skill_dim = getattr(args, "skill_dim", getattr(args, "latent_dim", 4))
        self.num_agents = getattr(args, "num_agents", 2)
        self.skill_concat_dim = self.skill_dim * self.num_agents

        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._layer_after_N = getattr(args, "layer_after_N", 0)
        self.tpdv = dict(dtype=torch.float32, device=device)

        share_obs_shape = get_shape_from_obs_space(share_obs_space)
        logger.trace(f"high-level critic share obs shape: {share_obs_shape}")

        if "Dict" in share_obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(args, share_obs_shape, cnn_layers_params=args.cnn_layers_params)
        else:
            self._mixed_obs = False
            self.base = (
                CNNBase(args, share_obs_shape, cnn_layers_params=args.cnn_layers_params)
                if len(share_obs_shape) == 3
                else MLPBase(
                    args,
                    share_obs_shape,
                    use_attn_internal=True,
                    use_cat_self=args.use_cat_self,
                )
            )

        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        if self._layer_after_N > 0:
            self.mlp_after = MLPLayer(
                input_size,
                input_size,
                self._layer_after_N,
                self._use_orthogonal,
                self._activation_id,
            )

        value_input_size = input_size + self.skill_concat_dim
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.q1 = init_(nn.Linear(value_input_size, 1))
        self.q2 = init_(nn.Linear(value_input_size, 1))

        self.to(device)

    def forward(self, share_obs, skills, rnn_states, masks):
        if not isinstance(share_obs, dict):
            nd = share_obs.ndim
            if nd == 5:
                # Use a representative agent's shared observation; for symmetric settings all agents see the same share_obs.
                share_obs = share_obs[:, 0]

        if self._mixed_obs:
            for key in share_obs.keys():
                share_obs[key] = check(share_obs[key]).to(**self.tpdv)
        else:
            share_obs = check(share_obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        skills = check(skills).to(**self.tpdv) if skills is not None else None
        if skills is None:
            skills = torch.zeros(share_obs.shape[0], self.skill_concat_dim, **self.tpdv)
        else:
            if skills.dim() > 2:
                skills = skills.view(skills.shape[0], -1)

        critic_features = self.base(share_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if self._layer_after_N > 0:
            critic_features = self.mlp_after(critic_features)

        critic_input = torch.cat([critic_features, skills], dim=1)
        q1 = self.q1(critic_input)
        q2 = self.q2(critic_input)
        return q1, q2, rnn_states
