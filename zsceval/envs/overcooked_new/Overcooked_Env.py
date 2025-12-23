import os
import pickle
import pprint
import time
from collections import defaultdict

import gym
import imageio
import numpy as np
import tqdm
from loguru import logger

from zsceval.envs.overcooked_new.script_agent.script_agent import SCRIPT_AGENTS
from zsceval.utils.train_util import setup_seed

from .src.overcooked_ai_py.mdp.actions import Action, Direction
from .src.overcooked_ai_py.mdp.overcooked_mdp import (
    EVENT_TYPES,
    SHAPED_INFOS,
    OvercookedGridworld,
)
from .src.overcooked_ai_py.mdp.overcooked_trajectory import (
    DEFAULT_TRAJ_KEYS,
    TIMESTEP_TRAJ_KEYS,
)
from .src.overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MediumLevelActionManager,
    MotionPlanner,
)
from .src.overcooked_ai_py.utils import append_dictionaries, mean_and_std_err
from .src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer

DEFAULT_ENV_PARAMS = {"horizon": 400}

MAX_HORIZON = 1e10


class OvercookedEnv:
    """
    An environment wrapper for the OvercookedGridworld Markov Decision Process.

    The environment keeps track of the current state of the agent, updates
    it as the agent takes actions, and provides rewards to the agent.

    E.g. of how to instantiate OvercookedEnv:
    > mdp = OvercookedGridworld(...)
    > env = OvercookedEnv.from_mdp(mdp, horizon=400)
    """

    #########################
    # INSTANTIATION METHODS #
    #########################

    def __init__(
        self,
        mdp_generator_fn,
        start_state_fn=None,
        horizon=MAX_HORIZON,
        evaluation: bool = False,
        mlam_params=NO_COUNTERS_PARAMS,
        info_level=0,
        num_mdp=1,
        initial_info={},
    ):
        """
        mdp_generator_fn (callable):    A no-argument function that returns a OvercookedGridworld instance
        start_state_fn (callable):      Function that returns start state for the MDP, called at each environment reset
        horizon (int):                  Number of steps before the environment returns done=True
        mlam_params (dict):             params for MediumLevelActionManager
        info_level (int):               Change amount of logging
        num_mdp (int):                  the number of mdp if we are using a list of mdps
        initial_info (dict):            the initial outside information feed into the generator function

        TODO: Potentially make changes based on this discussion
        https://github.com/HumanCompatibleAI/overcooked_ai/pull/22#discussion_r416786847
        """
        assert callable(mdp_generator_fn), (
            "OvercookedEnv takes in a OvercookedGridworld generator function. "
            "If trying to instantiate directly from a OvercookedGridworld "
            "instance, use the OvercookedEnv.from_mdp method"
        )
        self.num_mdp = num_mdp
        self.mdp_generator_fn = mdp_generator_fn
        self.horizon = horizon
        self._mlam = None
        self._mp = None
        self.mlam_params = mlam_params
        self.start_state_fn = start_state_fn
        self.evaluation = evaluation
        self.info_level = info_level
        self.reset(outside_info=initial_info)
        if self.horizon >= MAX_HORIZON and self.info_level > 0:
            print(
                "Environment has (near-)infinite horizon and no terminal states. \
                Reduce info level of OvercookedEnv to not see this message."
            )

    @property
    def mlam(self):
        if self._mlam is None:
            if self.info_level > 0:
                print("Computing MediumLevelActionManager")
            self._mlam = MediumLevelActionManager.from_pickle_or_compute(
                self.mdp, self.mlam_params, force_compute=False
            )
        return self._mlam

    @property
    def mp(self):
        if self._mp is None:
            if self._mlam is not None:
                self._mp = self.mlam.motion_planner
            else:
                if self.info_level > 0:
                    print("Computing MotionPlanner")
                self._mp = MotionPlanner.from_pickle_or_compute(
                    self.mdp,
                    self.mlam_params["counter_goals"],
                    force_compute=False,
                )
        return self._mp

    @staticmethod
    def from_mdp(
        mdp,
        start_state_fn=None,
        horizon=MAX_HORIZON,
        mlam_params=NO_COUNTERS_PARAMS,
        info_level=0,
        evaluation: bool = False,
    ):
        """
        Create an OvercookedEnv directly from a OvercookedGridworld mdp
        rather than a mdp generating function.
        """
        assert isinstance(mdp, OvercookedGridworld)
        mdp_generator_fn = lambda _: mdp
        return OvercookedEnv(
            mdp_generator_fn=mdp_generator_fn,
            start_state_fn=start_state_fn,
            horizon=horizon,
            mlam_params=mlam_params,
            info_level=info_level,
            num_mdp=1,
            evaluation=evaluation,
        )

    #####################
    # BASIC CLASS UTILS #
    #####################

    @property
    def env_params(self):
        """
        Env params should be though of as all of the params of an env WITHOUT the mdp.
        Alone, env_params is not sufficent to recreate a copy of the Env instance, but it is
        together with mdp_params (which is sufficient to build a copy of the Mdp instance).
        """
        return {
            "start_state_fn": self.start_state_fn,
            "horizon": self.horizon,
            "info_level": self.info_level,
        }

    def copy(self):
        # TODO: Add testing for checking that these util methods are up to date?
        return OvercookedEnv(
            mdp_generator_fn=self.mdp_generator_fn,
            start_state_fn=self.start_state_fn,
            horizon=self.horizon,
            info_level=self.info_level,
            num_mdp=self.num_mdp,
        )

    #############################
    # ENV VISUALIZATION METHODS #
    #############################

    def __repr__(self):
        """
        Standard way to view the state of an environment programatically
        is just to print the Env object
        """
        return self.mdp.state_string(self.state)

    def display_states(self, *states):
        old_state = self.state
        for s in states:
            self.state = s
            print(self)
        self.state = old_state

    def print_state_transition(self, a_t, r_t, env_info, fname=None, display_phi=False):
        """
        Terminal graphics visualization of a state transition.
        """
        # TODO: turn this into a "formatting action probs" function and add action symbols too
        action_probs = [
            None if "action_probs" not in agent_info.keys() else list(agent_info["action_probs"])
            for agent_info in env_info["agent_infos"]
        ]

        action_probs = [
            None if player_action_probs is None else [round(p, 2) for p in player_action_probs[0]]
            for player_action_probs in action_probs
        ]

        if display_phi:
            state_potential_str = "\nState potential = " + str(env_info["phi_s_prime"]) + "\t"
            potential_diff_str = (
                "Δ potential = " + str(0.99 * env_info["phi_s_prime"] - env_info["phi_s"]) + "\n"
            )  # Assuming gamma 0.99
        else:
            state_potential_str = ""
            potential_diff_str = ""

        output_string = "Timestep: {}\nJoint action taken: {} \t Reward: {} + shaping_factor * {}\nAction probs by index: {} {} {}\n{}\n".format(
            self.state.timestep,
            tuple(Action.ACTION_TO_CHAR[a] for a in a_t),
            r_t,
            env_info["shaped_r_by_agent"],
            action_probs,
            state_potential_str,
            potential_diff_str,
            self,
        )

        if fname is None:
            print(output_string)
        else:
            f = open(fname, "a")
            print(output_string, file=f)
            f.close()

    ###################
    # BASIC ENV LOGIC #
    ###################

    def step(self, joint_action, joint_agent_action_info=None, display_phi=False):
        """Performs a joint action, updating the environment state
        and providing a reward.

        On being done, stats about the episode are added to info:
            ep_sparse_r: the environment sparse reward, given only at soup delivery
            ep_shaped_r: the component of the reward that is due to reward shaped (excluding sparse rewards)
            ep_length: length of rollout
        """
        assert not self.is_done()
        if joint_agent_action_info is None:
            joint_agent_action_info = [{} for _ in range(self.mdp.num_players)]
        next_state, mdp_infos = self.mdp.get_state_transition(self.state, joint_action, display_phi, self.mp)

        # Update game_stats
        self._update_game_stats(mdp_infos)

        # Update state and done
        self.state = next_state
        done = self.is_done()
        env_info = self._prepare_info_dict(joint_agent_action_info, mdp_infos)

        if done:
            self._add_episode_info(env_info)

        timestep_sparse_reward = sum(mdp_infos["sparse_reward_by_agent"])
        return next_state, timestep_sparse_reward, done, env_info

    def lossless_state_encoding_mdp(self, state, old_dynamics=False):
        """
        Wrapper of the mdp's lossless_encoding
        """
        return self.mdp.lossless_state_encoding(state, self.horizon, old_dynamics=old_dynamics)

    def featurize_state_mdp(self, state, num_pots=2):
        """
        Wrapper of the mdp's featurize_state
        """
        return self.mdp.featurize_state(state, self.mlam, num_pots=num_pots)

    def reset(self, regen_mdp=True, outside_info={}):
        """
        Resets the environment. Does NOT reset the agent.
        Args:
            regen_mdp (bool): gives the option of not re-generating mdp on the reset,
                                which is particularly helpful with reproducing results on variable mdp
            outside_info (dict): the outside information that will be fed into the scheduling_fn (if used), which will
                                 in turn generate a new set of mdp_params that is used to regenerate mdp.
                                 Please note that, if you intend to use this arguments throughout the run,
                                 you need to have a "initial_info" dictionary with the same keys in the "env_params"
        """
        if regen_mdp:
            self.mdp = self.mdp_generator_fn(outside_info)
            self._mlam = None
            self._mp = None
        if self.start_state_fn is None:
            self.state = self.mdp.get_standard_start_state()
        # MARK
        elif type(self.start_state_fn) in [float, int]:
            p = np.random.uniform(0, 1)
            if p <= self.start_state_fn and not self.evaluation:
                self.state = self.mdp.get_random_start_state()
            else:
                self.state = self.mdp.get_standard_start_state()
        else:
            self.state = self.start_state_fn()
        self.state = self.state.deepcopy()
        if hasattr(self.mdp, "reset_subgoal_tracking"):
            self.mdp.reset_subgoal_tracking(self.state)

        events_dict = {k: [[] for _ in range(self.mdp.num_players)] for k in EVENT_TYPES}
        rewards_dict = {
            "cumulative_sparse_rewards_by_agent": np.array([0] * self.mdp.num_players),
            "cumulative_shaped_rewards_by_agent": np.array([0] * self.mdp.num_players),
            "cumulative_category_rewards_by_agent": np.zeros((self.mdp.num_players, len(SHAPED_INFOS))),
        }
        self.game_stats = {**events_dict, **rewards_dict}
        return self.state

    def is_done(self):
        """Whether the episode is over."""
        return self.state.timestep >= self.horizon or self.mdp.is_terminal(self.state)

    def potential(self, mlam, state=None, gamma=0.99):
        """
        Return the potential of the environment's current state, if no state is provided
        Otherwise return the potential of `state`
        args:
            mlam (MediumLevelActionManager): the mlam of self.mdp
            state (OvercookedState): the current state we are evaluating the potential on
            gamma (float): discount rate
        """
        state = state if state else self.state
        return self.mdp.potential_function(state, mp=mlam.motion_planner, gamma=gamma)

    def _prepare_info_dict(self, joint_agent_action_info, mdp_infos):
        """
        The normal timestep info dict will contain infos specifc to each agent's action taken,
        and reward shaping information.
        """
        # Get the agent action info, that could contain info about action probs, or other
        # custom user defined information
        env_info = {"agent_infos": [joint_agent_action_info[agent_idx] for agent_idx in range(self.mdp.num_players)]}
        # TODO: This can be further simplified by having all the mdp_infos copied over to the env_infos automatically
        env_info["sparse_r_by_agent"] = mdp_infos["sparse_reward_by_agent"]
        env_info["shaped_r_by_agent"] = mdp_infos["shaped_reward_by_agent"]
        env_info["shaped_info_by_agent"] = mdp_infos["shaped_info_by_agent"]
        env_info["phi_s"] = mdp_infos["phi_s"] if "phi_s" in mdp_infos else None
        env_info["phi_s_prime"] = mdp_infos["phi_s_prime"] if "phi_s_prime" in mdp_infos else None
        return env_info

    def _add_episode_info(self, env_info):
        env_info["episode"] = {
            "ep_game_stats": self.game_stats,
            "ep_sparse_r": sum(self.game_stats["cumulative_sparse_rewards_by_agent"]),
            "ep_shaped_r": sum(self.game_stats["cumulative_shaped_rewards_by_agent"]),
            "ep_sparse_r_by_agent": self.game_stats["cumulative_sparse_rewards_by_agent"],
            "ep_shaped_r_by_agent": self.game_stats["cumulative_shaped_rewards_by_agent"],
            "ep_category_r_by_agent": self.game_stats["cumulative_category_rewards_by_agent"],
            "ep_length": self.state.timestep,
        }
        return env_info

    def vectorize_shaped_info(self, shaped_info_by_agent):
        def vectorize(d: dict):
            # return np.array([v for k, v in d.items()])
            return np.array([d[k] for k in SHAPED_INFOS])

        shaped_info_by_agent = np.stack([vectorize(shaped_info) for shaped_info in shaped_info_by_agent])
        return shaped_info_by_agent

    def _update_game_stats(self, infos):
        """
        Update the game stats dict based on the events of the current step
        NOTE: the timer ticks after events are logged, so there can be events from time 0 to time self.horizon - 1
        """
        self.game_stats["cumulative_sparse_rewards_by_agent"] += np.array(infos["sparse_reward_by_agent"])
        self.game_stats["cumulative_shaped_rewards_by_agent"] += np.array(infos["shaped_reward_by_agent"])
        self.game_stats["cumulative_category_rewards_by_agent"] += self.vectorize_shaped_info(
            infos["shaped_info_by_agent"]
        )

        for event_type, bool_list_by_agent in infos["event_infos"].items():
            # For each event type, store the timestep if it occurred
            event_occurred_by_idx = [int(x) for x in bool_list_by_agent]
            for idx, event_by_agent in enumerate(event_occurred_by_idx):
                if event_by_agent:
                    self.game_stats[event_type][idx].append(self.state.timestep)

    ####################
    # TRAJECTORY LOGIC #
    ####################

    def execute_plan(self, start_state, joint_action_plan, display=False):
        """Executes action_plan (a list of joint actions) from a start
        state in the mdp and returns the resulting state."""
        self.state = start_state
        done = False
        if display:
            print(f"Starting state\n{self}")
        for joint_action in joint_action_plan:
            self.step(joint_action)
            done = self.is_done()
            if display:
                print(self)
            if done:
                break
        successor_state = self.state
        self.reset()
        return successor_state, done

    def run_agents(
        self,
        agent_pair,
        include_final_state=False,
        display=False,
        dir=None,
        display_phi=False,
        display_until=np.Inf,
    ):
        """
        Trajectory returned will a list of state-action pairs (s_t, joint_a_t, r_t, done_t, info_t).
        """
        assert self.state.timestep == 0, "Did not reset environment before running agents"
        trajectory = []
        done = False
        # default is to not print to file
        fname = None

        if dir != None:
            fname = dir + "/roll_out_" + str(time.time()) + ".txt"
            f = open(fname, "w+")
            print(self, file=f)
            f.close()
        while not done:
            s_t = self.state

            # Getting actions and action infos (optional) for both agents
            joint_action_and_infos = agent_pair.joint_action(s_t)
            a_t, a_info_t = zip(*joint_action_and_infos)
            assert all(a in Action.ALL_ACTIONS for a in a_t)
            assert all(type(a_info) is dict for a_info in a_info_t)

            s_tp1, r_t, done, info = self.step(a_t, a_info_t, display_phi)
            trajectory.append((s_t, a_t, r_t, done, info))

            if display and self.state.timestep < display_until:
                self.print_state_transition(a_t, r_t, info, fname, display_phi)

        assert len(trajectory) == self.state.timestep, f"{len(trajectory)} vs {self.state.timestep}"

        # Add final state
        if include_final_state:
            trajectory.append((s_tp1, (None, None), 0, True, None))

        total_sparse = sum(self.game_stats["cumulative_sparse_rewards_by_agent"])
        total_shaped = sum(self.game_stats["cumulative_shaped_rewards_by_agent"])
        return (
            np.array(trajectory, dtype=object),
            self.state.timestep,
            total_sparse,
            total_shaped,
        )

    def get_rollouts(
        self,
        agent_pair,
        num_games,
        display=False,
        dir=None,
        final_state=False,
        display_phi=False,
        display_until=np.Inf,
        metadata_fn=None,
        metadata_info_fn=None,
        info=True,
    ):
        """
        Simulate `num_games` number rollouts with the current agent_pair and returns processed
        trajectories.

        Returning excessive information to be able to convert trajectories to any required format
        (baselines, stable_baselines, etc)

        metadata_fn returns some metadata information computed at the end of each trajectory based on
        some of the trajectory data.

        NOTE: this is the standard trajectories format used throughout the codebase
        """
        trajectories = {k: [] for k in DEFAULT_TRAJ_KEYS}
        metadata_fn = (lambda x: {}) if metadata_fn is None else metadata_fn
        metadata_info_fn = (lambda x: "") if metadata_info_fn is None else metadata_info_fn
        range_iterator = tqdm.trange(num_games, desc="", leave=True) if info else range(num_games)
        for i in range_iterator:
            agent_pair.set_mdp(self.mdp)

            rollout_info = self.run_agents(
                agent_pair,
                display=display,
                dir=dir,
                include_final_state=final_state,
                display_phi=display_phi,
                display_until=display_until,
            )
            (
                trajectory,
                time_taken,
                tot_rews_sparse,
                _,
            ) = rollout_info
            obs, actions, rews, dones, infos = (
                trajectory.T[0],
                trajectory.T[1],
                trajectory.T[2],
                trajectory.T[3],
                trajectory.T[4],
            )
            trajectories["ep_states"].append(obs)
            trajectories["ep_actions"].append(actions)
            trajectories["ep_rewards"].append(rews)
            trajectories["ep_dones"].append(dones)
            trajectories["ep_infos"].append(infos)
            trajectories["ep_returns"].append(tot_rews_sparse)
            trajectories["ep_lengths"].append(time_taken)
            trajectories["mdp_params"].append(self.mdp.mdp_params)
            trajectories["env_params"].append(self.env_params)
            trajectories["metadatas"].append(metadata_fn(rollout_info))

            # we do not need to regenerate MDP if we are trying to generate a series of rollouts using the same MDP
            # Basically, the FALSE here means that we are using the same layout and starting positions
            # (if regen_mdp == True, resetting will call mdp_gen_fn to generate another layout & starting position)
            self.reset(regen_mdp=False)
            agent_pair.reset()

            if info:
                mu, se = mean_and_std_err(trajectories["ep_returns"])
                description = "Avg rew: {:.2f} (std: {:.2f}, se: {:.2f}); avg len: {:.2f}; ".format(
                    mu,
                    np.std(trajectories["ep_returns"]),
                    se,
                    np.mean(trajectories["ep_lengths"]),
                )
                description += metadata_info_fn(trajectories["metadatas"])
                range_iterator.set_description(description)
                range_iterator.refresh()

        # Converting to numpy arrays
        trajectories = {k: np.array(v) for k, v in trajectories.items()}

        # Merging all metadata dictionaries, assumes same keys throughout all
        trajectories["metadatas"] = append_dictionaries(trajectories["metadatas"])

        # TODO: should probably transfer check methods over to Env class
        from zsceval.envs.overcooked_new.src.overcooked_ai_py.agents.benchmarking import (
            AgentEvaluator,
        )

        AgentEvaluator.check_trajectories(trajectories, verbose=info)
        return trajectories

    ####################
    # TRAJECTORY UTILS #
    ####################

    @staticmethod
    def get_discounted_rewards(trajectories, gamma):
        rews = trajectories["ep_rewards"]
        horizon = rews.shape[1]
        return OvercookedEnv._get_discounted_rewards_with_horizon(rews, gamma, horizon)

    @staticmethod
    def _get_discounted_rewards_with_horizon(rewards_matrix, gamma, horizon):
        rewards_matrix = np.array(rewards_matrix)
        discount_array = [gamma**i for i in range(horizon)]
        rewards_matrix = rewards_matrix[:, :horizon]
        discounted_rews = np.sum(rewards_matrix * discount_array, axis=1)
        return discounted_rews

    @staticmethod
    def get_agent_infos_for_trajectories(trajectories, agent_idx):
        """
        Returns a dictionary of the form
        {
            "[agent_info_0]": [ [episode_values], [], ... ],
            "[agent_info_1]": [ [], [], ... ],
            ...
        }
        with as keys the keys returned by the agent in it's agent_info dictionary

        NOTE: deprecated
        """
        agent_infos = []
        for traj_idx in range(len(trajectories["ep_lengths"])):
            ep_infos = trajectories["ep_infos"][traj_idx]
            traj_agent_infos = [step_info["agent_infos"][agent_idx] for step_info in ep_infos]

            # Append all dictionaries together
            traj_agent_infos = append_dictionaries(traj_agent_infos)
            agent_infos.append(traj_agent_infos)

        # Append all dictionaries together once again
        agent_infos = append_dictionaries(agent_infos)
        agent_infos = {k: np.array(v) for k, v in agent_infos.items()}
        return agent_infos

    @staticmethod
    def proportion_stuck_time(trajectories, agent_idx, stuck_time=3):
        """
        Simple util for calculating a guess for the proportion of time in the trajectories
        during which the agent with the desired agent index was stuck.

        NOTE: deprecated
        """
        stuck_matrix = []
        for traj_idx in range(len(trajectories["ep_lengths"])):
            stuck_matrix.append([])
            obs = trajectories["ep_states"][traj_idx]
            for traj_timestep in range(stuck_time, trajectories["ep_lengths"][traj_idx]):
                if traj_timestep >= stuck_time:
                    recent_states = obs[traj_timestep - stuck_time : traj_timestep + 1]
                    recent_player_pos_and_or = [s.players[agent_idx].pos_and_or for s in recent_states]

                    if len({item for item in recent_player_pos_and_or}) == 1:
                        # If there is only one item in the last stuck_time steps, then we classify the agent as stuck
                        stuck_matrix[traj_idx].append(True)
                    else:
                        stuck_matrix[traj_idx].append(False)
                else:
                    stuck_matrix[traj_idx].append(False)
        return stuck_matrix


class Overcooked(gym.Env):
    """
    Wrapper for the Env class above that is SOMEWHAT compatible with the standard gym API.

    NOTE: Observations returned are in a dictionary format with various information that is
    necessary to be able to handle the multi-agent nature of the environment. There are probably
    better ways to handle this, but we found this to work with minor modifications to OpenAI Baselines.

    NOTE: The index of the main agent in the mdp is randomized at each reset of the environment, and
    is kept track of by the self.agent_idx attribute. This means that it is necessary to pass on this
    information in the output to know for which agent index featurizations should be made for other agents.

    For example, say one is training A0 paired with A1, and A1 takes a custom state featurization.
    Then in the runner.py loop in OpenAI Baselines, we will get the lossless encodings of the state,
    and the true Overcooked state. When we encode the true state to feed to A1, we also need to know
    what agent index it has in the environment (as encodings will be index dependent).
    """

    env_name = "Overcooked-v0"

    def __init__(
        self,
        all_args,
        run_dir,
        baselines_reproducible=True,
        featurize_type=None,  # (N-player-ver) Changed to None for dynamic default
        stuck_time=4,
        rank=None,
        evaluation=False,
        old_dynamics: bool = None,
    ):
        """
        base_env: OvercookedEnv
        featurize_fn(mdp, state): fn used to featurize states returned in the 'both_agent_obs' field
        """
        if baselines_reproducible:
            # NOTE:
            # This will cause all agent indices to be chosen in sync across simulation
            # envs (for each update, all envs will have index 0 or index 1).
            # This is to prevent the randomness of choosing agent indexes
            # from leaking when using subprocess-vec-env in baselines (which
            # seeding does not reach) i.e. having different results for different
            # runs with the same seed.
            # The effect of this should be negligible, as all other randomness is
            # controlled by the actual run seeds
            np.random.seed(0)
        self.all_args = all_args
        self.agent_idx = 0
        self._initial_reward_shaping_factor = all_args.initial_reward_shaping_factor
        self.reward_shaping_factor = all_args.reward_shaping_factor
        self.reward_shaping_horizon = all_args.reward_shaping_horizon
        self.use_phi = getattr(all_args, "use_phi", False)
        self.use_hsp = getattr(all_args, "use_hsp", True)
        self.store_traj = getattr(all_args, "store_traj", False)
        self.rank = rank
        self.random_index = all_args.random_index

        self.use_available_actions = getattr(all_args, "use_available_actions", True)
        # Only render if use_render is True AND this env's rank is within n_render_rollout_threads
        n_render_threads = getattr(all_args, "n_render_rollout_threads", 1)
        self.use_render = all_args.use_render and (rank is not None and rank < n_render_threads)
        self.num_agents = all_args.num_agents
        self.layout_name = all_args.layout_name
        self.episode_length = all_args.episode_length
        self.random_start_prob = getattr(all_args, "random_start_prob", 0.0)
        self.stuck_time = stuck_time
        self.history_sa = []
        self.traj_num = 0
        self.step_count = 0
        self.run_dir = run_dir

        if self.use_hsp:
            # HSP 관련 로직을 N명 에이전트에 맞게 일반화
            self.hsp_weights = []
            for i in range(self.num_agents):
                weight_str = getattr(all_args, f"w{i}", None)
                if weight_str:
                    self.hsp_weights.append(self.string2array(weight_str))
            
            w_dict = {f"w{i}": str(w) for i, w in enumerate(self.hsp_weights)}
            logger.debug("hsp weights:\n" + pprint.pformat(w_dict, compact=True, width=120))
            self.cumulative_hidden_reward = np.zeros(self.num_agents)
        
        # (N-player-ver) Set dynamic default for featurize_type based on num_agents
        if featurize_type is None:
            featurize_type = tuple(["ppo"] * self.num_agents)
        
        # Validate featurize_type length matches num_agents
        assert len(featurize_type) == self.num_agents, \
            f"Length of featurize_type ({len(featurize_type)}) does not match num_agents ({self.num_agents})."
        if not old_dynamics:
            self.old_dynamics = all_args.old_dynamics
        else:
            self.old_dynamics = old_dynamics
        mdp_params = {"layout_name": all_args.layout_name, "start_order_list": None}
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }
        # MARK: use reward shaping
        mdp_params.update(
            {
                "rew_shaping_params": rew_shaping_params,
                "old_dynamics": self.old_dynamics,
            }
        )
        env_params = {
            "horizon": all_args.episode_length,
            "evaluation": evaluation,
        }

        # if getattr(all_args, "stage", 1) == 1:
        #     rew_shaping_params = {
        #         "PLACEMENT_IN_POT_REW": 0,
        #         "DISH_PICKUP_REWARD": 3,
        #         "SOUP_PICKUP_REWARD": 5,
        #         "PICKUP_TOMATO_REWARD": 0,
        #         "DISH_DISP_DISTANCE_REW": 0,
        #         "POT_DISTANCE_REW": 0,
        #         "SOUP_DISTANCE_REW": 0,
        #         "USEFUL_TOMATO_PICKUP": 0,
        #         "FOLLOW_TOMATO": 0,
        #         "PLACE_FIRST_TOMATO": 0,
        #     }
        # TODO: tune these
        # else:
        #     if self.layout_name == "distant_tomato":
        #         rew_shaping_params = {
        #             "PLACEMENT_IN_POT_REW": 0,
        #             "DISH_PICKUP_REWARD": 3,
        #             "SOUP_PICKUP_REWARD": 5,
        #             "PICKUP_TOMATO_REWARD": 0,
        #             "DISH_DISP_DISTANCE_REW": 0,
        #             "POT_DISTANCE_REW": 0,
        #             "SOUP_DISTANCE_REW": 0,
        #             "USEFUL_TOMATO_PICKUP": 10,
        #             "FOLLOW_TOMATO": 5,
        #             "PLACE_FIRST_TOMATO": -10,
        #         }
        #     else:
        #         rew_shaping_params = {
        #             "PLACEMENT_IN_POT_REW": 0,
        #             "DISH_PICKUP_REWARD": 3,
        #             "SOUP_PICKUP_REWARD": 5,
        #             "PICKUP_TOMATO_REWARD": 0,
        #             "DISH_DISP_DISTANCE_REW": 0,
        #             "POT_DISTANCE_REW": 0,
        #             "SOUP_DISTANCE_REW": 0,
        #             "USEFUL_TOMATO_PICKUP": 0,
        #             "FOLLOW_TOMATO": 0,
        #             "PLACE_FIRST_TOMATO": 0,
        #         }
        # self.base_mdp = OvercookedGridworld.from_layout_name(
        #     all_args.layout_name, rew_shaping_params=rew_shaping_params
        # )
        self.base_mdp = OvercookedGridworld.from_layout_name(**mdp_params)
        self.base_env = OvercookedEnv.from_mdp(
            self.base_mdp,
            start_state_fn=self.random_start_prob if self.random_start_prob > 0 else None,
            **env_params,
        )
        self.use_agent_policy_id = dict(all_args._get_kwargs()).get(
            "use_agent_policy_id", False
        )  # Add policy id for loaded policy
        self.agent_policy_id = [-1.0 for _ in range(self.num_agents)]
        self.featurize_fn_ppo = lambda state: self.base_env.lossless_state_encoding_mdp(
            state, old_dynamics=self.old_dynamics
        )  # Encoding obs for PPO
        self.featurize_fn_bc = lambda state: self.base_env.featurize_state_mdp(state)  # Encoding obs for BC
        self.featurize_fn_mapping = {
            "ppo": self.featurize_fn_ppo,
            "bc": self.featurize_fn_bc,
        }
        self.reset_featurize_type(featurize_type=featurize_type)  # default agents are both ppo

        # (N-player-ver) Script agent 초기화
        # all_args에 N개의 정책 이름을 담은 리스트/튜플이 있다고 가정
        # 예: all_args.agent_policy_names = ["ppo", "ppo", "script:my_script"]
        
        # agent_policy_names 속성 확인 (없으면 기존 2인용 방식 사용)
        if hasattr(all_args, 'agent_policy_names'):
            # N명 버전: agent_policy_names 사용
            if len(all_args.agent_policy_names) != self.num_agents:
                logger.warning(
                    f"agent_policy_names length ({len(all_args.agent_policy_names)}) "
                    f"does not match num_agents ({self.num_agents}). Using non-scripted agents."
                )
                self.script_agent = [None] * self.num_agents
            elif self.all_args.algorithm_name == "population":
                assert not self.random_index
                # N명의 에이전트에 대해 script_agent 리스트 초기화
                self.script_agent = [None] * self.num_agents
                
                # N번 반복하며 agent_policy_names 리스트 사용
                for player_idx, policy_name in enumerate(all_args.agent_policy_names):
                    if policy_name is not None and policy_name.startswith("script:"):
                        # 스크립트 에이전트 이름 추출
                        script_name = policy_name[7:]
                        if script_name in SCRIPT_AGENTS:
                            self.script_agent[player_idx] = SCRIPT_AGENTS[script_name]()
                            # Note: reset will be called in reset() method
                        else:
                            logger.warning(f"Script agent '{script_name}' not found in SCRIPT_AGENTS.")
            else:
                # population 알고리즘이 아닐 경우 N명 모두 None으로 초기화
                self.script_agent = [None] * self.num_agents
        else:
            # (N-player-ver) 기존 2인용 방식 (하위 호환성) - N명 길이로 초기화
            logger.warning(
                "'agent_policy_names' not found in all_args. Falling back to 2-agent policy names "
                "(agent0_policy_name, agent1_policy_name) for compatibility, "
                f"but initializing script_agent list for {self.num_agents} agents."
            )
            # N명 길이로 초기화 (중요!)
            self.script_agent = [None] * self.num_agents
            
            # 기존 2인용 로직은 최대 2명까지만 채움
            if self.all_args.algorithm_name == "population":
                assert not self.random_index
                # agent0_policy_name, agent1_policy_name 사용 (최대 2개 인덱스만 접근)
                if hasattr(all_args, 'agent0_policy_name') and hasattr(all_args, 'agent1_policy_name'):
                    policy_names_legacy = [all_args.agent0_policy_name, all_args.agent1_policy_name]
                    for player_idx in range(min(self.num_agents, 2)):  # 최대 2번만 반복
                        policy_name = policy_names_legacy[player_idx]
                        if policy_name.startswith("script:"):
                            script_name = policy_name[7:]
                            if script_name in SCRIPT_AGENTS:
                                self.script_agent[player_idx] = SCRIPT_AGENTS[script_name]()
                                # Note: reset will be called in reset() method
                            else:
                                logger.warning(f"Script agent '{script_name}' not found in SCRIPT_AGENTS.")
            # else: already initialized with [None] * self.num_agents

    def reset_featurize_type(self, featurize_type=None):
        """
        (N-player-ver) Reset featurize type for all agents.
        If featurize_type is None, defaults to all "ppo".
        """
        # Set default if not provided
        if featurize_type is None:
            featurize_type = tuple(["ppo"] * self.num_agents)
        
        # Validate featurize_type length matches num_agents
        assert len(featurize_type) == self.num_agents, \
            f"Expected {self.num_agents} featurize types, got {len(featurize_type)}"
        self.featurize_type = featurize_type
        
        # Create featurize function for N agents
        self.featurize_fn = lambda state: [
            self.featurize_fn_mapping[f](state)[i] * (255 if f == "ppo" else 1)
            for i, f in enumerate(self.featurize_type)
        ]

        # Reset observation_space, share_observation_space and action_space
        self.observation_space = []
        self.share_observation_space = []
        self.action_space = []
        self._setup_observation_space()

        # Setup spaces for all N agents
        for i in range(self.num_agents): 
            self.observation_space.append(self._observation_space(featurize_type[i]))
            self.action_space.append(gym.spaces.Discrete(len(Action.ALL_ACTIONS)))
            self.share_observation_space.append(self._setup_share_observation_space())

    def _anneal(self, start_v, curr_t, end_t, end_v=0, start_t=0):
        if end_t == 0:
            # No annealing if horizon is zero
            return start_v
        else:
            off_t = curr_t - start_t
            # Calculate the new value based on linear annealing formula
            fraction = max(1 - float(off_t) / (end_t - start_t), 0)
            return fraction * start_v + (1 - fraction) * end_v

    def onehot2idx(self, onehot):
        idx = []
        for a in onehot:
            idx.append(np.argmax(a))
        return idx

    def string2array(self, weight):
        if isinstance(weight, list):
            return np.array(weight).astype(np.float32)
        w = []
        for s in weight.split(","):
            w.append(float(s))
        return np.array(w).astype(np.float32)

    def _action_convertor(self, action):
        return [a[0] for a in list(action)]

    def _observation_space(self, featurize_type):
        return {"ppo": self.ppo_observation_space, "bc": self.bc_observation_space}[featurize_type]

    def _setup_observation_space(self):
        dummy_state = self.base_env.mdp.get_standard_start_state()

        # ppo observation
        featurize_fn_ppo = lambda state: self.base_env.lossless_state_encoding_mdp(
            state, old_dynamics=self.old_dynamics
        )
        obs_shape = featurize_fn_ppo(dummy_state)[0].shape
        high = np.ones(obs_shape) * float("inf")
        low = np.ones(obs_shape) * 0
        self.ppo_observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)

        # bc observation
        featurize_fn_bc = lambda state: self.base_env.featurize_state_mdp(state)
        obs_shape = featurize_fn_bc(dummy_state)[0].shape
        high = np.ones(obs_shape) * 100
        low = np.ones(obs_shape) * -100
        self.bc_observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)

    def _setup_share_observation_space(self):
        """
        (N-player-ver) Setup shared observation space to match _gen_share_observation output.
        Calculates the shape based on global channels (not individual agent channels).
        """
        dummy_state = self.base_env.mdp.get_standard_start_state()
        mdp = self.base_env.mdp
        grid_shape = mdp.shape  # (H, W)

        # Calculate total global channels to match _gen_share_observation
        # 1. Global Player Features: 1 (all_player_loc) + 4 (orientations) = 5
        num_global_player_channels = 1 + len(Direction.ALL_DIRECTIONS)
        
        # 2. Base Map Features: 6 channels
        #    pot_loc, counter_loc, onion_disp_loc, tomato_disp_loc, dish_disp_loc, serve_loc
        num_base_map_channels = 6
            
        # 3. Variable Map Features: 8 channels
        #    onions_in_pot, tomatoes_in_pot, onions_in_soup, tomatoes_in_soup,
        #    soup_cook_time_remaining, dishes, onions, tomatoes
        num_variable_map_channels = 8
        
        # 4. Urgency Features: 1 channel
        num_urgency_channels = 1
        
        # Total global channels
        total_global_channels = (
            num_global_player_channels 
            + num_base_map_channels 
            + num_variable_map_channels 
            + num_urgency_channels
        )  # Expected: 5 + 6 + 8 + 1 = 20

        # Final shared observation shape (H, W, C_global)
        share_obs_shape = (grid_shape[0], grid_shape[1], total_global_channels)

        # Add policy ID channel if needed
        if self.use_agent_policy_id:
            final_channels = total_global_channels + 1
            share_obs_shape = (grid_shape[0], grid_shape[1], final_channels)

        # Create Gym space
        high = np.ones(share_obs_shape, dtype=np.float32) * float("inf")
        low = np.zeros(share_obs_shape, dtype=np.float32)

        return gym.spaces.Box(low, high, dtype=np.float32)

    def _set_agent_policy_id(self, agent_policy_id):
        self.agent_policy_id = agent_policy_id

# In Overcooked_Env.py
    
    def _gen_share_observation(self, state):
        """
        (N-player-ver) Generates a global shared observation.
        Player info uses 5 channels: 
        - 1 channel for all player locations.
        - 4 channels for all player orientations (spatial one-hot).
        Returns a stack of N identical copies of this global observation.
        """
        # --- Re-implement channel creation logic globally ---
        
        # Get base MDP and shape
        mdp = self.base_env.mdp # Or self.mdp depending on class structure
        grid_shape = mdp.shape
        num_players = self.num_agents # Or mdp.num_players

        # Define features similar to lossless_state_encoding
        # (excluding ego/partner specifics)
        base_map_features = [
            "pot_loc", "counter_loc", "onion_disp_loc", 
            "tomato_disp_loc", "dish_disp_loc", "serve_loc",
        ]
        variable_map_features = [
            "onions_in_pot", "tomatoes_in_pot", "onions_in_soup", 
            "tomatoes_in_soup", "soup_cook_time_remaining", 
            "dishes", "onions", "tomatoes",
        ]
        urgency_features = ["urgency"]
        
        # Define NEW global player features (5 channels)
        global_player_features = ["all_player_loc"] + [
            f"all_player_orientation_{Direction.DIRECTION_TO_INDEX[d]}" 
            for d in Direction.ALL_DIRECTIONS
        ]

        SHARED_LAYERS = global_player_features + base_map_features + variable_map_features + urgency_features
        state_mask_dict = {k: np.zeros(grid_shape) for k in SHARED_LAYERS}

        # --- Fill channels (adapted from lossless_state_encoding) ---

        def make_layer(position, value):
            layer = np.zeros(grid_shape)
            layer[position] = value
            return layer

        # MAP LAYERS (Copied from lossless_state_encoding logic)
        horizon = self.base_env.horizon # Get horizon
        if horizon - state.timestep < 40: # Use state passed to function
            state_mask_dict["urgency"] = np.ones(grid_shape)
        for loc in mdp.get_counter_locations(): state_mask_dict["counter_loc"][loc] = 1
        for loc in mdp.get_pot_locations(): state_mask_dict["pot_loc"][loc] = 1
        for loc in mdp.get_onion_dispenser_locations(): state_mask_dict["onion_disp_loc"][loc] = 1
        for loc in mdp.get_tomato_dispenser_locations(): state_mask_dict["tomato_disp_loc"][loc] = 1
        for loc in mdp.get_dish_dispenser_locations(): state_mask_dict["dish_disp_loc"][loc] = 1
        for loc in mdp.get_serving_locations(): state_mask_dict["serve_loc"][loc] = 1

        # GLOBAL PLAYER LAYERS (NEW logic)
        for player in state.players: # Iterate through all players
            player_position = player.position
            player_orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            
            # Mark position in the single 'all_player_loc' channel
            state_mask_dict["all_player_loc"][player_position] = 1 
            
            # Mark position in the corresponding orientation channel
            state_mask_dict[f"all_player_orientation_{player_orientation_idx}"][player_position] = 1

        # OBJECT & STATE LAYERS (Copied from lossless_state_encoding logic)
        # Need Counter if used in original code
        from collections import Counter # Make sure Counter is imported
        
        all_objects = state.all_objects_list
        for obj in all_objects:
            # (Copy the object processing logic exactly from lossless_state_encoding)
            # ... (soup, dish, onion, tomato logic) ...
            if obj.name == "soup":
                ingredients_dict = Counter(obj.ingredients)
                if obj.position in mdp.get_pot_locations():
                    if obj.is_idle:
                        state_mask_dict["onions_in_pot"] += make_layer(obj.position, ingredients_dict.get("onion", 0))
                        state_mask_dict["tomatoes_in_pot"] += make_layer(obj.position, ingredients_dict.get("tomato", 0))
                    else:
                        state_mask_dict["onions_in_soup"] += make_layer(obj.position, ingredients_dict.get("onion", 0))
                        state_mask_dict["tomatoes_in_soup"] += make_layer(obj.position, ingredients_dict.get("tomato", 0))
                        state_mask_dict["soup_cook_time_remaining"] += make_layer(obj.position, obj.cook_time_remaining)
                else:
                    state_mask_dict["onions_in_soup"] += make_layer(obj.position, ingredients_dict.get("onion", 0))
                    state_mask_dict["tomatoes_in_soup"] += make_layer(obj.position, ingredients_dict.get("tomato", 0))
                    state_mask_dict["soup_cook_time_remaining"] += make_layer(obj.position, obj.cook_time_remaining)
            elif obj.name == "dish":
                state_mask_dict["dishes"] += make_layer(obj.position, 1)
            elif obj.name == "onion":
                state_mask_dict["onions"] += make_layer(obj.position, 1)
            elif obj.name == "tomato":
                state_mask_dict["tomatoes"] += make_layer(obj.position, 1)
            else:
                # Ensure all object types are handled or raise an error
                # Consider adding robust error handling or logging for unrecognized objects
                print(f"Warning: Unrecognized object type '{obj.name}' encountered during observation generation.")
                # Depending on requirements, either raise ValueError("Unrecognized object") or continue

        # --- Stack layers ---
        global_state_mask_stack = np.array([state_mask_dict[layer_id] for layer_id in SHARED_LAYERS])
        # Transpose to (Height, Width, Channels)
        global_state_mask_stack = np.transpose(global_state_mask_stack, (1, 2, 0)).astype(np.float32) # Use float32

        # Optional: Multiply by 255 if needed (check if downstream code expects this)
        # global_state_mask_stack *= 255 

        # --- Return N identical copies stacked ---
        # The standard approach for shared observations in MARL frameworks
        # is often to provide the same global view to each agent's critic.
        share_obs_stack = np.stack([global_state_mask_stack] * num_players, axis=0)
        
        # Optional: Add policy ID if needed (adapt the old logic)
        # if self.use_agent_policy_id:
        #    policy_ids = np.array(self.agent_policy_id[:num_players]).reshape(num_players, 1, 1, 1)
        #    policy_id_features = np.ones_like(share_obs_stack[..., :1]) * policy_ids
        #    share_obs_stack = np.concatenate([share_obs_stack, policy_id_features], axis=-1)

        return share_obs_stack

    def _get_available_actions(self):
        available_actions = np.ones((self.num_agents, len(Action.ALL_ACTIONS)), dtype=np.uint8)
        if self.use_available_actions:
            state = self.base_env.state
            interact_index = Action.ACTION_TO_INDEX["interact"]
            for agent_idx in range(self.num_agents):
                player = state.players[agent_idx]
                pos = player.position
                o = player.orientation
                for move_i, move in enumerate(Direction.ALL_DIRECTIONS):
                    new_pos = Action.move_in_direction(pos, move)
                    if new_pos not in self.base_mdp.get_valid_player_positions() and o == move:
                        available_actions[agent_idx, move_i] = 0

                i_pos = Action.move_in_direction(pos, o)
                terrain_type = self.base_mdp.get_terrain_type_at_pos(i_pos)

                if (
                    terrain_type == " "
                    or (
                        terrain_type == "X"
                        and (
                            (not player.has_object() and not state.has_object(i_pos))
                            or (player.has_object() and state.has_object(i_pos))
                        )
                    )
                    or (terrain_type in ["O", "T", "D"] and player.has_object())
                    or (terrain_type == "S" and (not player.has_object() or player.get_object().name not in ["soup"]))
                ):
                    available_actions[agent_idx, interact_index] = 0
                if terrain_type == "P":
                    if player.has_object():
                        obj = player.get_object()
                        if obj.name not in [
                            "dish",
                            "onion",
                            "tomato",
                        ]:
                            available_actions[agent_idx, interact_index] = 0
                        # elif state.has_object(i_pos):
                        #     soup_obj = state.get_object(i_pos)
                        #     if obj.name in ["dish"]:
                        #         if not soup_obj.is_ready:
                        #             available_actions[agent_idx, interact_index] = 0
                        #     elif (
                        #             len(obj.ingredients) >= Recipe.MAX_NUM_INGREDIENTS
                        #         ):
                        #         available_actions[agent_idx, interact_index] = 0
                    elif not state.has_object(i_pos) or state.get_object(i_pos).is_ready or self.old_dynamics:
                        available_actions[agent_idx, interact_index] = 0

                # assert available_actions[agent_idx].sum() > 0
        return available_actions

    def step(self, action):
        """
        (N-player-ver)
        
        Args:
            action: Tuple/list of N action indices (0~5) for all agents
                    Example: [3, 1] for 2 agents
        
        Returns:
            both_agents_ob: Tuple of N observations
                            Example: (obs_agent0, obs_agent1), each (H, W, C)
            share_obs: Shared observation (N, H, W, C_share)
            reward: List of [r] for each agent
                    Example: [[0.5], [-0.1]] for 2 agents
            done: List of done flags
                  Example: [False, False]
            info: Dict containing:
                  - "shaped_info_timestep": List[Dict] - per-timestep event counts
                    Example: [{"onion_pickup": 1, "dish_pickup": 0, ...}, {...}]
                  - "shaped_info_by_agent": List[Dict] - cumulative episode counts
                  - "sparse_r": float - sparse reward (delivery)
                  - "shaped_r_by_agent": List[float] - dense reward per agent
                  - "detect_error": Dict[str, List[int]] - per-timestep error flags
                  - "all_agent_obs": np.array (N, H, W, C)
                  - "share_obs": np.array (N, H, W, C_share)
                  - "available_actions": np.array (N, num_actions)
            available_actions: np.array (N, num_actions) boolean mask
        """
        self.step_count += 1
        
        # Convert action to proper format
        action = self._action_convertor(action)
        assert len(action) == self.num_agents, f"Expected {self.num_agents} actions, got {len(action)}"
        
        # Check action validity
        assert all(self.action_space[0].contains(a) for a in action), "{!r} ({}) invalid".format(
            action,
            type(action),
        )

        # Convert action indices to Action objects for N agents
        joint_action = [Action.INDEX_TO_ACTION[a] for a in action]

        # Apply scripted agents if any
        for a in range(self.num_agents):
            if self.script_agent[a] is not None:
                joint_action[a] = self.script_agent[a].step(self.base_env.mdp, self.base_env.state, a)
        joint_action = tuple(joint_action)

        # Update history for stuck detection
        if self.stuck_time > 0:
            self.history_sa[-1][1] = joint_action

        # Base environment step
        if self.use_phi:
            raise NotImplementedError("N-player phi function not implemented")
        else:
            next_state, sparse_reward, done, info = self.base_env.step(joint_action, display_phi=False)

        # Calculate shaped rewards for N agents
        dense_reward = info["shaped_r_by_agent"]  # N-length list
        shaped_rewards_list = [0.0] * self.num_agents

        if self.use_hsp:
            # N-player HSP reward calculation (매 timestep 값만 사용)
            from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
            shaped_info_timestep = info["shaped_info_by_agent"]  # 매 timestep 값 (누적 전)
            vec_shaped_info = np.array([[agent_info[k] for k in SHAPED_INFOS] for agent_info in shaped_info_timestep]).astype(np.float32)

            hidden_rewards_for_log = np.zeros(self.num_agents)
            
            for i in range(self.num_agents):
                # 유연한 Weight 선택 로직
                # 1) 정책 관점 인덱스 i를 실제 환경 인덱스로 변환
                actual_agent_idx = (i + self.agent_idx) % self.num_agents
                
                # 2) 해당 에이전트의 weight 선택
                if actual_agent_idx < len(self.hsp_weights):
                    # 각 에이전트마다 개별 weight가 있는 경우 (독립적 파라미터 네트워크)
                    weight = self.hsp_weights[actual_agent_idx]
                elif len(self.hsp_weights) == 2:
                    # 2개의 weight만 있는 경우: main vs partner 구조 (기존 방식)
                    weight = self.hsp_weights[0] if (i == self.agent_idx) else self.hsp_weights[1]
                elif len(self.hsp_weights) == 1:
                    # 1개의 weight만 있는 경우: 모두 동일 (공유 파라미터 네트워크)
                    weight = self.hsp_weights[0]
                else:
                    raise ValueError(f"Invalid HSP weights configuration: {len(self.hsp_weights)} weights for {self.num_agents} agents")
                
                # Hidden reward 계산: dot(weight[:-1], events) + sparse_reward * weight[-1]
                hidden_r = np.dot(weight[:-1], vec_shaped_info[i]) + sparse_reward * weight[-1]
                hidden_rewards_for_log[i] = hidden_r
                
                # 최종 reward는 hidden_r과 동일 (모든 에이전트에 동일하게 적용)
                shaped_rewards_list[i] = hidden_r

            # 매 timestep의 hidden_r을 info에 추가 (누적값이 아닌 현재 timestep 값)
            info["hidden_r_by_agent"] = hidden_rewards_for_log.tolist()
            
            # Episode 누적을 위해 cumulative_hidden_reward 업데이트
            self.cumulative_hidden_reward += hidden_rewards_for_log
        else:
            # Basic shaped reward calculation for N players
            for i in range(self.num_agents):
                shaped_rewards_list[i] = sparse_reward + self.reward_shaping_factor * dense_reward[i]

        # Store per-timestep shaped info before accumulation
        info["shaped_info_timestep"] = info["shaped_info_by_agent"]
        
        # Update cumulative shaped info for N agents
        shaped_info = info["shaped_info_by_agent"]
        for i in range(self.num_agents):
            if shaped_info[i]:
                for k, v in shaped_info[i].items():
                    self.cumulative_shaped_info[i][k] += v
        info["shaped_info_by_agent"] = self.cumulative_shaped_info

        # Store trajectory if needed
        if self.store_traj:
            self.traj_to_store.append(info["shaped_info_by_agent"])
            self.traj_to_store.append(self.base_env.state.to_dict())

        # Prepare reward in required format
        reward = [[r] for r in shaped_rewards_list]

        # Update history
        self.history_sa = self.history_sa[1:] + [
            [next_state, None],
        ]

        # Calculate stuck info for N agents
        stuck_info = []
        for agent_id in range(self.num_agents):
            stuck, history_a = self.is_stuck(agent_id)
            if stuck:
                history_a_idxes = [Action.ACTION_TO_INDEX[a] for a in history_a]
                stuck_info.append([True, history_a_idxes])
            else:
                stuck_info.append([False, []])
        info["stuck"] = stuck_info

        # Render if needed
        if self.use_render:
            state = self.base_env.state
            self.traj["ep_states"][0].append(state)
            self.traj["ep_actions"][0].append(joint_action)
            self.traj["ep_rewards"][0].append(sparse_reward)
            self.traj["ep_dones"][0].append(done)
            self.traj["ep_infos"][0].append(info)
            if done:
                self.traj["ep_returns"].append(info["episode"]["ep_sparse_r"])
                self.traj["mdp_params"].append(self.base_mdp.mdp_params)
                self.traj["env_params"].append(self.base_env.env_params)
                self.render()

        # Handle episode termination
        if done:
            if self.store_traj:
                self._store_trajectory()
            if self.use_hsp:
                info["episode"]["ep_hidden_r_by_agent"] = self.cumulative_hidden_reward
            info["bad_transition"] = True
        else:
            info["bad_transition"] = False

        # Generate observations for N agents
        all_agent_obs = self.featurize_fn(next_state)
        
        # Return observations in tuple format (maintaining compatibility)
        both_agents_ob = tuple(all_agent_obs)

        # Generate shared observation
        share_obs = self._gen_share_observation(self.base_env.state)
        
        # Done flags for all agents
        done = [done] * self.num_agents
        
        # Get available actions for all agents
        available_actions = self._get_available_actions()

        # ===>>> info에 표준 정보 추가 <<<===
        info["all_agent_obs"] = np.array(all_agent_obs)  # (num_agents, H, W, C)
        info["share_obs"] = share_obs  # (num_agents, H, W, C_share)
        info["available_actions"] = available_actions  # (num_agents, num_actions)

        return both_agents_ob, share_obs, reward, done, info, available_actions

    def anneal_reward_shaping_factor(self, timesteps):
        """
        Set the current reward shaping factor such that we anneal linearly until self.reward_shaping_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        new_factor = self._anneal(self._initial_reward_shaping_factor, timesteps, self.reward_shaping_horizon)
        self.set_reward_shaping_factor(new_factor)

    def set_reward_shaping_factor(self, factor):
        self.reward_shaping_factor = factor

    def reset(self, reset_choose=True):
        """
        (N-player-ver, standard Gym format: obs, info)
        Resets the environment and randomizes the perspective agent index.
        Returns obs for the selected agent_idx and info dictionary containing
        share_obs, available_actions, and all_agent_obs.
        
        Returns:
            obs: Observation for the selected agent (current agent_idx)
            info: Dictionary containing:
                - 'all_agent_obs': numpy array of shape (num_agents, H, W, C)
                - 'share_obs': numpy array of shape (num_agents, H, W, C_share)
                - 'available_actions': numpy array of shape (num_agents, num_actions)
                - 'agent_idx': int, currently selected agent index
        """
        if reset_choose:
            self.traj_num += 1
            self.step_count = 0
            self.base_env.reset()
            # N명에 맞게 cumulative_shaped_info 초기화
            self.cumulative_shaped_info = [defaultdict(int) for _ in range(self.num_agents)]
            # HSP 누적 보상을 self.num_agents에 맞게 초기화
            if self.use_hsp:
                self.cumulative_hidden_reward = np.zeros(self.num_agents)

        # N명 중 랜덤하게 agent_idx 선택
        if self.random_index:
            self.agent_idx = np.random.choice(self.num_agents)
        else:
            self.agent_idx = 0  # 기본값 0

        # Scripted agent 리셋 (N명)
        for a in range(self.num_agents):
            if self.script_agent[a] is not None:
                self.script_agent[a].reset(self.base_env.mdp, self.base_env.state, a)

        self.mdp = self.base_env.mdp
        
        # featurize_fn이 N개의 observation 튜플/리스트를 반환한다고 가정
        all_agent_obs = self.featurize_fn(self.base_env.state)
        assert len(all_agent_obs) == self.num_agents, f"featurize_fn returned {len(all_agent_obs)} observations, expected {self.num_agents}"
        
        # history_sa 초기화 (N명 고려)
        if self.stuck_time > 0:
            self.history_sa = [None for _ in range(self.stuck_time - 1)] + [[self.base_env.state, None]]

        # 현재 agent_idx에 해당하는 관측치를 선택
        current_agent_obs = all_agent_obs[self.agent_idx]

        # Render 초기화
        if self.use_render:
            self.init_traj()

        # Trajectory 저장 초기화
        if self.store_traj:
            self.traj_to_store = []
            self.traj_to_store.append(self.base_env.state.to_dict())

        # HSP 관련 초기화 (N명)
        if self.use_hsp:
            self.cumulative_hidden_reward = np.zeros(self.num_agents)

        # 공유 관측 생성 (N명 버전)
        share_obs = self._gen_share_observation(self.base_env.state)
        
        # 사용 가능한 액션 (N명 버전)
        available_actions = self._get_available_actions()

        # ===>>> 표준 Gym info 딕셔너리 구성 <<<===
        info = {
            "all_agent_obs": np.array(all_agent_obs),  # (num_agents, H, W, C) - 모든 에이전트의 관측
            "share_obs": share_obs,  # (num_agents, H, W, C_share) - 공유 관측
            "available_actions": available_actions,  # (num_agents, num_actions) - 사용 가능한 액션
            "agent_idx": self.agent_idx  # 현재 선택된 에이전트 인덱스
        }
        
        # ===>>> 표준 Gym 반환값 (obs, info) <<<===
        return current_agent_obs, info

    def is_stuck(self, agent_id):
        if self.stuck_time == 0 or None in self.history_sa:
            return False, []
        history_s = [sa[0] for sa in self.history_sa]
        history_a = [sa[1][agent_id] for sa in self.history_sa[:-1]]  # last action is None
        player_s = [s.players[agent_id] for s in history_s]
        pos_and_ors = [p.pos_and_or for p in player_s]
        cur_po = pos_and_ors[-1]
        if all([po[0] == cur_po[0] and po[1] == cur_po[1] for po in pos_and_ors]):
            return True, history_a
        return False, []

    def init_traj(self):
        self.traj = {k: [] for k in DEFAULT_TRAJ_KEYS}
        for key in TIMESTEP_TRAJ_KEYS:
            self.traj[key].append([])

    def render(self):
        try:
            save_dir = f"{self.run_dir}/gifs/rank_{self.rank}_traj_num_{self.traj_num}"
            save_dir = os.path.expanduser(save_dir)
            StateVisualizer().display_rendered_trajectory(self.traj, img_directory_path=save_dir, ipython_display=False)
            for img_path in os.listdir(save_dir):
                img_path = save_dir + "/" + img_path
            imgs = []
            imgs_dir = os.listdir(save_dir)
            # Filter only PNG files before sorting
            imgs_dir = [f for f in imgs_dir if f.endswith('.png')]
            imgs_dir = sorted(imgs_dir, key=lambda x: int(x.split(".")[0]))
            for img_path in imgs_dir:
                img_path = save_dir + "/" + img_path
                try:
                    imgs.append(imageio.imread(img_path))
                except Exception as e:
                    print(f"Warning: Failed to read image {img_path}: {e}")
                    continue
            
            if not imgs:
                print("Warning: No images were successfully loaded, skipping gif creation")
            else:
                save_path = save_dir + f'/reward_{self.traj["ep_returns"][0]}.gif'
                imageio.mimsave(
                    save_path,
                    imgs,
                    duration=0.05,
                )
                print(f"save gifs in {save_path}")
            imgs_dir = os.listdir(save_dir)
            for img_path in imgs_dir:
                img_path = save_dir + "/" + img_path
                if "png" in img_path:
                    os.remove(img_path)
        except Exception as e:
            print("failed to render traj: ", e)

    def _store_trajectory(self):
        if not os.path.exists(f"{self.run_dir}/trajs/"):
            os.makedirs(f"{self.run_dir}/trajs/")
        save_dir = f"{self.run_dir}/trajs/traj_{self.rank}_{self.traj_num}.pkl"
        pickle.dump(self.traj_to_store, open(save_dir, "wb"))

    def seed(self, seed):
        setup_seed(seed)
        super().seed(seed)
