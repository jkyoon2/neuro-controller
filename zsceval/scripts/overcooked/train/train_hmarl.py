#!/usr/bin/env python

import argparse
import os
import pprint
import socket
import sys
from pathlib import Path

import setproctitle
import torch
import wandb
from loguru import logger

from zsceval.config import get_config
from zsceval.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocDummyBatchVecEnv
from zsceval.envs.overcooked.Overcooked_Env import Overcooked
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
from zsceval.overcooked_config import get_overcooked_args
from zsceval.utils.train_util import get_base_run_dir, setup_seed

os.environ["WANDB_DIR"] = os.getcwd() + "/wandb/"
os.environ["WANDB_CACHE_DIR"] = os.getcwd() + "/wandb/.cache/"
os.environ["WANDB_CONFIG_DIR"] = os.getcwd() + "/wandb/.config/"


def make_train_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.overcooked_version == "old":
                    env = Overcooked(all_args, run_dir, rank=rank)
                else:
                    env = Overcooked_new(all_args, run_dir, rank=rank)
            else:
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocDummyBatchVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)],
            all_args.dummy_batch_size,
        )


def make_eval_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.overcooked_version == "old":
                    env = Overcooked(all_args, run_dir, evaluation=True, rank=rank)
                else:
                    env = Overcooked_new(all_args, run_dir, evaluation=True, rank=rank)
            else:
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocDummyBatchVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)],
            all_args.dummy_batch_size,
        )


def parse_args(args, parser: argparse.ArgumentParser):
    parser = get_overcooked_args(parser)
    parser.add_argument("--skill_dim", type=int, default=4)
    parser.add_argument("--t_seg", type=int, default=5)
    parser.add_argument("--intrinsic_alpha", type=float, default=0.3)
    parser.add_argument("--vae_checkpoint_path", type=str, default="checkpoints/vae_epoch_004_1tasks_4d.pt")
    parser.add_argument("--high_buffer_size", type=int, default=50000)
    parser.add_argument("--high_batch_size", type=int, default=64)
    parser.add_argument("--intrinsic_scale", type=float, default=5.0)
    all_args = parser.parse_args(args)

    from zsceval.overcooked_config import OLD_LAYOUTS

    if all_args.layout_name in OLD_LAYOUTS:
        all_args.old_dynamics = True
    else:
        all_args.old_dynamics = False
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    base_run_dir = Path(get_base_run_dir())
    # Use sp/hmarl path (HMARL built on top of SP) and seed-specific subdirectory.
    run_dir = base_run_dir / all_args.env_name / all_args.layout_name / all_args.algorithm_name / all_args.experiment_name
    seed_run_dir = run_dir / f"seed{all_args.seed}"
    seed_run_dir.mkdir(parents=True, exist_ok=True)
    (seed_run_dir / "models").mkdir(exist_ok=True)
    (seed_run_dir / "gifs").mkdir(exist_ok=True)
    # Ensure downstream components see a plain string path
    all_args.run_dir = str(seed_run_dir)

    # sync render/gif flags: either one enables both
    wants_gif = getattr(all_args, "save_gifs", False)
    wants_render = getattr(all_args, "use_render", False)
    all_args.use_render = wants_render or wants_gif
    all_args.save_gifs = wants_render or wants_gif

    project_name = all_args.env_name if all_args.overcooked_version == "old" else all_args.env_name + "-new"
    run_dir = all_args.run_dir
    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=project_name,
            entity=all_args.wandb_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(all_args.seed),
            group=all_args.layout_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
            tags=all_args.wandb_tags,
        )
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-{all_args.env_name}_{all_args.layout_name}-{all_args.experiment_name}@{all_args.user_name}"
    )

    setup_seed(all_args.seed)

    envs = make_train_env(all_args, run_dir)
    eval_envs = make_eval_env(all_args, run_dir) if all_args.use_eval else None
    num_agents = all_args.num_agents

    logger.info(pprint.pformat(all_args.__dict__, compact=True))
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    from zsceval.runner.separated.hmarl_runner import HMARLRunner as Runner

    runner = Runner(config)
    runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish(quiet=True)
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir / "summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    main(sys.argv[1:])
