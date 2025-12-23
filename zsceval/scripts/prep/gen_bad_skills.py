#!/usr/bin/env python
import sys
from pathlib import Path

import torch
import yaml

# 프로젝트 루트 경로 설정
sys.path.append(str(Path(__file__).resolve().parents[3]))

from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args
from zsceval.utils.bad_skill_generator import BadSkillGenerator


def load_yaml_config(yaml_path: Path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = get_config()
    parser = get_overcooked_args(parser)

    default_config_path = Path(__file__).resolve().parents[1] / "overcooked" / "config" / "gen_bad_skills.yaml"
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(default_config_path),
        help="Path to the YAML config file for bad skill generation",
    )
    parser.add_argument(
        "--target_error_types",
        nargs="+",
        type=str,
        default=None,
        help="List of error types to collect (e.g., COLLISION ONION_COUNTER_REGRAB). Defaults to all.",
    )

    args = parser.parse_args()

    if args.config_path:
        yaml_config = load_yaml_config(Path(args.config_path))
        for key, value in yaml_config.items():
            setattr(args, key, value)
            print(f"[Config] {key}: {value}")

    # 데이터 생성에선 로깅 비활성화
    args.use_wandb = False

    if args.cuda and torch.cuda.is_available():
        print("Choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)

    output_path = getattr(args, "bad_skill_output", None)
    run_dir = Path(output_path).parent if output_path else Path("results")

    config = {
        "all_args": args,
        "device": device,
        "run_dir": run_dir,
        "num_agents": args.num_agents,
    }

    print("Initializing BadSkillGenerator...")
    generator = BadSkillGenerator(config)

    num_episodes = getattr(args, "num_episodes", 10)
    output_path = getattr(args, "bad_skill_output", None)

    print(f"Start generation for {num_episodes} episodes...")
    save_path = generator.generate_bad_skills(num_episodes=num_episodes, output_path=output_path)

    if save_path:
        print(f"Successfully saved bad skills to: {save_path}")
    else:
        print("No bad skills collected.")


if __name__ == "__main__":
    main()
