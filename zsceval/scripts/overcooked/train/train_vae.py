"""
VAE 학습 스크립트 (Task-Aware + KL Annealing)

- YAML 설정 파일 기반
- 디렉토리 구조(레이아웃 이름)를 기반으로 Task ID 자동 부여
- KL Beta Annealing 적용 (Posterior Collapse 방지)
"""

import argparse
import glob
import pickle
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

from zsceval.algorithms.hmarl.vae import SkillVAE


# ========================================================================
# Utility Functions
# ========================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def discover_layouts(root_path: Path, layout_filter: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Discover layouts from directory structure and assign Task IDs
    
    Args:
        root_path: Root VAE data directory
        layout_filter: Optional list of layout names to include
    
    Returns:
        Dictionary mapping layout_name -> task_id
    """
    layout_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    
    if layout_filter:
        layout_dirs = [d for d in layout_dirs if d.name in layout_filter]
    
    layout_to_id = {d.name: i for i, d in enumerate(layout_dirs)}
    return layout_to_id


def filter_data_files(
    all_files: List[str],
    layout_to_id: Dict[str, int],
    seed_filter: Optional[List[int]] = None,
    min_step: Optional[int] = None,
    max_step: Optional[int] = None
) -> List[str]:
    """
    Filter data files based on layout, seed, and step criteria
    
    Args:
        all_files: List of all .pkl file paths
        layout_to_id: Valid layout names
        seed_filter: Optional list of seeds to include
        min_step: Optional minimum checkpoint step
        max_step: Optional maximum checkpoint step
    
    Returns:
        Filtered list of file paths
    """
    filtered = []
    
    for f in all_files:
        f_path = Path(f)
        layout_name = f_path.parent.name
        
        # Check layout
        if layout_name not in layout_to_id:
            continue
        
        # Parse filename: seed{X}_step{Y}.pkl
        filename = f_path.stem
        try:
            parts = filename.split('_')
            seed = int(parts[0].replace('seed', ''))
            step = int(parts[1].replace('step', ''))
            
            # Apply filters
            if seed_filter and seed not in seed_filter:
                continue
            if min_step and step < min_step:
                continue
            if max_step and step > max_step:
                continue
            
            filtered.append(f)
            
        except (IndexError, ValueError):
            print(f"Warning: Skipping file with unexpected format: {filename}")
            continue
    
    return filtered


# ========================================================================
# Dataset
# ========================================================================

class LazyVAEDataset(Dataset):
    """
    메모리 효율적인 Lazy Loading VAE Dataset
    Returns: (obs, action, task_id)
    """
    
    def __init__(
        self, 
        buffer_files: List[str], 
        layout_to_id: Dict[str, int],
        k_timesteps: int = 5,
        action_dim: int = 6,
        max_cached_files: int = 10
    ):
        self.buffer_files = [Path(f) for f in buffer_files]
        self.layout_to_id = layout_to_id
        self.k = k_timesteps
        self.action_dim = action_dim
        self.max_cached_files = max_cached_files
        
        self.file_cache: OrderedDict[int, Dict] = OrderedDict()
        self.file_infos = []
        
        self._build_index()
    
    def _build_index(self):
        """Build index: global_idx -> (file, local_offset, task_id)"""
        current_offset = 0
        
        print(f"Indexing {len(self.buffer_files)} buffer files...")
        
        for file_path in tqdm(self.buffer_files, desc="Building index"):
            layout_name = file_path.parent.name
            if layout_name not in self.layout_to_id:
                continue
                
            task_id = self.layout_to_id[layout_name]
            
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                episodes = data['episodes']
                num_sequences = sum(
                    max(0, len(ep) - self.k + 1) for ep in episodes if len(ep) >= self.k
                )
                
                if num_sequences > 0:
                    self.file_infos.append({
                        'path': file_path,
                        'task_id': task_id,
                        'start': current_offset,
                        'end': current_offset + num_sequences,
                        'count': num_sequences
                    })
                    current_offset += num_sequences
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
        self.total_sequences = current_offset
        print(f"✓ Total sequences: {self.total_sequences:,}")
    
    def _load_file(self, file_idx: int) -> Dict:
        """Load file with LRU caching"""
        if file_idx in self.file_cache:
            self.file_cache.move_to_end(file_idx)
            return self.file_cache[file_idx]
        
        file_info = self.file_infos[file_idx]
        with open(file_info['path'], 'rb') as f:
            data = pickle.load(f)
            
        # Pre-convert to numpy
        data['episodes'] = [np.array(e) for e in data['episodes']]
        data['actions'] = [np.array(a) for a in data['actions']]
        
        self.file_cache[file_idx] = data
        self.file_cache.move_to_end(file_idx)
        
        if len(self.file_cache) > self.max_cached_files:
            self.file_cache.popitem(last=False)
        
        return data
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx: int):
        # Find target file
        target_file_idx = -1
        for i, info in enumerate(self.file_infos):
            if info['start'] <= idx < info['end']:
                target_file_idx = i
                break
        
        if target_file_idx == -1:
            raise IndexError(f"Index {idx} out of range")
            
        file_info = self.file_infos[target_file_idx]
        local_idx = idx - file_info['start']
        task_id = file_info['task_id']
        
        data = self._load_file(target_file_idx)
        
        # Find sequence in file
        current_seq_count = 0
        for ep_idx, episode in enumerate(data['episodes']):
            ep_len = len(episode)
            if ep_len < self.k:
                continue
            
            n_seqs = ep_len - self.k + 1
            
            if current_seq_count <= local_idx < current_seq_count + n_seqs:
                start_t = local_idx - current_seq_count
                end_t = start_t + self.k
                
                # Obs: (K, W, H, C) or (K, C, H, W)
                # Numpy Slice(View)를 텐서로 만들면 메모리 공유 에러가 발생하므로 
                obs_seq = episode[start_t:end_t]
                
                obs_tensor = torch.from_numpy(obs_seq).float()
                
                # 차원 순서 변경 (W, H, C) -> (C, H, W) 확인
                # Overcooked 데이터가 보통 (W, H, C)로 저장되어 있다면 permute 필요
                # 만약 저장된 데이터가 (C, H, W)라면 아래 줄은 주석 처리
                if obs_tensor.shape[-1] == 25: 
                    obs_tensor = obs_tensor.permute(0, 3, 2, 1) 
                
                # 정규화
                if obs_tensor.max() > 1.0:
                    obs_tensor = obs_tensor / 255.0
                    
                # Actions 처리
                action_seq = data['actions'][ep_idx][start_t:end_t]  
                action_tensor = torch.zeros(self.k, self.action_dim)
                for t, a in enumerate(action_seq):
                    if 0 <= a < self.action_dim:
                        action_tensor[t, int(a)] = 1.0
                        
                return obs_tensor, action_tensor, task_id
            
            current_seq_count += n_seqs
            
        raise RuntimeError("Sequence lookup failed")


# ========================================================================
# Training Loop
# ========================================================================

def train_vae(config: dict, args: argparse.Namespace):
    """Main training function"""
    
    # Set seed
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    # Device
    device = torch.device(config['training']['device'])
    
    # ========================================================================
    # 1. Discover Layouts & Build Task Mapping
    # ========================================================================
    print(f"\nScanning layouts in {config['paths']['vae_data_dir']}...")
    
    root_path = Path(config['paths']['vae_data_dir'])
    layout_to_id = discover_layouts(root_path, config['data_filter']['layouts'])
    num_tasks = len(layout_to_id)
    
    print(f"\n✓ Found {num_tasks} layouts (Tasks):")
    for name, lid in sorted(layout_to_id.items(), key=lambda x: x[1]):
        print(f"  [{lid}] {name}")
        
    if num_tasks == 0:
        raise ValueError("No layout directories found!")

    # ========================================================================
    # 2. Filter & Load Data Files
    # ========================================================================
    all_pkl_files = sorted(glob.glob(f"{config['paths']['vae_data_dir']}/**/*.pkl", recursive=True))
    
    filtered_files = filter_data_files(
        all_pkl_files,
        layout_to_id,
        seed_filter=config['data_filter']['seeds'],
        min_step=config['data_filter']['min_step'],
        max_step=config['data_filter']['max_step']
    )
    
    print(f"\n✓ Data files:")
    print(f"  Total found: {len(all_pkl_files)}")
    print(f"  After filtering: {len(filtered_files)}")
    if config['data_filter']['seeds']:
        print(f"  Seed filter: {config['data_filter']['seeds']}")
    if config['data_filter']['min_step'] or config['data_filter']['max_step']:
        print(f"  Step range: [{config['data_filter']['min_step'] or 'any'}, {config['data_filter']['max_step'] or 'any'}]")
    
    if len(filtered_files) == 0:
        raise ValueError("No data files after filtering!")
    
    # ========================================================================
    # 3. Create Dataset & DataLoader
    # ========================================================================
    dataset = LazyVAEDataset(
        buffer_files=filtered_files,
        layout_to_id=layout_to_id,
        k_timesteps=config['model']['k_timesteps'],
        action_dim=config['model']['action_dim']
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['training']['num_workers'], 
        pin_memory=True
    )
    
    # ========================================================================
    # 4. Initialize Model
    # ========================================================================
    print(f"\n✓ Initializing Task-Conditional VAE...")
    vae = SkillVAE(
        obs_channels=config['model']['obs_channels'],
        action_dim=config['model']['action_dim'],
        task_dim=num_tasks,
        hidden_dim=config['model']['hidden_dim'],
        latent_dim=config['model']['latent_dim'],
        t_seg=config['model']['k_timesteps']
    ).to(device)
    
    optimizer = Adam(vae.parameters(), lr=config['training']['lr'])
    total_steps = config['training']['epochs'] * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # ========================================================================
    # 5. Wandb Setup
    # ========================================================================
    if config['wandb']['use_wandb']:
        import wandb
        run_name = config['wandb']['run_name'] or f"vae_{num_tasks}tasks_{config['model']['latent_dim']}d"
        wandb.init(project=config['wandb']['project'], name=run_name, config=config)
        # 모델의 Gradient 변화도 추적 (선택사항)
        wandb.watch(vae, log="all", log_freq=100)
    
    # ========================================================================
    # 6. Training Loop
    # ========================================================================
    total_steps = len(dataloader) * config['training']['epochs']
    warmup_steps = config['annealing'].get('warmup_steps', 2000)
    n_cycles = config['annealing'].get('n_cycles', 4)
    cycle_ratio = config['annealing'].get('cycle_ratio', 0.5)
    beta_max = config['annealing']['beta_max']
    
    print(f"\n✓ Starting training... (Real-time W&B logging ON)")
    print(f"  Cyclical Annealing strategy:")
    print(f"    - Warmup: 0 ~ {warmup_steps:,} steps (Beta=0)")
    print(f"    - Cycles: {n_cycles} cycles")
    print(f"    - Cycle ratio: {cycle_ratio * 100}% increase, {(1-cycle_ratio) * 100}% maintain")
    print(f"    - Beta max: {beta_max}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print("")
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(1, config['training']['epochs'] + 1):
        vae.train()
        total_loss = 0
        recon_sum = 0
        kl_loss_sum = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config['training']['epochs']}")
        
        for batch_obs, batch_act, batch_task_ids in pbar:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)
            batch_task_ids = batch_task_ids.to(device)
            
            # Cyclical Annealing
            if global_step < warmup_steps:
                # Warmup: Beta=0
                beta = 0.0
            else:
                # Cyclical annealing after warmup
                cycle_len = (total_steps - warmup_steps) // n_cycles
                step_in_cycle = (global_step - warmup_steps) % cycle_len
                cycle_progress = step_in_cycle / cycle_len
                
                if cycle_progress < cycle_ratio:
                    # Linear increase: 0 -> beta_max
                    beta = beta_max * (cycle_progress / cycle_ratio)
                else:
                    # Maintain at beta_max
                    beta = beta_max
            
            # Convert Task ID to One-Hot
            task_onehot = F.one_hot(batch_task_ids, num_classes=num_tasks).float()
            
            # Get loss weights from config
            feature_weight = config.get('loss', {}).get('feature_weight', 30.0)
            action_weight = config.get('loss', {}).get('action_weight', 100.0)
            
            # Forward & Loss
            loss, info = vae.compute_loss(batch_obs, batch_act, task_onehot, beta=beta,
                                        feature_weight=feature_weight, action_weight=action_weight)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            recon_sum += info['recon']
            kl_loss_sum += info['kl']
            
            global_step += 1
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Recon': f"{info['recon']:.3f}",
                'KL': f"{info['kl']:.4f}",
                'β': f"{beta:.4f}"
            })
            
            # 실시간 W&B 로깅 (매 스텝 기록)
            # 조건문(%)을 제거하여 매 batch마다 즉시 그래프를 그립니다.
            if config['wandb']['use_wandb']:
                wandb.log({
                    "Train/Total_Loss": loss.item(),
                    "Train/Recon": info['recon'],
                    "Train/Recon_Feature": info['recon_f'],
                    "Train/Recon_Action": info['recon_a'],
                    "Train/KL_Loss": info['kl'],
                    "Hyperparam/Beta": beta,
                    "Hyperparam/LR": scheduler.get_last_lr()[0],
                }, step=global_step)
        
        # Epoch Summary
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch} | Loss: {avg_loss:.4f} | Beta: {beta:.4f}\n")
        
        # Prepare checkpoint directory
        save_dir = Path(config['paths']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        # Multi-task >> Generate unique filename
        layout_names = sorted(list(layout_to_id.keys()))

        # combine to a single string
        layout_str = "+".join(layout_names)
        
        # Common save dictionary
        save_dict = {
            'model': vae.state_dict(),
            'config': config,
            'layout_to_id': layout_to_id,
            'layout_name': layout_str,
            'layout_list': list(layout_to_id.keys()),
            'epoch': epoch,
            'loss': avg_loss,
            'best_loss': best_loss
        }
        
        # 1. Save Best Model (when loss improves)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_dict['best_loss'] = best_loss
            
            if args.save_path:
                best_path = args.save_path
            else:
                best_path = f"{save_dir}/vae_best_{num_tasks}tasks_{config['model']['latent_dim']}d.pt"
            torch.save(save_dict, best_path)
            print(f"  ✓ Saved BEST model to {best_path}")
        
        # 2. Save Periodic Checkpoint (every epoch)
        checkpoint_path = f"{save_dir}/vae_epoch_{epoch:03d}_{num_tasks}t_{layout_str}_{config['model']['latent_dim']}d.pt"
        torch.save(save_dict, checkpoint_path)
        print(f"  ✓ Saved checkpoint to {checkpoint_path}\n")

    print("✓ Training Finished!")
    if config['wandb']['use_wandb']:
        wandb.finish()


# ========================================================================
# Main Entry Point
# ========================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Task-Conditional VAE')
    
    parser.add_argument('--config', type=str, 
                        default='zsceval/scripts/overcooked/config/vae_config.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Override save path from config')
    
    # Config overrides (optional CLI overrides for quick testing)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.device:
        config['training']['device'] = args.device
    
    # Run training
    train_vae(config, args)


if __name__ == '__main__':
    main()
