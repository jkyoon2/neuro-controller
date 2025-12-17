import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class GridEncoder(nn.Module):
    """
    Simple CNN with LayerNorm to prevent feature collapse
    Input: (B, C, H, W) -> Output: (B, feature_dim)
    """
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 5 * 13, feature_dim), # 5x13 grid assumption
            nn.LayerNorm(feature_dim),           # [핵심] 0으로 죽는 것 방지
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class TrajectoryEncoder(nn.Module):
    def __init__(self, obs_channels, action_dim, task_dim, hidden_dim, latent_dim):
        super().__init__()
        self.feature_dim = 256
        
        # 1. Grid Feature Extractor
        self.cnn = GridEncoder(obs_channels, self.feature_dim)
        
        # 2. Embeddings for input balancing
        self.action_emb = nn.Linear(action_dim, 32)
        self.task_emb = nn.Linear(task_dim, 32)
        
        # 3. Temporal Encoder
        self.lstm = nn.LSTM(
            input_size=self.feature_dim + 32 + 32,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
    
    def forward(self, obs_seq, action_seq, task_id, return_features=False):
        B, T, C, H, W = obs_seq.shape
        
        # CNN (Time-distributed)
        obs_reshaped = obs_seq.contiguous().reshape(B * T, C, H, W)
        cnn_features = self.cnn(obs_reshaped).reshape(B, T, self.feature_dim)
        
        # Task ID: repeat to match B if needed
        if task_id.shape[0] != B:
            repeat_factor = B // task_id.shape[0]
            task_id = task_id.repeat(repeat_factor, 1)
        
        # Embeddings
        act_emb = self.action_emb(action_seq)  # (B, T, 32)
        task_emb = self.task_emb(task_id)  # (B, 32)
        task_seq_emb = task_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, 32)
        
        # Concat & LSTM
        lstm_input = torch.cat([cnn_features, act_emb, task_seq_emb], dim=-1)
        lstm_out, _ = self.lstm(lstm_input)
        
        # Mean Pooling
        pooled = lstm_out.mean(dim=1)
        mu = self.fc_mu(pooled)
        log_var = self.fc_logvar(pooled)
        
        if return_features:
            return mu, log_var, cnn_features
        return mu, log_var
    
    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std


class TrajectoryDecoder(nn.Module):
    """
    MLP Decoder: z -> All Timesteps Features & Actions
    """
    def __init__(self, latent_dim, task_dim, hidden_dim, feature_dim, action_dim, t_seg):
        super().__init__()
        self.t_seg = t_seg
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        decoder_hidden = hidden_dim * 2
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + task_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU()
        )
        
        self.feature_head = nn.Linear(decoder_hidden, t_seg * feature_dim)
        self.action_head = nn.Linear(decoder_hidden, t_seg * action_dim)
    
    def forward(self, z, task_id):
        z_task = torch.cat([z, task_id], dim=-1)
        common_feat = self.net(z_task)
        
        pred_features = self.feature_head(common_feat).view(-1, self.t_seg, self.feature_dim)
        pred_actions = self.action_head(common_feat).view(-1, self.t_seg, self.action_dim)
        
        return pred_features, pred_actions


class SkillVAE(nn.Module):
    def __init__(self, obs_channels, action_dim, task_dim, hidden_dim=128, latent_dim=16, t_seg=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.task_dim = task_dim
        self.t_seg = t_seg
        self.encoder = TrajectoryEncoder(obs_channels, action_dim, task_dim, hidden_dim, latent_dim)
        self.decoder = TrajectoryDecoder(latent_dim, task_dim, hidden_dim, 
                                       feature_dim=self.encoder.feature_dim,
                                       action_dim=action_dim,
                                       t_seg=t_seg)
    
    def forward(self, obs_seq, action_seq, task_id):
        mu, log_var, target_features = self.encoder(obs_seq, action_seq, task_id, return_features=True)
        z = self.encoder.sample(mu, log_var)
        pred_features, pred_actions = self.decoder(z, task_id)
        return z, mu, log_var, pred_features, pred_actions, target_features
    
    def encode(self, obs_seq, action_seq, task_id, deterministic: bool = False):
        """
        Encode trajectory to latent skill
        
        Args:
            obs_seq: (B, T, C, H, W) observation sequence
            action_seq: (B, T, action_dim) one-hot actions
            task_id: (B, task_dim) one-hot task identifier
            deterministic: if True, return mu; else sample
        
        Returns:
            z: (B, latent_dim) latent skill
        """
        mu, log_var = self.encoder(obs_seq, action_seq, task_id)
        if deterministic:
            return mu
        return self.encoder.sample(mu, log_var)
    
    def decode(self, z, task_id):
        """
        Decode latent skill to reconstructed features and actions
        
        Args:
            z: (B, latent_dim) latent skill
            task_id: (B, task_dim) one-hot task identifier
        
        Returns:
            pred_features: (B, T, feature_dim)
            pred_actions: (B, T, action_dim)
        """
        return self.decoder(z, task_id)

    def compute_loss(self, obs_seq, action_seq, task_id, beta: float, 
                     feature_weight: float = 300.0, action_weight: float = 10.0):
        """
        Feature and action reconstruction loss with scaling
        
        Args:
            feature_weight: Weight for feature reconstruction loss
            action_weight: Weight for action reconstruction loss
        """
        z, mu, log_var, pred_features, pred_actions, target_features = self.forward(obs_seq, action_seq, task_id)
        
        recon_feat_loss = F.mse_loss(pred_features, target_features)
        recon_act_loss = F.mse_loss(pred_actions, action_seq)
        
        scaled_feat_loss = recon_feat_loss * feature_weight
        scaled_act_loss = recon_act_loss * action_weight
        recon_loss = scaled_feat_loss + scaled_act_loss
        
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, {
            'loss': total_loss.item(),
            'recon': recon_loss.item(),
            'recon_f': scaled_feat_loss.item(),
            'recon_a': scaled_act_loss.item(),
            'kl': kl_loss.item(),
            'beta': beta
        }