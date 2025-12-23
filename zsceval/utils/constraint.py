import torch


class NegativeSkillBuffer:
    def __init__(self, device, threshold=0.5, penalty_weight=1.0):
        self.device = device
        self.negative_skills = None  # (N_bad, skill_dim)
        self.threshold = threshold
        self.penalty_weight = penalty_weight

    def add_skills(self, z_bad_batch):
        if z_bad_batch is None:
            return
        if not torch.is_tensor(z_bad_batch):
            z_bad_batch = torch.as_tensor(z_bad_batch, device=self.device, dtype=torch.float32)
        else:
            z_bad_batch = z_bad_batch.to(self.device, dtype=torch.float32)
        if z_bad_batch.ndim == 1:
            z_bad_batch = z_bad_batch.unsqueeze(0)

        if self.negative_skills is None:
            self.negative_skills = z_bad_batch
        else:
            self.negative_skills = torch.cat([self.negative_skills, z_bad_batch], dim=0)

    def compute_penalty(self, z_current):
        if self.negative_skills is None or self.negative_skills.numel() == 0:
            zeros = torch.zeros((z_current.shape[0], 1), device=self.device, dtype=torch.float32)
            return zeros, zeros

        if not torch.is_tensor(z_current):
            z_current = torch.as_tensor(z_current, device=self.device, dtype=torch.float32)
        else:
            z_current = z_current.to(self.device, dtype=torch.float32)
        if z_current.ndim == 1:
            z_current = z_current.unsqueeze(0)

        dists = torch.cdist(z_current, self.negative_skills, p=2)
        min_dists, _ = torch.min(dists, dim=1, keepdim=True)
        penalty = (min_dists < self.threshold).float() * self.penalty_weight
        return penalty, min_dists
