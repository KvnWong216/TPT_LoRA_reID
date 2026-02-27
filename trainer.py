import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class TargetSpecificTrainer:
    def __init__(self, model, lr=5e-5, margin=0.3): 
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.05)
        self.triplet_criterion = nn.TripletMarginLoss(margin=margin, p=2)

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc="Residual Learning")
        
        for anchor, positive, negatives in pbar:
            anchor, positive = anchor.to(self.device), positive.to(self.device)
            negatives = negatives.to(self.device)
            B, K, C, H, W = negatives.shape
            
            self.optimizer.zero_grad()
            
            _, emb_a = self.model(anchor, return_embeddings=True)
            _, emb_p = self.model(positive, return_embeddings=True)
            
            neg_flat = negatives.view(-1, C, H, W)
            _, emb_n_all = self.model(neg_flat, return_embeddings=True)
            emb_n_all = emb_n_all.view(B, K, -1)
            
            # 进行 L2 归一化
            emb_a = F.normalize(emb_a, p=2, dim=1)
            emb_p = F.normalize(emb_p, p=2, dim=1)
            emb_n_all = F.normalize(emb_n_all, p=2, dim=1)
            
            dist = torch.cdist(emb_a.unsqueeze(1), emb_n_all).squeeze(1)
            hard_idx = torch.argmin(dist, dim=1)
            emb_n_hard = emb_n_all[torch.arange(B), hard_idx]
            
            loss = self.triplet_criterion(emb_a, emb_p, emb_n_hard)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        return total_loss / len(dataloader)