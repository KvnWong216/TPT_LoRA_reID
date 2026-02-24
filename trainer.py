import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(dataloader)