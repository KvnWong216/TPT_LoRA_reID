import torch
from torch.utils.data import DataLoader, Dataset
from models.lora_resnet import get_lora_res
from utils.transforms import LightTransform
from trainer import Trainer
from PIL import Image
import numpy as np

# 临时模拟数据集
class DummyDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
    def __len__(self): return 100
    def __getitem__(self, idx):
        img = Image.fromarray(np.uint8(np.random.randint(0,255,(256,128,3))))
        return self.transform(img), torch.randint(0, 751, (1,)).item()

def main():
    model = get_lora_res(num_classes=751)
    model.print_trainable_parameters()
    
    train_transform = LightTransform(is_train=True)
    dataset = DummyDataset(transform=train_transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    trainer = Trainer(model)
    print("Starting training...")
    for epoch in range(3):
        loss = trainer.train_one_epoch(loader)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    model.save_pretrained("checkpoints/lora_adapter_v1")
    print("Weights saved!")

if __name__ == "__main__":
    main()