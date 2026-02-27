import torch
from torch.utils.data import DataLoader
from models.lora_resnet import get_lora_res
from dataset import TargetSpecificTripletDataset 
from trainer import TargetSpecificTrainer
from utils.transforms import LightTransform 
import argparse
from pathlib import Path
from utils.train_vis import TrainingVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA + OHEM Target-Specific Training")
    parser.add_argument('--data_root', type=str, default='./data', help='root directory of dataset')
    parser.add_argument('--seq_id', type=str, default='0001', help='terget sequence ID')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for nagtive samples')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--margin', type=float, default=0.6, help='Triplet Loss Margin')
    parser.add_argument('--num_negatives', type=int, default=6, help='OHEM candidate negative samples per batch')
    parser.add_argument('--anchor_ratio', type=float, default=0.08, help='training set proportion')
    return parser.parse_args()

def main():
    args = parse_args()
    
    transform_weak = LightTransform(is_train=False)
    transform_strong = LightTransform(is_train=True) 
    
    train_dataset = TargetSpecificTripletDataset(
        root_dir=args.data_root,
        seq_id=args.seq_id,
        anchor_ratio=args.anchor_ratio,
        num_negatives=args.num_negatives,
        transform_weak=transform_weak,
        transform_strong=transform_strong
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )

    print(f"🚀 初始化序列 {args.seq_id} 的专用微调任务...")
    print(f"📊 训练样本数: {len(train_dataset)}, 负样本候选池大小: {args.num_negatives}")

    model = get_lora_res(num_classes=2, r=4) 
    
    trainer = TargetSpecificTrainer(
        model=model,
        lr=args.lr,
        margin=args.margin
    )
    vis = TrainingVisualizer(save_dir=f"logs/{args.seq_id}")

    for epoch in range(args.epochs):
        avg_loss = trainer.train_one_epoch(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f}")

        vis.update(total=avg_loss, tri=avg_loss, ce=0.0) 
        print(f"Epoch {epoch+1} visualization updated.")

        if (epoch + 1) % 5 == 0:
            save_path = f"checkpoints/lora_{args.seq_id}_ep{epoch+1}"
            model.save_pretrained(save_path)
            print(f"💾 模型已保存至: {save_path}")

    final_path = f"checkpoints/lora_{args.seq_id}_final"
    model.save_pretrained(final_path)
    print(f"✅ 针对序列 {args.seq_id} 的 OHEM 微调已完成！")

if __name__ == "__main__":
    main()