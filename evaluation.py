import torch
import torch.nn.functional as F
import json
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from models.lora_resnet import get_lora_res

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_model(checkpoint_path=None, device='cpu', r=4):
    """针对 .safetensors 优化的强力加载逻辑"""
    model = get_lora_res(num_classes=2, r=r)
    
    if checkpoint_path:
        print(f">>> 正在尝试从 {checkpoint_path} 加载权重...")
        
        safe_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
        bin_path = os.path.join(checkpoint_path, "adapter_model.bin")
        
        try:
            model.lora_model = PeftModel.from_pretrained(
                model.lora_model.get_base_model(), 
                checkpoint_path
            )
            print(">>> [SUCCESS] PEFT 标准加载成功")
        except Exception as e:
            print(f">>> 标准加载失败，启动强力注入模式...")
            if os.path.exists(safe_path):
                state_dict = load_file(safe_path, device="cpu")
            elif os.path.exists(bin_path):
                state_dict = torch.load(bin_path, map_location="cpu")
            else:
                raise FileNotFoundError("找不到任何权重文件 (.safetensors 或 .bin)")
            
            set_peft_model_state_dict(model.lora_model, state_dict)
            print(">>> [SUCCESS] Safetensors 权重已强制注入")
            
    else:
        print(">>> 使用原始 ResNet50 (Baseline)")
        model.lora_model = model.lora_model.unload() 
    
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def extract_feature(model, img_path, bbox, transform, device):
    """提取特征，增加合法性校验"""
    try:
        if bbox[2] <= 0 or bbox[3] <= 0:
            return None
            
        full_img = Image.open(img_path).convert('RGB')
        x, y, w, h = bbox
        
        img_w, img_h = full_img.size
        left, top = max(0, x), max(0, y)
        right, bottom = min(img_w, x + w), min(img_h, y + h)
        
        if (right - left) < 5 or (bottom - top) < 5:
            return None

        crop = full_img.crop((left, top, right, bottom))
        img_tensor = transform(crop).unsqueeze(0).to(device)
        
        _, feat = model(img_tensor, return_embeddings=True)
        feat = F.normalize(feat, p=2, dim=1) 
        return feat
    except Exception:
        return None

def run_evaluation(seq_id, data_root, lora_path, r=4):
    device = get_device()
    print(f"正在使用设备: {device}")
    
    data_root = Path(data_root)
    json_path = data_root / "GTs" / f"{seq_id}.json"
    img_dir = data_root / "panoramic_images" / seq_id
    
    if not json_path.exists():
        print(f"跳过: 找不到 JSON 文件 {json_path}")
        return

    with open(json_path, 'r') as f:
        gt_data = json.load(f)
    
    sorted_frames = sorted(gt_data.keys())
    eval_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model_base = load_model(None, device, r=r)
    model_lora = load_model(lora_path, device, r=r)

    first_frame = sorted_frames[0]
    first_img_path = img_dir / f"{first_frame}.jpg"
    first_bbox = gt_data[first_frame]['bbox']
    
    gal_feat_base = extract_feature(model_base, first_img_path, first_bbox, eval_transform, device)
    gal_feat_lora = extract_feature(model_lora, first_img_path, first_bbox, eval_transform, device)

    sim_base, sim_lora, frames_idx = [], [], []

    for i, frame_idx in enumerate(tqdm(sorted_frames, desc=f"Eval {seq_id}")):
        if i % 5 != 0: continue 
        
        frame_info = gt_data.get(frame_idx, {})
        is_exist = frame_info.get("is_exist", 1)  
        bbox = frame_info.get('bbox', [0, 0, 0, 0])
        img_path = img_dir / f"{frame_idx}.jpg"

        if is_exist == 0 or bbox[2] <= 0 or bbox[3] <= 0:
            sim_base.append(0.0)
            sim_lora.append(0.0)
            frames_idx.append(i)
            continue

        f_base = extract_feature(model_base, img_path, bbox, eval_transform, device)
        f_lora = extract_feature(model_lora, img_path, bbox, eval_transform, device)
        
        if f_base is not None and f_lora is not None:
            s_b = torch.mm(gal_feat_base, f_base.t()).item()
            s_l = torch.mm(gal_feat_lora, f_lora.t()).item()
        else:
            s_b, s_l = 0.0, 0.0
            
        sim_base.append(s_b)
        sim_lora.append(s_l)
        frames_idx.append(i)

    valid_mask = [s > 0 for s in sim_base]
    valid_sim_b = [s for s, m in zip(sim_base, valid_mask) if m]
    valid_sim_l = [s for s, m in zip(sim_lora, valid_mask) if m]
    
    avg_v_b = sum(valid_sim_b)/len(valid_sim_b) if valid_sim_b else 0
    avg_v_l = sum(valid_sim_l)/len(valid_sim_l) if valid_sim_l else 0
    
    avg_all_b = sum(sim_base)/len(sim_base)
    avg_all_l = sum(sim_lora)/len(sim_lora)

    print("-" * 30)
    print(f">>> 序列 {seq_id} 评估完成")
    print(f"目标出现帧均值 (Visible Only): Baseline={avg_v_b:.3f}, LoRA={avg_v_l:.3f}, Gain={avg_v_l-avg_v_b:.3f}")
    print(f"全序列均值 (Total Stability): Baseline={avg_all_b:.3f}, LoRA={avg_all_l:.3f}, Gain={avg_all_l-avg_all_b:.3f}")
    print("-" * 30)

    plt.figure(figsize=(12, 6))
    plt.plot(frames_idx, sim_base, label='Baseline', color='gray', alpha=0.4, linewidth=1)
    plt.plot(frames_idx, sim_lora, label='Target-LoRA', color='blue', alpha=0.8, linewidth=1.5)
    
    train_limit = int(len(sorted_frames) * 0.05)
    plt.axvline(x=train_limit, color='red', linestyle='--', label='Train Boundary (5%)')
    
    plt.title(f'Seq {seq_id} Re-ID Consistency (r={r})')
    plt.xlabel('Frame Index')
    plt.ylabel('Cosine Similarity')
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend()
    
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/compare_{seq_id}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    for sid in ['0001', '0008']:
        run_evaluation(sid, './data', 'checkpoints/lora_0001_final', r=4)