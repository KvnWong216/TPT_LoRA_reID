import torch
import json
import random
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[0] + box1[2], box2[0] + box2[2]), min(box1[1] + box1[3], box2[1] + box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area <= 0: return 0.0
    box1_area, box2_area = box1[2] * box1[3], box2[2] * box2[3]
    return inter_area / float(box1_area + box2_area - inter_area)

class TargetSpecificTripletDataset(Dataset):
    def __init__(self, root_dir, seq_id, anchor_ratio=0.05, num_negatives=5, transform_weak=None, transform_strong=None):
        self.root_dir = Path(root_dir)
        self.seq_id = seq_id
        self.num_negatives = num_negatives
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
        
        json_path = self.root_dir / "GTs" / f"{seq_id}.json"
        self.img_dir = self.root_dir / "panoramic_images" / seq_id
        
        with open(json_path, 'r') as f:
            gt_data = json.load(f)
            
        sorted_frames = sorted(gt_data.keys())
        keep_len = max(1, int(len(sorted_frames) * anchor_ratio))
        train_frames = sorted_frames[:keep_len]
        
        self.bg_pool = []
        for f in sorted_frames:
            info = gt_data[f]
            if info.get("is_exist") == 0 and "bbox" in info:
                self.bg_pool.append({
                    'path': str(self.img_dir / f"{f}.jpg"),
                    'bbox': info['bbox']
                })

        self.samples = []
        for frame_idx in train_frames:
            frame_info = gt_data[frame_idx]
            if frame_info.get("is_exist", 1) == 1 and "bbox" in frame_info:
                self.samples.append({
                    'img_path': str(self.img_dir / f"{frame_idx}.jpg"),
                    'bbox': frame_info['bbox']
                })
        
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = None 

    def _get_hard_negatives(self, full_img, target_bbox):
        if self.face_cascade is None:
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
            
        img_w, img_h = full_img.size
        tx, ty, tw, th = target_bbox
        neg_crops = []

        if self.bg_pool:
            num_bg = min(len(self.bg_pool), 3)
            bg_samples = random.sample(self.bg_pool, num_bg)
            for bg_sample in bg_samples:
                try:
                    with Image.open(bg_sample['path']) as bg_img:
                        bx, by, bw, bh = bg_sample['bbox']
                        crop = bg_img.convert('RGB').crop((
                            max(0, bx), max(0, by), 
                            min(img_w, bx + bw), min(img_h, by + bh)
                        ))
                        neg_crops.append(crop)
                except: pass

        img_array = np.array(full_img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        for (nx, ny, nw, nh) in faces:
            if len(neg_crops) >= self.num_negatives: break
            if compute_iou(target_bbox, [nx, ny, nw, nh]) < 0.1:
                crop = full_img.crop((max(0, nx), max(0, ny), min(img_w, nx + nw), min(img_h, ny + nh)))
                neg_crops.append(crop)

        offsets = [(-tw, 0), (tw, 0), (0, -th), (0, th), (-tw, -th), (tw, th)]
        random.shuffle(offsets)
        for ox, oy in offsets:
            if len(neg_crops) >= self.num_negatives: break
            nx, ny = int(tx + ox), int(ty + oy)
            if 0 <= nx < img_w - tw and 0 <= ny < img_h - th:
                if compute_iou(target_bbox, [nx, ny, tw, th]) < 0.05:
                    neg_crops.append(full_img.crop((nx, ny, nx + tw, ny + th)))

        while len(neg_crops) < self.num_negatives:
            rx, ry = random.randint(0, img_w - tw), random.randint(0, img_h - th)
            if compute_iou(target_bbox, [rx, ry, tw, th]) < 0.01:
                neg_crops.append(full_img.crop((rx, ry, rx + tw, ry + th)))
                
        return neg_crops[:self.num_negatives]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        full_img = Image.open(sample['img_path']).convert('RGB')
        tx, ty, tw, th = sample['bbox']
        
        target_crop = full_img.crop((tx, ty, tx + tw, ty + th))
        if target_crop.size[0] == 0 or target_crop.size[1] == 0:
            target_crop = Image.new('RGB', (128, 256), (0, 0, 0))

        anchor = self.transform_weak(target_crop)
        positive = self.transform_strong(target_crop)
        neg_crops = self._get_hard_negatives(full_img, [tx, ty, tw, th])
        
        processed_negs = []
        for c in neg_crops:
            if c.size[0] > 0 and c.size[1] > 0:
                processed_negs.append(self.transform_weak(c))
        
        while len(processed_negs) < self.num_negatives:
            processed_negs.append(anchor.clone())
            
        return anchor, positive, torch.stack(processed_negs[:self.num_negatives])

    def __len__(self):
        return len(self.samples)