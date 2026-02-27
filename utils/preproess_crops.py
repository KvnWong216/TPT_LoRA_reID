import os
import json
import cv2
import mediapipe as mp
from PIL import Image
from tqdm import tqdm
from pathlib import Path

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def is_valid_face(crop_bgr):
    """使用 Mediapipe 检查小图中是否存在面部"""
    results = face_detection.process(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    return results.detections is not None

def process_anchor_data(seq_id, root_dir, output_dir, anchor_ratio=0.05):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    
    json_path = root_dir / "GTs" / f"{seq_id}.json"
    img_dir = root_dir / "panoramic_images" / seq_id
    
    with open(json_path, 'r') as f:
        gt_data = json.load(f)

    # 逻辑 1：计算前 5% 的帧索引
    all_frames = sorted([int(k) for k in gt_data.keys()])
    threshold_idx = all_frames[int(len(all_frames) * anchor_ratio)]
    
    print(f"Seq {seq_id}: 注册期截止至第 {threshold_idx} 帧")

    for frame_idx, frame_data in tqdm(gt_data.items(), desc=f"Scanning Seq {seq_id}"):
        if int(frame_idx) > threshold_idx:
            break  # 超过前 5% 直接跳过
            
        img_path = img_dir / f"{int(frame_idx)}.jpg"
        if not img_path.exists(): continue
        
        full_img_cv2 = cv2.imread(str(img_path))
        if full_img_cv2 is None: continue

        for obj in frame_data:
            if obj.get('category') == 'person':
                x, y, w, h = [int(v) for v in obj['bbox']]
                h_img, w_img = full_img_cv2.shape[:2]
                crop_bgr = full_img_cv2[max(0,y):min(h_img,y+h), max(0,x):min(w_img,x+w)]
                
                if crop_bgr.size == 0: continue

                if is_valid_face(crop_bgr):
                    save_dir = output_dir / seq_id / f"id_{obj['target_id']}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    save_path = save_dir / f"anchor_{int(frame_idx):05d}.jpg"
                    cv2.imwrite(str(save_path), crop_bgr)

if __name__ == "__main__":
    # TODO: define the sequence to be preprocessed.
    sequences = ['0001']
    for s in sequences:
        process_anchor_data(s, "./data", "./data/preprocessed", anchor_ratio=0.05)