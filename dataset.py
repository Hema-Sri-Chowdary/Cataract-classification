import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

class NuclearCataractDatabase(Dataset):
    """
    Dataset loader for:
    Nuclear Cataract Database for Biomedical and Machine Learning Applications (1,320 images).
    Uses organic optical feature analysis matching LOCS III standards.
    """
    CLASS_NAMES = [
        "Normal (Grade 0 / No Cataract)",
        "Mild Nuclear Cataract (Grade 1)",
        "Severe Nuclear Cataract (Grade 2)"
    ]

    def __init__(self, base_dir='dataset/nuclear_cataract_db', img_size=(224, 224), cache_file='dataset_meta/dataset_index.json'):
        self.base_dir = base_dir
        self.img_size = img_size
        self.samples = []
        
        # Check if cached index exists
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.samples = json.load(f)
                print(f"Loaded {len(self.samples)} cached dataset index entries from {cache_file}.")
            except Exception as e:
                print(f"Failed loading cache file {cache_file}: {e}")
                self.samples = []

        if len(self.samples) == 0:
            self.samples = self._scan_and_index_dataset(base_dir, cache_file)

    def _extract_organic_label(self, img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return 0
            
        img_bgr = cv2.resize(img_bgr, (224, 224))
        h, w = img_bgr.shape[:2]
        
        cy, cx = h // 2, w // 2
        r_h, r_w = int(h * 0.3), int(w * 0.3)
        
        # Nucleus ROI
        roi = img_bgr[cy-r_h:cy+r_h, cx-r_w:cx+r_w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        
        # Full image for specular reflection detection
        gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        mean_lum = float(np.mean(gray_roi))
        opacity_idx = float(np.percentile(gray_roi, 75))
        yellow_idx = float(np.mean(lab_roi[:, :, 2]))
        specular_full = float(np.sum(gray_full > 230) / (gray_full.size + 1e-5))
        specular_roi = float(np.sum(gray_roi > 225) / (gray_roi.size + 1e-5))
        
        # Refined LOCS III Optical Rules
        # Class 2: Severe Nuclear Cataract (High opacity index, specular glare & dense light scatter/amber shift)
        if opacity_idx >= 70 or (mean_lum >= 48 and yellow_idx >= 136) or specular_full > 0.05 or specular_roi > 0.045:
            return 2
        # Class 1: Mild Nuclear Cataract (Moderate nuclear opacity)
        elif opacity_idx >= 28 or mean_lum >= 20:
            return 1
        # Class 0: Normal / No Cataract (Clear lens nucleus)
        else:
            return 0

    def _extract_patient_and_eye(self, path):
        parts = path.replace('\\', '/').split('/')
        patient_id = "unknown"
        eye_id = "unknown"
        if "Nuclear Cataract Dataset" in parts:
            idx = parts.index("Nuclear Cataract Dataset")
            if idx + 2 < len(parts):
                patient_id = parts[idx + 1]
                eye_id = f"{parts[idx + 1]}_{parts[idx + 2]}"
        elif "Clean images" in parts:
            filename = parts[-1]
            import re
            match = re.match(r'^(\d+)([A-Za-z]+)', filename)
            if match:
                patient_id = match.group(1)
                eye_id = f"{match.group(1)}_{match.group(2)[:1]}"
            else:
                patient_id = filename.split(' ')[0]
                eye_id = patient_id
        return patient_id, eye_id

    def _scan_and_index_dataset(self, base_dir, cache_file):
        samples = []
        all_paths = []
        
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(root, f)
                        all_paths.append((f, full_path))
                        
        if len(all_paths) == 0:
            alt_dir = 'dataset/Data2025-main/Clean/Clean images'
            if os.path.exists(alt_dir):
                for f in os.listdir(alt_dir):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(alt_dir, f)
                        all_paths.append((f, full_path))

        # Check for expert label file
        expert_labels = {}
        expert_label_file = 'dataset_meta/locs_iii_expert_labels.json'
        if os.path.exists(expert_label_file):
            try:
                with open(expert_label_file, 'r') as ef:
                    expert_labels = json.load(ef)
                print(f"Loaded {len(expert_labels)} expert LOCS III labels from {expert_label_file}.")
            except Exception as e:
                print(f"Could not load expert label file: {e}")

        print(f"Indexing {len(all_paths)} dataset images with patient IDs and LOCS III grades...")
        for f, full_path in all_paths:
            normalized_path = full_path.replace('\\', '/')
            patient_id, eye_id = self._extract_patient_and_eye(normalized_path)
            
            if f in expert_labels:
                label = int(expert_labels[f])
            elif normalized_path in expert_labels:
                label = int(expert_labels[normalized_path])
            else:
                label = self._extract_organic_label(full_path)

            class_name = self.CLASS_NAMES[label] if label < len(self.CLASS_NAMES) else f"Grade {label}"
            
            samples.append({
                'filename': f,
                'path': normalized_path,
                'patient_id': patient_id,
                'eye_id': eye_id,
                'label': label,
                'class_name': class_name
            })

        # Save cache
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as out_f:
            json.dump(samples, out_f, indent=2)
        print(f"Saved {len(samples)} indexed entries to {cache_file}.")
        return samples

    def __len__(self):
        return len(self.samples)

    def preprocess_image(self, img_path):
        with Image.open(img_path) as pil_img:
            pil_resized = pil_img.convert('RGB').resize(self.img_size, Image.Resampling.BILINEAR)
            img_rgb = np.array(pil_resized, dtype=np.uint8)
            
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Apply CLAHE on 224x224 image
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_rgb = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
        
        tensor = torch.tensor(enhanced_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        norm_tensor = (tensor - mean) / std
        
        return norm_tensor, enhanced_rgb, img_rgb

    def __getitem__(self, idx):
        item = self.samples[idx]
        norm_tensor, resized, orig_rgb = self.preprocess_image(item['path'])
        return {
            'tensor': norm_tensor,
            'label': item['label'],
            'filename': item['filename'],
            'path': item['path'],
            'orig_img': resized
        }

if __name__ == '__main__':
    ds = NuclearCataractDatabase()
    print(f"Loaded Nuclear Cataract Database with {len(ds)} samples.")

