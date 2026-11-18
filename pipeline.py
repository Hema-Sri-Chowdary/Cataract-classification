import os
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.mkldnn.enabled = False

from models.attention_unet import AttentionUNet
from models.googlenet import CataractGoogLeNet
from xai.gradcam import GradCAM

class CataractAnalysisPipeline:
    """
    End-to-End Explainable AI Nuclear Cataract Detection and Grading Pipeline.
    Matching LOCS III Standards, 5-Panel Overview & Alignment Analysis.
    Supports robust multi-format image loading (RGBA, Grayscale, WebP, etc.).
    """
    CLASS_LABELS = [
        "Normal (LOCS III < 2.0)",
        "Mild (LOCS III 2.0 - 4.9)",
        "Severe (LOCS III >= 5.0)"
    ]

    CLASS_DESCRIPTIONS = [
        "Lens nucleus is clear with normal optical density (LOCS III NC < 2.0). No surgical intervention required.",
        "Early opacification observed in the central lens nucleus (LOCS III NC 2.0 - 4.9). Regular monitoring advised.",
        "Advanced nuclear sclerosis with marked opacification (LOCS III NC >= 5.0). Surgical phacoemulsification recommended."
    ]

    def __init__(self, unet_path='checkpoints/attention_unet_cataract.pth', googlenet_path='checkpoints/googlenet_cataract.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Attention U-Net
        self.unet = AttentionUNet().to(self.device)
        if os.path.exists(unet_path):
            try:
                self.unet.load_state_dict(torch.load(unet_path, map_location=self.device))
                print(f"Loaded Attention U-Net from {unet_path}")
            except Exception as e:
                print(f"Could not load Attention U-Net checkpoint: {e}")
        self.unet.eval()
        
        # Load GoogLeNet
        self.googlenet = CataractGoogLeNet(num_classes=4, pretrained=True).to(self.device)
        if os.path.exists(googlenet_path):
            try:
                self.googlenet.load_state_dict(torch.load(googlenet_path, map_location=self.device))
                print(f"Loaded GoogLeNet from {googlenet_path}")
            except Exception as e:
                print(f"Could not load GoogLeNet checkpoint: {e}")
        self.googlenet.eval()
        
        # Initialize Grad-CAM handler
        target_layer = self.googlenet.get_target_layer()
        self.gradcam = GradCAM(self.googlenet, target_layer)

    def preprocess_image(self, img_rgb, target_size=(224, 224)):
        """
        Robustly processes any input image array (RGBA, Gray, CMYK, uint8/float32).
        Applies CLAHE enhancement, resizing, and ImageNet normalization.
        """
        img_rgb = np.asarray(img_rgb, dtype=np.uint8)
        
        # Handle grayscale, RGBA, or multi-channel shapes
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        elif img_rgb.ndim == 3:
            if img_rgb.shape[2] == 4:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
            elif img_rgb.shape[2] == 1:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
            elif img_rgb.shape[2] > 3:
                img_rgb = img_rgb[:, :, :3]
                
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # CLAHE enhancement
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
        
        resized_rgb = cv2.resize(enhanced_bgr, target_size)
        
        tensor = torch.tensor(resized_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        norm_tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)
        
        return norm_tensor, resized_rgb, enhanced_bgr

    def segment_nucleus_attention_unet(self, norm_tensor, orig_resized_rgb):
        with torch.no_grad():
            mask_tensor = self.unet(norm_tensor)
            mask_np = mask_tensor.squeeze().cpu().numpy()
            
        h, w = orig_resized_rgb.shape[:2]
        
        # Raw probability from Attention U-Net (final_conv already includes sigmoid)
        prob = np.clip(mask_np, 0.0, 1.0)
        mask_binary = (prob > 0.45).astype(np.uint8) * 255
        
        # Morphological filtering to clean noise and extract smooth lens nucleus
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        
        # Create visual representation: glowing electric cyan nucleus ROI overlay with bright green boundary contour
        overlay = orig_resized_rgb.copy()
        overlay[mask_binary > 0] = [0, 220, 255] # Cyan highlight
        mask_visual = cv2.addWeighted(orig_resized_rgb, 0.50, overlay, 0.50, 0)
        
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(mask_visual, contours, -1, (0, 255, 128), 2)
            
        # Extract focused nucleus ROI
        roi_focused = orig_resized_rgb.copy()
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(c)
            if bw > 10 and bh > 10:
                crop = orig_resized_rgb[y:y+bh, x:x+bw]
                if crop.size > 0:
                    roi_focused = cv2.resize(crop, (w, h))

        return mask_binary, mask_visual, roi_focused

    def compute_alignment_metrics(self, mask_binary, cam_map):
        h, w = mask_binary.shape[:2]
        cam_resized = cv2.resize(cam_map, (w, h))
        
        mask_bool = (mask_binary > 100)
        cam_bool = (cam_resized > 0.22)
        
        intersection = np.logical_and(mask_bool, cam_bool).sum()
        union = np.logical_or(mask_bool, cam_bool).sum()
        
        raw_iou = float(intersection / max(union, 1))
        raw_dice = float(2 * intersection / max(mask_bool.sum() + cam_bool.sum(), 1))
        
        total_cam_energy = cam_resized.sum()
        mask_cam_energy = (cam_resized * mask_bool).sum()
        raw_coverage = float((mask_cam_energy / max(total_cam_energy, 1e-5)) * 100.0)
        
        return {
            'iou': round(raw_iou, 4),
            'dice': round(raw_dice, 4),
            'coverage': round(raw_coverage, 1)
        }

    def analyze_image(self, img_rgb):
        # 1. Preprocess Image safely
        norm_tensor, resized_rgb, enhanced_rgb = self.preprocess_image(img_rgb)
        
        # 2. Attention U-Net Segmentation & ROI Extraction
        mask_binary, mask_visual, roi_focused = self.segment_nucleus_attention_unet(norm_tensor, resized_rgb)
        
        # 3. Single-pass GoogLeNet Grading + Grad-CAM Explanation
        cam, pred_class, probs = self.gradcam.generate_heatmap(norm_tensor)
        confidence = float(probs[pred_class] * 100.0)
        
        # Raw Grad-CAM heatmap visualization
        h, w = mask_binary.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        if cam_resized.max() > 0:
            cam_resized = cam_resized / cam_resized.max()
            
        heatmap_rgb, gradcam_overlay_rgb = self.gradcam.overlay_heatmap(resized_rgb, cam_resized, alpha=0.55)

        # 4. Raw Alignment Metrics
        metrics = self.compute_alignment_metrics(mask_binary, cam_resized)

        # Base64 encodings
        orig_b64 = self.array_to_base64(resized_rgb)
        mask_b64 = self.array_to_base64(mask_visual)
        roi_b64 = self.array_to_base64(roi_focused)
        heatmap_b64 = self.array_to_base64(heatmap_rgb)
        overlay_b64 = self.array_to_base64(gradcam_overlay_rgb)

        result = {
            'predicted_class': pred_class,
            'predicted_label': self.CLASS_LABELS[pred_class],
            'confidence': round(confidence, 1),
            'description': self.CLASS_DESCRIPTIONS[pred_class],
            'metrics': metrics,
            'probabilities': {
                self.CLASS_LABELS[i].split(' (')[0]: round(float(probs[i] * 100.0), 1)
                for i in range(len(self.CLASS_LABELS))
            },
            'visualizations': {
                'original': f"data:image/png;base64,{orig_b64}",
                'mask': f"data:image/png;base64,{mask_b64}",
                'roi': f"data:image/png;base64,{roi_b64}",
                'heatmap': f"data:image/png;base64,{heatmap_b64}",
                'overlay': f"data:image/png;base64,{overlay_b64}"
            }
        }
        return result

    @staticmethod
    def array_to_base64(arr_rgb):
        arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', arr_bgr)
        return base64.b64encode(buffer).decode('utf-8')

if __name__ == '__main__':
    pipeline = CataractAnalysisPipeline()
    dummy_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    res = pipeline.analyze_image(dummy_img)
    print("Label:", res['predicted_label'])
