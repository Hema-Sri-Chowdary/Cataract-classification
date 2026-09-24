import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for GoogLeNet.
    Single-pass forward + backward with instant memory release.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        inp = input_tensor.detach().clone().requires_grad_(True)
        
        with torch.set_grad_enabled(True):
            output = self.model(inp)
            if output.shape[1] == 4:
                out_3class = torch.stack([
                    output[:, 0],
                    output[:, 1],
                    torch.max(output[:, 2], output[:, 3])
                ], dim=1)
            else:
                out_3class = output

            probs = F.softmax(out_3class, dim=1).detach().cpu().numpy()[0]
            
            if target_class is None:
                target_class = int(np.argmax(probs))
                
            score = out_3class[0, target_class]
            self.model.zero_grad()
            score.backward()
            
            gradients = self.gradients.detach().cpu().numpy()[0]
            activations = self.activations.detach().cpu().numpy()[0]
            
            self.gradients = None
            self.activations = None

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam, target_class, probs

    def overlay_heatmap(self, orig_img_rgb, cam, alpha=0.55, colormap=cv2.COLORMAP_JET):
        h, w = orig_img_rgb.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        cam_uint8 = np.uint8(255 * cam_resized)
        
        heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        
        overlay_rgb = cv2.addWeighted(orig_img_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
        return heatmap_rgb, overlay_rgb
