import numpy as np
import cv2
import torch
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from model import ShadowVisionEnhancer

class LunarPSRPipeline:
    def __init__(self, model_weights_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ShadowVisionEnhancer().to(self.device)
        self.model.eval()
        if model_weights_path:
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))

    def ccd_destripe(self, img):
        """Frequency-domain filtering to remove CCD vertical line artifacts."""
        f = fftshift(fft2(img))
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        
        # Mask vertical noise frequencies typical in Pushbroom OHRC detectors
        f[crow-2:crow+2, :] = 0
        
        img_destriped = np.real(ifft2(ifftshift(f)))
        return np.clip(img_destriped, 0, 1)

    def enhance_image(self, img_normalized):
        """Passes destriped low-photon image through the neural restorer."""
        # Pre-destripe
        clean_input = self.ccd_destripe(img_normalized)
        
        # Convert to Tensor
        tensor_in = torch.from_numpy(clean_input).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            enhanced_tensor = self.model(tensor_in)
            
        enhanced_img = enhanced_tensor.squeeze().cpu().numpy()
        return enhanced_img

    def detect_hazards_and_volatiles(self, enhanced_img, slope_threshold=15.0):
        """
        Calculates terrain hazards (>1.5m micro-boulders, steep slopes) 
        and flags high-reflectance volatile (water-ice) anomalies.
        """
        # Ensure float32 data type for OpenCV filter operations
        enhanced_img_32 = enhanced_img.astype(np.float32)

        # 1. Slope Estimation using Sobel Gradient Filters
        grad_x = cv2.Sobel(enhanced_img_32, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced_img_32, cv2.CV_32F, 0, 1, ksize=3)
        slope_map = np.sqrt(grad_x**2 + grad_y**2) * 100.0  # Normalized slope angle estimate
        
        hazard_mask = slope_map > slope_threshold

        # 2. Micro-Boulder Detection (Laplacian of Gaussian Spot Detector)
        blurred = cv2.GaussianBlur(enhanced_img_32, (5, 5), 0)
        boulders = cv2.Laplacian(blurred, cv2.CV_32F)
        boulder_mask = boulders > (boulders.mean() + 2.5 * boulders.std())

        # 3. Water-Ice Anomaly Tagging (Bright anomalous reflections inside PSR)
        ice_mask = enhanced_img_32 > (enhanced_img_32.mean() + 3.0 * enhanced_img_32.std())

        return {
            "hazard_mask": hazard_mask,
            "boulder_mask": boulder_mask,
            "ice_mask": ice_mask
        }

       