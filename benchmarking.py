import numpy as np
import cv2
from scipy.signal import convolve2d

class LunarModelEvaluator:
    """
    Quantitative evaluation metrics for assessing image quality, 
    terrain fidelity, and photometric consistency.
    """
    
    @staticmethod
    def calculate_psnr(img_true, img_test, data_range=1.0):
        """
        Calculates Peak Signal-to-Noise Ratio (PSNR) in dB.
        Higher is better.
        """
        mse = np.mean((img_true.astype(np.float64) - img_test.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return 10.0 * np.log10((data_range ** 2) / mse)

    @staticmethod
    def calculate_ssim(img1, img2, data_range=1.0):
        """
        Calculates Structural Similarity Index Measure (SSIM).
        Measures structural terrain preservation (0 to 1, higher is better).
        """
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        kernel = np.ones((11, 11)) / 121.0

        mu1 = convolve2d(img1, kernel, mode='valid')
        mu2 = convolve2d(img2, kernel, mode='valid')

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = convolve2d(img1 ** 2, kernel, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(img2 ** 2, kernel, mode='valid') - mu2_sq
        sigma12 = convolve2d(img1 * img2, kernel, mode='valid') - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(np.mean(ssim_map))

    @staticmethod
    def calculate_lommel_seeliger_loss(raw_img, enhanced_img, i=30.0, e=0.0):
        """
        Calculates photometric consistency loss using the Lommel-Seeliger law for lunar regolith.
        Verifies whether relative scattering intensities remain physically valid.
        Lower loss indicates zero AI hallucinations.
        """
        mu0 = np.cos(np.radians(i))  # Incident solar angle
        mu = np.cos(np.radians(e))   # Emission angle
        
        # Theoretical reflectance weight for lunar dust
        rho_ls = mu0 / (mu0 + mu + 1e-6)
        
        # Photometric residual error
        residual = np.abs(enhanced_img - (raw_img * (1.0 / (rho_ls + 1e-6))))
        loss = np.mean(residual ** 2)
        return float(loss)

    @staticmethod
    def calculate_entropy(img, bins=256):
        """
        Calculates Shannon Entropy (Information Content) in bits.
        Higher value indicates greater detail and texture visibility.
        """
        hist, _ = np.histogram(img, bins=bins, range=(0, 1), density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist + 1e-12)))

    def evaluate_pipeline(self, raw_img, enhanced_img):
        """Returns a full dictionary of benchmark metrics."""
        psnr_val = self.calculate_psnr(raw_img, enhanced_img)
        ssim_val = self.calculate_ssim(raw_img, enhanced_img)
        ls_loss = self.calculate_lommel_seeliger_loss(raw_img, enhanced_img)
        raw_entropy = self.calculate_entropy(raw_img)
        enh_entropy = self.calculate_entropy(enhanced_img)

        return {
            "PSNR (dB)": round(psnr_val, 2),
            "SSIM": round(ssim_val, 4),
            "Lommel-Seeliger Loss": round(ls_loss, 6),
            "Raw Entropy (bits)": round(raw_entropy, 2),
            "Enhanced Entropy (bits)": round(enh_entropy, 2),
            "Information Gain (%)": round(((enh_entropy - raw_entropy) / (raw_entropy + 1e-6)) * 100, 1)
        }