import numpy as np
import cv2
import matplotlib.pyplot as plt
from pipeline import LunarPSRPipeline

def generate_synthetic_ohrc_psr_image():
    """Generates a realistic ultra-low light simulated PSR crater image with noise & CCD striping."""
    np.random.seed(42)
    size = (512, 512)
    
    # Crater geometry base
    x = np.linspace(-1, 1, size[0])
    y = np.linspace(-1, 1, size[1])
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    crater_base = np.where(R < 0.7, 0.05 + 0.02 * np.cos(R * np.pi), 0.2)

    # Add simulated micro-boulders
    for _ in range(15):
        bx, by = np.random.randint(150, 350, 2)
        crater_base[bx:bx+4, by:by+4] += 0.08

    # Simulate extreme low-photon conditions + CCD Stripe Noise
    dark_psr = crater_base * 0.15
    stripes = np.zeros(size)
    stripes[:, ::8] = 0.03 # Periodic CCD line artifact
    noise = np.random.poisson(dark_psr * 100) / 1000.0 # Shot noise
    
    noisy_psr = np.clip(dark_psr + stripes + noise, 0, 1)
    return noisy_psr

if __name__ == "__main__":
    print("Initializing Project SHADOW-VISION Pipeline...")
    pipeline = LunarPSRPipeline()

    print("Generating simulated Chandrayaan OHRC PSR crater image...")
    raw_psr = generate_synthetic_ohrc_psr_image()

    print("Running Physics-Informed Denoising & Enhancement...")
    enhanced_psr = pipeline.enhance_image(raw_psr)

    print("Analyzing Terrain Hazards & Water-Ice Volatiles...")
    analysis = pipeline.detect_hazards_and_volatiles(enhanced_psr)

    # Plotting Results
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(raw_psr, cmap='gray')
    axes[0].set_title("1. Raw OHRC PSR (Low-Light & Noisy)")

    axes[1].imshow(enhanced_psr, cmap='magma')
    axes[1].set_title("2. SHADOW-VISION Enhanced (0.25m/px)")

    axes[2].imshow(analysis["hazard_mask"], cmap='Reds')
    axes[2].set_title("3. Hazardous Slopes & Slips")

    axes[3].imshow(analysis["ice_mask"], cmap='Blues')
    axes[3].set_title("4. Potential Water-Ice Signatures")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("shadow_vision_output.png")
    print("Processing complete! Output saved to 'shadow_vision_output.png'.")