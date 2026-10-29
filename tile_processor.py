import numpy as np
import torch
import cv2
from scipy.ndimage import gaussian_filter

class GigapixelTileProcessor:
    """
    Sliding window tile processor with seamless overlap blending 
    for massive satellite swaths.
    """
    def __init__(self, pipeline, tile_size=512, overlap=64):
        self.pipeline = pipeline
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

    def _generate_gaussian_weight_map(self, shape):
        h, w = shape
        mask = np.ones((h, w), dtype=np.float32)
        mask = gaussian_filter(mask, sigma=self.overlap / 2.0)
        mask = mask / mask.max()
        return mask

    def process_gigapixel_image(self, large_image, progress_callback=None):
        h_orig, w_orig = large_image.shape[:2]
        
        pad_h = (self.stride - (h_orig % self.stride)) % self.stride
        pad_w = (self.stride - (w_orig % self.stride)) % self.stride
        
        img_padded = np.pad(large_image, ((0, pad_h + self.overlap), (0, pad_w + self.overlap)), mode='reflect')
        h_pad, w_pad = img_padded.shape[:2]

        enhanced_accum = np.zeros((h_pad, w_pad), dtype=np.float32)
        hazard_accum = np.zeros((h_pad, w_pad), dtype=np.float32)
        boulder_accum = np.zeros((h_pad, w_pad), dtype=np.float32)
        ice_accum = np.zeros((h_pad, w_pad), dtype=np.float32)
        weight_accum = np.zeros((h_pad, w_pad), dtype=np.float32)

        weight_map = self._generate_gaussian_weight_map((self.tile_size, self.tile_size))

        y_steps = range(0, h_pad - self.tile_size + 1, self.stride)
        x_steps = range(0, w_pad - self.tile_size + 1, self.stride)
        total_tiles = len(y_steps) * len(x_steps)
        processed_tiles = 0

        for y in y_steps:
            for x in x_steps:
                tile = img_padded[y:y + self.tile_size, x:x + self.tile_size]

                enhanced_tile = self.pipeline.enhance_image(tile)
                analysis_tile = self.pipeline.detect_hazards_and_volatiles(enhanced_tile)

                hazard_tile = analysis_tile["hazard_mask"].astype(np.float32)
                boulder_tile = analysis_tile["boulder_mask"].astype(np.float32)
                ice_tile = analysis_tile["ice_mask"].astype(np.float32)

                enhanced_accum[y:y + self.tile_size, x:x + self.tile_size] += enhanced_tile * weight_map
                hazard_accum[y:y + self.tile_size, x:x + self.tile_size] += hazard_tile * weight_map
                boulder_accum[y:y + self.tile_size, x:x + self.tile_size] += boulder_tile * weight_map
                ice_accum[y:y + self.tile_size, x:x + self.tile_size] += ice_tile * weight_map
                weight_accum[y:y + self.tile_size, x:x + self.tile_size] += weight_map

                processed_tiles += 1
                if progress_callback:
                    progress_callback(processed_tiles / total_tiles)

        weight_accum = np.maximum(weight_accum, 1e-5)
        enhanced_final = (enhanced_accum / weight_accum)[:h_orig, :w_orig]
        hazard_final = ((hazard_accum / weight_accum)[:h_orig, :w_orig]) > 0.5
        boulder_final = ((boulder_accum / weight_accum)[:h_orig, :w_orig]) > 0.5
        ice_final = ((ice_accum / weight_accum)[:h_orig, :w_orig]) > 0.5

        return {
            "enhanced": enhanced_final,
            "hazard_mask": hazard_final,
            "boulder_mask": boulder_final,
            "ice_mask": ice_final
        }