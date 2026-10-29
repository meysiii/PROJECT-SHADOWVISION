import numpy as np
import cv2

class LandingZoneSelector:
    """
    Autonomous landing site selection engine for lunar landers/rovers.
    Evaluates terrain clearance, slope margins, and scientific proximity value.
    """
    def __init__(self, landing_radius_m=50.0, pixel_res=0.25):
        """
        Parameters:
            landing_radius_m: Target landing ellipse radius in meters (default: 50m).
            pixel_res: Spatial resolution in meters per pixel.
        """
        self.pixel_radius = int(landing_radius_m / pixel_res)
        self.pixel_res = pixel_res

    def find_optimal_landing_sites(self, enhanced_img, hazard_mask, ice_mask, top_k=3):
        """
        Scans the terrain map and identifies top-K safest landing ellipses 
        ranked by safety score and proximity to volatile ice targets.
        """
        h, w = enhanced_img.shape
        r = self.pixel_radius
        
        # Distance transform to find furthest points from hazards
        hazard_dist = cv2.distanceTransform((~hazard_mask).astype(np.uint8), cv2.DIST_L2, 5)
        
        # Distance transform to find proximity to scientific ice targets
        ice_dist = cv2.distanceTransform((~ice_mask).astype(np.uint8), cv2.DIST_L2, 5)

        candidates = []
        
        # Grid search stride
        stride = max(1, r // 2)
        
        for y in range(r, h - r, stride):
            for x in range(r, w - r, stride):
                # Extract landing patch candidate
                patch_haz = hazard_mask[y-r:y+r, x-r:x+r]
                
                # Hard Constraint: Zero hazard overlap allowed inside touchdown circle
                if np.sum(patch_haz) > 0:
                    continue

                # Safety Score: Distance from nearest slope hazard
                min_dist_to_hazard = hazard_dist[y, x] * self.pixel_res
                
                # Proximity Score: Distance to nearest water-ice volatile target
                dist_to_ice = ice_dist[y, x] * self.pixel_res
                
                # Local Terrain Flatness Score (Variance in intensity)
                patch_img = enhanced_img[y-r:y+r, x-r:x+r]
                flatness_score = 1.0 / (np.std(patch_img) + 1e-5)

                # Composite Landing Suitability Index
                composite_score = (min_dist_to_hazard * 2.0) + (flatness_score * 0.5) - (dist_to_ice * 0.1)

                candidates.append({
                    "center_pixel": (x, y),
                    "center_coords_m": ((x - w/2) * self.pixel_res, (y - h/2) * self.pixel_res),
                    "min_hazard_dist_m": round(min_dist_to_hazard, 2),
                    "dist_to_ice_m": round(dist_to_ice, 2),
                    "score": composite_score
                })

        # Sort candidate sites by composite suitability score
        candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
        
        # Non-maximum suppression to avoid overlapping candidate circles
        selected_sites = []
        for cand in candidates:
            cx, cy = cand["center_pixel"]
            overlap = False
            for prev in selected_sites:
                px, py = prev["center_pixel"]
                if np.hypot(cx - px, cy - py) < r:
                    overlap = True
                    break
            if not overlap:
                selected_sites.append(cand)
            if len(selected_sites) >= top_k:
                break

        return selected_sites

    def draw_landing_zones_on_map(self, img_rgb, landing_sites):
        """Annotates target landing circles onto a visualization map."""
        annotated = (img_rgb * 255).astype(np.uint8)
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        colors_list = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]  # Green = Site 1, Yellow = Site 2, Cyan = Site 3
        
        for idx, site in enumerate(landing_sites):
            cx, cy = site["center_pixel"]
            color = colors_list[idx % len(colors_list)]
            
            # Draw outer landing ellipse
            cv2.circle(annotated, (cx, cy), self.pixel_radius, color, 2)
            # Draw center crosshair
            cv2.line(annotated, (cx - 10, cy), (cx + 10, cy), color, 2)
            cv2.line(annotated, (cx, cy - 10), (cx, cy + 10), color, 2)
            
            # Label designation
            cv2.putText(
                annotated, f"Zone {idx+1}", (cx - 25, cy - self.pixel_radius - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

        return annotated