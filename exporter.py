import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
import os

class LunarGeoTIFFExporter:
    """
    Geospatial export utility for lunar orbital imagery.
    Encapsulates enhanced array outputs into PDS4/GIS-compliant GeoTIFFs.
    """
    def __init__(self, pixel_resolution=0.25, center_lat=-89.0, center_lon=0.0):
        """
        Parameters:
            pixel_resolution: Spatial resolution in meters per pixel (OHRC standard = 0.25 m/px).
            center_lat: Latitude coordinate of the lunar crater scene center.
            center_lon: Longitude coordinate of the lunar crater scene center.
        """
        self.pixel_resolution = pixel_resolution
        self.center_lat = center_lat
        self.center_lon = center_lon

        # Define Lunar Polar Stereographic CRS (Moon IAU2000 / South Pole)
        # Standard WKT/PROJ string for South Pole Lunar Polar Stereographic
        self.lunar_crs = CRS.from_proj4(
            f"+proj=stere +lat_0={self.center_lat} +lon_0={self.center_lon} "
            f"+k=1 +x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs"
        )

    def export_enhanced_geotiff(self, enhanced_arr, hazard_mask, ice_mask, output_filepath="shadow_vision_output.tif"):
        """
        Exports a multi-band GeoTIFF:
          Band 1: Enhanced Dynamic Range Image (Float32)
          Band 2: Steep Slope Hazard Mask (UInt8: 0 or 255)
          Band 3: Water-Ice Candidate Mask (UInt8: 0 or 255)
        """
        height, width = enhanced_arr.shape

        # Calculate Affine transformation matrix (Origin centered on scene)
        # Upper-left pixel coordinate in meters relative to scene center
        origin_x = -(width * self.pixel_resolution) / 2.0
        origin_y = (height * self.pixel_resolution) / 2.0

        transform = Affine.translation(origin_x, origin_y) * Affine.scale(
            self.pixel_resolution, -self.pixel_resolution
        )

        # Ensure correct data formats
        band1 = enhanced_arr.astype(np.float32)
        band2 = (hazard_mask.astype(np.uint8)) * 255
        band3 = (ice_mask.astype(np.uint8)) * 255

        # GeoTIFF metadata profile
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 3,  # 3 Bands
            'dtype': 'float32',
            'crs': self.lunar_crs,
            'transform': transform,
            'compress': 'lzw',
            'nodata': 0.0
        }

        # Write dataset to file
        with rasterio.open(output_filepath, 'w', **profile) as dst:
            dst.write(band1, 1)
            dst.set_band_description(1, "Enhanced_PSR_Radiance")

            # Write Band 2 & 3 as cast Float32 to maintain uniform file dtype
            dst.write(band2.astype(np.float32), 2)
            dst.set_band_description(2, "Hazard_Mask_Slopes_Boulders")

            dst.write(band3.astype(np.float32), 3)
            dst.set_band_description(3, "Water_Ice_Candidate_Anomalies")

        print(f"[GIS Engine] Successfully saved spatial GeoTIFF: {os.path.abspath(output_filepath)}")
        return output_filepath