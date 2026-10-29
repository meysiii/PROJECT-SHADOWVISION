import numpy as np
import plotly.graph_objects as go
import cv2

def render_3d_lunar_surface(enhanced_img, hazard_mask=None, ice_mask=None, downsample_size=(128, 128)):
    """
    Renders an interactive 3D mesh map of the lunar PSR surface 
    using normalized radiance intensity as elevation proxy.
    """
    # Downsample for smooth 60fps rendering in WebGL browser engine
    h, w = downsample_size
    img_resized = cv2.resize(enhanced_img, (w, h), interpolation=cv2.INTER_AREA)
    
    # Generate elevation coordinates (Z) based on enhanced radiance
    z_elevation = img_resized * 50.0  # Scale factor for realistic relief view
    x = np.linspace(0, w * 0.25, w)    # Real-world scale in meters (0.25m/px)
    y = np.linspace(0, h * 0.25, h)
    X, Y = np.meshgrid(x, y)

    # Base surface plot using Magma surface color mapping
    fig = go.Figure(data=[go.Surface(
        z=z_elevation,
        x=X,
        y=Y,
        surfacecolor=img_resized,
        colorscale='Magma',
        colorbar=dict(title='Radiance Intensity'),
        lighting=dict(ambient=0.4, diffuse=0.8, roughness=0.5, specular=0.2),
    )])

    # Overlay Hazardous Slopes as 3D Scatter Markers if provided
    if hazard_mask is not None:
        haz_resized = cv2.resize(hazard_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        haz_y, haz_x = np.where(haz_resized)
        if len(haz_x) > 0:
            fig.add_trace(go.Scatter3d(
                x=X[haz_y, haz_x],
                y=Y[haz_y, haz_x],
                z=z_elevation[haz_y, haz_x] + 1.0,
                mode='markers',
                marker=dict(size=3, color='red', opacity=0.8),
                name='Hazard Slopes (>15°)'
            ))

    # Overlay Water-Ice Candidates as Blue 3D Scatter Markers if provided
    if ice_mask is not None:
        ice_resized = cv2.resize(ice_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        ice_y, ice_x = np.where(ice_resized)
        if len(ice_x) > 0:
            fig.add_trace(go.Scatter3d(
                x=X[ice_y, ice_x],
                y=Y[ice_y, ice_x],
                z=z_elevation[ice_y, ice_x] + 1.5,
                mode='markers',
                marker=dict(size=4, color='cyan', opacity=0.9),
                name='Water-Ice Candidates'
            ))

    # Configure 3D Camera Controls and Layout
    fig.update_layout(
        title="Interactive 3D Lunar PSR Topography & Landing Hazards",
        scene=dict(
            xaxis_title="X (Meters)",
            yaxis_title="Y (Meters)",
            zaxis_title="Elevation Proxy (m)",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
            aspectratio=dict(x=1, y=1, z=0.4)
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        template="plotly_dark",
        height=600
    )

    return fig