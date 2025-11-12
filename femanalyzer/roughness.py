import numpy as np
from sklearn.decomposition import PCA

# --- Internal Core Functions ---
def _calculate_areal_parameters(deviations: np.ndarray):
    """Calculates ISO 25178 3D areal roughness parameters."""
    if deviations.size == 0:
        return {"Sa": 0.0, "Sq": 0.0, "Sz": 0.0}
    Sa = np.mean(np.abs(deviations))
    Sq = np.sqrt(np.mean(deviations ** 2))
    Sz = np.max(deviations) - np.min(deviations)
    return {"Sa": Sa, "Sq": Sq, "Sz": Sz}

def _remove_form_mean(points: np.ndarray) -> np.ndarray:
    """Removes form by subtracting the mean of the Z-coordinate."""
    z_heights = points[:, 2]
    return z_heights - np.mean(z_heights)

def _remove_form_planar_pca(points: np.ndarray) -> np.ndarray:
    """Removes the planar 'form' using Principal Component Analysis (PCA)."""
    pca = PCA(n_components=3)
    pca.fit(points)
    normal_vector = pca.components_[2] # Normal to the best-fit plane
    centered_points = points - pca.mean_
    deviations = np.dot(centered_points, normal_vector)
    return deviations

# --- Public API Function ---
def analyze_surface_roughness(points: np.ndarray, detrend_method: str = 'plane'):
    """
    Analyzes the 3D surface roughness of a point cloud.

    Args:
        points (np.ndarray): (N, 3) array of surface node coordinates.
        detrend_method (str): 'plane' (default) or 'mean'.

    Returns:
        dict: A dictionary containing Sa, Sq, and Sz parameters.
    """
    if points.size == 0 or points.shape[0] == 0:
        print("Warning: Received empty point array. Skipping roughness analysis.")
        return {"Sa": 0.0, "Sq": 0.0, "Sz": 0.0}

    print(f"Analyzing {points.shape[0]} points using '{detrend_method}' detrending...")
    
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must be (N, 3), but got shape {points.shape}")

    if detrend_method == 'plane':
        deviations = _remove_form_planar_pca(points)
    elif detrend_method == 'mean':
        deviations = _remove_form_mean(points)
    else:
        raise ValueError(f"Unknown detrend_method: '{detrend_method}'.")

    parameters = _calculate_areal_parameters(deviations)
    
    print(f"  Sa (Arithmetical Mean): {parameters['Sa']:.6e} m")
    print(f"  Sq (RMS Height):        {parameters['Sq']:.6e} m")
    print(f"  Sz (Max Height):        {parameters['Sz']:.6e} m")
    
    return parameters