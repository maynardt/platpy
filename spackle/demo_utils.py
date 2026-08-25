import copy
import urllib.request
import zipfile
import cv2
import numpy as np
import dask.array as da
from pathlib import Path


def fetch_zenodo_bundle(dest, url):
    """
    Download and extract a SPACKLE bundle from Zenodo (one-time; skips if already present).

    Parameters
    ----------
    dest : str or Path — local directory to extract into
    url  : str — direct Zenodo file URL (.zip/content)
    """
    dest = Path(dest)
    if dest.exists():
        return
    zip_path = dest.parent / (dest.name + '.zip')
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} from Zenodo...")
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            # Normalize Windows backslash paths so extraction works on Mac/Linux
            member.filename = member.filename.replace('\\', '/')
            z.extract(member, dest.parent)
    zip_path.unlink()
    print(f"✅ Ready at {dest}")

# Standardized colors for the demo
RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)
CYAN = (0, 255, 255, 255)


def create_demo_base_layer(channel, spots_um: np.ndarray, spot_radius_um: float = 80, spot_color: tuple = RED):
    """
    Creates an 8-bit RGBA copy of a channel, drops the data into the green channel, 
    and bakes max-brightness colored spots directly into the image array.
    """
    raw_img = channel.image
    raw_img = raw_img.compute() if isinstance(raw_img, da.Array) else raw_img

    # Normalize the 16-bit array to 8-bit
    valid_pixels = raw_img[raw_img > 0]
    p_hi = np.percentile(valid_pixels, 99.9) if len(valid_pixels) > 0 else 1.0
    norm_8bit = np.clip((raw_img / p_hi) * 255, 0, 255).astype(np.uint8)

    # Create an RGBA image: Drop NeuN data into the Green channel, set Alpha to 255
    rgba_image = np.zeros((*norm_8bit.shape, 4), dtype=np.uint8)
    rgba_image[..., 1] = norm_8bit  # Green channel
    rgba_image[..., 3] = 255        # Alpha channel

    # Bake spots
    spot_r_native = max(1, int(spot_radius_um / channel.mpp)) 
    for px in channel.microns_to_pixels(spots_um).round().astype(int):
        cv2.circle(rgba_image, (int(px[0]), int(px[1])), radius=spot_r_native, color=spot_color, thickness=-1)

    # Assign this new RGBA array to our throwaway channel
    new_channel = copy.deepcopy(channel)
    new_channel.image = np.ascontiguousarray(rgba_image)
    new_channel.pyramid = {}  
    
    return new_channel


def detect_circles(panel: np.ndarray, lower_rgba: list, upper_rgba: list, min_area: int = 500) -> np.ndarray:
    """
    Finds the sub-pixel centroids of color-matched blobs in an RGBA image.
    Used to mathematically verify that rendered spots appear exactly where 
    they belong in physical space.
    """
    # Ensure panel is 8-bit RGBA before thresholding
    if panel.dtype != np.uint8:
        if panel.max() <= 1.0:
            panel = (panel * 255).astype(np.uint8)
        else:
            panel = (panel / panel.max() * 255).astype(np.uint8)

    mask = cv2.inRange(
        panel, 
        np.array(lower_rgba, dtype=np.uint8),
        np.array(upper_rgba, dtype=np.uint8)
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centers = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] > 0:
            # Center of mass (centroid)
            centers.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])
            
    if not centers:
        print("⚠️ WARNING: No circles detected matching the color bounds.")
        return np.empty((0, 2))
            
    return np.array(centers)


def match_to_predicted(detected: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Ensures that an array of detected coordinate centers is sorted to match 
    the order of the original predicted coordinates.
    """
    if len(detected) == 0:
        raise ValueError("Cannot match predictions: No circles were detected! "
                         "Check that your ViewPort tint matches the OpenCV detection bounds.")
        
    return np.array([
        detected[np.argmin(np.linalg.norm(detected - p, axis=1))]
        for p in predicted.astype(float)
    ])