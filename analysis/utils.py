import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from collections import defaultdict
from typing import Dict, Union, Tuple
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split


class FingerprintLoader:
    def __init__(self):
        self.subject_to_frgp = None
        pass

    def extract_subject_frgp_map(self, root_dir:str) -> Dict:
        """
        Traverse the directory tree rooted at `root_dir` and extract a mapping from subject IDs 
        to their associated friction ridge generalized position (FRGP) codes based on image filenames.

        The filenames are expected to follow the format:
            SUBJECT_DEVICE_RESOLUTION_CAPTURE_FRGP.EXT
        For example:
            00002303_A_roll_01.png

        Args:
            root_dir (str): The root directory containing the image files organized in subfolders.

        Returns:
            dict[str, list[str]]: A dictionary where each key is a subject ID (SUBJECT), and the value 
            is a list of FRGP codes associated with that subject.

        Notes:
            - Only files with common image extensions (.png) are considered.
            - Files that do not conform to the expected naming format are skipped with a warning.
        """
        self.subject_to_frgp = defaultdict(set)

        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if not filename.lower().endswith(('.png', '.npy')):
                    break

                name, _ = os.path.splitext(filename)
                parts = name.split('_')
                
                if len(parts) < 4:
                    print(f"Skipping unrecognized format: {filename}")
                    continue
                
                subject = parts[0]
                frgp = parts[-1]

                self.subject_to_frgp[subject].add(frgp)
                
        self.subject_to_frgp = dict(self.subject_to_frgp)
        return self.subject_to_frgp

    def _build_filename(self, subject:str, frgp:str, device='A', capture='roll', extension='.png') -> str:
        """
        Construct a fingerprint image filename based on subject ID and FRGP code.

        Args:
            subject (str): Unique identifier for the study participant.
            frgp (str): Friction ridge generalized position code (e.g., '01', '02').
            device (str, optional): Device code. Default is 'A'.
            capture (str, optional): Capture type. Default is 'roll'.

        Returns:
            str: A string representing the filename in the format:
                SUBJECT_DEVICE_CAPTURE_FRGP.png
                SUBJECT_DEVICE_CAPTURE_FRGP.npy
        """
        return f"{subject}_{device}_{capture}_{frgp}{extension}"

    def load_fingerprints(self, subject:str, root_dir:str, device='A', capture='roll', resize:Union[None,float]=None):
        """
        Load all fingerprint image data for a given subject.

        Args:
            subject (str): Unique identifier for the study participant.
            root_dir (str): Root directory where fingerprint images are stored.
            device (str, optional): Device code. Default is 'A'.
            capture (str, optional): Capture type. Default is 'roll'.

        Returns:
            np.ndarray: A 2D array where each row is a flattened fingerprint image. Shape: (num_images, height * width)
        """
        fingerprints = []
        
        for frgp in self.subject_to_frgp[subject]:
            filename = self._build_filename(subject, frgp, device, capture)
            for dirpath, _, filenames in os.walk(root_dir):
                if filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) 
                    if img is not None:
                        # fingerprints.append(img.flatten())
                        if resize is not None:
                            height, width = img.shape
                            new_width = int(width * resize)  
                            new_height = int(height * resize) 
                            new_size = (new_width, new_height)
                            img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
                        fingerprints.append(img)
                    else:
                        print(f"Warning: Failed to read image {file_path}")
                    break  # stop searching once file is found
        if fingerprints:
            return fingerprints
            # return np.vstack(fingerprints)
        else:
            return np.empty((0, 0))

def center_crop_to_smallest(images:List[np.ndarray])->List[np.ndarray]:
    """
    Center-crop all images in the list to match the smallest image's dimensions.

    Args:
        images (list of np.ndarray): List of grayscale or color images (as NumPy arrays).

    Returns:
        list of np.ndarray: List of cropped images with uniform shape.
    """
    # Find the smallest height and width
    min_height = min(img.shape[0] for img in images)
    min_width = min(img.shape[1] for img in images)
    
    cropped_images = []

    for img in images:
        h, w = img.shape[:2]

        top = (h - min_height) // 2
        left = (w - min_width) // 2

        # Crop the image
        cropped = img[top:top + min_height, left:left + min_width]
        cropped_images.append(cropped)

    return cropped_images, (min_height, min_width)

def get_features(
    root_dir: str,
    seed: int = 42,
    flatten: bool = True,
    num_subjects: int = 10,
    resize: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Load and preprocess fingerprint features from a given directory.

    Args:
        root_dir (str): Root directory containing fingerprint images.
        seed (int): Random seed for reproducibility.
        flatten (bool): Whether to flatten each image to a 1D vector.
        num_subjects (int): Number of unique individuals to sample.
        resize (float): Resize factor for loaded fingerprint images.

    Returns:
        features (np.ndarray): Array of shape (N, D) or (N, 1, H, W) depending on flatten.
        labels (np.ndarray): Array of shape (N,) with subject labels.
        img_size (Tuple[int, int]): The final image size (H, W) after cropping.
    """
    loader = FingerprintLoader()
    subject_fingerprint_map = loader.extract_subject_frgp_map(root_dir)

    rng = np.random.default_rng(seed)
    selected_subjects = set(rng.choice(list(subject_fingerprint_map.keys()), num_subjects, replace=False))

    features: List[np.ndarray] = []
    labels: List[int] = []
    
    for subject_idx, subject_id in enumerate(selected_subjects):
        fingerprints = loader.load_fingerprints(subject_id, root_dir, resize=resize)
        features.extend(fingerprints)
        labels.extend([subject_idx] * len(fingerprints))

    labels = np.array(labels, dtype=np.uint8)
    features, img_size = center_crop_to_smallest(features)

    if flatten:
        features = np.array([f.flatten() for f in features])
    else:
        features = np.expand_dims(np.array(features), axis=1)  # shape: (N, 1, H, W)

    return features, labels, img_size