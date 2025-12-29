"""
Thermal Image Preprocessing Module
Handles FLIR thermal camera data preprocessing and augmentation
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import albumentations as A
from dataclasses import dataclass
import logging

@dataclass
class ThermalPreprocessorConfig:
    """Configuration for thermal image preprocessing"""
    input_size: Tuple[int, int] = (640, 512)
    output_size: Tuple[int, int] = (640, 640)
    normalize_range: Tuple[float, float] = (0.0, 255.0)
    apply_hist_eq: bool = True
    denoise_strength: float = 0.1
    clip_thermal_range: bool = True
    min_temp: float = -20.0
    max_temp: float = 120.0

class ThermalPreprocessor:
    """Preprocessing pipeline for thermal images"""
    
    def __init__(self, config: ThermalPreprocessorConfig = None):
        self.config = config or ThermalPreprocessorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Define augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.MotionBlur(blur_limit=7, p=0.2),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def load_thermal_image(self, image_path: str) -> np.ndarray:
        """Load thermal image with temperature data preservation"""
        try:
            # Read as 16-bit for temperature precision
            if image_path.endswith('.tiff') or image_path.endswith('.tif'):
                image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
            else:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            return image.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def normalize_thermal(self, image: np.ndarray) -> np.ndarray:
        """Normalize thermal image to 0-255 range"""
        if self.config.clip_thermal_range:
            image = np.clip(image, self.config.min_temp, self.config.max_temp)
        
        # Convert to 8-bit
        image_normalized = cv2.normalize(
            image, None, 
            alpha=self.config.normalize_range[0],
            beta=self.config.normalize_range[1],
            norm_type=cv2.NORM_MINMAX
        )
        
        return image_normalized.astype(np.uint8)
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast enhancement"""
        if self.config.apply_hist_eq:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        return image
    
    def denoise_thermal(self, image: np.ndarray) -> np.ndarray:
        """Apply thermal-specific denoising"""
        # Non-local means denoising for thermal images
        denoised = cv2.fastNlMeansDenoising(
            image,
            h=self.config.denoise_strength * 30,
            templateWindowSize=7,
            searchWindowSize=21
        )
        return denoised
    
    def enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges in thermal images"""
        # Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
        
        # Laplacian for edge enhancement
        laplacian = cv2.Laplacian(blurred, cv2.CV_32F)
        
        # Convert back to uint8 and combine
        laplacian_abs = cv2.convertScaleAbs(laplacian)
        
        # Add enhanced edges to original
        enhanced = cv2.addWeighted(image, 0.7, laplacian_abs, 0.3, 0)
        
        return enhanced
    
    def resize_with_padding(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image with aspect ratio preservation"""
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scaling factor
        scale = min(target_h / h, target_w / w)
        
        # New dimensions
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.zeros((target_h, target_w), dtype=image.dtype)
        
        # Calculate padding
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        
        # Place resized image in center
        padded[top:top+new_h, left:left+new_w] = resized
        
        return padded
    
    def preprocess_pipeline(self, image_path: str, apply_augmentation: bool = False,
                           bboxes: Optional[List] = None, labels: Optional[List] = None) -> dict:
        """Complete preprocessing pipeline for thermal images"""
        
        # Load image
        raw_image = self.load_thermal_image(image_path)
        
        # Normalize to 8-bit
        normalized = self.normalize_thermal(raw_image)
        
        # Apply histogram equalization
        enhanced = self.apply_histogram_equalization(normalized)
        
        # Denoise
        denoised = self.denoise_thermal(enhanced)
        
        # Edge enhancement
        final_image = self.enhance_edges(denoised)
        
        # Resize with padding
        resized = self.resize_with_padding(final_image, self.config.output_size)
        
        # Apply augmentations if requested
        if apply_augmentation and bboxes is not None and labels is not None:
            augmented = self.augmentation_pipeline(
                image=resized,
                bboxes=bboxes,
                class_labels=labels
            )
            resized = augmented['image']
            bboxes = augmented['bboxes']
            labels = augmented['class_labels']
        
        # Convert to 3-channel for YOLO
        final_output = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        # Calculate scale factors for bounding boxes
        h, w = final_image.shape[:2]
        target_h, target_w = self.config.output_size
        
        scale_h = target_h / h
        scale_w = target_w / w
        
        return {
            'image': final_output,
            'original_shape': (h, w),
            'processed_shape': final_output.shape[:2],
            'scale_factors': (scale_w, scale_h),
            'bboxes': bboxes if bboxes else [],
            'labels': labels if labels else []
        }
    
    def batch_preprocess(self, image_paths: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Process batch of images efficiently"""
        processed_images = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for path in batch_paths:
                result = self.preprocess_pipeline(path)
                batch_images.append(result['image'])
            
            processed_images.extend(batch_images)
            
            self.logger.info(f"Processed {i+len(batch_paths)}/{len(image_paths)} images")
        
        return processed_images
    
    def extract_thermal_features(self, image: np.ndarray) -> dict:
        """Extract thermal-specific features for analysis"""
        features = {}
        
        # Temperature statistics
        features['mean_temp'] = np.mean(image)
        features['std_temp'] = np.std(image)
        features['max_temp'] = np.max(image)
        features['min_temp'] = np.min(image)
        
        # Thermal gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        features['gradient_magnitude'] = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        # Thermal contrast
        features['thermal_contrast'] = np.max(image) - np.min(image)
        
        # Hot spot detection
        _, binary = cv2.threshold(image, 0.8 * np.max(image), 255, cv2.THRESH_BINARY)
        features['hot_spot_area'] = np.sum(binary > 0) / (image.size)
        
        return features

if __name__ == "__main__":
    # Example usage
    config = ThermalPreprocessorConfig(
        input_size=(640, 512),
        output_size=(640, 640),
        apply_hist_eq=True
    )
    
    preprocessor = ThermalPreprocessor(config)
    
    # Process single image
    result = preprocessor.preprocess_pipeline("sample_thermal.jpg")
    
    print(f"Original shape: {result['original_shape']}")
    print(f"Processed shape: {result['processed_shape']}")
    print(f"Number of bboxes: {len(result['bboxes'])}")
