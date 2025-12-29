"""
Thermal Image Augmentation Module
Specialized augmentations for thermal imagery
"""

import cv2
import numpy as np
import random
from typing import Tuple, List
import albumentations as A
from albumentations.core.composition import Compose
import logging

class ThermalAugmentor:
    """Augmentation pipeline specifically designed for thermal images"""
    
    def __init__(self, augmentation_mode: str = 'standard'):
        """
        Args:
            augmentation_mode: 'standard', 'aggressive', or 'light'
        """
        self.mode = augmentation_mode
        self.logger = logging.getLogger(__name__)
        
        # Define augmentations based on mode
        self.augmentation_pipeline = self._create_pipeline()
        
    def _create_pipeline(self) -> Compose:
        """Create augmentation pipeline based on mode"""
        
        if self.mode == 'aggressive':
            return A.Compose([
                # Geometric transformations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=45, p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=0,
                    p=0.5
                ),
                A.Perspective(scale=(0.05, 0.1), p=0.3),
                
                # Thermal-specific augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.7
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                
                # Weather simulations
                self._simulate_fog(p=0.2),
                self._simulate_rain(p=0.1),
                self._simulate_snow(p=0.1),
                
                # Sensor noise
                self._add_thermal_noise(p=0.4),
                self._simulate_temperature_shift(p=0.3),
                
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.3
            ))
            
        elif self.mode == 'light':
            return A.Compose([
                A.HorizontalFlip(p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.5
            ))
            
        else:  # standard
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=5,
                    p=0.4
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.6
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.4),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.4),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.RandomGamma(gamma_limit=(90, 110), p=0.4),
                self._add_thermal_noise(p=0.3),
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'],
                min_visibility=0.4
            ))
    
    def _add_thermal_noise(self, p: float = 0.5):
        """Add thermal-specific noise patterns"""
        def thermal_noise_transform(image, **kwargs):
            if random.random() < p:
                h, w = image.shape[:2]
                
                # Add fixed pattern noise (common in thermal sensors)
                pattern_noise = np.zeros((h, w), dtype=np.float32)
                
                # Vertical banding (common in thermal cameras)
                for i in range(0, w, 10):
                    band_width = random.randint(1, 3)
                    band_strength = random.uniform(0.5, 2.0)
                    pattern_noise[:, i:i+band_width] += band_strength
                
                # Random hot/cold pixels
                num_hot_pixels = random.randint(5, 20)
                for _ in range(num_hot_pixels):
                    x, y = random.randint(0, w-1), random.randint(0, h-1)
                    pattern_noise[y, x] += random.uniform(5.0, 15.0)
                
                # Add the noise
                image = image.astype(np.float32) + pattern_noise
                image = np.clip(image, 0, 255)
                
            return image.astype(np.uint8)
        
        return A.Lambda(name="ThermalNoise", image=thermal_noise_transform, p=p)
    
    def _simulate_fog(self, p: float = 0.3):
        """Simulate fog/atmospheric effects"""
        def fog_transform(image, **kwargs):
            if random.random() < p:
                h, w = image.shape[:2]
                
                # Create fog effect
                fog_intensity = random.uniform(0.1, 0.4)
                fog_base = np.full((h, w), random.uniform(100, 150), dtype=np.float32)
                
                # Add gradient
                y_coords = np.linspace(0, 1, h)[:, np.newaxis]
                fog_gradient = fog_intensity * (1 - y_coords)  # More fog at bottom
                
                # Apply fog
                image = image.astype(np.float32)
                image = image * (1 - fog_gradient) + fog_base * fog_gradient
                image = np.clip(image, 0, 255)
                
            return image.astype(np.uint8)
        
        return A.Lambda(name="FogSimulation", image=fog_transform, p=p)
    
    def _simulate_temperature_shift(self, p: float = 0.4):
        """Simulate temperature variations"""
        def temperature_transform(image, **kwargs):
            if random.random() < p:
                # Global temperature shift
                temp_shift = random.uniform(-10, 10)
                image = image.astype(np.float32) + temp_shift
                
                # Local hot/cold spots
                if random.random() < 0.3:
                    h, w = image.shape[:2]
                    center_x = random.randint(w//4, 3*w//4)
                    center_y = random.randint(h//4, 3*h//4)
                    radius = random.randint(20, min(h, w)//4)
                    
                    y, x = np.ogrid[:h, :w]
                    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                    spot_strength = random.uniform(-20, 20)
                    image[mask] += spot_strength
                
                image = np.clip(image, 0, 255)
            
            return image.astype(np.uint8)
        
        return A.Lambda(name="TemperatureShift", image=temperature_transform, p=p)
    
    def augment_image(self, image: np.ndarray, bboxes: List = None, 
                     labels: List = None) -> dict:
        """
        Apply augmentations to a single image
        
        Args:
            image: Input thermal image
            bboxes: List of bounding boxes
            labels: List of class labels
            
        Returns:
            Dictionary with augmented image and updated bboxes/labels
        """
        if bboxes is None:
            bboxes = []
        if labels is None:
            labels = []
        
        try:
            # Convert grayscale to 3 channels if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Apply augmentations
            augmented = self.augmentation_pipeline(
                image=image,
                bboxes=bboxes,
                class_labels=labels
            )
            
            return {
                'image': augmented['image'],
                'bboxes': augmented['bboxes'],
                'labels': augmented['class_labels']
            }
            
        except Exception as e:
            self.logger.error(f"Augmentation failed: {e}")
            return {
                'image': image,
                'bboxes': bboxes,
                'labels': labels
            }
    
    def augment_batch(self, images: List[np.ndarray], 
                     bboxes_list: List[List] = None,
                     labels_list: List[List] = None) -> List[dict]:
        """
        Augment batch of images
        
        Args:
            images: List of input images
            bboxes_list: List of bbox lists
            labels_list: List of label lists
            
        Returns:
            List of augmentation results
        """
        results = []
        
        if bboxes_list is None:
            bboxes_list = [[] for _ in range(len(images))]
        if labels_list is None:
            labels_list = [[] for _ in range(len(images))]
        
        for i, (img, bboxes, labels) in enumerate(zip(images, bboxes_list, labels_list)):
            result = self.augment_image(img, bboxes, labels)
            results.append(result)
            
            if (i + 1) % 100 == 0:
                self.logger.info(f"Augmented {i+1}/{len(images)} images")
        
        return results
    
    def create_augmented_dataset(self, original_images: List[str], 
                                output_dir: str, num_augmentations: int = 3):
        """
        Create augmented dataset from original images
        
        Args:
            original_images: List of paths to original images
            output_dir: Directory to save augmented images
            num_augmentations: Number of augmentations per image
        """
        import os
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating augmented dataset with {num_augmentations} augmentations per image")
        
        for img_idx, img_path in enumerate(original_images):
            # Load image
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                self.logger.warning(f"Failed to load image: {img_path}")
                continue
            
            # Generate augmentations
            for aug_idx in range(num_augmentations):
                augmented = self.augment_image(image)
                
                # Save augmented image
                img_name = Path(img_path).stem
                output_path = output_dir / f"{img_name}_aug{aug_idx:03d}.jpg"
                cv2.imwrite(str(output_path), augmented['image'])
            
            if (img_idx + 1) % 10 == 0:
                self.logger.info(f"Processed {img_idx+1}/{len(original_images)} images")
        
        self.logger.info(f"Augmented dataset created in {output_dir}")

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create augmentor
    augmentor = ThermalAugmentor(mode='standard')
    
    # Load sample image
    sample_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    # Apply augmentations
    augmented = augmentor.augment_image(sample_image)
    
    print(f"Original shape: {sample_image.shape}")
    print(f"Augmented shape: {augmented['image'].shape}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sample_image, cmap='hot')
    axes[0].set_title('Original')
    axes[1].imshow(augmented['image'], cmap='hot')
    axes[1].set_title('Augmented')
    plt.show()
