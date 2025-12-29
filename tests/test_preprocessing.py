"""
Tests for thermal image preprocessing
"""

import unittest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from data_preprocessing.thermal_preprocessor import ThermalPreprocessor, ThermalPreprocessorConfig

class TestThermalPreprocessor(unittest.TestCase):
    """Test thermal image preprocessing"""
    
    def setUp(self):
        """Setup test data"""
        # Create a synthetic thermal image
        self.test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        # Create preprocessor
        self.config = ThermalPreprocessorConfig(
            input_size=(256, 256),
            output_size=(256, 256),
            normalize_range=(0, 255),
            apply_hist_eq=True,
            denoise_strength=0.1
        )
        self.preprocessor = ThermalPreprocessor(self.config)
    
    def test_load_thermal_image(self):
        """Test image loading"""
        # Create temporary test image
        test_path = Path('test_thermal.png')
        cv2.imwrite(str(test_path), self.test_image)
        
        try:
            # Test loading
            loaded_image = self.preprocessor.load_thermal_image(str(test_path))
            
            # Check properties
            self.assertEqual(loaded_image.dtype, np.float32)
            self.assertEqual(loaded_image.shape, self.test_image.shape)
            
            # Cleanup
            test_path.unlink()
            
        except Exception as e:
            if test_path.exists():
                test_path.unlink()
            self.fail(f"Failed to load image: {e}")
    
    def test_normalize_thermal(self):
        """Test thermal normalization"""
        # Create test data with wide range
        test_data = np.array([-10, 0, 50, 100, 150], dtype=np.float32)
        
        # Test normalization
        normalized = self.preprocessor.normalize_thermal(test_data)
        
        # Check range
        self.assertGreaterEqual(normalized.min(), 0)
        self.assertLessEqual(normalized.max(), 255)
        self.assertEqual(normalized.dtype, np.uint8)
    
    def test_histogram_equalization(self):
        """Test histogram equalization"""
        # Create low contrast image
        low_contrast = np.random.randint(100, 120, (100, 100), dtype=np.uint8)
        
        # Apply equalization
        equalized = self.preprocessor.apply_histogram_equalization(low_contrast)
        
        # Check properties
        self.assertEqual(equalized.shape, low_contrast.shape)
        self.assertEqual(equalized.dtype, np.uint8)
        
        # Equalized image should have better contrast
        orig_contrast = low_contrast.std()
        eq_contrast = equalized.std()
        self.assertGreater(eq_contrast, orig_contrast * 0.5)  # At least 50% improvement
    
    def test_denoise_thermal(self):
        """Test thermal denoising"""
        # Create noisy image
        clean = np.random.randint(100, 150, (100, 100), dtype=np.uint8)
        noise = np.random.randn(100, 100) * 20
        noisy = np.clip(clean + noise, 0, 255).astype(np.uint8)
        
        # Apply denoising
        denoised = self.preprocessor.denoise_thermal(noisy)
        
        # Check properties
        self.assertEqual(denoised.shape, noisy.shape)
        self.assertEqual(denoised.dtype, np.uint8)
        
        # Denoised should have less noise
        noise_level_before = np.std(noisy - clean)
        noise_level_after = np.std(denoised - clean)
        self.assertLess(noise_level_after, noise_level_before * 1.5)  # Should reduce noise
    
    def test_resize_with_padding(self):
        """Test resize with aspect ratio preservation"""
        # Create test image
        test_img = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
        target_size = (256, 256)
        
        # Resize
        resized = self.preprocessor.resize_with_padding(test_img, target_size)
        
        # Check properties
        self.assertEqual(resized.shape, target_size)
        self.assertEqual(resized.dtype, test_img.dtype)
        
        # Check aspect ratio preservation (should have black borders)
        non_zero_pixels = np.sum(resized > 0)
        total_pixels = resized.size
        self.assertLess(non_zero_pixels / total_pixels, 1.0)  # Should have padding
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline"""
        # Create test image path
        test_path = Path('test_pipeline.png')
        cv2.imwrite(str(test_path), self.test_image)
        
        try:
            # Run pipeline
            result = self.preprocessor.preprocess_pipeline(
                str(test_path),
                apply_augmentation=False
            )
            
            # Check result structure
            self.assertIn('image', result)
            self.assertIn('original_shape', result)
            self.assertIn('processed_shape', result)
            self.assertIn('scale_factors', result)
            
            # Check image properties
            self.assertEqual(result['image'].shape[2], 3)  # Should be 3-channel
            self.assertEqual(result['image'].dtype, np.uint8)
            self.assertEqual(result['processed_shape'], (256, 256))
            
            # Cleanup
            test_path.unlink()
            
        except Exception as e:
            if test_path.exists():
                test_path.unlink()
            self.fail(f"Pipeline failed: {e}")
    
    def test_extract_thermal_features(self):
        """Test thermal feature extraction"""
        # Create test image
        test_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Extract features
        features = self.preprocessor.extract_thermal_features(test_img)
        
        # Check feature structure
        expected_features = ['mean_temp', 'std_temp', 'max_temp', 'min_temp',
                           'gradient_magnitude', 'thermal_contrast', 'hot_spot_area']
        
        for feat in expected_features:
            self.assertIn(feat, features)
            self.assertIsInstance(features[feat], (int, float, np.number))
    
    def test_batch_preprocess(self):
        """Test batch preprocessing"""
        # Create multiple test images
        test_paths = []
        for i in range(3):
            path = Path(f'test_batch_{i}.png')
            img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            cv2.imwrite(str(path), img)
            test_paths.append(str(path))
        
        try:
            # Process batch
            processed = self.preprocessor.batch_preprocess(test_paths, batch_size=2)
            
            # Check results
            self.assertEqual(len(processed), 3)
            for img in processed:
                self.assertEqual(img.shape[2], 3)  # 3 channels
                self.assertEqual(img.dtype, np.uint8)
            
            # Cleanup
            for path in test_paths:
                Path(path).unlink()
                
        except Exception as e:
            # Cleanup on error
            for path in test_paths:
                if Path(path).exists():
                    Path(path).unlink()
            self.fail(f"Batch processing failed: {e}")

class TestThermalAugmentor(unittest.TestCase):
    """Test thermal image augmentation"""
    
    def setUp(self):
        """Setup test data"""
        from data_preprocessing.data_augmentation import ThermalAugmentor
        
        self.augmentor = ThermalAugmentor(mode='standard')
        self.test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        self.test_bboxes = [[50, 50, 150, 150], [100, 100, 200, 200]]
        self.test_labels = ['human', 'vehicle']
    
    def test_augment_image(self):
        """Test single image augmentation"""
        # Test without bboxes
        result = self.augmentor.augment_image(self.test_image)
        
        self.assertIn('image', result)
        self.assertIn('bboxes', result)
        self.assertIn('labels', result)
        
        self.assertEqual(result['image'].shape[2], 3)  # RGB
        self.assertEqual(len(result['bboxes']), 0)
        self.assertEqual(len(result['labels']), 0)
        
        # Test with bboxes
        result = self.augmentor.augment_image(
            self.test_image,
            bboxes=self.test_bboxes,
            labels=self.test_labels
        )
        
        self.assertEqual(len(result['bboxes']), len(self.test_bboxes))
        self.assertEqual(len(result['labels']), len(self.test_labels))
    
    def test_augmentation_types(self):
        """Test different augmentation modes"""
        modes = ['standard', 'aggressive', 'light']
        
        for mode in modes:
            from data_preprocessing.data_augmentation import ThermalAugmentor
            augmentor = ThermalAugmentor(mode=mode)
            
            result = augmentor.augment_image(self.test_image)
            self.assertEqual(result['image'].shape[2], 3)
    
    def test_thermal_noise(self):
        """Test thermal-specific noise addition"""
        # This tests that noise is properly added
        original = self.test_image.copy()
        result = self.augmentor.augment_image(original)
        
        # Images should be different after augmentation
        self.assertFalse(np.array_equal(original, result['image'][:, :, 0]))
    
    def test_augment_batch(self):
        """Test batch augmentation"""
        images = [self.test_image] * 3
        bboxes_list = [self.test_bboxes] * 3
        labels_list = [self.test_labels] * 3
        
        results = self.augmentor.augment_batch(
            images, 
            bboxes_list, 
            labels_list
        )
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('image', result)
            self.assertIn('bboxes', result)
            self.assertIn('labels', result)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
