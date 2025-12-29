"""
Dataset Splitting and Management for Thermal Object Detection
"""

import os
import random
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

class ThermalDatasetSplitter:
    """Split and manage thermal object detection dataset"""
    
    def __init__(self, base_dir: str, seed: int = 42):
        """
        Args:
            base_dir: Base directory containing dataset
            seed: Random seed for reproducibility
        """
        self.base_dir = Path(base_dir)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        self.logger = logging.getLogger(__name__)
        
        # Directory structure
        self.raw_dir = self.base_dir / 'raw_thermal'
        self.processed_dir = self.base_dir / 'processed'
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def discover_dataset(self) -> Dict:
        """Discover images and annotations in dataset"""
        dataset_info = {
            'images': [],
            'annotations': [],
            'class_distribution': {},
            'image_sizes': []
        }
        
        # Look for images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(self.raw_dir.rglob(f'*{ext}')))
            image_files.extend(list(self.raw_dir.rglob(f'*{ext.upper()}')))
        
        dataset_info['images'] = sorted(image_files)
        
        # Look for annotations (PASCAL VOC format)
        annotation_files = list(self.raw_dir.rglob('*.xml'))
        dataset_info['annotations'] = sorted(annotation_files)
        
        # Parse annotations to get class distribution
        class_counts = {}
        total_objects = 0
        
        for ann_file in annotation_files:
            try:
                classes = self._parse_voc_annotation(ann_file)
                for cls in classes:
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                    total_objects += 1
            except Exception as e:
                self.logger.warning(f"Failed to parse {ann_file}: {e}")
        
        # Calculate percentages
        for cls, count in class_counts.items():
            dataset_info['class_distribution'][cls] = {
                'count': count,
                'percentage': (count / total_objects * 100) if total_objects > 0 else 0
            }
        
        self.logger.info(f"Found {len(dataset_info['images'])} images")
        self.logger.info(f"Found {len(dataset_info['annotations'])} annotations")
        self.logger.info(f"Class distribution: {dataset_info['class_distribution']}")
        
        return dataset_info
    
    def _parse_voc_annotation(self, xml_path: Path) -> List[str]:
        """Parse PASCAL VOC XML annotation file"""
        import xml.etree.ElementTree as ET
        
        classes = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            classes.append(class_name)
        
        return classes
    
    def split_dataset(self, dataset_info: Dict, 
                     split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                     stratify: bool = True) -> Dict:
        """
        Split dataset into train/val/test sets
        
        Args:
            dataset_info: Dataset information dictionary
            split_ratios: (train, val, test) ratios
            stratify: Whether to stratify by class
            
        Returns:
            Dictionary with split information
        """
        train_ratio, val_ratio, test_ratio = split_ratios
        
        # Get image-annotation pairs
        image_annotation_pairs = []
        
        for img_path in dataset_info['images']:
            # Find corresponding annotation
            ann_path = img_path.with_suffix('.xml')
            if not ann_path.exists():
                # Try other naming conventions
                ann_path = img_path.parent / f"{img_path.stem}.xml"
            
            if ann_path.exists():
                image_annotation_pairs.append((img_path, ann_path))
            else:
                self.logger.warning(f"No annotation found for {img_path}")
        
        self.logger.info(f"Found {len(image_annotation_pairs)} valid image-annotation pairs")
        
        if stratify:
            # Get class labels for stratification
            labels = []
            for _, ann_path in image_annotation_pairs:
                try:
                    classes = self._parse_voc_annotation(ann_path)
                    # Use most frequent class in image for stratification
                    if classes:
                        from collections import Counter
                        most_common = Counter(classes).most_common(1)[0][0]
                        labels.append(most_common)
                    else:
                        labels.append('unknown')
                except:
                    labels.append('unknown')
            
            # First split: train vs (val + test)
            train_idx, temp_idx = train_test_split(
                range(len(image_annotation_pairs)),
                test_size=(val_ratio + test_ratio),
                stratify=labels,
                random_state=self.seed
            )
            
            # Adjust split for val/test
            adjusted_val_ratio = val_ratio / (val_ratio + test_ratio)
            
            # Second split: val vs test
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=(1 - adjusted_val_ratio),
                stratify=[labels[i] for i in temp_idx],
                random_state=self.seed
            )
        else:
            # Random split without stratification
            indices = list(range(len(image_annotation_pairs)))
            random.shuffle(indices)
            
            train_size = int(len(indices) * train_ratio)
            val_size = int(len(indices) * val_ratio)
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]
        
        # Organize splits
        splits = {
            'train': [image_annotation_pairs[i] for i in train_idx],
            'val': [image_annotation_pairs[i] for i in val_idx],
            'test': [image_annotation_pairs[i] for i in test_idx]
        }
        
        # Log statistics
        self.logger.info(f"Train set: {len(splits['train'])} images")
        self.logger.info(f"Validation set: {len(splits['val'])} images")
        self.logger.info(f"Test set: {len(splits['test'])} images")
        
        # Calculate split distribution
        split_distribution = self._calculate_split_distribution(splits)
        self.logger.info(f"Split distribution: {split_distribution}")
        
        return splits
    
    def _calculate_split_distribution(self, splits: Dict) -> Dict:
        """Calculate class distribution in each split"""
        distribution = {}
        
        for split_name, pairs in splits.items():
            class_counts = {}
            total_objects = 0
            
            for _, ann_path in pairs:
                try:
                    classes = self._parse_voc_annotation(ann_path)
                    for cls in classes:
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                        total_objects += 1
                except:
                    continue
            
            distribution[split_name] = {
                'total_images': len(pairs),
                'total_objects': total_objects,
                'class_counts': class_counts
            }
        
        return distribution
    
    def organize_splits(self, splits: Dict, output_format: str = 'yolo'):
        """
        Organize dataset splits in required format
        
        Args:
            splits: Dictionary with train/val/test splits
            output_format: 'yolo' or 'pascal_voc'
        """
        # Create directory structure
        for split in ['train', 'val', 'test']:
            split_dir = self.processed_dir / split
            split_dir.mkdir(exist_ok=True)
            
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            images_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)
        
        # Process each split
        for split_name, pairs in splits.items():
            self.logger.info(f"Processing {split_name} split...")
            
            split_stats = {
                'images_copied': 0,
                'labels_converted': 0,
                'classes_found': set()
            }
            
            for img_idx, (img_path, ann_path) in enumerate(pairs):
                try:
                    # Copy image
                    dest_img_path = self.processed_dir / split_name / 'images' / img_path.name
                    shutil.copy2(img_path, dest_img_path)
                    
                    # Convert annotation
                    if output_format == 'yolo':
                        self._convert_to_yolo(ann_path, dest_img_path, 
                                            self.processed_dir / split_name / 'labels')
                    elif output_format == 'pascal_voc':
                        dest_ann_path = self.processed_dir / split_name / 'labels' / ann_path.name
                        shutil.copy2(ann_path, dest_ann_path)
                    
                    split_stats['images_copied'] += 1
                    
                    # Track classes
                    classes = self._parse_voc_annotation(ann_path)
                    split_stats['classes_found'].update(classes)
                    
                    if (img_idx + 1) % 100 == 0:
                        self.logger.info(f"  Processed {img_idx+1}/{len(pairs)} images")
                        
                except Exception as e:
                    self.logger.error(f"Failed to process {img_path}: {e}")
            
            self.logger.info(f"{split_name} split complete:")
            self.logger.info(f"  Images: {split_stats['images_copied']}")
            self.logger.info(f"  Classes found: {sorted(split_stats['classes_found'])}")
        
        # Create dataset YAML file for YOLO
        self._create_dataset_yaml(list(split_stats['classes_found']))
    
    def _convert_to_yolo(self, xml_path: Path, img_path: Path, output_dir: Path):
        """Convert PASCAL VOC annotation to YOLO format"""
        import xml.etree.ElementTree as ET
        
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size_elem = root.find('size')
        img_width = int(size_elem.find('width').text)
        img_height = int(size_elem.find('height').text)
        
        # Class mapping (should be configured)
        class_mapping = {
            'human': 0,
            'animal': 1,
            'vehicle': 2
        }
        
        # Prepare YOLO format lines
        yolo_lines = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_mapping:
                continue
                
            class_id = class_mapping[class_name]
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (normalized center x, center y, width, height)
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Write YOLO annotation file
        if yolo_lines:
            output_path = output_dir / f"{img_path.stem}.txt"
            with open(output_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
    
    def _create_dataset_yaml(self, classes: List[str]):
        """Create dataset YAML file for YOLO training"""
        yaml_content = {
            'path': str(self.processed_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(classes),
            'names': classes
        }
        
        yaml_path = self.processed_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        self.logger.info(f"Dataset YAML created at {yaml_path}")
    
    def analyze_balance(self, splits: Dict) -> pd.DataFrame:
        """Analyze dataset balance and suggest augmentation strategies"""
        analysis = []
        
        for split_name, pairs in splits.items():
            class_counts = {}
            
            for _, ann_path in pairs:
                try:
                    classes = self._parse_voc_annotation(ann_path)
                    for cls in classes:
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                except:
                    continue
            
            total = sum(class_counts.values())
            
            for cls, count in class_counts.items():
                analysis.append({
                    'split': split_name,
                    'class': cls,
                    'count': count,
                    'percentage': (count / total * 100) if total > 0 else 0
                })
        
        df = pd.DataFrame(analysis)
        
        # Identify imbalances
        pivot_df = df.pivot_table(index='class', columns='split', values='count', fill_value=0)
        
        self.logger.info("\nDataset Balance Analysis:")
        self.logger.info(pivot_df.to_string())
        
        return df

if __name__ == "__main__":
    # Example usage
    splitter = ThermalDatasetSplitter(base_dir='data')
    
    # Discover dataset
    dataset_info = splitter.discover_dataset()
    
    # Split dataset
    splits = splitter.split_dataset(
        dataset_info, 
        split_ratios=(0.7, 0.2, 0.1),
        stratify=True
    )
    
    # Organize splits
    splitter.organize_splits(splits, output_format='yolo')
    
    # Analyze balance
    analysis_df = splitter.analyze_balance(splits)
