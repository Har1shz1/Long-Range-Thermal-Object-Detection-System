"""
Metrics calculation and analysis for thermal object detection
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import json

@dataclass
class DetectionMetrics:
    """Container for detection metrics"""
    precision: float
    recall: float
    f1_score: float
    ap: float  # Average Precision
    tp: int    # True Positives
    fp: int    # False Positives
    fn: int    # False Negatives

class ThermalMetricsCalculator:
    """Calculate comprehensive metrics for thermal object detection"""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all counters"""
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)
        self.scores = defaultdict(list)
        self.matches = defaultdict(list)
        self.class_names = set()
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate area of intersection
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate area of both boxes
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate union area
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, 
               predictions: List[Dict], 
               ground_truth: List[Dict],
               image_id: str = None):
        """
        Update metrics with new predictions and ground truth
        
        Args:
            predictions: List of dicts with keys: bbox, class_name, confidence
            ground_truth: List of dicts with keys: bbox, class_name
            image_id: Optional image identifier
        """
        # Group by class
        pred_by_class = defaultdict(list)
        gt_by_class = defaultdict(list)
        
        for pred in predictions:
            pred_by_class[pred['class_name']].append(pred)
        
        for gt in ground_truth:
            gt_by_class[gt['class_name']].append(gt)
        
        # Process each class
        all_classes = set(list(pred_by_class.keys()) + list(gt_by_class.keys()))
        self.class_names.update(all_classes)
        
        for class_name in all_classes:
            preds = pred_by_class[class_name]
            gts = gt_by_class[class_name]
            
            # Sort predictions by confidence
            preds_sorted = sorted(preds, key=lambda x: x['confidence'], reverse=True)
            
            # Initialize matching arrays
            matched_gt = [False] * len(gts)
            
            # Match predictions to ground truth
            for pred_idx, pred in enumerate(preds_sorted):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gts):
                    if matched_gt[gt_idx]:
                        continue
                    
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Check if match meets IoU threshold
                if best_iou >= self.iou_threshold:
                    self.tp[class_name] += 1
                    matched_gt[best_gt_idx] = True
                    self.scores[class_name].append({
                        'confidence': pred['confidence'],
                        'matched': True
                    })
                else:
                    self.fp[class_name] += 1
                    self.scores[class_name].append({
                        'confidence': pred['confidence'],
                        'matched': False
                    })
            
            # Count false negatives (unmatched ground truth)
            self.fn[class_name] += sum(not matched for matched in matched_gt)
    
    def calculate_precision_recall(self, class_name: str) -> Tuple[List[float], List[float]]:
        """Calculate precision-recall curve for a class"""
        scores = self.scores.get(class_name, [])
        
        if not scores:
            return [1.0], [0.0]
        
        # Sort by confidence
        scores_sorted = sorted(scores, key=lambda x: x['confidence'], reverse=True)
        
        tp_cumulative = 0
        fp_cumulative = 0
        total_tp = self.tp[class_name]
        total_fp = self.fp[class_name]
        
        precision_vals = []
        recall_vals = []
        
        for score in scores_sorted:
            if score['matched']:
                tp_cumulative += 1
            else:
                fp_cumulative += 1
            
            precision = tp_cumulative / (tp_cumulative + fp_cumulative) if (tp_cumulative + fp_cumulative) > 0 else 0
            recall = tp_cumulative / total_tp if total_tp > 0 else 0
            
            precision_vals.append(precision)
            recall_vals.append(recall)
        
        return precision_vals, recall_vals
    
    def calculate_ap(self, precision: List[float], recall: List[float]) -> float:
        """Calculate Average Precision using 11-point interpolation"""
        # 11-point interpolation
        interp_recall = np.linspace(0, 1, 11)
        interp_precision = []
        
        for r in interp_recall:
            # Get maximum precision where recall >= r
            precisions_at_r = [p for p, rec in zip(precision, recall) if rec >= r]
            if precisions_at_r:
                interp_precision.append(max(precisions_at_r))
            else:
                interp_precision.append(0)
        
        return np.mean(interp_precision)
    
    def calculate_class_metrics(self, class_name: str) -> DetectionMetrics:
        """Calculate all metrics for a specific class"""
        tp = self.tp.get(class_name, 0)
        fp = self.fp.get(class_name, 0)
        fn = self.fn.get(class_name, 0)
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate AP
        precision_vals, recall_vals = self.calculate_precision_recall(class_name)
        ap = self.calculate_ap(precision_vals, recall_vals)
        
        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            ap=ap,
            tp=tp,
            fp=fp,
            fn=fn
        )
    
    def calculate_overall_metrics(self) -> Dict[str, float]:
        """Calculate overall metrics across all classes"""
        total_tp = sum(self.tp.values())
        total_fp = sum(self.fp.values())
        total_fn = sum(self.fn.values())
        
        # Macro-averaged metrics
        class_metrics = {}
        aps = []
        
        for class_name in self.class_names:
            metrics = self.calculate_class_metrics(class_name)
            class_metrics[class_name] = metrics
            aps.append(metrics.ap)
        
        # Calculate mAP
        map_score = np.mean(aps) if aps else 0
        
        # Overall precision, recall, F1
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) \
                    if (overall_precision + overall_recall) > 0 else 0
        
        return {
            'mAP': map_score,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'class_metrics': {k: vars(v) for k, v in class_metrics.items()}
        }
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Generate confusion matrix"""
        if not self.class_names:
            return np.zeros((1, 1))
        
        class_list = sorted(self.class_names)
        n_classes = len(class_list)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        # Note: This is simplified - real confusion matrix would track misclassifications
        # For now, diagonal shows TP for each class
        for i, class_name in enumerate(class_list):
            cm[i, i] = self.tp.get(class_name, 0)
        
        return cm
    
    def generate_report(self, output_path: str = None) -> Dict:
        """Generate comprehensive metrics report"""
        overall_metrics = self.calculate_overall_metrics()
        confusion_matrix = self.get_confusion_matrix()
        
        report = {
            'overall_metrics': overall_metrics,
            'confusion_matrix': confusion_matrix.tolist(),
            'class_names': sorted(list(self.class_names)),
            'iou_threshold': self.iou_threshold,
            'summary': {
                'mAP': f"{overall_metrics['mAP']:.3f}",
                'precision': f"{overall_metrics['precision']:.3f}",
                'recall': f"{overall_metrics['recall']:.3f}",
                'f1_score': f"{overall_metrics['f1_score']:.3f}"
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report

class PerformanceAnalyzer:
    """Analyze system performance metrics"""
    
    @staticmethod
    def calculate_inference_stats(inference_times: List[float]) -> Dict:
        """Calculate inference time statistics"""
        if not inference_times:
            return {}
        
        times_array = np.array(inference_times)
        
        return {
            'mean': float(np.mean(times_array)),
            'median': float(np.median(times_array)),
            'std': float(np.std(times_array)),
            'min': float(np.min(times_array)),
            'max': float(np.max(times_array)),
            'p95': float(np.percentile(times_array, 95)),
            'p99': float(np.percentile(times_array, 99)),
            'fps': float(1 / np.mean(times_array)) if np.mean(times_array) > 0 else 0
        }
    
    @staticmethod
    def calculate_memory_stats() -> Dict:
        """Calculate memory usage statistics"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def calculate_power_estimate(inference_time: float, 
                                model_complexity: float = 1.0) -> Dict:
        """Estimate power consumption (simplified)"""
        # Base power consumption estimates (in watts)
        BASE_POWER = {
            'raspberry_pi_4': 3.0,
            'jetson_nano': 5.0,
            'desktop_gpu': 150.0
        }
        
        # Inference power multiplier
        inference_power = {
            'idle': 1.0,
            'inference': 1.5,
            'heavy_processing': 2.0
        }
        
        # Calculate estimated power
        estimated_power = {
            'device': 'raspberry_pi_4',
            'base_power_w': BASE_POWER['raspberry_pi_4'],
            'inference_multiplier': inference_power['inference'],
            'estimated_power_w': BASE_POWER['raspberry_pi_4'] * inference_power['inference'],
            'energy_per_inference_j': (BASE_POWER['raspberry_pi_4'] * 
                                     inference_power['inference'] * inference_time)
        }
        
        return estimated_power

# Example usage
if __name__ == "__main__":
    # Create calculator
    calculator = ThermalMetricsCalculator(iou_threshold=0.5)
    
    # Sample predictions and ground truth
    predictions = [
        {'bbox': [50, 50, 150, 150], 'class_name': 'human', 'confidence': 0.95},
        {'bbox': [200, 100, 250, 180], 'class_name': 'vehicle', 'confidence': 0.87},
        {'bbox': [80, 200, 120, 240], 'class_name': 'animal', 'confidence': 0.76}
    ]
    
    ground_truth = [
        {'bbox': [45, 45, 155, 155], 'class_name': 'human'},
        {'bbox': [195, 95, 255, 185], 'class_name': 'vehicle'},
        {'bbox': [85, 195, 125, 245], 'class_name': 'animal'}
    ]
    
    # Update metrics
    calculator.update(predictions, ground_truth)
    
    # Calculate overall metrics
    metrics = calculator.calculate_overall_metrics()
    
    print("Overall Metrics:")
    print(f"  mAP: {metrics['mAP']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")
    
    # Generate report
    report = calculator.generate_report('metrics_report.json')
    print(f"\nReport saved to metrics_report.json")
