"""
Visualization utilities for thermal object detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class ThermalVisualizer:
    """Visualization tools for thermal detection results"""
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['human', 'animal', 'vehicle']
        self.colors = {
            'human': (0, 255, 0),    # Green
            'animal': (255, 165, 0), # Orange
            'vehicle': (0, 0, 255),  # Red
            'default': (255, 255, 0) # Yellow
        }
    
    def plot_thermal_image(self, image: np.ndarray, 
                          title: str = "Thermal Image",
                          cmap: str = 'hot',
                          figsize: Tuple[int, int] = (10, 8)):
        """Plot thermal image with temperature scale"""
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar with temperature scale
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Temperature Intensity', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_detections(self, image: np.ndarray, 
                       detections: List[Dict],
                       show_confidence: bool = True,
                       show_class: bool = True,
                       figsize: Tuple[int, int] = (12, 8)):
        """Plot image with detection bounding boxes"""
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            vis_image = image.copy()
        
        # Draw each detection
        for det in detections:
            bbox = det.get('bbox', [])
            class_name = det.get('class_name', 'unknown')
            confidence = det.get('confidence', 0.0)
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                color = self.colors.get(class_name, self.colors['default'])
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Create label
                label = class_name
                if show_confidence:
                    label += f" {confidence:.1%}"
                
                # Draw label background
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                cv2.rectangle(
                    vis_image,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    color,
                    -1  # Filled
                )
                
                # Draw label text
                cv2.putText(
                    vis_image,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(vis_image)
        ax.set_title(f"Detections: {len(detections)} objects", 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        return fig, vis_image
    
    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict],
                               metric_name: str = 'mAP'):
        """Plot comparison of metrics across different models/runs"""
        models = list(metrics_dict.keys())
        metrics = [metrics_dict[m].get(metric_name, 0) for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, metrics, color=sns.color_palette("viridis", len(models)))
        
        ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, metrics):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                            normalize: bool = True,
                            figsize: Tuple[int, int] = (8, 6)):
        """Plot confusion matrix"""
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.class_names,
               yticklabels=self.class_names,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        
        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, precision: List[float], 
                                   recall: List[float],
                                   ap: float = None,
                                   figsize: Tuple[int, int] = (8, 6)):
        """Plot precision-recall curve"""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(recall, precision, 'b-', linewidth=2)
        ax.fill_between(recall, precision, alpha=0.2)
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        
        if ap is not None:
            title = f'Precision-Recall Curve (AP = {ap:.3f})'
        else:
            title = 'Precision-Recall Curve'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_inference_timeline(self, timestamps: List[float],
                               inference_times: List[float],
                               detections_per_frame: List[int] = None,
                               figsize: Tuple[int, int] = (12, 8)):
        """Plot inference performance over time"""
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Inference time
        ax1 = axes[0]
        ax1.plot(timestamps, inference_times, 'b-', linewidth=2, alpha=0.7)
        ax1.axhline(y=np.mean(inference_times), color='r', linestyle='--',
                   label=f'Mean: {np.mean(inference_times):.3f}s')
        ax1.set_title('Inference Time Over Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xlabel('Frame')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Detections per frame
        if detections_per_frame:
            ax2 = axes[1]
            ax2.plot(timestamps, detections_per_frame, 'g-', linewidth=2, alpha=0.7)
            ax2.axhline(y=np.mean(detections_per_frame), color='r', linestyle='--',
                       label=f'Mean: {np.mean(detections_per_frame):.1f}')
            ax2.set_title('Detections Per Frame', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Number of Detections')
            ax2.set_xlabel('Frame')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self, metrics: Dict, 
                                    confusion_matrix: np.ndarray):
        """Create interactive Plotly dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Metrics', 'Confusion Matrix',
                           'Per-Class Performance', 'Precision-Recall'),
            specs=[[{'type': 'bar'}, {'type': 'heatmap'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # 1. Overall metrics
        overall_metrics = ['Precision', 'Recall', 'F1-Score', 'mAP']
        values = [metrics.get('precision', 0), metrics.get('recall', 0),
                 metrics.get('f1_score', 0), metrics.get('mAP', 0)]
        
        fig.add_trace(
            go.Bar(x=overall_metrics, y=values, 
                  marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7']),
            row=1, col=1
        )
        
        # 2. Confusion matrix
        fig.add_trace(
            go.Heatmap(z=confusion_matrix,
                      x=self.class_names,
                      y=self.class_names,
                      colorscale='Blues',
                      showscale=True),
            row=1, col=2
        )
        
        # 3. Per-class metrics
        class_metrics = metrics.get('class_metrics', {})
        if class_metrics:
            precision_vals = [class_metrics[c].get('precision', 0) for c in self.class_names]
            recall_vals = [class_metrics[c].get('recall', 0) for c in self.class_names]
            
            fig.add_trace(
                go.Bar(name='Precision', x=self.class_names, y=precision_vals,
                      marker_color='#FF6B6B'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(name='Recall', x=self.class_names, y=recall_vals,
                      marker_color='#4ECDC4'),
                row=2, col=1
            )
        
        # 4. Precision-Recall curve (placeholder)
        fig.add_trace(
            go.Scatter(x=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                      y=[1.0, 0.95, 0.9, 0.85, 0.8, 0.75],
                      mode='lines',
                      name='PR Curve'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True,
                         title_text="Model Performance Dashboard")
        
        return fig
    
    def save_visualization(self, fig, filename: str, 
                          formats: List[str] = ['png', 'pdf']):
        """Save visualization in multiple formats"""
        for fmt in formats:
            save_path = f"{filename}.{fmt}"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
    
    def create_thermal_heatmap_video(self, frames: List[np.ndarray],
                                    output_path: str,
                                    fps: int = 10):
        """Create video visualization of thermal heatmaps"""
        if not frames:
            return
        
        # Get dimensions from first frame
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Normalize to 0-255
            frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply colormap
            frame_color = cv2.applyColorMap(frame_norm.astype(np.uint8), cv2.COLORMAP_HOT)
            
            # Add frame number
            cv2.putText(frame_color, f"Frame: {frames.index(frame)}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame_color)
        
        out.release()
        print(f"Thermal video saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Create sample data
    visualizer = ThermalVisualizer()
    
    # Sample thermal image
    sample_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    # Sample detections
    detections = [
        {'bbox': [50, 50, 150, 150], 'class_name': 'human', 'confidence': 0.95},
        {'bbox': [200, 100, 250, 180], 'class_name': 'vehicle', 'confidence': 0.87},
        {'bbox': [80, 200, 120, 240], 'class_name': 'animal', 'confidence': 0.76}
    ]
    
    # Plot thermal image
    fig1 = visualizer.plot_thermal_image(sample_image, "Sample Thermal Image")
    plt.show()
    
    # Plot detections
    fig2, _ = visualizer.plot_detections(sample_image, detections)
    plt.show()
