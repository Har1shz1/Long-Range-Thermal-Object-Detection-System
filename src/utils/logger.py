"""
Logging configuration for thermal object detection system
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import json

class ThermalLogger:
    """Custom logger for thermal detection system"""
    
    def __init__(self, 
                 name: str = "thermal_detection",
                 log_level: str = "INFO",
                 log_dir: str = "logs",
                 console_output: bool = True,
                 file_output: bool = True):
        """
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to save log files
            console_output: Whether to log to console
            file_output: Whether to log to file
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Add console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(simple_formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler
        if file_output:
            log_file = self.log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
        
        # Store metadata
        self.metadata = {
            'start_time': datetime.now().isoformat(),
            'logger_name': name,
            'log_level': log_level,
            'log_file': str(log_file) if file_output else None
        }
        
        # Log initialization
        self.logger.info(f"Logger initialized: {name}")
        self.logger.info(f"Log level: {log_level}")
        if file_output:
            self.logger.info(f"Log file: {log_file}")
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance"""
        return self.logger
    
    def log_system_info(self):
        """Log system information"""
        import platform
        import psutil
        import torch
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cpu_count': psutil.cpu_count(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            system_info['cuda_version'] = torch.version.cuda
            system_info['gpu_name'] = torch.cuda.get_device_name(0)
            system_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        self.logger.info("System Information:")
        for key, value in system_info.items():
            self.logger.info(f"  {key}: {value}")
        
        return system_info
    
    def log_training_start(self, config: Dict):
        """Log training start information"""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 60)
        
        self.logger.info("Training Configuration:")
        for key, value in config.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"    {sub_key}: {sub_value}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_training_progress(self, epoch: int, total_epochs: int,
                            train_loss: float, val_metrics: Dict = None):
        """Log training progress"""
        progress = (epoch + 1) / total_epochs * 100
        
        message = f"Epoch {epoch+1}/{total_epochs} ({progress:.1f}%) - "
        message += f"Train Loss: {train_loss:.4f}"
        
        if val_metrics:
            message += f" - Val mAP: {val_metrics.get('mAP', 0):.3f}"
            message += f" - Precision: {val_metrics.get('precision', 0):.3f}"
            message += f" - Recall: {val_metrics.get('recall', 0):.3f}"
        
        self.logger.info(message)
    
    def log_training_complete(self, best_metrics: Dict, training_time: float):
        """Log training completion"""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 60)
        
        self.logger.info(f"Training Time: {training_time:.2f} hours")
        self.logger.info(f"Best mAP: {best_metrics.get('mAP', 0):.3f}")
        self.logger.info(f"Best Precision: {best_metrics.get('precision', 0):.3f}")
        self.logger.info(f"Best Recall: {best_metrics.get('recall', 0):.3f}")
        
        if 'class_metrics' in best_metrics:
            self.logger.info("Per-Class Best Metrics:")
            for class_name, metrics in best_metrics['class_metrics'].items():
                self.logger.info(f"  {class_name}: AP={metrics.get('ap', 0):.3f}, "
                               f"P={metrics.get('precision', 0):.3f}, "
                               f"R={metrics.get('recall', 0):.3f}")
    
    def log_inference_start(self, model_info: Dict, input_info: Dict):
        """Log inference start information"""
        self.logger.info("=" * 60)
        self.logger.info("INFERENCE STARTED")
        self.logger.info("=" * 60)
        
        self.logger.info("Model Information:")
        for key, value in model_info.items():
            self.logger.info(f"  {key}: {value}")
        
        self.logger.info("Input Information:")
        for key, value in input_info.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_inference_results(self, frame_count: int, 
                            detections: List[Dict],
                            inference_time: float):
        """Log inference results for a frame"""
        if frame_count % 10 == 0:  # Log every 10th frame
            fps = 1 / inference_time if inference_time > 0 else 0
            
            message = f"Frame {frame_count}: "
            message += f"{len(detections)} detections, "
            message += f"Time: {inference_time*1000:.1f}ms, "
            message += f"FPS: {fps:.1f}"
            
            # Log detection details
            if detections:
                class_counts = {}
                for det in detections:
                    class_name = det.get('class_name', 'unknown')
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                message += " - Classes: " + ", ".join(
                    [f"{k}:{v}" for k, v in class_counts.items()]
                )
            
            self.logger.info(message)
    
    def log_deployment_info(self, hardware_info: Dict, software_info: Dict):
        """Log deployment information"""
        self.logger.info("=" * 60)
        self.logger.info("DEPLOYMENT INFORMATION")
        self.logger.info("=" * 60)
        
        self.logger.info("Hardware Configuration:")
        for key, value in hardware_info.items():
            self.logger.info(f"  {key}: {value}")
        
        self.logger.info("Software Configuration:")
        for key, value in software_info.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        error_message = f"ERROR{' - ' + context if context else ''}: {str(error)}"
        self.logger.error(error_message, exc_info=True)
        
        # Log additional debugging info
        import traceback
        self.logger.debug(f"Error traceback:\n{traceback.format_exc()}")
    
    def log_warning(self, warning: str, context: str = ""):
        """Log warning with context"""
        warning_message = f"WARNING{' - ' + context if context else ''}: {warning}"
        self.logger.warning(warning_message)
    
    def log_performance_metrics(self, metrics: Dict):
        """Log performance metrics"""
        self.logger.info("Performance Metrics:")
        
        if 'inference' in metrics:
            infer = metrics['inference']
            self.logger.info(f"  Inference - Mean: {infer.get('mean', 0):.3f}s, "
                           f"FPS: {infer.get('fps', 0):.1f}, "
                           f"Std: {infer.get('std', 0):.3f}s")
        
        if 'memory' in metrics:
            memory = metrics['memory']
            self.logger.info(f"  Memory - RSS: {memory.get('rss_mb', 0):.1f}MB, "
                           f"VMS: {memory.get('vms_mb', 0):.1f}MB, "
                           f"{memory.get('percent', 0):.1f}%")
        
        if 'power' in metrics:
            power = metrics['power']
            self.logger.info(f"  Power - Estimated: {power.get('estimated_power_w', 0):.1f}W, "
                           f"Per inference: {power.get('energy_per_inference_j', 0):.3f}J")
    
    def save_log_summary(self, output_path: Optional[str] = None):
        """Save log summary to file"""
        if not output_path:
            output_path = self.log_dir / f"{self.name}_summary.json"
        
        summary = {
            'metadata': self.metadata,
            'end_time': datetime.now().isoformat(),
            'duration_minutes': (
                datetime.fromisoformat(datetime.now().isoformat()) - 
                datetime.fromisoformat(self.metadata['start_time'])
            ).total_seconds() / 60
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Log summary saved to {output_path}")
        return summary

# Global logger instance for easy import
_global_logger = None

def setup_global_logger(**kwargs):
    """Setup global logger instance"""
    global _global_logger
    _global_logger = ThermalLogger(**kwargs)
    return _global_logger

def get_logger() -> ThermalLogger:
    """Get global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = ThermalLogger()
    return _global_logger

# Example usage
if __name__ == "__main__":
    # Create logger
    logger = ThermalLogger(
        name="thermal_training",
        log_level="INFO",
        log_dir="training_logs"
    )
    
    # Log system info
    logger.log_system_info()
    
    # Log training start
    config = {
        'model': 'yolov5s',
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.001
    }
    logger.log_training_start(config)
    
    # Log training progress
    for epoch in range(10):
        train_loss = 0.1 * (1 - epoch/10)  # Simulated loss
        val_metrics = {'mAP': 0.5 + epoch*0.05, 'precision': 0.6, 'recall': 0.55}
        logger.log_training_progress(epoch, 10, train_loss, val_metrics)
    
    # Log training complete
    best_metrics = {'mAP': 0.95, 'precision': 0.96, 'recall': 0.94}
    logger.log_training_complete(best_metrics, training_time=2.5)
    
    # Save summary
    summary = logger.save_log_summary()
    print(f"Logging complete. Summary: {summary}")
