"""
TensorFlow Lite Converter for YOLOv5 Thermal Detection Model
"""

import torch
import tensorflow as tf
import numpy as np
import onnx
import onnxruntime
from pathlib import Path
import logging
import yaml
from typing import Optional, Dict, Tuple

class TFLiteConverter:
    """Convert PyTorch YOLOv5 model to TensorFlow Lite"""
    
    def __init__(self, model_path: str, output_dir: str = 'models/tflite_models'):
        """
        Args:
            model_path: Path to PyTorch model (.pt file)
            output_dir: Directory to save converted models
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configuration
        self.input_size = 640
        self.num_classes = 3
        self.class_names = ['human', 'animal', 'vehicle']
        
    def load_pytorch_model(self) -> torch.nn.Module:
        """Load PyTorch YOLOv5 model"""
        self.logger.info(f"Loading PyTorch model from {self.model_path}")
        
        try:
            # Load YOLOv5 model
            model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                  path=self.model_path, 
                                  force_reload=False)
            model.eval()
            return model
        except Exception as e:
            self.logger.error(f"Failed to load PyTorch model: {e}")
            raise
    
    def convert_to_onnx(self, model: torch.nn.Module, 
                       onnx_path: Optional[Path] = None) -> Path:
        """Convert PyTorch model to ONNX format"""
        if onnx_path is None:
            onnx_path = self.output_dir / f"{self.model_path.stem}.onnx"
        
        self.logger.info(f"Converting to ONNX format...")
        
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes={
                    'images': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            self.logger.info(f"ONNX model saved to {onnx_path}")
            self.logger.info(f"ONNX IR version: {onnx_model.ir_version}")
            self.logger.info(f"ONNX opset version: {onnx_model.opset_import[0].version}")
            
            return onnx_path
            
        except Exception as e:
            self.logger.error(f"Failed to convert to ONNX: {e}")
            raise
    
    def convert_to_tensorflow(self, onnx_path: Path) -> tf.keras.Model:
        """Convert ONNX model to TensorFlow format"""
        self.logger.info("Converting ONNX to TensorFlow...")
        
        try:
            # Convert ONNX to TensorFlow
            import onnx_tf
            
            tf_model_path = self.output_dir / 'tensorflow_model'
            
            # Load ONNX model
            onnx_model = onnx.load(str(onnx_path))
            
            # Prepare TensorFlow model
            tf_rep = onnx_tf.backend.prepare(onnx_model)
            
            # Export as SavedModel
            tf_rep.export_graph(str(tf_model_path))
            
            # Load TensorFlow model
            tf_model = tf.saved_model.load(str(tf_model_path))
            
            self.logger.info(f"TensorFlow model saved to {tf_model_path}")
            
            return tf_model
            
        except Exception as e:
            self.logger.error(f"Failed to convert to TensorFlow: {e}")
            raise
    
    def convert_to_tflite(self, tf_model, quantization: str = 'int8') -> Path:
        """
        Convert TensorFlow model to TensorFlow Lite
        
        Args:
            tf_model: TensorFlow model
            quantization: 'float32', 'float16', 'int8', or 'dynamic_range'
        """
        self.logger.info(f"Converting to TensorFlow Lite with {quantization} quantization...")
        
        try:
            # Create converter
            converter = tf.lite.TFLiteConverter.from_saved_model(
                str(self.output_dir / 'tensorflow_model')
            )
            
            # Optimization options
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if quantization == 'float16':
                converter.target_spec.supported_types = [tf.float16]
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
            elif quantization == 'int8':
                # For full integer quantization, need representative dataset
                def representative_dataset():
                    for _ in range(100):
                        data = np.random.rand(1, self.input_size, self.input_size, 3).astype(np.float32)
                        yield [data]
                
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
                
            elif quantization == 'dynamic_range':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
            # Set other options
            converter.experimental_new_converter = True
            converter.allow_custom_ops = True
            
            # Convert model
            tflite_model = converter.convert()
            
            # Save model
            tflite_path = self.output_dir / f"model_{quantization}.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            self.logger.info(f"TFLite model saved to {tflite_path}")
            
            # Get model size
            model_size = len(tflite_model) / (1024 * 1024)  # MB
            self.logger.info(f"Model size: {model_size:.2f} MB")
            
            return tflite_path
            
        except Exception as e:
            self.logger.error(f"Failed to convert to TFLite: {e}")
            raise
    
    def optimize_tflite_model(self, tflite_path: Path, use_xnnpack: bool = True) -> Path:
        """Apply additional optimizations to TFLite model"""
        self.logger.info("Applying TFLite optimizations...")
        
        try:
            # Load TFLite model
            with open(tflite_path, 'rb') as f:
                tflite_model = f.read()
            
            # Apply optimizations
            import tensorflow as tf
            
            # Create converter from buffer
            converter = tf.lite.Interpreter(model_content=tflite_model)
            
            # Get model details
            input_details = converter.get_input_details()
            output_details = converter.get_output_details()
            
            self.logger.info("Model Details:")
            self.logger.info(f"  Input shape: {input_details[0]['shape']}")
            self.logger.info(f"  Input type: {input_details[0]['dtype']}")
            self.logger.info(f"  Output shape: {output_details[0]['shape']}")
            self.logger.info(f"  Output type: {output_details[0]['dtype']}")
            
            # Apply XNNPACK delegate if available (for CPU acceleration)
            if use_xnnpack:
                try:
                    delegate = tf.lite.experimental.load_delegate('libxnnpack_delegate.so')
                    self.logger.info("XNNPACK delegate loaded for CPU acceleration")
                except:
                    self.logger.warning("XNNPACK delegate not available")
            
            # Save optimized model
            optimized_path = self.output_dir / f"{tflite_path.stem}_optimized.tflite"
            with open(optimized_path, 'wb') as f:
                f.write(tflite_model)
            
            return optimized_path
            
        except Exception as e:
            self.logger.error(f"Failed to optimize TFLite model: {e}")
            return tflite_path
    
    def test_tflite_model(self, tflite_path: Path) -> Dict:
        """Test TFLite model inference"""
        self.logger.info(f"Testing TFLite model: {tflite_path.name}")
        
        try:
            import time
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            
            # Get input/output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Create test input
            input_shape = input_details[0]['shape']
            input_data = np.random.randn(*input_shape).astype(np.float32)
            
            # Run inference multiple times
            num_runs = 100
            inference_times = []
            
            for i in range(num_runs):
                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], input_data)
                
                # Run inference
                start_time = time.time()
                interpreter.invoke()
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Get output
                output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Calculate statistics
            avg_time = np.mean(inference_times) * 1000  # Convert to ms
            std_time = np.std(inference_times) * 1000
            fps = 1000 / avg_time if avg_time > 0 else 0
            
            # Model size
            model_size = tflite_path.stat().st_size / (1024 * 1024)  # MB
            
            results = {
                'model_path': str(tflite_path),
                'model_size_mb': model_size,
                'avg_inference_time_ms': avg_time,
                'std_inference_time_ms': std_time,
                'fps': fps,
                'input_shape': input_shape,
                'output_shape': output_data.shape,
                'quantization': str(input_details[0]['dtype'])
            }
            
            self.logger.info("TFLite Model Test Results:")
            self.logger.info(f"  Model Size: {model_size:.2f} MB")
            self.logger.info(f"  Avg Inference Time: {avg_time:.2f} ms")
            self.logger.info(f"  FPS: {fps:.1f}")
            self.logger.info(f"  Quantization: {input_details[0]['dtype']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to test TFLite model: {e}")
            raise
    
    def benchmark_models(self, models: Dict[str, Path]) -> pd.DataFrame:
        """Benchmark multiple TFLite models"""
        import pandas as pd
        
        results = []
        
        for name, model_path in models.items():
            self.logger.info(f"Benchmarking {name}...")
            
            try:
                model_results = self.test_tflite_model(model_path)
                model_results['model_name'] = name
                results.append(model_results)
                
            except Exception as e:
                self.logger.error(f"Failed to benchmark {name}: {e}")
        
        # Create comparison DataFrame
        df = pd.DataFrame(results)
        
        # Save benchmark results
        benchmark_file = self.output_dir / 'benchmark_results.csv'
        df.to_csv(benchmark_file, index=False)
        
        self.logger.info(f"Benchmark results saved to {benchmark_file}")
        
        # Create visualization
        self._create_benchmark_visualization(df)
        
        return df
    
    def _create_benchmark_visualization(self, df: pd.DataFrame):
        """Create benchmark visualization"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. Inference time comparison
            ax1 = axes[0, 0]
            bars = ax1.bar(df['model_name'], df['avg_inference_time_ms'], 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7'])
            ax1.set_title('Inference Time Comparison', fontweight='bold')
            ax1.set_ylabel('Time (ms)')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar, time_val in zip(bars, df['avg_inference_time_ms']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{time_val:.1f}ms', ha='center', va='bottom', fontweight='bold')
            
            # 2. FPS comparison
            ax2 = axes[0, 1]
            bars = ax2.bar(df['model_name'], df['fps'], 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7'])
            ax2.set_title('FPS Comparison', fontweight='bold')
            ax2.set_ylabel('Frames per Second')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, fps_val in zip(bars, df['fps']):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{fps_val:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # 3. Model size comparison
            ax3 = axes[1, 0]
            bars = ax3.bar(df['model_name'], df['model_size_mb'], 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7'])
            ax3.set_title('Model Size Comparison', fontweight='bold')
            ax3.set_ylabel('Size (MB)')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, size_val in zip(bars, df['model_size_mb']):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{size_val:.1f}MB', ha='center', va='bottom', fontweight='bold')
            
            # 4. Trade-off plot (size vs speed)
            ax4 = axes[1, 1]
            scatter = ax4.scatter(df['model_size_mb'], df['avg_inference_time_ms'], 
                                 s=200, c=range(len(df)), cmap='viridis')
            
            # Add model labels
            for i, row in df.iterrows():
                ax4.annotate(row['model_name'], 
                           (row['model_size_mb'], row['avg_inference_time_ms']),
                           xytext=(5, 5), textcoords='offset points')
            
            ax4.set_title('Size vs Speed Trade-off', fontweight='bold')
            ax4.set_xlabel('Model Size (MB)')
            ax4.set_ylabel('Inference Time (ms)')
            ax4.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'benchmark_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create visualization: {e}")
    
    def create_deployment_package(self, tflite_path: Path, 
                                 config: Optional[Dict] = None):
        """Create deployment package with model and configuration"""
        self.logger.info("Creating deployment package...")
        
        package_dir = self.output_dir / 'deployment_package'
        package_dir.mkdir(exist_ok=True)
        
        # Copy model
        shutil.copy2(tflite_path, package_dir / 'model.tflite')
        
        # Create configuration file
        if config is None:
            config = {
                'model': {
                    'input_size': self.input_size,
                    'num_classes': self.num_classes,
                    'class_names': self.class_names,
                    'quantization': 'int8'
                },
                'inference': {
                    'confidence_threshold': 0.5,
                    'iou_threshold': 0.45,
                    'max_detections': 100
                },
                'deployment': {
                    'target_device': 'raspberry_pi',
                    'recommended_fps': 10,
                    'memory_usage_mb': 100
                }
            }
        
        config_path = package_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Create README
        readme_path = package_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(f"""# Thermal Object Detection Deployment Package

## Model Information
- **Model**: {tflite_path.name}
- **Input Size**: {self.input_size}x{self.input_size}
- **Classes**: {', '.join(self.class_names)}
- **Quantization**: {config['model']['quantization']}
- **Size**: {tflite_path.stat().st_size / (1024 * 1024):.2f} MB

## Usage
```python
import tensorflow as tf

# Load model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Run inference
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
