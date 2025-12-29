"""
Real-time Thermal Object Detection with YOLOv5
"""

import cv2
import numpy as np
import torch
import time
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from queue import Queue
from threading import Thread
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DetectionConfig:
    """Configuration for real-time detection"""
    model_path: str = 'models/trained_weights/best.pt'
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size: int = 640
    max_detections: int = 100
    classes: Optional[List[int]] = None
    agnostic_nms: bool = False

@dataclass
class DetectionResult:
    """Container for detection results"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    thermal_intensity: float = 0.0

class ThermalDetector:
    """Real-time thermal object detector"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.device = torch.device(self.config.device)
        self.model = self.load_model()
        
        # Warm up model
        self.warmup()
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Class names (adjust based on your dataset)
        self.class_names = ['human', 'animal', 'vehicle']
        
        # Color map for classes
        self.colors = {
            'human': (0, 255, 0),    # Green
            'animal': (255, 165, 0), # Orange
            'vehicle': (0, 0, 255)   # Red
        }
    
    def load_model(self) -> torch.nn.Module:
        """Load YOLOv5 model"""
        self.logger.info(f"Loading model from {self.config.model_path}")
        
        try:
            # Load model
            model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                  path=self.config.model_path, 
                                  force_reload=False)
            
            # Configure model
            model.conf = self.config.confidence_threshold
            model.iou = self.config.iou_threshold
            model.max_det = self.config.max_detections
            model.classes = self.config.classes
            model.agnostic = self.config.agnostic_nms
            
            # Move to device
            model.to(self.device)
            
            # Set to evaluation mode
            model.eval()
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def warmup(self, warmup_iterations: int = 10):
        """Warm up model with dummy data"""
        self.logger.info("Warming up model...")
        
        dummy_input = torch.randn(1, 3, self.config.image_size, 
                                 self.config.image_size).to(self.device)
        
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = self.model(dummy_input)
        
        self.logger.info("Model warmup complete")
    
    def preprocess_thermal(self, image: np.ndarray) -> np.ndarray:
        """Preprocess thermal image for YOLO"""
        # Convert to 3 channels if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize to model input size
        original_h, original_w = image.shape[:2]
        
        # Calculate scale factors
        scale = min(self.config.image_size / original_w, 
                   self.config.image_size / original_h)
        
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), 
                            interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((self.config.image_size, self.config.image_size, 3), 
                        114, dtype=np.uint8)  # 114 is padding value
        
        # Calculate padding
        dx = (self.config.image_size - new_w) // 2
        dy = (self.config.image_size - new_h) // 2
        
        # Place resized image in center
        padded[dy:dy+new_h, dx:dx+new_w] = resized
        
        # Normalize
        padded = padded.astype(np.float32) / 255.0
        
        # Convert to torch tensor
        tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device), (dx, dy, scale, original_w, original_h)
    
    def postprocess_detections(self, predictions: torch.Tensor, 
                              meta_info: Tuple) -> List[DetectionResult]:
        """Postprocess model predictions"""
        dx, dy, scale, original_w, original_h = meta_info
        
        detections = []
        
        if predictions is not None and len(predictions):
            for pred in predictions[0]:
                # Extract prediction components
                x1, y1, x2, y2, conf, cls = pred[:6].cpu().numpy()
                
                # Skip if confidence below threshold
                if conf < self.config.confidence_threshold:
                    continue
                
                # Convert from padded coordinates to original image coordinates
                x1 = (x1 - dx) / scale
                y1 = (y1 - dy) / scale
                x2 = (x2 - dx) / scale
                y2 = (y2 - dy) / scale
                
                # Clip to image boundaries
                x1 = max(0, min(x1, original_w))
                y1 = max(0, min(y1, original_h))
                x2 = max(0, min(x2, original_w))
                y2 = max(0, min(y2, original_h))
                
                # Get class name
                class_id = int(cls)
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
                
                # Create detection result
                detection = DetectionResult(
                    bbox=[x1, y1, x2, y2],
                    confidence=float(conf),
                    class_id=class_id,
                    class_name=class_name
                )
                
                detections.append(detection)
        
        return detections
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect objects in thermal image"""
        # Preprocess
        tensor, meta_info = self.preprocess_thermal(image)
        
        # Inference
        with torch.no_grad():
            start_time = time.time()
            predictions = self.model(tensor)
            inference_time = time.time() - start_time
        
        # Postprocess
        detections = self.postprocess_detections(predictions.xyxy, meta_info)
        
        # Update FPS
        self.update_fps(inference_time)
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[DetectionResult]]:
        """Detect objects in batch of images"""
        batch_tensors = []
        batch_meta = []
        
        # Preprocess all images
        for img in images:
            tensor, meta = self.preprocess_thermal(img)
            batch_tensors.append(tensor)
            batch_meta.append(meta)
        
        # Stack batch
        batch = torch.cat(batch_tensors, dim=0)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(batch)
        
        # Postprocess each image
        all_detections = []
        for i, pred in enumerate(predictions.xyxy):
            detections = self.postprocess_detections(pred, batch_meta[i])
            all_detections.append(detections)
        
        return all_detections
    
    def update_fps(self, inference_time: float):
        """Update FPS calculation"""
        self.frame_count += 1
        
        # Calculate average FPS over last 30 frames
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
    
    def draw_detections(self, image: np.ndarray, 
                       detections: List[DetectionResult],
                       show_confidence: bool = True,
                       show_class: bool = True) -> np.ndarray:
        """Draw detection results on image"""
        result = image.copy()
        
        # Convert to color if grayscale
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            class_name = det.class_name
            confidence = det.confidence
            
            # Get color for class
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name}"
            if show_confidence:
                label += f" {confidence:.2f}"
            
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(
                result,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                result,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        # Draw FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            result,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Draw detection count
        count_text = f"Detections: {len(detections)}"
        cv2.putText(
            result,
            count_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return result
    
    def process_video(self, video_path: str, 
                     output_path: Optional[str] = None,
                     show_video: bool = True):
        """Process video file for object detection"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        
        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Resolution: {width}x{height}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = self.detect(frame)
            total_detections += len(detections)
            
            # Draw detections
            annotated = self.draw_detections(frame, detections)
            
            # Write to output
            if writer:
                writer.write(annotated)
            
            # Show video
            if show_video:
                cv2.imshow('Thermal Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Log progress
            if frame_count % 30 == 0:
                self.logger.info(
                    f"Processed {frame_count} frames, "
                    f"Average FPS: {self.fps:.1f}"
                )
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        self.logger.info(f"Processing complete. Total frames: {frame_count}")
        self.logger.info(f"Total detections: {total_detections}")
        self.logger.info(f"Average FPS: {self.fps:.1f}")

class AsyncDetector:
    """Asynchronous detector for real-time applications"""
    
    def __init__(self, detector: ThermalDetector, max_queue_size: int = 10):
        self.detector = detector
        self.input_queue = Queue(maxsize=max_queue_size)
        self.output_queue = Queue(maxsize=max_queue_size)
        self.running = False
        self.thread = None
    
    def start(self):
        """Start asynchronous detection thread"""
        self.running = True
        self.thread = Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop detection thread"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _detection_loop(self):
        """Main detection loop"""
        while self.running:
            try:
                # Get image from queue (non-blocking with timeout)
                image = self.input_queue.get(timeout=0.1)
                
                # Detect objects
                detections = self.detector.detect(image)
                
                # Put results in output queue
                self.output_queue.put(detections, timeout=0.1)
                
            except Exception as e:
                # Queue empty or other error
                continue
    
    def detect_async(self, image: np.ndarray):
        """Submit image for asynchronous detection"""
        self.input_queue.put(image, timeout=0.1)
    
    def get_results(self, timeout: float = 0.1) -> Optional[List[DetectionResult]]:
        """Get detection results if available"""
        try:
            return self.output_queue.get(timeout=timeout)
        except:
            return None

def main():
    """Main function for demonstration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Thermal Object Detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or video path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for results')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Setup detector
    config = DetectionConfig(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    detector = ThermalDetector(config)
    
    # Check if input is image or video
    if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Image detection
        image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image: {args.input}")
            return
        
        detections = detector.detect(image)
        annotated = detector.draw_detections(image, detections)
        
        if args.output:
            cv2.imwrite(args.output, annotated)
            print(f"Results saved to: {args.output}")
        
        cv2.imshow('Detection Results', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Print detections
        for det in detections:
            print(f"{det.class_name}: {det.confidence:.2f} at {det.bbox}")
    
    else:
        # Video detection
        detector.process_video(args.input, args.output)

if __name__ == '__main__':
    main()
