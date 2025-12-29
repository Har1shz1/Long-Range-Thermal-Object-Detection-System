import cv2
import numpy as np
import time
import threading
from queue import Queue
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
import logging
from datetime import datetime

class VideoProcessor:
    """Process video files and streams for object detection"""
    
    def __init__(self, detector, config: Optional[Dict] = None):
        """
        Args:
            detector: Object detector instance
            config: Processing configuration
        """
        self.detector = detector
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Processing state
        self.processing = False
        self.current_frame = 0
        self.total_frames = 0
        
        # Performance tracking
        self.fps_history = []
        self.processing_times = []
        
        # Output queues
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        
    def _default_config(self) -> Dict:
        """Default processing configuration"""
        return {
            'show_video': True,
            'save_output': False,
            'output_format': 'mp4',
            'output_fps': 30,
            'draw_boxes': True,
            'show_confidence': True,
            'show_fps': True,
            'show_class': True,
            'confidence_threshold': 0.5,
            'max_frame_size': (1280, 720),
            'skip_frames': 0,  # Process every nth frame
            'batch_size': 1
        }
    
    def process_video_file(self, video_path: str, 
                          output_path: Optional[str] = None,
                          callback: Optional[Callable] = None) -> Dict:
        """
        Process a video file
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video
            callback: Callback function for progress updates
            
        Returns:
            Dictionary with processing statistics
        """
        self.logger.info(f"Processing video: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return {}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video Properties: {width}x{height}, {fps} FPS, {self.total_frames} frames")
        
        # Setup video writer if saving output
        writer = None
        if output_path and self.config['save_output']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_size = (width, height)
            writer = cv2.VideoWriter(output_path, fourcc, fps, output_size)
        
        # Processing statistics
        stats = {
            'total_frames': self.total_frames,
            'processed_frames': 0,
            'total_detections': 0,
            'processing_time': 0,
            'average_fps': 0,
            'detections_per_frame': [],
            'class_distribution': {},
            'start_time': time.time()
        }
        
        self.processing = True
        self.current_frame = 0
        
        frame_count = 0
        frame_skip = self.config['skip_frames']
        
        try:
            while self.processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if configured
                if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                start_time = time.time()
                
                # Convert to grayscale if thermal image
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Assume thermal image in grayscale (all channels same)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                # Detect objects
                detections = self.detector.detect(frame)
                
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # Update statistics
                stats['processed_frames'] += 1
                stats['total_detections'] += len(detections)
                stats['detections_per_frame'].append(len(detections))
                
                # Update class distribution
                for det in detections:
                    class_name = det.class_name
                    stats['class_distribution'][class_name] = \
                        stats['class_distribution'].get(class_name, 0) + 1
                
                # Draw detections if configured
                if self.config['draw_boxes']:
                    frame = self.detector.draw_detections(
                        frame, 
                        detections,
                        show_confidence=self.config['show_confidence'],
                        show_class=self.config['show_class']
                    )
                
                # Add FPS counter if configured
                if self.config['show_fps']:
                    current_fps = 1.0 / processing_time if processing_time > 0 else 0
                    self.fps_history.append(current_fps)
                    
                    fps_text = f"FPS: {current_fps:.1f}"
                    cv2.putText(
                        frame,
                        fps_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                
                # Add frame counter
                frame_text = f"Frame: {frame_count}/{self.total_frames}"
                cv2.putText(
                    frame,
                    frame_text,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Write frame to output
                if writer:
                    writer.write(frame)
                
                # Display frame if configured
                if self.config['show_video']:
                    # Resize if too large
                    if frame.shape[1] > self.config['max_frame_size'][0] or \
                       frame.shape[0] > self.config['max_frame_size'][1]:
                        scale = min(
                            self.config['max_frame_size'][0] / frame.shape[1],
                            self.config['max_frame_size'][1] / frame.shape[0]
                        )
                        new_width = int(frame.shape[1] * scale)
                        new_height = int(frame.shape[0] * scale)
                        display_frame = cv2.resize(frame, (new_width, new_height))
                    else:
                        display_frame = frame
                    
                    cv2.imshow('Thermal Detection', display_frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("Processing stopped by user")
                        break
                
                # Call progress callback
                if callback and frame_count % 10 == 0:
                    progress = (frame_count / self.total_frames) * 100
                    callback(progress, frame_count, detections)
                
                frame_count += 1
                
                # Log progress
                if frame_count % 100 == 0:
                    avg_fps = frame_count / (time.time() - stats['start_time'])
                    self.logger.info(
                        f"Processed {frame_count}/{self.total_frames} frames "
                        f"({progress:.1f}%), FPS: {avg_fps:.1f}, "
                        f"Detections: {stats['total_detections']}"
                    )
        
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted")
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
        finally:
            # Cleanup
            self.processing = False
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Calculate final statistics
            end_time = time.time()
            stats['processing_time'] = end_time - stats['start_time']
            stats['average_fps'] = stats['processed_frames'] / stats['processing_time'] \
                if stats['processing_time'] > 0 else 0
            
            if self.processing_times:
                stats['avg_processing_time'] = np.mean(self.processing_times)
                stats['std_processing_time'] = np.std(self.processing_times)
            
            if self.fps_history:
                stats['avg_fps'] = np.mean(self.fps_history)
                stats['min_fps'] = np.min(self.fps_history)
                stats['max_fps'] = np.max(self.fps_history)
            
            # Calculate detection statistics
            if stats['detections_per_frame']:
                stats['avg_detections_per_frame'] = np.mean(stats['detections_per_frame'])
                stats['max_detections_per_frame'] = np.max(stats['detections_per_frame'])
            
            self.logger.info("Video processing complete")
            self.logger.info(f"Statistics: {stats}")
            
            return stats
    
    def process_rtsp_stream(self, stream_url: str, 
                           duration: Optional[float] = None,
                           output_path: Optional[str] = None) -> Dict:
        """
        Process RTSP stream
        
        Args:
            stream_url: RTSP stream URL
            duration: Maximum processing duration in seconds
            output_path: Path to save output
            
        Returns:
            Processing statistics
        """
        self.logger.info(f"Processing RTSP stream: {stream_url}")
        
        # Open stream
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            self.logger.error(f"Failed to open RTSP stream: {stream_url}")
            return {}
        
        # Setup writer if needed
        writer = None
        if output_path and self.config['save_output']:
            # Get stream properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        stats = {
            'processed_frames': 0,
            'total_detections': 0,
            'start_time': time.time(),
            'class_distribution': {}
        }
        
        self.processing = True
        start_time = time.time()
        
        try:
            while self.processing:
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    self.logger.info(f"Duration limit reached: {duration} seconds")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from stream")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                detections = self.detector.detect(frame)
                
                # Update statistics
                stats['processed_frames'] += 1
                stats['total_detections'] += len(detections)
                
                for det in detections:
                    class_name = det.class_name
                    stats['class_distribution'][class_name] = \
                        stats['class_distribution'].get(class_name, 0) + 1
                
                # Draw detections
                if self.config['draw_boxes']:
                    frame = self.detector.draw_detections(frame, detections)
                
                # Display
                if self.config['show_video']:
                    cv2.imshow('RTSP Stream - Thermal Detection', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Save frame
                if writer:
                    writer.write(frame)
                
                # Log progress
                if stats['processed_frames'] % 100 == 0:
                    elapsed = time.time() - stats['start_time']
                    fps = stats['processed_frames'] / elapsed
                    self.logger.info(
                        f"Processed {stats['processed_frames']} frames, "
                        f"FPS: {fps:.1f}, Detections: {stats['total_detections']}"
                    )
        
        except KeyboardInterrupt:
            self.logger.info("Stream processing interrupted")
        except Exception as e:
            self.logger.error(f"Error processing stream: {e}")
        finally:
            self.processing = False
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Calculate final statistics
            stats['processing_time'] = time.time() - stats['start_time']
            stats['average_fps'] = stats['processed_frames'] / stats['processing_time'] \
                if stats['processing_time'] > 0 else 0
            
            self.logger.info(f"Stream processing complete")
            self.logger.info(f"Statistics: {stats}")
            
            return stats
    
    def batch_process_videos(self, video_paths: List[str], 
                           output_dir: str,
                           parallel: bool = False) -> Dict:
        """
        Process multiple videos in batch
        
        Args:
            video_paths: List of video file paths
            output_dir: Directory to save outputs
            parallel: Process videos in parallel
            
        Returns:
            Batch processing statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        batch_stats = {
            'total_videos': len(video_paths),
            'processed_videos': 0,
            'failed_videos': 0,
            'total_frames': 0,
            'total_detections': 0,
            'video_results': []
        }
        
        self.logger.info(f"Batch processing {len(video_paths)} videos")
        
        if parallel:
            # Parallel processing
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                for video_path in video_paths:
                    video_name = Path(video_path).stem
                    output_path = output_dir / f"{video_name}_processed.mp4"
                    
                    future = executor.submit(
                        self.process_video_file,
                        video_path,
                        str(output_path),
                        None
                    )
                    futures.append((video_path, future))
                
                for video_path, future in futures:
                    try:
                        result = future.result(timeout=3600)  # 1 hour timeout
                        batch_stats['video_results'].append({
                            'video': video_path,
                            'result': result
                        })
                        batch_stats['processed_videos'] += 1
                        
                        if result:
                            batch_stats['total_frames'] += result.get('processed_frames', 0)
                            batch_stats['total_detections'] += result.get('total_detections', 0)
                        
                        self.logger.info(f"Completed: {video_path}")
                        
                    except Exception as e:
                        batch_stats['failed_videos'] += 1
                        self.logger.error(f"Failed to process {video_path}: {e}")
        
        else:
            # Sequential processing
            for video_path in video_paths:
                try:
                    video_name = Path(video_path).stem
                    output_path = output_dir / f"{video_name}_processed.mp4"
                    
                    self.logger.info(f"Processing: {video_path}")
                    
                    result = self.process_video_file(
                        video_path,
                        str(output_path),
                        None
                    )
                    
                    batch_stats['video_results'].append({
                        'video': video_path,
                        'result': result
                    })
                    batch_stats['processed_videos'] += 1
                    
                    if result:
                        batch_stats['total_frames'] += result.get('processed_frames', 0)
                        batch_stats['total_detections'] += result.get('total_detections', 0)
                    
                    self.logger.info(f"Completed: {video_path}")
                    
                except Exception as e:
                    batch_stats['failed_videos'] += 1
                    self.logger.error(f"Failed to process {video_path}: {e}")
        
        # Generate batch report
        self._generate_batch_report(batch_stats, output_dir)
        
        self.logger.info(f"Batch processing complete")
        self.logger.info(f"Processed: {batch_stats['processed_videos']}/"
                        f"{batch_stats['total_videos']} videos")
        
        return batch_stats
    
    def _generate_batch_report(self, batch_stats: Dict, output_dir: Path):
        """Generate batch processing report"""
        report_file = output_dir / 'batch_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Batch Video Processing Report\n\n")
            
            f.write(f"## Summary\n")
            f.write(f"- **Total Videos**: {batch_stats['total_videos']}\n")
            f.write(f"- **Successfully Processed**: {batch_stats['processed_videos']}\n")
            f.write(f"- **Failed**: {batch_stats['failed_videos']}\n")
            f.write(f"- **Total Frames**: {batch_stats['total_frames']:,}\n")
            f.write(f"- **Total Detections**: {batch_stats['total_detections']:,}\n\n")
            
            f.write("## Per-Video Results\n")
            f.write("| Video | Frames | Detections | FPS | Processing Time |\n")
            f.write("|-------|--------|------------|-----|-----------------|\n")
            
            for video_result in batch_stats['video_results']:
                result = video_result['result']
                if result:
                    video_name = Path(video_result['video']).name
                    frames = result.get('processed_frames', 0)
                    detections = result.get('total_detections', 0)
                    fps = result.get('average_fps', 0)
                    proc_time = result.get('processing_time', 0)
                    
                    f.write(f"| {video_name} | {frames:,} | {detections:,} | "
                           f"{fps:.1f} | {proc_time:.1f}s |\n")
            
            f.write("\n## Class Distribution\n")
            
            # Aggregate class distribution
            class_dist = {}
            for video_result in batch_stats['video_results']:
                result = video_result['result']
                if result and 'class_distribution' in result:
                    for cls, count in result['class_distribution'].items():
                        class_dist[cls] = class_dist.get(cls, 0) + count
            
            for cls, count in sorted(class_dist.items()):
                percentage = (count / batch_stats['total_detections'] * 100) \
                            if batch_stats['total_detections'] > 0 else 0
                f.write(f"- **{cls}**: {count:,} ({percentage:.1f}%)\n")
        
        self.logger.info(f"Batch report saved to {report_file}")

class MultiThreadedProcessor:
    """Multi-threaded video processor for high performance"""
    
    def __init__(self, detector, num_workers: int = 4):
        """
        Args:
            detector: Object detector instance
            num_workers: Number of worker threads
        """
        self.detector = detector
        self.num_workers = num_workers
        
        self.logger = logging.getLogger(__name__)
        
        # Queues for frame processing
        self.input_queue = Queue(maxsize=100)
        self.output_queue = Queue(maxsize=100)
        
        # Worker threads
        self.workers = []
        self.running = False
        
    def start_workers(self):
        """Start worker threads"""
        self.running = True
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started {self.num_workers} worker threads")
    
    def stop_workers(self):
        """Stop worker threads"""
        self.running = False
        
        # Clear queues to unblock workers
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        self.logger.info("Worker threads stopped")
    
    def _worker_loop(self, worker_id: int):
        """Worker thread processing loop"""
        self.logger.debug(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get frame from queue
                frame_data = self.input_queue.get(timeout=1)
                if frame_data is None:
                    continue
                
                frame_id, frame = frame_data
                
                # Process frame
                detections = self.detector.detect(frame)
                
                # Put result in output queue
                self.output_queue.put((frame_id, frame, detections))
                
                self.input_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
    
    def process_video_stream(self, video_source, callback=None):
        """Process video stream with multiple workers"""
        self.start_workers()
        
        # Open video source
        if isinstance(video_source, str):
            cap = cv2.VideoCapture(video_source)
        else:
            cap = video_source
        
        frame_id = 0
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Put frame in input queue
                self.input_queue.put((frame_id, frame))
                frame_id += 1
                
                # Get results from output queue
                while not self.output_queue.empty():
                    result_id, result_frame, detections = self.output_queue.get_nowait()
                    
                    # Call callback with results
                    if callback:
                        callback(result_id, result_frame, detections)
                    
                    self.output_queue.task_done()
                
                # Control frame rate
                time.sleep(0.01)  # 100 FPS max
        
        except KeyboardInterrupt:
            self.logger.info("Stream processing interrupted")
        except Exception as e:
            self.logger.error(f"Stream processing error: {e}")
        finally:
            self.stop_workers()
            cap.release()

if __name__ == "__main__":
    # Example usage
    from src.inference.realtime_inference import ThermalDetector, DetectionConfig
    
    # Initialize detector
    detector_config = DetectionConfig(
        model_path='models/trained_weights/best.pt',
        confidence_threshold=0.5
    )
    detector = ThermalDetector(detector_config)
    
    # Create video processor
    processor_config = {
        'show_video': True,
        'save_output': True,
        'draw_boxes': True,
        'show_fps': True,
        'confidence_threshold': 0.5
    }
    
    processor = VideoProcessor(detector, processor_config)
    
    # Process single video
    stats = processor.process_video_file(
        video_path='input_video.mp4',
        output_path='output_video.mp4'
    )
    
    print(f"Processing complete!")
    print(f"Processed {stats['processed_frames']} frames")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Average FPS: {stats['average_fps']:.1f}")
