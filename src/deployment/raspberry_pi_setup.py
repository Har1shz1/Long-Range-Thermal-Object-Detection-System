"""
Raspberry Pi Deployment Script for Thermal Detection System
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import subprocess
import signal
import threading

# Raspberry Pi specific imports
try:
    import RPi.GPIO as GPIO
    RASPBERRY_PI = True
except ImportError:
    RASPBERRY_PI = False
    print("Running in simulation mode (not on Raspberry Pi)")

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

class RaspberryPiDeployment:
    """Deploy thermal detection system on Raspberry Pi"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_hardware()
        self.running = False
        
    def load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load deployment configuration"""
        default_config = {
            'model_path': 'models/tflite_models/model.tflite',
            'camera_type': 'flir',  # 'flir', 'pi', 'simulation'
            'resolution': (640, 512),
            'fps': 10,
            'confidence_threshold': 0.5,
            'voice_alerts': True,
            'save_detections': True,
            'output_dir': 'detections',
            'log_level': 'INFO',
            'gpio_led_pin': 17,
            'gpio_button_pin': 27,
            'thermal_range': (-20, 120),
            'power_saving': True,
            'network_mode': 'wifi',  # 'wifi', 'ethernet', 'offline'
            'upload_interval': 300,  # seconds
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def setup_logging(self):
        """Setup logging for Raspberry Pi"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Raspberry Pi deployment initialized")
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
    
    def setup_hardware(self):
        """Setup Raspberry Pi hardware components"""
        if not RASPBERRY_PI:
            self.logger.warning("Not running on Raspberry Pi - hardware setup skipped")
            return
        
        try:
            # Setup GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup LED pin
            GPIO.setup(self.config['gpio_led_pin'], GPIO.OUT)
            GPIO.output(self.config['gpio_led_pin'], GPIO.LOW)
            
            # Setup button pin
            GPIO.setup(self.config['gpio_button_pin'], GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # Add button callback
            GPIO.add_event_detect(
                self.config['gpio_button_pin'],
                GPIO.FALLING,
                callback=self.button_callback,
                bouncetime=300
            )
            
            self.logger.info("GPIO setup complete")
            
        except Exception as e:
            self.logger.error(f"Failed to setup GPIO: {e}")
    
    def setup_camera(self):
        """Setup thermal camera based on configuration"""
        camera_type = self.config['camera_type'].lower()
        
        if camera_type == 'flir':
            return self.setup_flir_camera()
        elif camera_type == 'pi':
            return self.setup_picamera()
        else:
            return self.setup_simulation_camera()
    
    def setup_flir_camera(self):
        """Setup FLIR thermal camera"""
        try:
            # This is a simplified version - actual FLIR setup depends on your camera model
            # For FLIR Lepton on Raspberry Pi
            
            # Check if SPI is enabled
            if not self.check_spi_enabled():
                self.logger.error("SPI not enabled. Enable with 'sudo raspi-config'")
                return None
            
            # Import FLIR specific libraries
            try:
                import spidev
                import numpy as np
                
                # Setup SPI
                spi = spidev.SpiDev()
                spi.open(0, 0)  # SPI bus 0, device 0
                spi.max_speed_hz = 20000000  # 20 MHz
                
                self.logger.info("FLIR camera SPI setup complete")
                
                # Return a camera object with capture method
                class FLIRCamera:
                    def __init__(self, spi_obj, resolution):
                        self.spi = spi_obj
                        self.resolution = resolution
                    
                    def capture_frame(self):
                        # Simplified capture - real implementation varies by camera
                        # Read SPI data
                        data = self.spi.xfer2([0] * (self.resolution[0] * self.resolution[1] * 2))
                        
                        # Convert to numpy array (16-bit temperature data)
                        frame = np.frombuffer(bytes(data), dtype=np.uint16)
                        frame = frame.reshape(self.resolution[::-1])  # Height, Width
                        
                        return frame
                    
                    def close(self):
                        self.spi.close()
                
                return FLIRCamera(spi, self.config['resolution'])
                
            except ImportError as e:
                self.logger.error(f"Failed to import FLIR dependencies: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to setup FLIR camera: {e}")
            return None
    
    def setup_picamera(self):
        """Setup Raspberry Pi camera"""
        if not PICAMERA_AVAILABLE:
            self.logger.error("Picamera2 not available")
            return None
        
        try:
            picam2 = Picamera2()
            
            # Configure camera
            config = picam2.create_still_configuration(
                main={"size": self.config['resolution'], "format": "RGB888"},
                buffer_count=2
            )
            picam2.configure(config)
            
            picam2.start()
            time.sleep(2)  # Allow camera to warm up
            
            self.logger.info("PiCamera setup complete")
            return picam2
            
        except Exception as e:
            self.logger.error(f"Failed to setup PiCamera: {e}")
            return None
    
    def setup_simulation_camera(self):
        """Setup simulation camera for testing"""
        self.logger.info("Using simulation camera")
        
        import numpy as np
        
        class SimulationCamera:
            def __init__(self, resolution):
                self.resolution = resolution
                self.frame_count = 0
            
            def capture_frame(self):
                # Generate simulated thermal image
                height, width = self.resolution
                
                # Create base thermal pattern
                frame = np.random.randn(height, width) * 10 + 25  # ~25°C mean
                
                # Add "hot" objects periodically
                self.frame_count += 1
                if self.frame_count % 30 == 0:  # Every 30 frames
                    # Add simulated human
                    center_y, center_x = np.random.randint(100, height-100), np.random.randint(100, width-100)
                    size = np.random.randint(30, 80)
                    
                    # Create elliptical heat signature
                    y, x = np.ogrid[-size:size+1, -size:size+1]
                    mask = (x*x)/(size*size) + (y*y)/(size*0.7*size*0.7) <= 1
                    
                    y_start = max(0, center_y - size)
                    y_end = min(height, center_y + size + 1)
                    x_start = max(0, center_x - size)
                    x_end = min(width, center_x + size + 1)
                    
                    mask_y_start = size - (center_y - y_start) if center_y - size < 0 else 0
                    mask_y_end = size + (y_end - center_y) if center_y + size + 1 > height else 2*size + 1
                    mask_x_start = size - (center_x - x_start) if center_x - size < 0 else 0
                    mask_x_end = size + (x_end - center_x) if center_x + size + 1 > width else 2*size + 1
                    
                    frame[y_start:y_end, x_start:x_end] += mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end] * 15
                
                return frame
            
            def close(self):
                pass
        
        return SimulationCamera(self.config['resolution'][::-1])  # (width, height) -> (height, width)
    
    def check_spi_enabled(self) -> bool:
        """Check if SPI interface is enabled on Raspberry Pi"""
        if not RASPBERRY_PI:
            return True  # Assume enabled for simulation
        
        try:
            # Check if SPI is enabled in config
            result = subprocess.run(['raspi-config', 'nonint', 'get_spi'], 
                                  capture_output=True, text=True)
            return result.stdout.strip() == '0'
        except:
            # Fallback: check device tree
            return Path('/dev/spidev0.0').exists()
    
    def setup_network(self):
        """Setup network connectivity"""
        network_mode = self.config['network_mode']
        
        if network_mode == 'offline':
            self.logger.info("Running in offline mode")
            return
        
        try:
            # Check WiFi connectivity
            if network_mode == 'wifi':
                result = subprocess.run(['iwgetid', '-r'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    ssid = result.stdout.strip()
                    self.logger.info(f"Connected to WiFi: {ssid}")
                else:
                    self.logger.warning("Not connected to WiFi")
            
            # Check internet connectivity
            result = subprocess.run(['ping', '-c', '1', '8.8.8.8'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("Internet connectivity: OK")
            else:
                self.logger.warning("No internet connectivity")
                
        except Exception as e:
            self.logger.error(f"Network check failed: {e}")
    
    def setup_power_saving(self):
        """Enable power saving features"""
        if not self.config['power_saving'] or not RASPBERRY_PI:
            return
        
        try:
            # Disable HDMI to save power
            subprocess.run(['tvservice', '-o'], check=False)
            
            # Set CPU governor to powersave
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'w') as f:
                f.write('powersave')
            
            # Reduce GPU memory
            with open('/boot/config.txt', 'a') as f:
                f.write('\ngpu_mem=16\n')
            
            self.logger.info("Power saving features enabled")
            
        except Exception as e:
            self.logger.error(f"Failed to setup power saving: {e}")
    
    def setup_voice_alerts(self):
        """Setup voice alert system"""
        if not self.config['voice_alerts']:
            return None
        
        try:
            from src.deployment.voice_alert_system import VoiceAlertSystem
            
            voice_system = VoiceAlertSystem()
            voice_system.initialize()
            
            self.logger.info("Voice alert system initialized")
            return voice_system
            
        except Exception as e:
            self.logger.error(f"Failed to setup voice alerts: {e}")
            return None
    
    def button_callback(self, channel):
        """Handle button press events"""
        self.logger.info(f"Button pressed on GPIO {channel}")
        
        # Toggle system state
        self.running = not self.running
        
        # Blink LED to confirm
        self.blink_led(3, 0.2)
        
        if self.running:
            self.logger.info("System started via button press")
        else:
            self.logger.info("System stopped via button press")
    
    def blink_led(self, times: int, interval: float = 0.5):
        """Blink LED for visual feedback"""
        if not RASPBERRY_PI:
            return
        
        for _ in range(times):
            GPIO.output(self.config['gpio_led_pin'], GPIO.HIGH)
            time.sleep(interval)
            GPIO.output(self.config['gpio_led_pin'], GPIO.LOW)
            time.sleep(interval)
    
    def setup_detection_system(self):
        """Setup the object detection system"""
        try:
            # Import and setup detector
            from src.inference.realtime_inference import ThermalDetector, DetectionConfig
            
            config = DetectionConfig(
                model_path=self.config['model_path'],
                confidence_threshold=self.config['confidence_threshold'],
                device='cpu'  # Raspberry Pi uses CPU
            )
            
            detector = ThermalDetector(config)
            self.logger.info("Detection system initialized")
            
            return detector
            
        except Exception as e:
            self.logger.error(f"Failed to setup detection system: {e}")
            return None
    
    def create_output_directories(self):
        """Create directories for storing outputs"""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (output_dir / 'images').mkdir(exist_ok=True)
        (output_dir / 'videos').mkdir(exist_ok=True)
        (output_dir / 'logs').mkdir(exist_ok=True)
        
        return output_dir
    
    def run_detection_loop(self, detector, camera, voice_system, output_dir):
        """Main detection loop"""
        self.logger.info("Starting detection loop")
        
        frame_count = 0
        detection_count = 0
        last_upload_time = time.time()
        
        # Turn on LED to indicate running
        if RASPBERRY_PI:
            GPIO.output(self.config['gpio_led_pin'], GPIO.HIGH)
        
        try:
            while self.running:
                # Capture frame
                frame_start = time.time()
                thermal_frame = camera.capture_frame()
                
                # Process frame
                if thermal_frame is not None:
                    # Convert to 8-bit for visualization
                    frame_8bit = self.normalize_thermal_frame(thermal_frame)
                    
                    # Detect objects
                    detections = detector.detect(frame_8bit)
                    
                    # Handle detections
                    if detections:
                        detection_count += len(detections)
                        self.handle_detections(detections, voice_system, frame_8bit, output_dir, frame_count)
                    
                    # Save frame if configured
                    if self.config['save_detections'] and frame_count % 30 == 0:
                        self.save_frame(frame_8bit, detections, output_dir, frame_count)
                    
                    # Upload data periodically
                    if time.time() - last_upload_time > self.config['upload_interval']:
                        self.upload_data(output_dir)
                        last_upload_time = time.time()
                    
                    frame_count += 1
                    
                    # Log progress
                    if frame_count % 100 == 0:
                        self.logger.info(
                            f"Processed {frame_count} frames, "
                            f"{detection_count} total detections, "
                            f"FPS: {detector.fps:.1f}"
                        )
                
                # Control frame rate
                frame_time = time.time() - frame_start
                target_frame_time = 1.0 / self.config['fps']
                
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)
        
        except KeyboardInterrupt:
            self.logger.info("Detection loop interrupted")
        except Exception as e:
            self.logger.error(f"Error in detection loop: {e}")
        finally:
            # Turn off LED
            if RASPBERRY_PI:
                GPIO.output(self.config['gpio_led_pin'], GPIO.LOW)
    
    def normalize_thermal_frame(self, frame):
        """Normalize thermal frame to 8-bit range"""
        import cv2
        import numpy as np
        
        # Clip to thermal range
        min_temp, max_temp = self.config['thermal_range']
        frame_clipped = np.clip(frame, min_temp, max_temp)
        
        # Normalize to 0-255
        frame_normalized = cv2.normalize(
            frame_clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        
        return frame_normalized
    
    def handle_detections(self, detections, voice_system, frame, output_dir, frame_count):
        """Handle detected objects"""
        for det in detections:
            self.logger.info(
                f"Detected {det.class_name} with confidence {det.confidence:.2f}"
            )
            
            # Trigger voice alert
            if voice_system and det.confidence > 0.7:
                alert_message = f"{det.class_name} detected with {det.confidence:.0%} confidence"
                voice_system.speak(alert_message)
            
            # Save high-confidence detections
            if det.confidence > 0.8 and self.config['save_detections']:
                self.save_detection_image(frame, det, output_dir, frame_count)
    
    def save_frame(self, frame, detections, output_dir, frame_count):
        """Save frame with detections"""
        import cv2
        from src.inference.realtime_inference import ThermalDetector
        
        # Create temporary detector for visualization
        temp_config = type('Config', (), {
            'model_path': self.config['model_path'],
            'confidence_threshold': self.config['confidence_threshold']
        })()
        
        temp_detector = ThermalDetector(temp_config)
        annotated = temp_detector.draw_detections(frame, detections)
        
        # Save image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / 'images' / f"frame_{timestamp}_{frame_count:06d}.jpg"
        cv2.imwrite(str(filename), annotated)
    
    def save_detection_image(self, frame, detection, output_dir, frame_count):
        """Save cropped detection image"""
        import cv2
        
        x1, y1, x2, y2 = map(int, detection.bbox)
        
        # Crop detection
        cropped = frame[y1:y2, x1:x2]
        
        if cropped.size > 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = output_dir / 'images' / f"detection_{detection.class_name}_{timestamp}_{frame_count}.jpg"
            cv2.imwrite(str(filename), cropped)
    
    def upload_data(self, output_dir):
        """Upload data to cloud/server"""
        if self.config['network_mode'] == 'offline':
            return
        
        self.logger.info("Uploading data...")
        # Implement your upload logic here
        # This could be FTP, SCP, AWS S3, etc.
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up resources...")
        
        if RASPBERRY_PI:
            GPIO.cleanup()
        
        self.running = False
    
    def run(self):
        """Main deployment run method"""
        self.logger.info("Starting Raspberry Pi deployment")
        
        try:
            # Setup components
            self.setup_network()
            self.setup_power_saving()
            
            camera = self.setup_camera()
            if not camera:
                self.logger.error("Failed to setup camera")
                return
            
            voice_system = self.setup_voice_alerts()
            detector = self.setup_detection_system()
            if not detector:
                self.logger.error("Failed to setup detector")
                return
            
            output_dir = self.create_output_directories()
            
            # Blink LED to indicate ready
            self.blink_led(2, 0.3)
            
            # Start detection loop
            self.running = True
            self.run_detection_loop(detector, camera, voice_system, output_dir)
            
        except KeyboardInterrupt:
            self.logger.info("Deployment interrupted by user")
        except Exception as e:
            self.logger.error(f"Deployment error: {e}")
        finally:
            self.cleanup()
            self.logger.info("Deployment stopped")

def signal_handler(sig, frame):
    """Handle termination signals"""
    print("\nReceived termination signal")
    sys.exit(0)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Raspberry Pi Thermal Detection Deployment')
    parser.add_argument('--config', type=str, default='configs/deployment_config.yaml',
                       help='Path to deployment configuration file')
    parser.add_argument('--simulate', action='store_true',
                       help='Run in simulation mode (no hardware required)')
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run deployment
    deployment = RaspberryPiDeployment(args.config)
    deployment.run()

if __name__ == '__main__':
    main()
