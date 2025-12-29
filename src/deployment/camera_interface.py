"""
Thermal Camera Interface for Raspberry Pi
Handles communication with FLIR and other thermal cameras
"""

import time
import threading
import queue
from enum import Enum
from typing import Optional, Dict, List, Tuple, Callable
import logging
import numpy as np
from dataclasses import dataclass

# Try to import Raspberry Pi specific libraries
try:
    import RPi.GPIO as GPIO
    import spidev
    import smbus2
    RASPBERRY_PI = True
except ImportError:
    RASPBERRY_PI = False
    print("Running in simulation mode (not on Raspberry Pi)")

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

class CameraType(Enum):
    """Supported camera types"""
    FLIR_LEPTON = "flir_lepton"
    RPI_CAMERA = "rpi_camera"
    SIMULATION = "simulation"
    USB_CAMERA = "usb_camera"

@dataclass
class CameraConfig:
    """Camera configuration"""
    camera_type: CameraType
    resolution: Tuple[int, int] = (640, 512)
    fps: int = 10
    spi_bus: int = 0
    spi_device: int = 0
    i2c_bus: int = 1
    i2c_address: int = 0x2A
    thermal_range: Tuple[float, float] = (-20.0, 120.0)  # Celsius
    auto_gain: bool = True
    auto_exposure: bool = True

@dataclass
class ThermalFrame:
    """Thermal frame container"""
    data: np.ndarray  # Thermal data (temperature values)
    timestamp: float
    metadata: Dict
    ambient_temp: Optional[float] = None
    shutter_temp: Optional[float] = None
    fpa_temp: Optional[float] = None

class ThermalCamera:
    """Base class for thermal cameras"""
    
    def __init__(self, config: CameraConfig):
        """
        Args:
            config: Camera configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Frame buffer
        self.frame_buffer = queue.Queue(maxsize=10)
        self.latest_frame = None
        
        # Camera state
        self.initialized = False
        self.capturing = False
        self.capture_thread = None
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
    def initialize(self) -> bool:
        """Initialize camera"""
        raise NotImplementedError
    
    def capture_frame(self) -> Optional[ThermalFrame]:
        """Capture single frame"""
        raise NotImplementedError
    
    def start_capture(self):
        """Start continuous capture"""
        if self.capturing:
            self.logger.warning("Capture already running")
            return
        
        self.capturing = True
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        self.capture_thread.start()
        self.logger.info("Camera capture started")
    
    def stop_capture(self):
        """Stop continuous capture"""
        self.capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
        self.logger.info("Camera capture stopped")
    
    def _capture_loop(self):
        """Continuous capture loop"""
        self.logger.info("Starting capture loop")
        
        frame_times = []
        
        while self.capturing:
            try:
                start_time = time.time()
                
                # Capture frame
                frame = self.capture_frame()
                if frame is not None:
                    # Update statistics
                    self.frame_count += 1
                    self.latest_frame = frame
                    
                    # Put frame in buffer (non-blocking)
                    try:
                        self.frame_buffer.put(frame, block=False)
                    except queue.Full:
                        # Remove oldest frame
                        try:
                            self.frame_buffer.get_nowait()
                            self.frame_buffer.put(frame, block=False)
                        except:
                            pass
                    
                    # Calculate FPS
                    frame_times.append(start_time)
                    # Keep only last 30 frames for FPS calculation
                    if len(frame_times) > 30:
                        frame_times.pop(0)
                    
                    if len(frame_times) > 1:
                        time_range = frame_times[-1] - frame_times[0]
                        self.fps = len(frame_times) / time_range if time_range > 0 else 0
                
                # Control frame rate
                elapsed = time.time() - start_time
                target_time = 1.0 / self.config.fps
                
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
    
    def get_frame(self, timeout: float = 1.0) -> Optional[ThermalFrame]:
        """Get frame from buffer"""
        try:
            return self.frame_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_latest_frame(self) -> Optional[ThermalFrame]:
        """Get latest frame"""
        return self.latest_frame
    
    def get_statistics(self) -> Dict:
        """Get camera statistics"""
        elapsed = time.time() - self.start_time
        return {
            'total_frames': self.frame_count,
            'fps': self.fps,
            'uptime': elapsed,
            'buffer_size': self.frame_buffer.qsize(),
            'camera_type': self.config.camera_type.value,
            'resolution': self.config.resolution
        }

class FLIRLeptonCamera(ThermalCamera):
    """FLIR Lepton thermal camera implementation"""
    
    def __init__(self, config: CameraConfig):
        super().__init__(config)
        
        if not RASPBERRY_PI:
            self.logger.warning("Not on Raspberry Pi, using simulation mode")
            self.simulation_mode = True
        else:
            self.simulation_mode = False
            self.spi = None
            self.i2c = None
        
        # Lepton specific parameters
        self.lepton_version = 3.5
        self.radiometry_enabled = False
        
    def initialize(self) -> bool:
        """Initialize FLIR Lepton camera"""
        if self.simulation_mode:
            self.logger.info("FLIR Lepton simulation initialized")
            self.initialized = True
            return True
        
        try:
            # Initialize SPI
            self.spi = spidev.SpiDev()
            self.spi.open(self.config.spi_bus, self.config.spi_device)
            self.spi.max_speed_hz = 20000000  # 20 MHz
            self.spi.mode = 0b11  # SPI mode 3
            
            # Initialize I2C for control
            self.i2c = smbus2.SMBus(self.config.i2c_bus)
            
            # Check if Lepton is connected
            if not self._check_lepton():
                self.logger.error("FLIR Lepton not detected")
                return False
            
            # Enable radiometry mode for temperature readings
            self._enable_radiometry()
            
            # Set initial settings
            self._set_gain_mode(self.config.auto_gain)
            self._set_exposure_mode(self.config.auto_exposure)
            
            self.logger.info("FLIR Lepton camera initialized successfully")
            self.initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FLIR Lepton: {e}")
            return False
    
    def _check_lepton(self) -> bool:
        """Check if Lepton is connected"""
        try:
            # Try to read version register
            version = self._read_i2c_register(0x00)
            return version is not None
        except:
            return False
    
    def _read_i2c_register(self, register: int) -> Optional[int]:
        """Read I2C register"""
        try:
            return self.i2c.read_byte_data(self.config.i2c_address, register)
        except:
            return None
    
    def _write_i2c_register(self, register: int, value: int):
        """Write I2C register"""
        try:
            self.i2c.write_byte_data(self.config.i2c_address, register, value)
        except Exception as e:
            self.logger.error(f"Failed to write I2C register: {e}")
    
    def _enable_radiometry(self):
        """Enable radiometry mode for temperature readings"""
        try:
            # Send radiometry enable command
            self._write_i2c_register(0x01, 0x01)
            self.radiometry_enabled = True
            self.logger.info("Radiometry mode enabled")
        except Exception as e:
            self.logger.error(f"Failed to enable radiometry: {e}")
    
    def _set_gain_mode(self, auto: bool):
        """Set gain mode"""
        try:
            mode = 0x01 if auto else 0x00
            self._write_i2c_register(0x02, mode)
            self.logger.info(f"Gain mode set to {'auto' if auto else 'manual'}")
        except Exception as e:
            self.logger.warning(f"Failed to set gain mode: {e}")
    
    def _set_exposure_mode(self, auto: bool):
        """Set exposure mode"""
        try:
            mode = 0x01 if auto else 0x00
            self._write_i2c_register(0x03, mode)
            self.logger.info(f"Exposure mode set to {'auto' if auto else 'manual'}")
        except Exception as e:
            self.logger.warning(f"Failed to set exposure mode: {e}")
    
    def _read_spi_frame(self) -> Optional[np.ndarray]:
        """Read frame from SPI"""
        try:
            width, height = self.config.resolution
            
            # Lepton sends 16-bit words
            num_words = width * height
            data = self.spi.xfer2([0] * (num_words * 2))
            
            # Convert to numpy array
            frame_data = np.frombuffer(bytes(data), dtype=np.uint16)
            frame_data = frame_data.reshape((height, width))
            
            # Apply scaling factor for temperature (Lepton 3.5: 0.01°C per count)
            temperature_data = frame_data.astype(np.float32) * 0.01
            
            return temperature_data
            
        except Exception as e:
            self.logger.error(f"Failed to read SPI frame: {e}")
            return None
    
    def _read_temperatures(self) -> Dict:
        """Read temperature sensors"""
        try:
            temps = {}
            
            # Read ambient temperature
            ambient_raw = self._read_i2c_register(0x10)
            if ambient_raw is not None:
                temps['ambient'] = ambient_raw * 0.1  # Convert to Celsius
            
            # Read shutter temperature
            shutter_raw = self._read_i2c_register(0x11)
            if shutter_raw is not None:
                temps['shutter'] = shutter_raw * 0.1
            
            # Read FPA temperature
            fpa_raw = self._read_i2c_register(0x12)
            if fpa_raw is not None:
                temps['fpa'] = fpa_raw * 0.1
            
            return temps
            
        except:
            return {}
    
    def capture_frame(self) -> Optional[ThermalFrame]:
        """Capture single thermal frame"""
        if not self.initialized:
            self.logger.error("Camera not initialized")
            return None
        
        if self.simulation_mode:
            return self._capture_simulation_frame()
        
        try:
            # Read temperature sensors
            temperatures = self._read_temperatures()
            
            # Read thermal data
            thermal_data = self._read_spi_frame()
            if thermal_data is None:
                return None
            
            # Clip to thermal range
            min_temp, max_temp = self.config.thermal_range
            thermal_data = np.clip(thermal_data, min_temp, max_temp)
            
            # Create frame
            frame = ThermalFrame(
                data=thermal_data,
                timestamp=time.time(),
                metadata={
                    'camera_type': 'flir_lepton',
                    'version': self.lepton_version,
                    'radiometry': self.radiometry_enabled,
                    'resolution': thermal_data.shape
                },
                ambient_temp=temperatures.get('ambient'),
                shutter_temp=temperatures.get('shutter'),
                fpa_temp=temperatures.get('fpa')
            )
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Failed to capture frame: {e}")
            return None
    
    def _capture_simulation_frame(self) -> ThermalFrame:
        """Capture simulation frame (for testing)"""
        width, height = self.config.resolution
        
        # Create simulated thermal data
        base_temp = 25.0  # Ambient temperature
        
        # Add random noise
        thermal_data = np.random.randn(height, width) * 2.0 + base_temp
        
        # Add simulated objects
        if self.frame_count % 30 == 0:  # Add object every 30 frames
            # Simulate human (warmer)
            human_y = np.random.randint(height//4, 3*height//4)
            human_x = np.random.randint(width//4, 3*width//4)
            human_size = np.random.randint(20, 40)
            
            y, x = np.ogrid[-human_size:human_size+1, -human_size:human_size+1]
            mask = x*x + y*y <= human_size*human_size
            
            start_y = max(0, human_y - human_size)
            end_y = min(height, human_y + human_size + 1)
            start_x = max(0, human_x - human_size)
            end_x = min(width, human_x + human_size + 1)
            
            mask_y_start = human_size - (human_y - start_y) if human_y - human_size < 0 else 0
            mask_y_end = human_size + (end_y - human_y) if human_y + human_size + 1 > height else 2*human_size + 1
            mask_x_start = human_size - (human_x - start_x) if human_x - human_size < 0 else 0
            mask_x_end = human_size + (end_x - human_x) if human_x + human_size + 1 > width else 2*human_size + 1
            
            thermal_data[start_y:end_y, start_x:end_x] += mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end] * 10.0
        
        # Clip to thermal range
        min_temp, max_temp = self.config.thermal_range
        thermal_data = np.clip(thermal_data, min_temp, max_temp)
        
        # Create frame
        frame = ThermalFrame(
            data=thermal_data,
            timestamp=time.time(),
            metadata={
                'camera_type': 'flir_lepton_simulation',
                'simulation': True,
                'resolution': (width, height)
            },
            ambient_temp=base_temp,
            shutter_temp=base_temp + 5.0,
            fpa_temp=base_temp + 2.0
        )
        
        return frame
    
    def close(self):
        """Close camera connections"""
        if self.spi:
            self.spi.close()
        if self.i2c:
            self.i2c.close()
        
        self.logger.info("FLIR Lepton camera closed")

class RaspberryPiCamera(ThermalCamera):
    """Raspberry Pi Camera implementation (for visual camera)"""
    
    def __init__(self, config: CameraConfig):
        super().__init__(config)
        
        if not PICAMERA_AVAILABLE:
            self.logger.error("Picamera2 not available")
            self.camera = None
        else:
            self.camera = None
    
    def initialize(self) -> bool:
        """Initialize Raspberry Pi Camera"""
        if not PICAMERA_AVAILABLE:
            self.logger.error("Picamera2 not available")
            return False
        
        try:
            self.camera = Picamera2()
            
            # Configure camera
            config = self.camera.create_still_configuration(
                main={
                    "size": self.config.resolution,
                    "format": "RGB888"
                },
                buffer_count=2
            )
            self.camera.configure(config)
            
            # Start camera
            self.camera.start()
            time.sleep(2)  # Allow camera to warm up
            
            self.logger.info("Raspberry Pi Camera initialized")
            self.initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Raspberry Pi Camera: {e}")
            return False
    
    def capture_frame(self) -> Optional[ThermalFrame]:
        """Capture single frame"""
        if not self.initialized or self.camera is None:
            return None
        
        try:
            # Capture frame
            frame_data = self.camera.capture_array()
            
            # Convert to grayscale (simulate thermal-like image)
            if len(frame_data.shape) == 3:
                # Convert RGB to grayscale
                gray_data = np.mean(frame_data, axis=2).astype(np.float32)
            else:
                gray_data = frame_data.astype(np.float32)
            
            # Normalize to temperature-like range (simulation)
            min_val = np.min(gray_data)
            max_val = np.max(gray_data)
            
            if max_val > min_val:
                # Scale to thermal range
                thermal_range = self.config.thermal_range[1] - self.config.thermal_range[0]
                normalized = (gray_data - min_val) / (max_val - min_val)
                temperature_data = normalized * thermal_range + self.config.thermal_range[0]
            else:
                temperature_data = np.full_like(gray_data, self.config.thermal_range[0])
            
            # Create frame
            frame = ThermalFrame(
                data=temperature_data,
                timestamp=time.time(),
                metadata={
                    'camera_type': 'rpi_camera',
                    'original_shape': frame_data.shape,
                    'resolution': self.config.resolution
                }
            )
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Failed to capture frame: {e}")
            return None
    
    def close(self):
        """Close camera"""
        if self.camera:
            self.camera.stop()
            self.camera.close()
        
        self.logger.info("Raspberry Pi Camera closed")

class USBCamera(ThermalCamera):
    """USB Camera implementation"""
    
    def __init__(self, config: CameraConfig, device_id: int = 0):
        super().__init__(config)
        self.device_id = device_id
        self.capture_device = None
    
    def initialize(self) -> bool:
        """Initialize USB Camera"""
        try:
            import cv2
            
            self.capture_device = cv2.VideoCapture(self.device_id)
            
            if not self.capture_device.isOpened():
                self.logger.error(f"Failed to open USB camera {self.device_id}")
                return False
            
            # Set camera properties
            width, height = self.config.resolution
            self.capture_device.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture_device.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.capture_device.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            self.logger.info(f"USB Camera {self.device_id} initialized")
            self.initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize USB camera: {e}")
            return False
    
    def capture_frame(self) -> Optional[ThermalFrame]:
        """Capture single frame"""
        if not self.initialized or self.capture_device is None:
            return None
        
        try:
            import cv2
            
            ret, frame_data = self.capture_device.read()
            if not ret:
                self.logger.warning("Failed to read frame from USB camera")
                return None
            
            # Convert to grayscale (simulate thermal)
            if len(frame_data.shape) == 3:
                gray_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
            else:
                gray_data = frame_data
            
            # Convert to float and normalize
            gray_data = gray_data.astype(np.float32)
            
            # Normalize to temperature-like range
            min_val = np.min(gray_data)
            max_val = np.max(gray_data)
            
            if max_val > min_val:
                thermal_range = self.config.thermal_range[1] - self.config.thermal_range[0]
                normalized = (gray_data - min_val) / (max_val - min_val)
                temperature_data = normalized * thermal_range + self.config.thermal_range[0]
            else:
                temperature_data = np.full_like(gray_data, self.config.thermal_range[0])
            
            # Create frame
            frame = ThermalFrame(
                data=temperature_data,
                timestamp=time.time(),
                metadata={
                    'camera_type': 'usb_camera',
                    'device_id': self.device_id,
                    'resolution': gray_data.shape
                }
            )
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Failed to capture frame from USB camera: {e}")
            return None
    
    def close(self):
        """Close camera"""
        if self.capture_device:
            self.capture_device.release()
        
        self.logger.info("USB Camera closed")

class CameraManager:
    """Manage multiple cameras"""
    
    def __init__(self):
        self.cameras = {}
        self.logger = logging.getLogger(__name__)
    
    def create_camera(self, camera_type: CameraType, 
                     config: Optional[CameraConfig] = None) -> Optional[ThermalCamera]:
        """Create camera instance"""
        if config is None:
            config = CameraConfig(camera_type=camera_type)
        
        camera = None
        
        if camera_type == CameraType.FLIR_LEPTON:
            camera = FLIRLeptonCamera(config)
        elif camera_type == CameraType.RPI_CAMERA:
            camera = RaspberryPiCamera(config)
        elif camera_type == CameraType.USB_CAMERA:
            camera = USBCamera(config)
        elif camera_type == CameraType.SIMULATION:
            camera = FLIRLeptonCamera(config)  # Use FLIR simulation
        
        if camera:
            camera_id = f"{camera_type.value}_{len(self.cameras)}"
            self.cameras[camera_id] = camera
            self.logger.info(f"Created camera {camera_id}")
            return camera
        
        return None
    
    def initialize_all(self) -> bool:
        """Initialize all cameras"""
        success = True
        
        for camera_id, camera in self.cameras.items():
            if camera.initialize():
                self.logger.info(f"Camera {camera_id} initialized")
            else:
                self.logger.error(f"Failed to initialize camera {camera_id}")
                success = False
        
        return success
    
    def start_all(self):
        """Start all cameras"""
        for camera_id, camera in self.cameras.items():
            camera.start_capture()
            self.logger.info(f"Camera {camera_id} started")
    
    def stop_all(self):
        """Stop all cameras"""
        for camera_id, camera in self.cameras.items():
            camera.stop_capture()
            self.logger.info(f"Camera {camera_id} stopped")
    
    def close_all(self):
        """Close all cameras"""
        for camera_id, camera in self.cameras.items():
            camera.close()
            self.logger.info(f"Camera {camera_id} closed")
    
    def get_camera(self, camera_id: str) -> Optional[ThermalCamera]:
        """Get camera by ID"""
        return self.cameras.get(camera_id)
    
    def get_statistics(self) -> Dict:
        """Get statistics from all cameras"""
        stats = {}
        
        for camera_id, camera in self.cameras.items():
            stats[camera_id] = camera.get_statistics()
        
        return stats

def normalize_thermal_frame(frame: ThermalFrame, 
                          output_range: Tuple[int, int] = (0, 255)) -> np.ndarray:
    """
    Normalize thermal frame to 8-bit range for display
    
    Args:
        frame: Thermal frame
        output_range: Output range (min, max)
        
    Returns:
        Normalized 8-bit image
    """
    import cv2
    
    thermal_data = frame.data
    
    # Clip to camera's thermal range
    min_temp, max_temp = frame.metadata.get('thermal_range', (-20, 120))
    clipped_data = np.clip(thermal_data, min_temp, max_temp)
    
    # Normalize to output range
    output_min, output_max = output_range
    normalized = (clipped_data - min_temp) / (max_temp - min_temp)
    scaled = normalized * (output_max - output_min) + output_min
    
    # Convert to 8-bit
    result = scaled.astype(np.uint8)
    
    return result

def apply_color_map(frame: np.ndarray, colormap: str = 'hot') -> np.ndarray:
    """
    Apply color map to thermal image
    
    Args:
        frame: 8-bit thermal image
        colormap: Color map name
        
    Returns:
        Color-mapped image
    """
    import cv2
    
    colormaps = {
        'hot': cv2.COLORMAP_HOT,
        'jet': cv2.COLORMAP_JET,
        'plasma': cv2.COLORMAP_PLASMA,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'inferno': cv2.COLORMAP_INFERNO,
        'magma': cv2.COLORMAP_MAGMA,
        'cool': cv2.COLORMAP_COOL,
        'spring': cv2.COLORMAP_SPRING,
        'summer': cv2.COLORMAP_SUMMER,
        'autumn': cv2.COLORMAP_AUTUMN,
        'winter': cv2.COLORMAP_WINTER,
        'bone': cv2.COLORMAP_BONE,
        'copper': cv2.COLORMAP_COPPER,
        'rainbow': cv2.COLORMAP_RAINBOW
    }
    
    if colormap.lower() in colormaps:
        colored = cv2.applyColorMap(frame, colormaps[colormap.lower()])
    else:
        # Default to hot colormap
        colored = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
    
    return colored

if __name__ == "__main__":
    # Example usage
    import logging
    import cv2
    
    logging.basicConfig(level=logging.INFO)
    
    # Create camera manager
    manager = CameraManager()
    
    # Create FLIR camera (will use simulation if not on Raspberry Pi)
    flir_config = CameraConfig(
        camera_type=CameraType.FLIR_LEPTON,
        resolution=(160, 120),  # Lepton 3.5 native resolution
        fps=9
    )
    
    flir_camera = manager.create_camera(CameraType.FLIR_LEPTON, flir_config)
    
    # Initialize and start camera
    if flir_camera and flir_camera.initialize():
        flir_camera.start_capture()
        
        print("Camera started. Press 'q' to quit.")
        
        try:
            while True:
                # Get frame
                frame = flir_camera.get_frame(timeout=1.0)
                if frame is None:
                    continue
                
                # Normalize for display
                normalized = normalize_thermal_frame(frame)
                colored = apply_color_map(normalized, 'hot')
                
                # Display
                cv2.imshow('Thermal Camera', colored)
                
                # Display FPS
                fps = flir_camera.fps
                cv2.putText(colored, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Display temperature info
                if frame.ambient_temp is not None:
                    temp_text = f"Ambient: {frame.ambient_temp:.1f}°C"
                    cv2.putText(colored, temp_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Thermal Camera', colored)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            flir_camera.stop_capture()
            flir_camera.close()
            cv2.destroyAllWindows()
            
            # Print statistics
            stats = flir_camera.get_statistics()
            print("\nCamera Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    else:
        print("Failed to initialize camera")
