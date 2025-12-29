"""
Voice Alert System for Thermal Object Detection
Provides audio warnings for detected objects
"""

import pyttsx3
import threading
import queue
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

class AlertPriority(Enum):
    """Priority levels for alerts"""
    CRITICAL = 4    # Immediate threat (human with weapon, fast vehicle)
    HIGH = 3        # High threat (human approaching, vehicle moving fast)
    MEDIUM = 2      # Medium threat (human detected, animal close)
    LOW = 1         # Low threat (animal detected, stationary vehicle)
    INFO = 0        # Information only (system status)

@dataclass
class AlertMessage:
    """Alert message container"""
    priority: AlertPriority
    message: str
    class_name: str
    confidence: float
    distance: Optional[float] = None  # meters
    direction: Optional[str] = None   # compass direction
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class VoiceAlertSystem:
    """Voice alert system for object detection warnings"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize TTS engine
        self.tts_engine = self._initialize_tts()
        
        # Alert queue
        self.alert_queue = queue.PriorityQueue()
        
        # Alert history
        self.alert_history = []
        self.max_history_size = 100
        
        # Processing thread
        self.processing_thread = None
        self.running = False
        
        # Alert patterns
        self.alert_patterns = self._create_alert_patterns()
        
        # Cooldown timers
        self.cooldown_timers = {}
        self.cooldown_duration = self.config['cooldown_duration']
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'voice_rate': 150,           # Words per minute
            'voice_volume': 0.9,         # Volume (0.0 to 1.0)
            'voice_id': 0,               # Voice index
            'min_confidence': 0.5,       # Minimum confidence for alerts
            'max_alerts_per_minute': 10, # Rate limiting
            'cooldown_duration': 5.0,    # Seconds between similar alerts
            'distance_thresholds': {
                'human': {
                    'critical': 50,      # meters
                    'high': 100,
                    'medium': 200,
                    'low': 300
                },
                'vehicle': {
                    'critical': 100,
                    'high': 200,
                    'medium': 300,
                    'low': 500
                },
                'animal': {
                    'critical': 30,
                    'high': 50,
                    'medium': 100,
                    'low': 200
                }
            }
        }
    
    def _initialize_tts(self) -> pyttsx3.Engine:
        """Initialize text-to-speech engine"""
        try:
            engine = pyttsx3.init()
            
            # Configure voice properties
            engine.setProperty('rate', self.config['voice_rate'])
            engine.setProperty('volume', self.config['voice_volume'])
            
            # Get available voices
            voices = engine.getProperty('voices')
            if voices:
                # Try to find a clear, natural voice
                if self.config['voice_id'] < len(voices):
                    engine.setProperty('voice', voices[self.config['voice_id']].id)
                else:
                    engine.setProperty('voice', voices[0].id)
            
            self.logger.info("TTS engine initialized successfully")
            return engine
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS engine: {e}")
            raise
    
    def _create_alert_patterns(self) -> Dict:
        """Create alert message patterns"""
        return {
            'human': {
                'critical': [
                    "CRITICAL! Human detected at {distance} meters, moving {direction}",
                    "IMMEDIATE THREAT! Person at {distance} meters {direction}",
                    "ALERT! Hostile presence {distance} meters {direction}"
                ],
                'high': [
                    "Warning. Human detected at {distance} meters, {direction}",
                    "Person sighted at {distance} meters, moving {direction}",
                    "Human presence {distance} meters {direction}"
                ],
                'medium': [
                    "Human detected at {distance} meters",
                    "Person at {distance} meters",
                    "Human signature {distance} meters"
                ],
                'low': [
                    "Human in area, {distance} meters",
                    "Person detected in vicinity",
                    "Human presence noted"
                ]
            },
            'animal': {
                'critical': [
                    "CAUTION! Animal very close, {distance} meters {direction}",
                    "Wildlife alert! Animal at {distance} meters",
                    "Animal proximity warning, {distance} meters"
                ],
                'high': [
                    "Animal detected at {distance} meters {direction}",
                    "Wildlife presence {distance} meters",
                    "Animal sighted {distance} meters"
                ],
                'medium': [
                    "Animal at {distance} meters",
                    "Wildlife detected {distance} meters",
                    "Animal presence {distance} meters"
                ],
                'low': [
                    "Animal in area",
                    "Wildlife detected",
                    "Animal presence noted"
                ]
            },
            'vehicle': {
                'critical': [
                    "EMERGENCY! Vehicle approaching fast, {distance} meters {direction}",
                    "HIGH SPEED VEHICLE! {distance} meters {direction}",
                    "RAPID APPROACH! Vehicle {distance} meters {direction}"
                ],
                'high': [
                    "Vehicle detected moving {direction}, {distance} meters",
                    "Approaching vehicle {distance} meters {direction}",
                    "Vehicle movement {distance} meters {direction}"
                ],
                'medium': [
                    "Vehicle detected at {distance} meters",
                    "Car or truck {distance} meters",
                    "Vehicle presence {distance} meters"
                ],
                'low': [
                    "Vehicle in vicinity",
                    "Car detected in area",
                    "Vehicle presence noted"
                ]
            }
        }
    
    def calculate_priority(self, class_name: str, confidence: float, 
                         distance: Optional[float] = None,
                         speed: Optional[float] = None) -> AlertPriority:
        """
        Calculate alert priority based on detection parameters
        
        Args:
            class_name: Object class (human, animal, vehicle)
            confidence: Detection confidence (0.0 to 1.0)
            distance: Distance in meters (optional)
            speed: Speed in m/s (optional)
            
        Returns:
            AlertPriority level
        """
        # Start with confidence-based priority
        if confidence >= 0.9:
            base_priority = AlertPriority.HIGH
        elif confidence >= 0.7:
            base_priority = AlertPriority.MEDIUM
        elif confidence >= self.config['min_confidence']:
            base_priority = AlertPriority.LOW
        else:
            return AlertPriority.INFO
        
        # Adjust based on distance if available
        if distance is not None and class_name in self.config['distance_thresholds']:
            thresholds = self.config['distance_thresholds'][class_name]
            
            if distance <= thresholds['critical']:
                distance_priority = AlertPriority.CRITICAL
            elif distance <= thresholds['high']:
                distance_priority = AlertPriority.HIGH
            elif distance <= thresholds['medium']:
                distance_priority = AlertPriority.MEDIUM
            else:
                distance_priority = AlertPriority.LOW
            
            # Use the higher priority (more critical)
            if distance_priority.value > base_priority.value:
                base_priority = distance_priority
        
        # Adjust based on speed if available
        if speed is not None:
            if speed > 10:  # Fast moving (m/s)
                if base_priority.value < AlertPriority.CRITICAL.value:
                    base_priority = AlertPriority.CRITICAL
            elif speed > 5:  # Medium speed
                if base_priority.value < AlertPriority.HIGH.value:
                    base_priority = AlertPriority.HIGH
        
        return base_priority
    
    def generate_alert_message(self, class_name: str, priority: AlertPriority,
                             distance: Optional[float] = None,
                             direction: Optional[str] = None,
                             confidence: float = 0.0) -> str:
        """
        Generate alert message based on parameters
        
        Args:
            class_name: Object class
            priority: Alert priority
            distance: Distance in meters
            direction: Compass direction
            confidence: Detection confidence
            
        Returns:
            Formatted alert message
        """
        # Check if class has alert patterns
        if class_name not in self.alert_patterns:
            class_name = 'human'  # Default
        
        # Check if priority has patterns
        priority_name = priority.name.lower()
        if priority_name not in self.alert_patterns[class_name]:
            # Find closest available priority
            available = list(self.alert_patterns[class_name].keys())
            priority_name = available[0]
        
        # Get patterns for this class and priority
        patterns = self.alert_patterns[class_name][priority_name]
        
        # Select random pattern (for variety)
        pattern = np.random.choice(patterns)
        
        # Format the message
        message = pattern
        
        # Add distance if available
        if distance is not None:
            # Round distance to nearest 10 meters for clarity
            rounded_distance = round(distance / 10) * 10
            message = message.replace('{distance}', f'{rounded_distance:.0f}')
        else:
            message = message.replace('{distance}', 'unknown distance')
        
        # Add direction if available
        if direction is not None:
            message = message.replace('{direction}', direction)
        else:
            message = message.replace('{direction}', '')
        
        # Clean up extra spaces
        message = ' '.join(message.split())
        
        # Add confidence if low
        if confidence < 0.7:
            confidence_percent = int(confidence * 100)
            message = f"{message}. Confidence {confidence_percent} percent."
        
        return message
    
    def create_alert(self, class_name: str, confidence: float,
                    distance: Optional[float] = None,
                    direction: Optional[str] = None,
                    speed: Optional[float] = None) -> Optional[AlertMessage]:
        """
        Create and queue an alert
        
        Args:
            class_name: Object class
            confidence: Detection confidence
            distance: Distance in meters
            direction: Compass direction
            speed: Speed in m/s
            
        Returns:
            AlertMessage if queued, None if filtered
        """
        # Check minimum confidence
        if confidence < self.config['min_confidence']:
            return None
        
        # Calculate priority
        priority = self.calculate_priority(class_name, confidence, distance, speed)
        
        # Check cooldown
        alert_key = f"{class_name}_{priority.value}"
        current_time = time.time()
        
        if alert_key in self.cooldown_timers:
            last_alert_time = self.cooldown_timers[alert_key]
            if current_time - last_alert_time < self.config['cooldown_duration']:
                self.logger.debug(f"Alert cooldown active for {alert_key}")
                return None
        
        # Generate message
        message = self.generate_alert_message(
            class_name, priority, distance, direction, confidence
        )
        
        # Create alert
        alert = AlertMessage(
            priority=priority,
            message=message,
            class_name=class_name,
            confidence=confidence,
            distance=distance,
            direction=direction,
            timestamp=current_time
        )
        
        # Add to queue (negative priority for correct ordering)
        self.alert_queue.put((-priority.value, alert))
        
        # Update cooldown timer
        self.cooldown_timers[alert_key] = current_time
        
        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history_size:
            self.alert_history.pop(0)
        
        self.logger.info(f"Alert created: {priority.name} - {message}")
        
        return alert
    
    def process_alerts(self):
        """Process alert queue and speak alerts"""
        self.logger.info("Starting alert processing thread")
        
        alert_count = 0
        last_minute_time = time.time()
        
        while self.running:
            try:
                # Get next alert (blocking with timeout)
                priority, alert = self.alert_queue.get(timeout=1)
                
                # Rate limiting
                current_time = time.time()
                if current_time - last_minute_time > 60:
                    # Reset counter every minute
                    alert_count = 0
                    last_minute_time = current_time
                
                if alert_count >= self.config['max_alerts_per_minute']:
                    self.logger.warning("Alert rate limit exceeded, skipping alert")
                    self.alert_queue.task_done()
                    continue
                
                # Speak the alert
                self.speak_alert(alert)
                
                alert_count += 1
                self.alert_queue.task_done()
                
                # Small delay between alerts
                time.sleep(0.5)
                
            except queue.Empty:
                # No alerts in queue
                continue
            except Exception as e:
                self.logger.error(f"Error processing alert: {e}")
                if not self.alert_queue.empty():
                    self.alert_queue.task_done()
    
    def speak_alert(self, alert: AlertMessage):
        """Speak an alert message"""
        try:
            self.logger.info(f"Speaking alert: {alert.message}")
            
            # Speak the message
            self.tts_engine.say(alert.message)
            self.tts_engine.runAndWait()
            
            # Add emphasis for critical alerts
            if alert.priority == AlertPriority.CRITICAL:
                time.sleep(0.2)
                self.tts_engine.say("Warning!")
                self.tts_engine.runAndWait()
            
        except Exception as e:
            self.logger.error(f"Failed to speak alert: {e}")
    
    def start(self):
        """Start the alert system"""
        if self.running:
            self.logger.warning("Alert system already running")
            return
        
        self.running = True
        self.processing_thread = threading.Thread(
            target=self.process_alerts,
            daemon=True
        )
        self.processing_thread.start()
        
        self.logger.info("Voice alert system started")
    
    def stop(self):
        """Stop the alert system"""
        self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        self.logger.info("Voice alert system stopped")
    
    def test_system(self):
        """Test the voice alert system with sample alerts"""
        self.logger.info("Testing voice alert system...")
        
        test_alerts = [
            ('human', 0.95, 50, 'north', 2.0),
            ('vehicle', 0.88, 200, 'east', 15.0),
            ('animal', 0.75, 30, 'west', 1.0),
            ('human', 0.65, 150, 'south', 0.5),
        ]
        
        print("\n" + "="*60)
        print("VOICE ALERT SYSTEM TEST")
        print("="*60)
        
        for class_name, confidence, distance, direction, speed in test_alerts:
            print(f"\nTest Alert:")
            print(f"  Class: {class_name}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Distance: {distance}m")
            print(f"  Direction: {direction}")
            print(f"  Speed: {speed}m/s")
            
            alert = self.create_alert(
                class_name=class_name,
                confidence=confidence,
                distance=distance,
                direction=direction,
                speed=speed
            )
            
            if alert:
                print(f"  Priority: {alert.priority.name}")
                print(f"  Message: {alert.message}")
                
                # Speak the alert
                self.speak_alert(alert)
                time.sleep(1)  # Pause between test alerts
            else:
                print("  Alert filtered (cooldown or low confidence)")
        
        print("\nTest complete!")
    
    def get_alert_history(self, limit: int = 10) -> List[AlertMessage]:
        """Get recent alert history"""
        return self.alert_history[-limit:] if self.alert_history else []
    
    def get_statistics(self) -> Dict:
        """Get alert system statistics"""
        if not self.alert_history:
            return {}
        
        # Calculate statistics
        total_alerts = len(self.alert_history)
        
        # Priority distribution
        priority_dist = {}
        for alert in self.alert_history:
            priority_name = alert.priority.name
            priority_dist[priority_name] = priority_dist.get(priority_name, 0) + 1
        
        # Class distribution
        class_dist = {}
        for alert in self.alert_history:
            class_name = alert.class_name
            class_dist[class_name] = class_dist.get(class_name, 0) + 1
        
        # Time range
        if self.alert_history:
            first_alert = self.alert_history[0].timestamp
            last_alert = self.alert_history[-1].timestamp
            time_range = last_alert - first_alert
        else:
            time_range = 0
        
        stats = {
            'total_alerts': total_alerts,
            'priority_distribution': priority_dist,
            'class_distribution': class_dist,
            'time_range_seconds': time_range,
            'alerts_per_minute': total_alerts / (time_range / 60) if time_range > 0 else 0
        }
        
        return stats

class DirectionCalculator:
    """Calculate direction from coordinates"""
    
    @staticmethod
    def calculate_direction(dx: float, dy: float) -> str:
        """
        Calculate compass direction from displacement
        
        Args:
            dx: X displacement (east-west)
            dy: Y displacement (north-south)
            
        Returns:
            Compass direction (e.g., "north", "southeast")
        """
        if dx == 0 and dy == 0:
            return "stationary"
        
        # Calculate angle in radians
        angle = np.arctan2(dy, dx)
        
        # Convert to degrees and adjust for compass
        angle_deg = np.degrees(angle)
        compass_angle = (90 - angle_deg) % 360
        
        # Map to compass directions
        directions = [
            ('north', 0, 22.5),
            ('northeast', 22.5, 67.5),
            ('east', 67.5, 112.5),
            ('southeast', 112.5, 157.5),
            ('south', 157.5, 202.5),
            ('southwest', 202.5, 247.5),
            ('west', 247.5, 292.5),
            ('northwest', 292.5, 337.5),
            ('north', 337.5, 360)
        ]
        
        for direction, start, end in directions:
            if start <= compass_angle < end:
                return direction
        
        return "unknown"

if __name__ == "__main__":
    # Example usage
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create voice alert system
    alert_system = VoiceAlertSystem()
    
    # Start the system
    alert_system.start()
    
    try:
        # Run test
        alert_system.test_system()
        
        # Keep running for manual testing
        print("\n" + "="*60)
        print("Manual testing mode")
        print("Enter alerts in format: class confidence distance direction speed")
        print("Example: human 0.8 100 north 2.0")
        print("Type 'quit' to exit")
        print("="*60)
        
        while True:
            user_input = input("\nEnter alert: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            try:
                parts = user_input.split()
                if len(parts) >= 2:
                    class_name = parts[0]
                    confidence = float(parts[1])
                    distance = float(parts[2]) if len(parts) > 2 else None
                    direction = parts[3] if len(parts) > 3 else None
                    speed = float(parts[4]) if len(parts) > 4 else None
                    
                    alert = alert_system.create_alert(
                        class_name=class_name,
                        confidence=confidence,
                        distance=distance,
                        direction=direction,
                        speed=speed
                    )
                    
                    if alert:
                        print(f"Alert created: {alert.message}")
                    else:
                        print("Alert filtered")
                else:
                    print("Invalid input format")
                    
            except ValueError:
                print("Invalid input values")
            except Exception as e:
                print(f"Error: {e}")
    
    finally:
        # Stop the system
        alert_system.stop()
        
        # Print statistics
        stats = alert_system.get_statistics()
        print("\n" + "="*60)
        print("SYSTEM STATISTICS")
        print("="*60)
        for key, value in stats.items():
            print(f"{key}: {value}")
