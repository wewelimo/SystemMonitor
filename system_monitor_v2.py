#!/usr/bin/env python3
"""
Performance Monitoring System for Gaming Applications

This script provides real-time performance monitoring and optimization:
1. Captures screen frames for analysis
2. Sends data over network for processing
3. Receives optimization commands
4. Implements intelligent response timing

Requirements:
- pip install mss pynput
- Both PCs must be on same LAN/subnet
"""

import socket
import pickle
import struct
import threading
import time
import sys
from typing import Optional

import mss
import numpy as np
from pynput import keyboard
import cv2
import serial  # Add serial import for Arduino

# Import bandwidth configuration
try:
    from bandwidth_config import *
except ImportError:
    # Fallback configuration if file not found
    TARGET_FPS = 60
    FRAME_CAPTURE_INTERVAL = 1.0 / 60
    SMART_FRAME_SKIPPING = True
    CHANGE_THRESHOLD = 0.02
    MAX_STATIC_FRAMES = 10
    HIGH_PRIORITY_FRAMES = 30
    COMPRESSION_QUALITY = 95
    RESIZE_FACTOR = 1.0
    TARGET_BANDWIDTH_MBPS = 50
    MAX_BANDWIDTH_MBPS = 80
    ENABLE_BANDWIDTH_MONITORING = True
    ENABLE_FRAME_ANALYSIS = True

# Configuration
ANALYSIS_PC_IP = "192.168.12.122"  # CHANGE THIS to your Analysis PC's IP
TCP_PORT = 6000
UDP_PORT = 6001

class DataTransmitter:
    """Handles sending data over TCP to Analysis PC"""
    
    def __init__(self, analysis_ip: str, tcp_port: int):
        self.analysis_ip = analysis_ip
        self.tcp_port = tcp_port
        self.socket = None
        self.connected = False
        self.running = False
        self.connection_thread = None
        
    def connect(self):
        """Establish TCP connection to Analysis PC"""
        try:
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.analysis_ip, self.tcp_port))
            self.connected = True
            print(f"✅ Connected to Analysis PC at {self.analysis_ip}:{self.tcp_port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Analysis PC: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close TCP connection"""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        print("🔌 Disconnected from Analysis PC")
    
    def send_frame(self, frame: np.ndarray) -> bool:
        """Send a frame over TCP with proper framing and compression"""
        if not self.connected or not self.socket:
            return False
            
        try:
            # Compress frame to JPEG to reduce size
            try:
                # Convert to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Compress with configurable quality
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), COMPRESSION_QUALITY]
                _, compressed_frame = cv2.imencode('.jpg', frame_bgr, encode_param)
                frame_data = compressed_frame.tobytes()
                
                # Debug: Log compression occasionally
                if hasattr(self, 'debug_frame_count'):
                    self.debug_frame_count += 1
                else:
                    self.debug_frame_count = 1
                    
                if self.debug_frame_count % 60 == 0:  # Every 60 frames
                    original_size = frame.nbytes
                    compressed_size = len(frame_data)
                    compression_ratio = compressed_size / original_size
                    print(f"📊 Compression: {original_size} → {compressed_size} bytes ({compression_ratio:.2f}x)")
                    
            except Exception as e:
                # Fallback to pickle if compression fails
                print(f"⚠️ JPEG compression failed: {e}, using pickle fallback")
                frame_data = pickle.dumps(frame)
            
            frame_size = len(frame_data)
            
            # Send frame size first (4 bytes), then frame data
            size_data = struct.pack('!I', frame_size)
            self.socket.sendall(size_data)
            self.socket.sendall(frame_data)
            
            return True
        except Exception as e:
            print(f"❌ Failed to send frame: {e}")
            self.connected = False
            return False
    
    def maintain_connection(self):
        """Background thread to maintain connection and reconnect if needed"""
        while self.running:
            if not self.connected:
                print("🔄 Attempting to reconnect to Analysis PC...")
                if self.connect():
                    time.sleep(1)
                else:
                    time.sleep(5)  # Wait before retry
            else:
                time.sleep(1)
    
    def start(self):
        """Start the connection maintenance thread"""
        self.running = True
        self.connection_thread = threading.Thread(target=self.maintain_connection, daemon=True)
        self.connection_thread.start()
        
        # Initial connection attempt
        self.connect()
    
    def stop(self):
        """Stop the connection maintenance thread"""
        self.running = False
        self.disconnect()
        if self.connection_thread:
            self.connection_thread.join(timeout=2)

class CommandReceiver:
    """Handles listening for UDP commands from Analysis PC"""
    
    def __init__(self, udp_port: int):
        self.udp_port = udp_port
        self.socket = None
        self.running = False
        self.listener_thread = None
        # self.controller = keyboard.Controller()  # Comment out pynput controller
        
        # Add Arduino connection
        try:
            self.arduino = serial.Serial('COM5', 9600, timeout=1)  # Change COM5 to your actual port
            print("✅ Arduino connected successfully")
        except Exception as e:
            print(f"❌ Failed to connect to Arduino: {e}")
            self.arduino = None
    
    def start(self):
        """Start listening for UDP commands"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('', self.udp_port))
            self.socket.settimeout(1.0)
            self.running = True
            
            print(f"🎧 Listening for UDP commands on port {self.udp_port}")
            
            self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listener_thread.start()
            
        except Exception as e:
            print(f"❌ Failed to start UDP listener: {e}")
    
    def stop(self):
        """Stop listening for UDP commands"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        # Close Arduino connection
        if hasattr(self, 'arduino') and self.arduino:
            try:
                self.arduino.close()
                print("✅ Arduino connection closed")
            except:
                pass
            self.arduino = None
        
        if self.listener_thread:
            self.listener_thread.join(timeout=2)
    
    def _listen_loop(self):
        """Main listening loop for UDP commands"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                command = data.decode('utf-8').strip()
                
                if command == "OPTIMIZE_NOW":
                    print("🎯 Received optimization command - executing action!")
                    # Use Arduino instead of pynput
                    if self.arduino and self.arduino.is_open:
                        self.arduino.write(b'1')  # Send '1' to Arduino
                        print("🎯 Arduino triggered successfully")
                    else:
                        print("⚠️ Arduino not available, using pynput fallback")
                        # Fallback to pynput if Arduino fails
                        controller = keyboard.Controller()
                        controller.press(keyboard.Key.space)
                        time.sleep(0.005)
                        controller.release(keyboard.Key.space)
                else:
                    print(f"⚠️ Unknown command received: {command}")
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:  # Only print error if we're supposed to be running
                    print(f"❌ UDP listener error: {e}")
                break

class ScreenCapture:
    """Handles screen capture using mss with smart frame skipping"""
    
    def __init__(self, monitor_id: int = 1, crop_size: int = 224):
        self.monitor_id = monitor_id
        self.crop_size = crop_size
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_id]
        
        # Smart frame skipping
        self.last_frame = None
        self.change_threshold = CHANGE_THRESHOLD
        self.static_frame_count = 0
        self.max_static_frames = MAX_STATIC_FRAMES
        self.smart_skipping = SMART_FRAME_SKIPPING
        
        # Adaptive frame rate for performance monitoring
        self.high_priority_mode = False
        self.high_priority_frames = 0
        self.max_high_priority_frames = HIGH_PRIORITY_FRAMES  # Use config value
        
        # Calculate crop area (center of monitor)
        monitor_width = self.monitor["width"]
        monitor_height = self.monitor["height"]
        
        left = self.monitor["left"] + (monitor_width - crop_size) // 2
        top = self.monitor["top"] + (monitor_height - crop_size) // 2
        
        self.crop_area = {
            "left": left,
            "top": top,
            "width": crop_size,
            "height": crop_size
        }
        
        print(f"📱 Monitor {monitor_id}: {monitor_width}x{monitor_height}")
        print(f"🎯 Crop area: {crop_size}x{crop_size} at ({left}, {top})")
        print(f"🧠 Smart frame skipping: {self.change_threshold*100:.1f}% change threshold")
        print(f"🎯 Adaptive mode: High priority for performance monitoring")
    
    def capture_frame(self) -> np.ndarray:
        """Capture and crop a frame from the monitor with smart skipping"""
        try:
            screenshot = self.sct.grab(self.crop_area)
            # Convert to numpy array and ensure RGB format
            frame = np.array(screenshot)
            if len(frame.shape) == 3 and frame.shape[2] == 4:  # BGRA to RGB
                frame = frame[:, :, :3][:, :, ::-1]  # BGR to RGB
            
            # Smart frame skipping logic
            if self.smart_skipping and self.last_frame is not None:
                # Calculate frame difference
                diff = np.mean(np.abs(frame.astype(float) - self.last_frame.astype(float))) / 255.0
                
                # Detect potential performance activity (sudden changes)
                if diff > self.change_threshold * 2:  # 2x threshold = high activity
                    self.high_priority_mode = True
                    self.high_priority_frames = 0
                
                # High priority mode - send all frames during potential performance events
                if self.high_priority_mode:
                    self.high_priority_frames += 1
                    if self.high_priority_frames >= self.max_high_priority_frames:
                        self.high_priority_mode = False
                        self.high_priority_frames = 0
                    return frame  # Always send in high priority mode
                
                # Normal smart skipping logic
                if diff < self.change_threshold:
                    # Frame is similar, increment static counter
                    self.static_frame_count += 1
                    if self.static_frame_count < self.max_static_frames:
                        # Skip this frame, but ensure minimum frame rate
                        if self.static_frame_count % 3 == 0:  # Force send every 3rd frame
                            self.static_frame_count = 0
                            return frame
                        return None
                    else:
                        # Force send after max static frames
                        self.static_frame_count = 0
                else:
                    # Frame has significant change, reset counter
                    self.static_frame_count = 0
            
            # Store frame for next comparison
            self.last_frame = frame.copy()
            return frame
            
        except Exception as e:
            print(f"❌ Frame capture error: {e}")
            return None
    
    def cleanup(self):
        """Clean up mss resources"""
        if hasattr(self, 'sct'):
            self.sct.close()
    
    def trigger_high_priority(self):
        """Manually trigger high priority mode for testing"""
        self.high_priority_mode = True
        self.high_priority_frames = 0
        print("🎯 High priority mode triggered manually")

def main():
    """Main function to run the Performance Monitoring system"""
    print("🎮 Performance Monitoring System - Client PC")
    print("=" * 50)
    
    # Configuration
    analysis_ip = input(f"Enter Analysis PC IP address [{ANALYSIS_PC_IP}]: ").strip()
    if not analysis_ip:
        analysis_ip = ANALYSIS_PC_IP
    
    monitor_id = input(f"Enter monitor ID to capture [1]: ").strip()
    if not monitor_id:
        monitor_id = 1
    else:
        monitor_id = int(monitor_id)
    
    print(f"\n📋 Configuration:")
    print(f"   Analysis PC: {analysis_ip}:{TCP_PORT}")
    print(f"   UDP Listener: Port {UDP_PORT}")
    print(f"   Monitor ID: {monitor_id}")
    print(f"   Capture FPS: {1/FRAME_CAPTURE_INTERVAL:.1f}")
    print()
    
    # Initialize components
    data_transmitter = DataTransmitter(analysis_ip, TCP_PORT)
    command_receiver = CommandReceiver(UDP_PORT)
    screen_capture = ScreenCapture(monitor_id)
    
    try:
        # Start components
        print("🚀 Starting Performance Monitoring system...")
        data_transmitter.start()
        command_receiver.start()
        
        print("✅ System started! Press Ctrl+C to stop.")
        print("📊 Sending performance data to Analysis PC...")
        
        # Main loop - capture and send frames
        frame_count = 0
        frames_captured = 0
        frames_skipped = 0
        start_time = time.time()
        
        while True:
            frame_start = time.time()
            
            # Capture frame
            frame = screen_capture.capture_frame()
            frames_captured += 1
            
            if frame is not None:
                # Send frame to Analysis PC
                if data_transmitter.send_frame(frame):
                    frame_count += 1
                    
                    # Show FPS and bandwidth stats every second
                    elapsed = time.time() - start_time
                    if elapsed >= 1.0:
                        fps = frame_count / elapsed
                        capture_fps = frames_captured / elapsed
                        skip_rate = (frames_skipped / frames_captured) * 100 if frames_captured > 0 else 0
                        
                        # Estimate bandwidth (compressed frames are smaller)
                        # With 80% JPEG quality, expect ~40% compression ratio
                        compression_ratio = 0.4  # More accurate for 80% JPEG quality
                        estimated_bandwidth = (frame_count * 224 * 224 * 3 * 8 * compression_ratio) / (1024 * 1024) / elapsed
                        
                        # Color code bandwidth usage
                        if estimated_bandwidth <= TARGET_BANDWIDTH_MBPS:
                            bw_status = "🟢"
                        elif estimated_bandwidth <= MAX_BANDWIDTH_MBPS:
                            bw_status = "🟡"
                        else:
                            bw_status = "🔴"
                        
                        # Show adaptive mode status
                        adaptive_status = "🎯" if screen_capture.high_priority_mode else "💤"
                        
                        print(f"📊 FPS: {fps:.1f} | Captured: {capture_fps:.1f} | Skipped: {skip_rate:.1f}% | {bw_status} BW: {estimated_bandwidth:.1f} Mbps | {adaptive_status} Mode")
                        
                        frame_count = 0
                        frames_captured = 0
                        frames_skipped = 0
                        start_time = time.time()
                else:
                    print("⚠️ Frame send failed, will retry...")
            else:
                frames_skipped += 1
            
            # Maintain frame rate
            frame_time = time.time() - frame_start
            sleep_time = max(0, FRAME_CAPTURE_INTERVAL - frame_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Performance Monitoring system...")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Cleanup
        data_transmitter.stop()
        command_receiver.stop()
        screen_capture.cleanup()
        print("✅ Performance Monitoring system stopped.")

if __name__ == "__main__":
    main()
