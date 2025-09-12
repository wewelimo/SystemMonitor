#!/usr/bin/env python3
"""
Analysis PC Script for Performance Monitoring Networked System

This script runs on the Analysis PC and:
1. Receives frames over TCP from Client PC
2. Processes frames with AI model to detect performance events
3. Sends UDP commands back to Client PC when optimization needed
4. Provides Gradio web UI for configuration and monitoring

Requirements:
- pip install gradio onnxruntime numpy opencv-python
- AI model files in models/ folder
- Both PCs must be on same LAN/subnet
"""

import os
import socket
import pickle
import struct
import threading
import time
import sys
from typing import Optional, Tuple, Dict, Any

import gradio as gr
import numpy as np
import cv2

# Import the AI model from the existing codebase
from core.AI_model import AI_model

# Configuration
CLIENT_PC_IP = "192.168.12.177"  # CHANGE THIS to your Client PC's IP
TCP_PORT = 6000
UDP_PORT = 6001
BUFFER_SIZE = 1024 * 1024  # 1MB buffer for frame data

class FrameReceiver:
    """Handles receiving frames over TCP from Client PC"""
    
    def __init__(self, tcp_port: int):
        self.tcp_port = tcp_port
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.running = False
        self.receiver_thread = None
        self.frame_callback = None
        
    def set_frame_callback(self, callback):
        """Set callback function to handle received frames"""
        self.frame_callback = callback
    
    def start(self):
        """Start TCP server to receive frames"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('', self.tcp_port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0)
            self.running = True
            
            print(f"🎧 TCP server started on port {self.tcp_port}")
            print("⏳ Waiting for Client PC connection...")
            
            self.receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receiver_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start TCP server: {e}")
            self.running = False
            return False
    
    def stop(self):
        """Stop TCP server"""
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
            
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None
        
        if self.receiver_thread:
            self.receiver_thread.join(timeout=2)
    
    def _receive_loop(self):
        """Main loop to receive frames from Client PC"""
        while self.running:
            try:
                # Accept new connection
                if self.client_socket is None:
                    try:
                        self.client_socket, self.client_address = self.server_socket.accept()
                        self.client_socket.settimeout(1.0)
                        print(f"✅ Client PC connected from {self.client_address}")
                    except socket.timeout:
                        continue
                    except Exception as e:
                        if self.running:
                            print(f"❌ Connection accept error: {e}")
                        continue
                
                # Receive frame data
                try:
                    # First receive frame size (4 bytes)
                    size_data = self.client_socket.recv(4)
                    if not size_data:
                        raise ConnectionError("Connection closed by client")
                    
                    frame_size = struct.unpack('!I', size_data)[0]
                    
                    # Then receive frame data
                    frame_data = b''
                    while len(frame_data) < frame_size:
                        chunk = self.client_socket.recv(min(frame_size - len(frame_data), BUFFER_SIZE))
                        if not chunk:
                            raise ConnectionError("Connection closed while receiving frame")
                        frame_data += chunk
                    
                    # Deserialize frame (handle both compressed and uncompressed)
                    try:
                        # Try to decode as JPEG first (compressed frames)
                        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            # Convert BGR to RGB for consistency
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # Only log occasionally to avoid spam
                            if hasattr(self, 'log_counter'):
                                self.log_counter += 1
                            else:
                                self.log_counter = 1
                            
                            if self.log_counter % 120 == 0:  # Every 120 frames
                                print(f"📸 JPEG frame decoded: {frame.shape}, size: {len(frame_data)} bytes")
                        else:
                            # Fallback to pickle (uncompressed frames)
                            frame = pickle.loads(frame_data)
                            if hasattr(self, 'log_counter'):
                                self.log_counter += 1
                            else:
                                self.log_counter = 1
                            
                            if self.log_counter % 120 == 0:  # Every 120 frames
                                print(f"📸 Pickle frame decoded: {frame.shape}, size: {len(frame_data)} bytes")
                    except Exception as e:
                        # Fallback to pickle if JPEG decoding fails
                        try:
                            frame = pickle.loads(frame_data)
                            if hasattr(self, 'log_counter'):
                                self.log_counter += 1
                            else:
                                self.log_counter = 1
                            
                            if self.log_counter % 120 == 0:  # Every 120 frames
                                print(f"📸 Pickle fallback frame: {frame.shape}, size: {len(frame_data)} bytes")
                        except Exception as e2:
                            print(f"❌ Failed to decode frame: JPEG error: {e}, Pickle error: {e2}")
                            print(f"   Frame data size: {len(frame_data)} bytes, first 10 bytes: {frame_data[:10]}")
                            continue  # Skip this frame
                    
                    # Call callback with received frame
                    if self.frame_callback:
                        self.frame_callback(frame)
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"❌ Frame receive error: {e}")
                    # Only close connection for serious errors, not frame decode issues
                    if "Connection closed" in str(e) or "Connection reset" in str(e):
                        try:
                            self.client_socket.close()
                        except:
                            pass
                        self.client_socket = None
                        self.client_address = None
                        print("🔄 Waiting for new Client PC connection...")
                    else:
                        print("⚠️ Frame error, continuing...")
                        continue
                    
            except Exception as e:
                if self.running:
                    print(f"❌ Receiver loop error: {e}")
                break

class CommandSender:
    """Handles sending UDP commands to Client PC"""
    
    def __init__(self, client_ip: str, udp_port: int):
        self.client_ip = client_ip
        self.udp_port = udp_port
        self.socket = None
        
    def connect(self):
        """Initialize UDP socket"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"📡 UDP sender ready for {self.client_ip}:{self.udp_port}")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize UDP sender: {e}")
            return False
    
    def send_command(self, command: str) -> bool:
        """Send a command to Client PC"""
        if not self.socket:
            print(f"❌ UDP socket not initialized")
            return False
            
        try:
            self.socket.sendto(command.encode('utf-8'), (self.client_ip, self.udp_port))
            return True
        except Exception as e:
            print(f"❌ Failed to send command: {e}")
            return False
    
    def cleanup(self):
        """Clean up UDP socket"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None

class NetworkedAIProcessor:
    """Main class that integrates frame reception, AI processing, and command sending"""
    
    def __init__(self, client_ip: str, model_path: str, use_gpu: bool, nb_cpu_threads: int):
        self.client_ip = client_ip
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.nb_cpu_threads = nb_cpu_threads
        
        # Initialize components
        self.frame_receiver = FrameReceiver(TCP_PORT)
        self.command_sender = CommandSender(client_ip, UDP_PORT)
        self.ai_model = None
        
        # State variables
        self.last_frame = None
        self.last_prediction = "No AI model loaded"
        self.last_confidence = {}
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.total_frames_received = 0
        
        # Cooldown mechanism to prevent command spam
        self.last_command_time = 0.0
        self.command_cooldown = 0.5  # 0.5 second cooldown like app.py
        self.cooldown_log_counter = 0  # Reduce cooldown message spam
        self.last_skill_check_type = None  # Track last skill check type for cooldown
        
        # Group similar skill check types to prevent cooldown bypass
        self.skill_check_groups = {
            'repair-heal': ['repair-heal (great)', 'repair-heal (ante-frontier)', 'repair-heal (out)'],
            'full black': ['full black (great)', 'full black (out)'],
            'wiggle': ['wiggle (great)', 'wiggle (out)'],
            'struggle': ['struggle (great)', 'struggle (out)'],
            'run': ['run (great)', 'run (out)'],
            'merc': ['merc (great)', 'merc (out)']
        }
        
        # Confidence threshold to prevent false positives
        self.min_confidence = 0.95  # Only act on 95%+ confidence detections
        
        # Bandwidth monitoring
        self.bandwidth_counter = 0
        self.bandwidth_start_time = time.time()
        self.current_bandwidth = 0.0
        
        # Set frame callback
        self.frame_receiver.set_frame_callback(self._process_frame)
        
        # Don't load AI model yet - wait for configuration via UI
        self.ai_model = None
    
    def _get_skill_check_group(self, desc: str) -> str:
        """Get the base skill check group for a description"""
        for group_name, variants in self.skill_check_groups.items():
            if desc in variants:
                return group_name
        # If no group found, return the description as-is (fallback)
        return desc
    
    def _load_ai_model(self):
        """Load the AI model"""
        if not self.model_path:
            print("⚠️ No AI model path specified, skipping model loading")
            self.ai_model = None
            return
            
        try:
            print(f"🤖 Loading AI model: {self.model_path}")
            
            # Create AI model without starting the monitor (we'll use network frames instead)
            # Note: monitor_id doesn't matter since we disable the monitor immediately
            self.ai_model = AI_model(self.model_path, self.use_gpu, self.nb_cpu_threads, monitor_id=0)
            
            # Stop the monitor immediately since we don't need it for network frames
            if hasattr(self.ai_model, 'monitor') and self.ai_model.monitor:
                print(f"🔧 Stopping local monitor: {type(self.ai_model.monitor)}")
                self.ai_model.monitor.stop()
                self.ai_model.monitor = None
                print("✅ Disabled local monitor (using network frames)")
            else:
                print("⚠️ No monitor found to disable")
            
            execution_provider = self.ai_model.check_provider()
            print(f"✅ AI model loaded successfully using {execution_provider}")
        except Exception as e:
            print(f"❌ Failed to load AI model: {e}")
            self.ai_model = None
    
    def _process_frame(self, frame: np.ndarray):
        """Process received frame with AI model"""
        # Store frame for UI display (this is the frame from Client PC)
        self.last_frame = frame.copy()
        self.total_frames_received += 1
        
        # Update bandwidth counter
        self.bandwidth_counter += 1
        
        # Only log every 1000 frames to reduce spam (much cleaner)
        if self.total_frames_received % 1000 == 0:
            # Calculate bandwidth (assuming compressed frames)
            elapsed = time.time() - self.bandwidth_start_time
            if elapsed >= 5.0:  # Show bandwidth every 5 seconds instead of every second
                # Estimate received bandwidth (compressed frames are smaller)
                compression_ratio = 0.4  # 80% JPEG quality = ~40% size
                self.current_bandwidth = (self.bandwidth_counter * 224 * 224 * 3 * 8 * compression_ratio) / (1024 * 1024) / elapsed
                self.bandwidth_counter = 0
                self.bandwidth_start_time = time.time()
            
            print(f"📸 Frame #{self.total_frames_received} | 📊 BW: {self.current_bandwidth:.1f} Mbps")
        
        if self.ai_model is None:
            # Update FPS counter even without AI model
            self.fps_counter += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed >= 1.0:
                self.current_fps = self.fps_counter / elapsed
                self.fps_counter = 0
                self.fps_start_time = time.time()
            return
        
        try:
            # Process frame with AI model
            pred, desc, probs, should_hit = self.ai_model.predict(frame)
            
            # Store results
            self.last_prediction = desc
            self.last_confidence = probs
            
            # Update FPS counter
            self.fps_counter += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed >= 1.0:
                self.current_fps = self.fps_counter / elapsed
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Send command if performance event detected and cooldown has passed
            if should_hit:
                current_time = time.time()
                
                # Get confidence for this detection
                detection_confidence = probs.get(desc, 0.0)
                
                # Skip low-confidence detections to prevent false positives
                if detection_confidence < self.min_confidence:
                    # Only log every 50th low confidence detection to reduce spam
                    self.cooldown_log_counter += 1
                    if self.cooldown_log_counter % 50 == 1:
                        print(f"⚠️ Low confidence detection '{desc}' ({detection_confidence:.3f}) - ignoring to prevent false positives [showing 1 of {self.cooldown_log_counter}]")
                    return
                
                # Get the skill check group (e.g., 'repair-heal' for 'repair-heal (great)')
                current_group = self._get_skill_check_group(desc)
                last_group = self._get_skill_check_group(self.last_skill_check_type) if self.last_skill_check_type else None
                
                # Check if this is a different skill check GROUP (bypass cooldown) or same group (respect cooldown)
                is_different_group = (current_group != last_group)
                
                if is_different_group or (current_time - self.last_command_time >= self.command_cooldown):
                    # Clean, minimal command logging
                    success = self.command_sender.send_command("OPTIMIZE_NOW")
                    if success:
                        print(f"🎯 {desc} - Command sent ✅")
                        self.last_command_time = current_time  # Update cooldown timer
                        self.last_skill_check_type = desc  # Update skill check type
                        self.cooldown_log_counter = 0  # Reset spam counter
                    else:
                        print(f"❌ Failed to send command to {self.command_sender.client_ip}:{self.command_sender.udp_port}")
                else:
                    # Still detecting same skill check GROUP in cooldown period - no logging needed
                    self.cooldown_log_counter += 1
                    # Removed cooldown spam messages for cleaner output
            # Only log non-None predictions occasionally to reduce spam
            elif desc != "None" and self.total_frames_received % 300 == 0:  # Much less frequent
                print(f"📊 No action needed for: {desc} (confidence: {probs.get(desc, 0):.3f})")
                
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
    
    def start(self):
        """Start the networked AI processing system"""
        print("🚀 Starting Detection PC system...")
        
        # Initialize UDP sender
        if not self.command_sender.connect():
            print("❌ Failed to initialize UDP sender")
            return False
        
        # Start frame receiver
        if not self.frame_receiver.start():
            print("❌ Failed to start frame receiver")
            return False
        
        print("✅ Detection PC system started!")
        print(f"   📡 UDP sender: {self.command_sender.client_ip}:{self.command_sender.udp_port}")
        print(f"   🎧 TCP receiver: Port {self.frame_receiver.tcp_port}")
        return True
    
    def stop(self):
        """Stop the networked AI processing system"""
        print("🛑 Stopping Detection PC system...")
        self.frame_receiver.stop()
        self.command_sender.cleanup()
        
        if self.ai_model:
            try:
                self.ai_model.cleanup()
            except:
                pass
            self.ai_model = None
        
        print("✅ Detection PC system stopped.")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status for UI"""
        return {
            "fps": self.current_fps,
            "last_frame": self.last_frame,
            "last_prediction": self.last_prediction,
            "last_confidence": self.last_confidence,
            "connected": self.frame_receiver.client_socket is not None,
            "ai_model_loaded": self.ai_model is not None,
            "system_ready": self.frame_receiver.running and self.command_sender.socket is not None,
            "total_frames": self.total_frames_received
        }
    
    def is_ready(self) -> bool:
        """Check if the system is ready to process frames"""
        return (self.frame_receiver.running and 
                self.command_sender.socket is not None and 
                self.ai_model is not None)

def create_gradio_interface(processor: NetworkedAIProcessor):
    """Create Gradio web interface"""
    
    def update_ui():
        """Update UI with current status"""
        status = processor.get_status()
        
        fps = status["fps"]
        frame = status["last_frame"]
        prediction = status["last_prediction"] or "No prediction yet"
        confidence = status["last_confidence"] or {}
        connected = status["connected"]
        ai_model_loaded = status["ai_model_loaded"]
        total_frames = status.get("total_frames", 0)
        
        # Debug: Print frame info occasionally
        if frame is not None and total_frames > 0 and total_frames % 120 == 0:
            print(f"🖼️ UI displaying frame: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
        
        # Format confidence as string
        if confidence:
            conf_str = "\n".join([f"{k}: {v:.3f}" for k, v in confidence.items()])
        else:
            conf_str = "No confidence data"
        
        # Determine connection status
        if connected:
            if ai_model_loaded:
                status_text = "🟢 Connected & AI Ready"
            else:
                status_text = "🟡 Connected (No AI Model)"
        else:
            if ai_model_loaded:
                status_text = "🟡 AI Ready (Waiting for Client PC)"
            else:
                status_text = "🟡 Ready to Start"
        
        # Add system status info
        if not status.get("system_ready", False):
            status_text += " - System Starting..."
        
        return fps, frame, prediction, conf_str, status_text, total_frames, processor.current_bandwidth
    
    def start_system(game_ip, model_path, device, cpu_threads):
        """Start the networked system"""
        try:
            if not model_path:
                return gr.Error("Please select an AI model file")
            
            use_gpu = (device == "GPU")
            nb_cpu_threads = int(cpu_threads)
            
            # Stop existing processor if running
            processor.stop()
            
            # Update processor configuration
            processor.client_ip = game_ip
            processor.model_path = model_path
            processor.use_gpu = use_gpu
            processor.nb_cpu_threads = nb_cpu_threads
            
            # Load AI model with new configuration
            processor._load_ai_model()
            
            # Start system
            if processor.start():
                return gr.Info("System started successfully!")
            else:
                return gr.Error("Failed to start system")
        except Exception as e:
            return gr.Error(f"Error starting system: {e}")
    
    def stop_system():
        """Stop the networked system"""
        try:
            processor.stop()
            return gr.Info("System stopped")
        except Exception as e:
            return gr.Error(f"Error stopping system: {e}")
    
    # Find available AI models
    models_folder = "models"
    model_files = []
    if os.path.exists(models_folder):
        model_files = [(f, f'{models_folder}/{f}') for f in os.listdir(models_folder) 
                      if f.endswith(".onnx") or f.endswith(".trt")]
    
    if not model_files:
        model_files = [("No models found", None)]
    else:
        model_files.insert(0, ("Select AI model", None))
    
    # Create interface
    with gr.Blocks(title="Performance Analysis - Analysis PC") as interface:
        gr.Markdown("<h1 style='text-align: center;'>Performance Analysis - Analysis PC</h1>")
        gr.Markdown("Networked AI processing system for gaming performance optimization")
        
        # Add helpful instructions
        with gr.Row():
            gr.Markdown("""
            **📋 Setup Instructions:**
            1. Enter the Client PC's IP address
            2. Select an AI model file from the models folder
            3. Choose CPU or GPU device
            4. Click "Start System"
            5. Start the Client PC script on the other machine
            """)
        
        with gr.Row():
            with gr.Column(variant="panel"):
                gr.Markdown("**System Configuration**")
                
                game_ip = gr.Textbox(
                    label="Client PC IP Address",
                    value=CLIENT_PC_IP,
                    placeholder="Enter Client PC IP address"
                )
                
                model_path = gr.Dropdown(
                    choices=model_files,
                    value=model_files[0][1] if model_files else None,
                    label="AI Model File"
                )
                
                device = gr.Radio(
                    choices=["CPU", "GPU"],
                    value="CPU",
                    label="Device"
                )
                
                cpu_threads = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=4,
                    step=1,
                    label="CPU Threads (CPU mode only)"
                )
                
                with gr.Row():
                    start_btn = gr.Button("Start System", variant="primary")
                    stop_btn = gr.Button("Stop System", variant="stop")
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                
                connection_status = gr.Textbox(
                    label="Connection Status",
                    value="🟡 Ready to Start",
                    interactive=False
                )
            
            with gr.Column(variant="panel"):
                gr.Markdown("**Live Monitoring**")
                
                fps_display = gr.Number(
                    label="Processing FPS",
                    value=0.0,
                    interactive=False
                )
                
                total_frames_display = gr.Number(
                    label="Total Frames Received",
                    value=0,
                    interactive=False
                )
                
                bandwidth_display = gr.Number(
                    label="Received Bandwidth (Mbps)",
                    value=0.0,
                    interactive=False
                )
                
                frame_display = gr.Image(
                    label="Last Received Frame",
                    height=300,
                    interactive=False
                )
                
                prediction_display = gr.Textbox(
                    label="Last Prediction",
                    value="No prediction yet",
                    interactive=False
                )
                
                confidence_display = gr.Textbox(
                    label="Confidence Scores",
                    value="No confidence data",
                    interactive=False,
                    lines=5
                )
        
        # Event handlers
        start_btn.click(
            fn=start_system,
            inputs=[game_ip, model_path, device, cpu_threads],
            outputs=None
        )
        
        stop_btn.click(
            fn=stop_system,
            inputs=None,
            outputs=None
        )
        
        # Add refresh button for troubleshooting
        refresh_btn.click(
            fn=lambda: update_ui(),
            outputs=[fps_display, frame_display, prediction_display, confidence_display, connection_status, total_frames_display, bandwidth_display]
        )
        
        # Auto-update UI
        interface.load(update_ui, outputs=[fps_display, frame_display, prediction_display, confidence_display, connection_status, total_frames_display, bandwidth_display])
        

    
    return interface

def main():
    """Main function to run the Detection PC system"""
    print("🤖 Performance Analysis - Analysis PC")
    print("=" * 50)
    
    # Configuration
    client_ip = input(f"Enter Client PC IP address [{CLIENT_PC_IP}]: ").strip()
    if not client_ip:
        client_ip = CLIENT_PC_IP
    
    print(f"\n📋 Configuration:")
    print(f"   Client PC: {client_ip}:{UDP_PORT}")
    print(f"   TCP Server: Port {TCP_PORT}")
    print()
    
    # Create processor
    processor = NetworkedAIProcessor(client_ip, None, False, 4)  # Will be configured via UI
    
    # Create and launch Gradio interface
    interface = create_gradio_interface(processor)
    
    try:
        print("🌐 Launching Gradio web interface...")
        interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Detection PC system...")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        processor.stop()
        print("✅ Detection PC system stopped.")

if __name__ == "__main__":
    main()
