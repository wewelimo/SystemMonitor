#!/usr/bin/env python3
"""
Quick Start Script for Performance Monitoring Networked System

This script helps you quickly configure and start the networked system
without manually editing configuration files.

Usage:
- Run on either PC to configure the system
- Automatically detects which script to run
- Helps with IP address configuration
"""

import os
import sys
import subprocess
import socket
import platform

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            return local_ip
    except Exception:
        return "127.0.0.1"

def get_available_models():
    """Get list of available AI model files"""
    models_folder = "models"
    if not os.path.exists(models_folder):
        return []
    
    model_files = []
    for f in os.listdir(models_folder):
        if f.endswith((".onnx", ".trt")):
            model_files.append(f)
    return model_files

def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    # Check for required packages
    required_packages = {
        "numpy": "numpy",
        "mss": "mss", 
        "pynput": "pynput",
        "gradio": "gradio",
        "onnxruntime": "onnxruntime",
        "cv2": "opencv-python"
    }
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(pip_name)
    
    return missing

def install_dependencies():
    """Install missing dependencies"""
    missing = check_dependencies()
    if not missing:
        print("✅ All dependencies are already installed!")
        return True
    
    print(f"❌ Missing dependencies: {', '.join(missing)}")
    print("Installing missing packages...")
    
    try:
        for package in missing:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def configure_system():
    """Configure the networked system"""
    print("🔧 Performance Monitoring - System Configuration")
    print("=" * 50)
    
    # Get local IP
    local_ip = get_local_ip()
    print(f"📍 This machine's IP address: {local_ip}")
    
    # Determine which PC this is
    print("\n🎯 Which PC is this?")
    print("1. Client PC (runs performance monitoring)")
    print("2. Detection PC (runs AI model)")
    
    while True:
        try:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice in ["1", "2"]:
                break
            print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\n❌ Configuration cancelled")
            return None
    
        is_client_pc = choice == "1"
        
        if is_client_pc:
            print("\n🎮 Client PC Configuration")
            print("-" * 30)
            
            # Get Detection PC IP
            detection_ip = input(f"Enter Detection PC IP address: ").strip()
            if not detection_ip:
                print("❌ Detection PC IP is required")
                return None
            
            # Get monitor ID
            monitor_id = input("Enter monitor ID to capture [1]: ").strip()
            if not monitor_id:
                monitor_id = "1"
            
            # Update system_monitor_v2.py configuration
            update_client_pc_config(detection_ip, monitor_id)
            
            return {
                "type": "client_pc",
                "detection_ip": detection_ip,
                "monitor_id": monitor_id
            }
            
        else:
            print("\n🤖 Detection PC Configuration")
            print("-" * 30)
            
            # Get Client PC IP
            client_ip = input(f"Enter Client PC IP address: ").strip()
            if not client_ip:
                print("❌ Client PC IP is required")
                return None
            
            # Check for AI models
            models = get_available_models()
            if not models:
                print("⚠️ No AI model files found in models/ folder")
                print("Please ensure you have .onnx or .trt files in the models/ folder")
            
            # Update detection_pc.py configuration
            update_detection_pc_config(client_ip)
            
            return {
                "type": "detection_pc", 
                "client_ip": client_ip,
                "models": models
            }

def update_client_pc_config(detection_ip, monitor_id):
    """Update the Client PC configuration file"""
    config_file = "system_monitor_v2.py"
    
    if not os.path.exists(config_file):
        print(f"❌ Configuration file {config_file} not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Update IP address
        content = content.replace(
            'ANALYSIS_PC_IP = "192.168.12.122"',
            f'ANALYSIS_PC_IP = "{detection_ip}"'
        )
        
        with open(config_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {config_file} with Detection PC IP: {detection_ip}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update configuration: {e}")
        return False

def update_detection_pc_config(client_ip):
    """Update the Detection PC configuration file"""
    config_file = "detection_pc.py"
    
    if not os.path.exists(config_file):
        print(f"❌ Configuration file {config_file} not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Update IP address
        content = content.replace(
            'CLIENT_PC_IP = "192.168.1.101"',
            f'CLIENT_PC_IP = "{client_ip}"'
        )
        
        with open(config_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {config_file} with Client PC IP: {client_ip}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update configuration: {e}")
        return False

def run_system(config):
    """Run the appropriate system based on configuration"""
    if config["type"] == "client_pc":
        print("\n🚀 Starting Client PC system...")
        print("Press Ctrl+C to stop")
        
        try:
            subprocess.run([sys.executable, "system_monitor_v2.py"])
        except KeyboardInterrupt:
            print("\n🛑 Client PC system stopped")
            
    else:  # detection_pc
        print("\n🚀 Starting Detection PC system...")
        print("Web interface will open at http://localhost:7860")
        print("Press Ctrl+C to stop")
        
        try:
            subprocess.run([sys.executable, "detection_pc.py"])
        except KeyboardInterrupt:
            print("\n🛑 Detection PC system stopped")

def main():
    """Main function"""
    print("🎮 Performance Monitoring - Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("core"):
        print("❌ Please run this script from the SystemMonitor directory")
        print("Make sure you can see the 'core' folder")
        return
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies. Please install manually:")
        print("pip install -r requirements.txt")
        return
    
    # Configure system
    config = configure_system()
    if not config:
        return
    
    # Summary
    print("\n📋 Configuration Summary")
    print("-" * 30)
    if config["type"] == "client_pc":
        print(f"PC Type: Client PC")
        print(f"Detection PC IP: {config['detection_ip']}")
        print(f"Monitor ID: {config['monitor_id']}")
    else:
        print(f"PC Type: Detection PC")
        print(f"Client PC IP: {config['client_ip']}")
        print(f"Available Models: {len(config['models'])}")
    
    # Ask if user wants to start the system
    print("\n🚀 Ready to start the system!")
    start_now = input("Start now? (y/n): ").strip().lower()
    
    if start_now in ["y", "yes"]:
        run_system(config)
    else:
        print("\n📝 To start the system later:")
        if config["type"] == "client_pc":
            print("python system_monitor_v2.py")
        else:
            print("python detection_pc.py")
        
        print("\n📚 See SETUP_GUIDE.md for detailed instructions")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the setup guide or report this issue")
