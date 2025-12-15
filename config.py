#!/usr/bin/env python3
"""
Configuration file for DBD Auto Skill Check Detection System
"""

# Network Configuration
NETWORK_CONFIG = {
    # Game PC Configuration
    "GAME_PC": {
        "DETECTION_IP": "192.168.1.100",  # IP of Detection PC
        "TCP_PORT": 6000,                 # Port for frame transmission
        "UDP_PORT": 6001,                 # Port for commands
    },
    
    # Detection PC Configuration
    "DETECTION_PC": {
        "GAME_IP": "192.168.1.50",       # IP of Game PC
        "TCP_PORT": 6000,                 # Port for frame reception
        "UDP_PORT": 6001,                 # Port for commands
        "GRADIO_HOST": "0.0.0.0",        # Host for Gradio interface
        "GRADIO_PORT": 7860,              # Port for Gradio interface
    }
}

# AI Model Configuration
MODEL_CONFIG = {
    "DEFAULT_MODEL_PATH": "models/model.onnx",
    "DEFAULT_DEVICE": "CPU",              # CPU or GPU
    "CROP_SIZE": 224,                     # Input image size
    "CONFIDENCE_THRESHOLD": 0.8,         # Default confidence threshold
    "AUTO_PRESS_ENABLED": False,          # Default auto-press state
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "TARGET_FPS": 60,                     # Target frame rate
    "FRAME_UPDATE_INTERVAL": 30,          # Update FPS every N frames
    "PROCESSING_TIME_WINDOW": 100,        # Keep last N processing times
    "STATUS_REFRESH_INTERVAL": 2,         # UI refresh interval (seconds)
}

# Screen Capture Configuration
CAPTURE_CONFIG = {
    "MONITOR_ID": 1,                      # Primary monitor
    "CENTER_REGION_SIZE": 224,            # Size of captured region
    "FRAME_FORMAT": "RGB",                # Frame color format
}

# Logging Configuration
LOGGING_CONFIG = {
    "LEVEL": "INFO",                      # Log level (DEBUG, INFO, WARNING, ERROR)
    "FORMAT": "%(asctime)s - %(levelname)s - %(message)s",
    "FILE": None,                         # Log file path (None for console only)
}

# Auto-press Configuration
AUTO_PRESS_CONFIG = {
    "ENABLED": False,                     # Default auto-press state
    "CONFIDENCE_THRESHOLD": 0.8,         # Minimum confidence for auto-press
    "KEY_PRESS_DURATION": 0.05,          # Duration of key press (seconds)
    "KEY": "space",                       # Key to press
}

# Validation functions
def validate_ip_address(ip):
    """Validate IP address format"""
    import re
    pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if not re.match(pattern, ip):
        return False
    
    parts = ip.split('.')
    return all(0 <= int(part) <= 255 for part in parts)

def validate_port(port):
    """Validate port number"""
    return isinstance(port, int) and 1024 <= port <= 65535

def validate_config():
    """Validate configuration values"""
    errors = []
    
    # Validate network config
    for pc_type, config in NETWORK_CONFIG.items():
        for key, value in config.items():
            if key.endswith('_IP'):
                if not validate_ip_address(value):
                    errors.append(f"Invalid IP address in {pc_type}.{key}: {value}")
            elif key.endswith('_PORT'):
                if not validate_port(value):
                    errors.append(f"Invalid port in {pc_type}.{key}: {value}")
    
    # Validate model config
    if not (0.0 <= MODEL_CONFIG["CONFIDENCE_THRESHOLD"] <= 1.0):
        errors.append("Confidence threshold must be between 0.0 and 1.0")
    
    # Validate performance config
    if MODEL_CONFIG["CROP_SIZE"] <= 0:
        errors.append("Crop size must be positive")
    
    if PERFORMANCE_CONFIG["TARGET_FPS"] <= 0:
        errors.append("Target FPS must be positive")
    
    return errors

# Get configuration with validation
def get_validated_config():
    """Get configuration with validation"""
    errors = validate_config()
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix these errors before running the system.")
        return None
    
    return {
        "NETWORK": NETWORK_CONFIG,
        "MODEL": MODEL_CONFIG,
        "PERFORMANCE": PERFORMANCE_CONFIG,
        "CAPTURE": CAPTURE_CONFIG,
        "LOGGING": LOGGING_CONFIG,
        "AUTO_PRESS": AUTO_PRESS_CONFIG,
    }

if __name__ == "__main__":
    # Test configuration
    config = get_validated_config()
    if config:
        print("Configuration is valid!")
        print(f"Game PC will connect to: {config['NETWORK']['GAME_PC']['DETECTION_IP']}:{config['NETWORK']['GAME_PC']['TCP_PORT']}")
        print(f"Detection PC will listen on: {config['NETWORK']['DETECTION_PC']['GRADIO_HOST']}:{config['NETWORK']['DETECTION_PC']['GRADIO_PORT']}")
    else:
        print("Configuration validation failed!")
