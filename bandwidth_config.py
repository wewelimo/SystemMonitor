#!/usr/bin/env python3
"""
OPTIMIZED Bandwidth Configuration for Performance Monitoring System

This configuration maintains 60 FPS and accuracy while reducing bandwidth.
Uses aggressive compression and smart frame skipping for optimal performance.
"""

# Frame Rate Settings
TARGET_FPS = 60                    # Target frames per second
FRAME_CAPTURE_INTERVAL = 1.0 / 60  # Capture interval in seconds

# Smart Frame Skipping Settings (Aggressive but Smart)
SMART_FRAME_SKIPPING = True        # Enable intelligent frame skipping
CHANGE_THRESHOLD = 0.01           # Balanced sensitivity - Skip more frames but not too aggressive
MAX_STATIC_FRAMES = 6              # Moderate skipping - Skip longer but maintain responsiveness

# Advanced Settings
COMPRESSION_QUALITY = 80           # Lower JPEG quality for smaller files (was 95)
RESIZE_FACTOR = 1.0                # Frame resize factor

# Bandwidth Targets (Optimized for 60 FPS)
TARGET_BANDWIDTH_MBPS = 40         # Target bandwidth (was 80) - Much lower!
MAX_BANDWIDTH_MBPS = 60            # Maximum allowed bandwidth (was 120) - Much lower!

# Performance Monitoring
ENABLE_BANDWIDTH_MONITORING = True
ENABLE_FRAME_ANALYSIS = True

# Adaptive Mode Settings
HIGH_PRIORITY_FRAMES = 30          # Shorter high priority for faster response (was 45)
