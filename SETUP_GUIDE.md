# Performance Monitoring - Networked System Setup Guide

This guide explains how to set up and run the two-PC networked system for performance monitoring and optimization.

## System Overview

The networked system consists of two components:

1. **Client PC**: Captures performance frames and sends them over TCP to Detection PC
2. **Detection PC**: Receives frames, processes them with AI, and sends UDP commands back

## Network Architecture

```
Client PC (192.168.1.101)          Detection PC (192.168.1.100)
     |                                    |
     |                                    |
     |-- TCP 6000 (frames) ------------> |
     |                                    |
     |<-- UDP 6001 (commands) -----------|
```

- **TCP Port 6000**: Reliable frame transmission (Client PC → Detection PC)
- **UDP Port 6001**: Low-latency commands (Detection PC → Client PC)

## Prerequisites

### Both PCs Need:
- Python 3.8+ installed
- Both PCs on same LAN/subnet
- Network connectivity between PCs

### Client PC Requirements:
- Performance monitoring application running
- Monitor access for screen capture
- Python packages: `mss`, `pynput`, `numpy`

### Detection PC Requirements:
- AI model files in `models/` folder
- Python packages: `gradio`, `onnxruntime`, `numpy`, `opencv-python`
- Sufficient CPU/GPU for AI inference

## Installation

### 1. Install Python Dependencies

**Client PC:**
```bash
pip install mss pynput numpy
```

**Detection PC:**
```bash
pip install gradio onnxruntime numpy opencv-python
```

### 2. Verify AI Models

Ensure the `models/` folder contains your AI model files:
- `model.onnx` (ONNX format)
- Or `model.trt` (TensorRT format)

### 3. Configure IP Addresses

Edit the IP addresses in both scripts:

**Game PC (`game_pc.py`):**
```python
DETECTION_PC_IP = "192.168.1.100"  # Change to your Detection PC's IP
```

**Detection PC (`detection_pc.py`):**
```python
CLIENT_PC_IP = "192.168.1.101"  # Change to your Client PC's IP
```

## Network Testing

Before running the full system, test network connectivity:

### 1. Test Network Connectivity

**On Detection PC:**
```bash
python network_test.py server
```

**On Game PC:**
```bash
python network_test.py client
```

Enter the Detection PC's IP when prompted.

### 2. Expected Results

If successful, you should see:
- ✅ TCP connection successful
- ✅ UDP connection successful
- Messages received on both sides

## Firewall Configuration

### Windows Firewall

**Detection PC (TCP 6000 inbound):**
1. Open Windows Defender Firewall
2. Click "Advanced settings"
3. Select "Inbound Rules" → "New Rule"
4. Choose "Port" → "TCP" → "Specific local ports: 6000"
5. Allow the connection
6. Apply to all profiles
7. Name: "DBD Detection PC TCP"

**Game PC (UDP 6001 inbound):**
1. Follow same steps but choose "UDP" and port "6001"
2. Name: "DBD Game PC UDP"

### Alternative: Command Line

**Detection PC (run as Administrator):**
```cmd
netsh advfirewall firewall add rule name="DBD Detection PC TCP" dir=in action=allow protocol=TCP localport=6000
```

**Game PC (run as Administrator):**
```cmd
netsh advfirewall firewall add rule name="DBD Game PC UDP" dir=in action=allow protocol=UDP localport=6001
```

## Running the System

### 1. Start Detection PC First

**On Detection PC:**
```bash
python detection_pc.py
```

This will:
- Start TCP server on port 6000
- Launch Gradio web interface on http://localhost:7860
- Wait for Game PC connection

### 2. Start Game PC

**On Game PC:**
```bash
python system_monitor_v2.py
```

This will:
- Connect to Detection PC
- Start capturing frames
- Listen for UDP commands

### 3. Configure via Web Interface

1. Open http://localhost:7860 in your browser
2. Enter Client PC IP address
3. Select AI model file
4. Choose device (CPU/GPU)
5. Click "Start System"

## Configuration Options

### Frame Capture Settings

**Game PC (`game_pc.py`):**
```python
FRAME_CAPTURE_INTERVAL = 1.0 / 30  # 30 FPS capture rate
```

### AI Model Settings

**Detection PC (via web UI):**
- **Device**: CPU or GPU
- **CPU Threads**: 1-8 (CPU mode only)
- **Model File**: Select from models folder

### Network Settings

**Ports (configurable in both scripts):**
```python
TCP_PORT = 6000    # Frame transmission
UDP_PORT = 6001    # Command transmission
```

## Troubleshooting

### Common Issues

#### 1. Connection Refused
**Symptoms:** "Failed to connect to Detection PC"
**Solutions:**
- Verify Detection PC is running
- Check IP address is correct
- Ensure firewall allows TCP 6000 inbound on Detection PC
- Test with `network_test.py`

#### 2. UDP Commands Not Received
**Symptoms:** No key presses happening
**Solutions:**
- Check firewall allows UDP 6001 inbound on Client PC
- Verify Client PC IP in Detection PC configuration
- Test UDP connectivity with `network_test.py`

#### 3. High Latency
**Symptoms:** Delayed skill check responses
**Solutions:**
- Reduce frame capture FPS (increase `FRAME_CAPTURE_INTERVAL`)
- Use GPU mode if available
- Check network quality between PCs
- Ensure both PCs on same subnet

#### 4. AI Model Loading Failed
**Symptoms:** "Failed to load AI model"
**Solutions:**
- Verify model file exists in `models/` folder
- Check model file format (ONNX or TensorRT)
- Ensure all dependencies installed
- Check GPU drivers if using GPU mode

### Debug Mode

Enable verbose logging by modifying the scripts:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Network Diagnostics

**Check connectivity:**
```bash
# On Game PC
ping 192.168.1.100

# On Detection PC  
ping 192.168.1.101
```

**Check ports:**
```bash
# On Detection PC
netstat -an | findstr :6000

# On Game PC
netstat -an | findstr :6001
```

## Performance Optimization

### Frame Rate Tuning

**For lower latency:**
- Reduce `FRAME_CAPTURE_INTERVAL` (higher FPS)
- Use GPU mode if available
- Optimize network between PCs

**For lower CPU usage:**
- Increase `FRAME_CAPTURE_INTERVAL` (lower FPS)
- Use CPU mode with fewer threads
- Reduce frame resolution

### Network Optimization

**Best practices:**
- Use wired Ethernet connection
- Ensure both PCs on same switch/router
- Avoid VPN or complex network routing
- Monitor network latency with ping

## Integration with Existing Code

### How It Works

The networked system integrates with the existing `dbd_autoSkillCheck` codebase:

1. **Frame Input**: Instead of local screen capture, frames come over TCP
2. **AI Processing**: Uses existing `AI_model.py` unchanged
3. **Key Output**: Instead of local key press, sends UDP command

### Key Changes Made

- **Game PC**: Replaces `Monitoring_mss` with network frame sender
- **Detection PC**: Replaces local capture with TCP frame receiver
- **AI Model**: No changes needed - processes frames the same way
- **Key Press**: Replaces local `PressKey` with UDP command

### Extending the System

**Add new commands:**
1. Modify `CommandListener` in `system_monitor_v2.py`
2. Add new command types in `CommandSender` in `detection_pc.py`
3. Update command handling logic

**Add new frame processing:**
1. Extend `_process_frame` in `detection_pc.py`
2. Add new AI model outputs
3. Implement additional decision logic

## Security Considerations

### Network Security

- **Local Network Only**: System designed for local LAN use
- **No Authentication**: Basic network-level security only
- **Port Exposure**: Only necessary ports exposed
- **Firewall Rules**: Restrict to specific IP addresses if needed

### Enhanced Security (Optional)

**IP Whitelisting:**
```python
ALLOWED_IPS = ["192.168.1.101"]  # Only allow specific Client PC

def is_allowed_ip(ip):
    return ip in ALLOWED_IPS
```

**Command Validation:**
```python
VALID_COMMANDS = ["PRESS_SPACE"]  # Only allow specific commands

def is_valid_command(cmd):
    return cmd in VALID_COMMANDS
```

## Support and Maintenance

### Log Files

Enable logging to file:
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dbd_network.log'),
        logging.StreamHandler()
    ]
)
```

### Monitoring

**Key metrics to monitor:**
- Frame processing FPS
- Network latency
- Connection stability
- AI model performance
- Error rates

### Updates

**Regular maintenance:**
- Update Python packages
- Check for AI model updates
- Monitor network performance
- Review firewall rules

## Conclusion

This networked system provides a robust, low-latency solution for distributed skill check detection. The modular design makes it easy to extend and maintain, while the comprehensive error handling ensures reliable operation.

For additional support or feature requests, refer to the main project repository or create issues for specific problems encountered during setup or operation.
