# 🚀 Bandwidth Optimization Guide

## 📊 **Current Performance**
- **Accuracy**: 43 greats, 0 goods, 5 misses ✅
- **Current Bandwidth**: ~130+ Mbps
- **Target Bandwidth**: 50-80 Mbps

## ⚙️ **Configuration Options**

### **1. Frame Rate Reduction (Safest)**
```python
# In bandwidth_config.py
TARGET_FPS = 60                    # Was 120, now 60 (50% reduction)
FRAME_CAPTURE_INTERVAL = 1.0 / 60  # 16.67ms between frames
```

### **2. Smart Frame Skipping**
```python
SMART_FRAME_SKIPPING = True        # Enable intelligent frame skipping
CHANGE_THRESHOLD = 0.02            # 2% change threshold
MAX_STATIC_FRAMES = 10             # Max frames to skip when static
```

### **3. Threshold Tuning**
- **CHANGE_THRESHOLD**: 
  - `0.01` = Very sensitive (more frames, higher accuracy)
  - `0.02` = Balanced (recommended)
  - `0.05` = Less sensitive (fewer frames, lower bandwidth)

- **MAX_STATIC_FRAMES**:
  - `5` = Aggressive skipping (lower bandwidth)
  - `10` = Balanced (recommended)
  - `15` = Conservative skipping (higher bandwidth)

## 🔧 **Quick Tuning Guide**

### **For Lower Bandwidth (40-60 Mbps):**
```python
TARGET_FPS = 45
CHANGE_THRESHOLD = 0.03
MAX_STATIC_FRAMES = 8
```

### **For Balanced Performance (60-80 Mbps):**
```python
TARGET_FPS = 60
CHANGE_THRESHOLD = 0.02
MAX_STATIC_FRAMES = 10
```

### **For Maximum Accuracy (80+ Mbps):**
```python
TARGET_FPS = 90
CHANGE_THRESHOLD = 0.015
MAX_STATIC_FRAMES = 12
```

## 📈 **Expected Results**

| Setting | Bandwidth | Accuracy Impact | Recommendation |
|---------|-----------|----------------|----------------|
| 60 FPS + Smart Skip | 50-70 Mbps | Minimal | ✅ **Start Here** |
| 45 FPS + Smart Skip | 30-50 Mbps | Low | ⚠️ Test First |
| 90 FPS + Smart Skip | 70-90 Mbps | None | 🟡 If bandwidth allows |

## 🧪 **Testing Process**

1. **Start with current settings** (60 FPS + Smart Skip)
2. **Monitor accuracy** for 10-20 skill checks
3. **If accuracy drops**: Increase FPS or lower threshold
4. **If bandwidth too high**: Increase threshold or lower FPS
5. **Fine-tune** until you find the sweet spot

## 🚨 **Warning Signs**

- **Too many "goods"**: Lower CHANGE_THRESHOLD or increase FPS
- **Too many "misses"**: Lower CHANGE_THRESHOLD or increase FPS
- **High bandwidth**: Increase CHANGE_THRESHOLD or decrease FPS

## 💡 **Pro Tips**

1. **Monitor the console output** for bandwidth status
2. **Green bandwidth** = Optimal
3. **Yellow bandwidth** = Acceptable
4. **Red bandwidth** = Too high, adjust settings
5. **Test in actual gameplay** before finalizing settings

## 🔄 **Quick Reset**

If you mess up the settings, just delete `bandwidth_config.py` and restart - it will use safe defaults!
