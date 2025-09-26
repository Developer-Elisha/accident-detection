
# 🚦 Real-Time Accident Detection

A **real-time accident detection system** using YOLOv8 and Streamlit.  
The system detects vehicles and people in videos or live camera feeds, tracks them, and identifies collisions (accidents). Detected objects are highlighted with **green boxes**, and collisions are highlighted with **red boxes**. Notifications appear in real-time when an accident is detected.

---

## Features

- ✅ Real-time detection of vehicles and people
- ✅ Live webcam support and video file upload
- ✅ Collision/accident detection
- ✅ Green boxes for normal objects, red boxes for accidents
- ✅ Live notifications for accidents
- ✅ JSON output with accident timestamps

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/accident-detection.git
cd accident-detection
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Download the YOLOv8 model (if not downloaded automatically):

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.0/yolov8n.pt
```

---

## Usage

Run the Streamlit app:

```bash
streamlit run live_detection_app.py
```

### Options in the App:

1. **Upload Video**: Upload a `.mp4`, `.avi`, or `.mov` file for accident detection.  
2. **Live Camera**: Detect accidents in real-time using your webcam.

---

## How It Works

1. **Object Detection**: Uses [YOLOv8](https://github.com/ultralytics/ultralytics) to detect vehicles (car, bus, motorbike, truck) and people.  
2. **Tracking**: `TinyTracker` tracks objects across frames using Intersection-over-Union (IoU).  
3. **Accident Detection**: Collisions are detected when two objects overlap significantly (ignores small overlaps like shadows).  
4. **Notifications**: Real-time alerts in Streamlit when an accident occurs.  
5. **Output**: Accident timestamps are saved to `accidents.json`.

---

## File Structure

```
accident-detection/
│
├─ live_detection_app.py   # Main Streamlit app
├─ yolov8n.pt             # YOLOv8 model (auto-download)
├─ accidents.json         # JSON output of detected accidents
├─ README.md              # Project documentation
├─ requirements.txt       # Install all required Packages
```

---

## Notes

- Make sure your camera works and is accessible by your system.  
- Adjust `MIN_OVERLAP_AREA` in `live_detection_app.py` to reduce false positives (like shadows).  
- Works best in well-lit environments.

---

## License

© 2025 Elisha Noel. All rights reserved.
