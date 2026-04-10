import streamlit as st
import cv2
import numpy as np
import tempfile
# import jsona
import time
import torch
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- Page Config ---
st.set_page_config(
    page_title="Watch Tower - Accident Detection",
    page_icon="Favicon.png",
    layout="wide"
)

# --- Logo and Title ---
st.image("Watch_Tower.png", width=120)
st.title("🚦 Real-Time Accident Detection")


# --- Simple Tracker ---
class TinyTracker:
    def __init__(self, iou_th=0.3, max_miss=10):
        self.iou_th = iou_th
        self.max_miss = max_miss
        self.next_id = 1
        self.tracks = {}

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        aa = (ax2 - ax1) * (ay2 - ay1)
        bb = (bx2 - bx1) * (by2 - by1)
        return inter / (aa + bb - inter + 1e-6)

    def update(self, boxes):
        results = []
        new_tracks = {}
        for b in boxes:
            assigned = False
            for tid, info in self.tracks.items():
                if self._iou(info['box'], b) > self.iou_th:
                    new_tracks[tid] = {'box': b}
                    results.append((tid, b))
                    assigned = True
                    break
            if not assigned:
                tid = self.next_id
                self.next_id += 1
                new_tracks[tid] = {'box': b}
                results.append((tid, b))
        self.tracks = new_tracks
        return results


# --- Constants ---
FRAME_SKIP = 2
MIN_OVERLAP_AREA = 500


# --- Detection Function for Uploaded Video ---
def detect_accidents(video_source, stop_flag):
    model = YOLO("yolov8n.pt")
    if torch.cuda.is_available():
        model.to("cuda")

    cap = cv2.VideoCapture(video_source)
    tracker = TinyTracker()
    accidents = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0

    stframe = st.empty()
    accident_placeholder = st.empty()

    while cap.isOpened() and not stop_flag():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            results = model(frame, conf=0.25, verbose=False)[0]
            boxes = []
            for b, c in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
                if int(c) in [0, 2, 3, 5, 7]:  # person, car, bike, bus, truck
                    x1, y1, x2, y2 = map(int, b)
                    boxes.append((x1, y1, x2, y2))
            tracks = tracker.update(boxes)
        else:
            tracks = tracker.update([])

        accident_ids = set()
        new_accident = False
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                id1, b1 = tracks[i]
                id2, b2 = tracks[j]
                x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                overlap_area = x_overlap * y_overlap
                if overlap_area > MIN_OVERLAP_AREA:
                    accident_ids.update([id1, id2])
                    if len(accidents) == 0 or frame_idx / fps - accidents[-1] > 1:
                        accidents.append(frame_idx / fps)
                        new_accident = True

        # Draw boxes
        for tid, box in tracks:
            x1, y1, x2, y2 = box
            color = (0, 0, 255) if tid in accident_ids else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        if new_accident:
            accident_placeholder.warning("⚠️ Accident detected!")

        frame_idx += 1
        time.sleep(0.01)

    cap.release()
    with open("accidents.json", "w") as f:
        json.dump(accidents, f)


# --- Streamlit UI ---
mode = st.radio("Choose Mode:", ["Upload Video", "Live Camera"])

# --- Upload Video Mode ---
if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        detect_accidents(tfile.name, stop_flag=lambda: False)


# --- Live Camera Mode (WebRTC Streaming) ---
elif mode == "Live Camera":
    st.markdown("### 🎥 Live Accident Detection (WebRTC)")

    model = YOLO("yolov8n.pt")
    if torch.cuda.is_available():
        model.to("cuda")

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.tracker = TinyTracker()
            self.last_accident_time = 0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # YOLO detection
            results = model(img, conf=0.25, verbose=False)[0]
            boxes = []
            for b, c in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
                if int(c) in [0, 2, 3, 5, 7]:
                    x1, y1, x2, y2 = map(int, b)
                    boxes.append((x1, y1, x2, y2))

            tracks = self.tracker.update(boxes)
            accident_ids = set()
            new_accident = False

            # Detect overlapping boxes
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    id1, b1 = tracks[i]
                    id2, b2 = tracks[j]
                    x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                    y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                    overlap_area = x_overlap * y_overlap
                    if overlap_area > MIN_OVERLAP_AREA:
                        accident_ids.update([id1, id2])
                        now = time.time()
                        if now - self.last_accident_time > 1:
                            self.last_accident_time = now
                            new_accident = True

            # Draw bounding boxes
            for tid, box in tracks:
                x1, y1, x2, y2 = box
                color = (0, 0, 255) if tid in accident_ids else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, f"ID {tid}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if new_accident:
                st.warning("⚠️ Accident detected!")

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="live",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )
