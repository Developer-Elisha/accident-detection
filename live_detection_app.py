import streamlit as st
import cv2
import numpy as np
import tempfile
import json
from ultralytics import YOLO
import torch

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

# --- Detection Function ---
FRAME_SKIP = 2           # process every 2nd frame for speed
MIN_OVERLAP_AREA = 500   # min area for collision to avoid false positives

def detect_accidents(video_source):
    # Load model
    model = YOLO("yolov8n.pt")
    if torch.cuda.is_available():
        model.to("cuda").fuse().half()  # GPU optimizations

    cap = cv2.VideoCapture(video_source)
    tracker = TinyTracker()
    accidents = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0

    stframe = st.empty()            
    accident_placeholder = st.empty()  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection only on every N-th frame
        if frame_idx % FRAME_SKIP == 0:
            results = model(frame, conf=0.25, verbose=False)[0]
            boxes = []
            for b, c in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
                if int(c) in [0, 2, 3, 5, 7]:
                    x1, y1, x2, y2 = map(int, b)
                    boxes.append((x1, y1, x2, y2))
            tracks = tracker.update(boxes)
        else:
            tracks = tracker.update([])

        # detect collisions
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

        # draw boxes
        for tid, box in tracks:
            x1, y1, x2, y2 = box
            color = (0, 0, 255) if tid in accident_ids else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        if new_accident:
            accident_placeholder.warning("⚠️ Accident detected! Be cautious!")

        frame_idx += 1

    cap.release()
    with open("accidents.json", "w") as f:
        json.dump(accidents, f)
    st.success("✅ Finished. Accident timestamps saved to accidents.json")

# --- Streamlit UI ---
st.title("🚦 Real-Time Accident Detection with Notifications")

mode = st.radio("Choose Mode:", ["Upload Video", "Live Camera"])

if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        detect_accidents(tfile.name)

elif mode == "Live Camera":
    if st.button("Start Live Camera"):
        detect_accidents(0)
